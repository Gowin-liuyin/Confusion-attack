"""Acceptance tests for S08: weight attacks enumerate model layers.

Gate-IA / Attn-IA used to read ``public.layers`` - the hidden-state *sampling*
tuple (e.g. ``(1, 5, 10, 15)``) - and thus saw only 4 of a 16-layer model.
These tests force the all-layer enumeration, record missing/unusable layers and
pin the chunk-size-independent tie ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.data.contracts import CapabilityError
from obfuscation_attacks.evaluation.real_weight_attacks import (
    ATTN_WEIGHTS,
    GATE_WEIGHT,
    attn_public_candidates,
    gate_public_candidates,
    weight_layers,
)
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit


class _WeightStore:
    def __init__(self, clean: dict, deployment: dict) -> None:
        self.clean = clean
        self.deployment = deployment

    def get(self, name: str, side: str) -> np.ndarray:
        return (self.clean if side == "clean" else self.deployment)[name]


def _public(*, n_layers: int, sampled: tuple, clean: dict, deployment: dict,
            E: np.ndarray, E_dep: np.ndarray, hidden: int,
            n_heads: int = 1, n_kv_heads: int = 1) -> PublicDeployment:
    return PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=int(E.shape[0]), hidden_size=hidden, n_layers=n_layers,
        layers=tuple(sampled), max_len=None, tokenizer_id="fixture",
        n_heads=n_heads, n_kv_heads=n_kv_heads,
        E_server=E_dep, E_clean=E,
        weight_source=_WeightStore(clean, deployment),
        allow_clean_weights=True,
    )


# ---------------------------------------------------------------------------
# Gate-IA fixture: layers 0-2 cannot separate tokens 0 and 1, layer 3 can.
# ---------------------------------------------------------------------------
def _gate_fixture(*, missing_layers: tuple = (), bad_shape_layers: tuple = (),
                  n_layers: int = 4,
                  sampled: tuple = (0, 1, 2)) -> tuple[PublicDeployment, np.ndarray]:
    hidden = 2
    E = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
    perm = np.array([3, 1, 0, 2], dtype=np.int64)
    E_dep = np.empty_like(E)
    E_dep[perm] = E
    gates = {
        0: np.eye(2, dtype=np.float32),
        1: np.eye(2, dtype=np.float32),
        2: np.eye(2, dtype=np.float32),
        3: np.array([[1, 0], [0, 2]], dtype=np.float32),
    }
    clean: dict[str, np.ndarray] = {}
    deployment: dict[str, np.ndarray] = {}
    for layer, gate in gates.items():
        if layer in missing_layers:
            continue
        if n_layers > 0 and layer >= n_layers:
            continue
        name = f"model.layers.{layer}.mlp.gate_proj.weight"
        clean[name] = np.zeros((3, 3), np.float32) if layer in bad_shape_layers else gate
        deployment[name] = clean[name]
    public = _public(
        n_layers=n_layers, sampled=sampled, clean=clean, deployment=deployment,
        E=E, E_dep=E_dep, hidden=hidden,
    )
    return public, perm


def test_gate_default_uses_every_model_layer():
    public, perm = _gate_fixture()  # sampled tuple (0, 1, 2) omits layer 3
    out = gate_public_candidates(public, [0, 1], top_k=1)
    assert out["layers"] == [0, 1, 2, 3]
    assert out["missing_weight_layers"] == []
    # Only layer 3 separates the two tokens; the sampling tuple cannot.
    assert out["candidates"][0] == [int(perm[0])]
    assert out["candidates"][1] == [int(perm[1])]


def test_gate_sampling_tuple_alone_cannot_recover():
    public, perm = _gate_fixture()
    out = gate_public_candidates(public, [0, 1], layers=(0, 1, 2), top_k=1)
    assert out["layers"] == [0, 1, 2]
    assert out["candidates"][0] != [int(perm[0])]  # exact tie, wrong row wins
    assert out["candidates"][0] == [1]


def test_gate_missing_layer_is_recorded_and_skipped():
    public, perm = _gate_fixture(missing_layers=(1,))
    out = gate_public_candidates(public, [0, 1], top_k=1)
    assert out["layers"] == [0, 2, 3]
    record = out["missing_weight_layers"][0]
    assert record["layer"] == 1
    assert record["code"] == "weight_missing"
    assert record["reason"]
    assert "gate_proj" in record["reason"]
    assert out["candidates"][0] == [int(perm[0])]


def test_gate_unusable_shape_is_recorded_not_silently_skipped():
    public, _ = _gate_fixture(bad_shape_layers=(2,))
    out = gate_public_candidates(public, [0], top_k=1)
    assert out["layers"] == [0, 1, 3]
    record = out["missing_weight_layers"][0]
    assert record["layer"] == 2
    assert record["code"] == "shape_invalid"
    assert "unusable" in record["reason"]
    assert "hidden axis" in record["reason"]


def test_gate_all_layers_missing_raises():
    public, _ = _gate_fixture(missing_layers=(0, 1, 2, 3))
    with pytest.raises(CapabilityError, match="no usable weight layer"):
        gate_public_candidates(public, [0], top_k=1)


def test_gate_explicit_layers_are_used_verbatim():
    public, perm = _gate_fixture()
    out = gate_public_candidates(public, [0], layers=(1,), top_k=1)
    assert out["layers"] == [1]
    assert out["missing_weight_layers"] == []
    # Only layer 1 participates: tokens 0 and 1 tie and the lower row wins.
    assert out["candidates"][0] == [min(int(perm[0]), int(perm[1]))]


def test_gate_explicit_layer_bypasses_the_availability_check():
    public, _ = _gate_fixture(missing_layers=(1,))
    with pytest.raises(KeyError):
        gate_public_candidates(public, [0], layers=(1,), top_k=1)


def test_gate_ranking_is_chunk_size_independent():
    public, _ = _gate_fixture()
    types = [0, 1, 2, 3]
    baseline = gate_public_candidates(public, types, top_k=4, chunk=None)
    for chunk in (1, 2, 3, 5):
        out = gate_public_candidates(public, types, top_k=4, chunk=chunk)
        assert out["candidates"] == baseline["candidates"]


# ---------------------------------------------------------------------------
# weight_layers enumeration / fallback
# ---------------------------------------------------------------------------
def test_weight_layers_enumerates_n_layers_not_the_sampling_tuple():
    public, _ = _gate_fixture(sampled=(1, 3))
    used, missing = weight_layers(public, required=(GATE_WEIGHT,))
    assert used == [0, 1, 2, 3]
    assert missing == []


def test_weight_layers_accepts_a_bare_suffix():
    public, _ = _gate_fixture()
    used, missing = weight_layers(public, required=("mlp.gate_proj.weight",))
    assert used == [0, 1, 2, 3]
    assert missing == []


def test_weight_layers_falls_back_to_sampled_layers_without_n_layers():
    public, _ = _gate_fixture(n_layers=0, sampled=(1, 3))
    used, missing = weight_layers(public, required=(GATE_WEIGHT,))
    assert used == [1, 3]
    assert missing == []


def test_weight_layers_needs_an_enumeration_source():
    public, _ = _gate_fixture(n_layers=0, sampled=())
    with pytest.raises(CapabilityError, match="cannot enumerate"):
        weight_layers(public, required=(GATE_WEIGHT,))


def test_weight_layers_derives_head_dim_when_the_view_has_none():
    """S08-3: build_public_view exposes no ``head_dim`` field, so the q/k shape
    check must fall back to ``hidden_size // n_heads`` from the config.

    The 2048/32/8 example has ``q_dim == k_dim == 32`` (the cross-check alone
    cannot reject it), but the frozen llama-3.2-1b head_dim is 64.
    """
    hidden, n_heads, n_kv_heads = 2048, 32, 8
    E = np.zeros((1, hidden), dtype=np.float32)

    def public_for(weights: dict) -> PublicDeployment:
        return _public(
            n_layers=1, sampled=(), clean=weights, deployment=weights,
            E=E, E_dep=E, hidden=hidden, n_heads=n_heads, n_kv_heads=n_kv_heads,
        )

    wrong = {
        "model.layers.0.self_attn.q_proj.weight": np.zeros((1024, hidden), np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((256, hidden), np.float32),
    }
    used, missing = weight_layers(public_for(wrong), required=ATTN_WEIGHTS)
    assert used == []
    assert [record["layer"] for record in missing] == [0]
    assert "head_dim" in missing[0]["reason"]

    correct = {  # frozen llama-3.2-1b: q (32*64, 2048), k (8*64, 2048)
        "model.layers.0.self_attn.q_proj.weight": np.zeros((2048, hidden), np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((512, hidden), np.float32),
    }
    assert weight_layers(public_for(correct), required=ATTN_WEIGHTS) == ([0], [])

    # A future view that does expose head_dim keeps precedence over the ratio.
    explicit = public_for(wrong)
    explicit.head_dim = 32
    assert weight_layers(explicit, required=ATTN_WEIGHTS) == ([0], [])


# ---------------------------------------------------------------------------
# Attn-IA fixture (V >> d_head so every RoPE Gram is invertible)
# ---------------------------------------------------------------------------
V, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM = 20, 8, 2, 1, 4


def _attn_fixture(*, n_layers: int = 4, missing_k_layers: tuple = (),
                  drop_all: bool = False) -> tuple[PublicDeployment, np.ndarray]:
    rng = np.random.default_rng(7)
    E = rng.normal(size=(V, HIDDEN)).astype(np.float32)
    perm = rng.permutation(V).astype(np.int64)
    E_dep = np.empty_like(E)
    E_dep[perm] = E
    clean: dict[str, np.ndarray] = {}
    deployment: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        if drop_all:
            continue
        q_name = f"model.layers.{layer}.self_attn.q_proj.weight"
        q = rng.normal(size=(N_HEADS * HEAD_DIM, HIDDEN)).astype(np.float32)
        clean[q_name] = q
        deployment[q_name] = 3.0 * q
        if layer in missing_k_layers:
            continue
        k_name = f"model.layers.{layer}.self_attn.k_proj.weight"
        k = rng.normal(size=(N_KV_HEADS * HEAD_DIM, HIDDEN)).astype(np.float32)
        clean[k_name] = k
        deployment[k_name] = 2.0 * k
    public = _public(
        n_layers=n_layers, sampled=(1,), clean=clean, deployment=deployment,
        E=E, E_dep=E_dep, hidden=HIDDEN, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
    )
    return public, perm


def test_attn_default_uses_every_model_layer():
    public, perm = _attn_fixture()  # sampled tuple (1,) only
    types = [0, 3, 7, 11]
    out = attn_public_candidates(public, types, top_k=1)
    assert out["layers"] == [0, 1, 2, 3]
    assert out["missing_weight_layers"] == []
    assert out["invariant_dims"] == 4 * 6  # 4 layers x (2 q heads + 1 k head) x 2 blocks
    for token in types:
        assert out["candidates"][int(token)] == [int(perm[token])]


def test_attn_layer_missing_key_weight_is_recorded_and_skipped():
    public, perm = _attn_fixture(missing_k_layers=(2,))
    out = attn_public_candidates(public, [0, 5], top_k=1)
    assert out["layers"] == [0, 1, 3]
    assert [record["layer"] for record in out["missing_weight_layers"]] == [2]
    assert "k_proj" in out["missing_weight_layers"][0]["reason"]
    assert out["invariant_dims"] == 3 * 6
    for token in (0, 5):
        assert out["candidates"][int(token)] == [int(perm[token])]


def test_attn_all_layers_missing_raises():
    public, _ = _attn_fixture(drop_all=True)
    with pytest.raises(CapabilityError, match="no usable weight layer"):
        attn_public_candidates(public, [0], top_k=1)


def test_attn_explicit_layers_are_used_verbatim():
    public, perm = _attn_fixture()
    out = attn_public_candidates(public, [0], layers=(2,), top_k=1)
    assert out["layers"] == [2]
    assert out["missing_weight_layers"] == []
    assert out["invariant_dims"] == 6  # one layer only
    assert out["candidates"][0] == [int(perm[0])]


def test_attn_rankings_do_not_depend_on_chunking():
    public, _ = _attn_fixture()
    types = [0, 2, 4, 6]
    baseline = attn_public_candidates(public, types, top_k=3, chunk=None)
    for chunk in (1, 3, 5):
        out = attn_public_candidates(public, types, top_k=3, chunk=chunk)
        assert out["candidates"] == baseline["candidates"]


# ---------------------------------------------------------------------------
# ATTN-IA-SILENT-LAYER-DROP: a layer with no invertible Gram block contributes
# zero invariant columns and must not be reported as covered.
# ---------------------------------------------------------------------------
def _attn_zero_fixture(*, zero_layers: tuple) -> tuple[PublicDeployment, np.ndarray]:
    V, hidden = 12, 4  # n_heads == n_kv_heads == 1 -> head_dim 4, two blocks/group
    rng = np.random.default_rng(11)
    E = rng.normal(size=(V, hidden)).astype(np.float32)
    perm = rng.permutation(V).astype(np.int64)
    E_dep = np.empty_like(E)
    E_dep[perm] = E
    clean: dict[str, np.ndarray] = {}
    deployment: dict[str, np.ndarray] = {}
    for layer in range(2):
        q = rng.normal(size=(hidden, hidden)).astype(np.float32)
        k = rng.normal(size=(hidden, hidden)).astype(np.float32)
        if layer in zero_layers:
            q = np.zeros_like(q)
            k = np.zeros_like(k)
        q_name = f"model.layers.{layer}.self_attn.q_proj.weight"
        k_name = f"model.layers.{layer}.self_attn.k_proj.weight"
        clean[q_name], deployment[q_name] = q, 3.0 * q
        clean[k_name], deployment[k_name] = k, 2.0 * k
    public = _public(
        n_layers=2, sampled=(1,), clean=clean, deployment=deployment,
        E=E, E_dep=E_dep, hidden=hidden, n_heads=1, n_kv_heads=1,
    )
    return public, perm


def test_attn_zero_weight_layer_is_dropped_and_recorded():
    public, perm = _attn_zero_fixture(zero_layers=(1,))
    out = attn_public_candidates(public, [0, 3], top_k=1)
    assert out["layers"] == [0]
    assert out["invariant_dims"] == 4  # only layer 0: 2 q blocks + 2 k blocks
    record = out["missing_weight_layers"][0]
    assert record["layer"] == 1
    assert record["code"] == "zero_columns"
    assert "0 of the invariant columns" in record["reason"]
    skipped = out["skipped_gram_blocks"]
    assert skipped and {entry["layer"] for entry in skipped} == {1}
    assert {entry["side"] for entry in skipped} == {"clean", "deployment"}
    # All-zero weights give a singular (non-finite condition) Gram block.
    assert {entry["code"] for entry in skipped} == {"singular_gram"}
    assert all(entry["reason"] for entry in skipped)
    for token in (0, 3):
        assert out["candidates"][int(token)] == [int(perm[token])]


def test_attn_all_zero_weight_layers_raise():
    public, _ = _attn_zero_fixture(zero_layers=(0, 1))
    with pytest.raises(CapabilityError, match="no query/key Gram block"):
        attn_public_candidates(public, [0], top_k=1)


def test_attn_partially_skipped_layer_is_kept_and_block_recorded():
    V, hidden = 12, 4
    rng = np.random.default_rng(13)
    E = rng.normal(size=(V, hidden)).astype(np.float32)
    perm = rng.permutation(V).astype(np.int64)
    E_dep = np.empty_like(E)
    E_dep[perm] = E
    q = rng.normal(size=(hidden, hidden)).astype(np.float32)
    q[[0, 2]] = 0.0  # projected columns 0 and 2 vanish -> exactly singular (0, 2) block
    k = rng.normal(size=(hidden, hidden)).astype(np.float32)
    clean = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
    }
    deployment = {
        "model.layers.0.self_attn.q_proj.weight": 3.0 * q,
        "model.layers.0.self_attn.k_proj.weight": 2.0 * k,
    }
    public = _public(
        n_layers=1, sampled=(0,), clean=clean, deployment=deployment,
        E=E, E_dep=E_dep, hidden=hidden, n_heads=1, n_kv_heads=1,
    )
    out = attn_public_candidates(public, [0, 3], top_k=1)
    # One q block of layer 0 is singular; the layer still contributes (k blocks
    # plus the surviving q block), so it stays in ``layers``.
    assert out["layers"] == [0]
    assert out["missing_weight_layers"] == []
    assert out["invariant_dims"] == 3
    skipped = {
        (record["side"], record["group"], tuple(record["block"]), record["code"])
        for record in out["skipped_gram_blocks"]
    }
    assert skipped == {
        ("clean", "q_proj", (0, 2), "singular_gram"),
        ("deployment", "q_proj", (0, 2), "singular_gram"),
    }
    for token in (0, 3):
        assert out["candidates"][int(token)] == [int(perm[token])]


def test_weight_layers_emits_side_and_head_dim_mismatch_codes():
    hidden = 0  # no config ratio and no head_dim: only the cross-check can flag
    q_name = "model.layers.0.self_attn.q_proj.weight"
    k_name = "model.layers.0.self_attn.k_proj.weight"
    E = np.zeros((1, 1), dtype=np.float32)

    def public_for(clean_q, deployment_q, k) -> PublicDeployment:
        return _public(
            n_layers=1, sampled=(0,),
            clean={q_name: clean_q, k_name: k},
            deployment={q_name: deployment_q, k_name: k},
            E=E, E_dep=E, hidden=hidden, n_heads=1, n_kv_heads=1,
        )

    side = public_for(
        np.zeros((8, 4), np.float32), np.zeros((6, 4), np.float32),
        np.zeros((8, 4), np.float32),
    )
    used, missing = weight_layers(side, required=ATTN_WEIGHTS)
    assert used == []
    assert missing[0]["code"] == "side_mismatch"
    assert missing[0]["reason"]

    head_dim = public_for(
        np.zeros((8, 4), np.float32), np.zeros((8, 4), np.float32),
        np.zeros((6, 4), np.float32),
    )
    used, missing = weight_layers(head_dim, required=ATTN_WEIGHTS)
    assert used == []
    assert missing[0]["code"] == "head_dim_mismatch"
    assert "head_dim" in missing[0]["reason"]
