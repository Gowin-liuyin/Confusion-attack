"""Acceptance tests for F01: Gate/Attn candidates come from the public full table."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.evaluation.real_weight_attacks import (
    attn_public_candidates,
    gate_public_candidates,
)
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D, L, HEADS, HEAD_DIM = 20, 8, 2, 2, 4


class WeightSource:
    def __init__(self, clean: dict, deployment: dict) -> None:
        self.clean = clean
        self.deployment = deployment

    def get(self, name: str, side: str) -> np.ndarray:
        return (self.clean if side == "clean" else self.deployment)[name]


class SpyPublic(PublicDeployment):
    """A public view whose forbidden attributes raise if touched."""

    @property
    def perm(self):
        raise AssertionError("attack read the secret permutation")

    @property
    def inv_perm(self):
        raise AssertionError("attack read the inverse permutation")

    @property
    def secret_metadata(self):
        raise AssertionError("attack read secret metadata")


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(V, D)).astype(np.float32)
    perm = rng.permutation(V).astype(np.int64)
    E_dep = np.empty_like(E)
    E_dep[perm] = E

    clean: dict[str, np.ndarray] = {}
    deployment: dict[str, np.ndarray] = {}
    for layer in range(L):
        # HF convention: gate_proj is (intermediate, hidden), so the invariant
        # transposes it to (hidden, intermediate); the obfuscation permutes the
        # intermediate columns.
        gate = rng.normal(size=(6, D)).astype(np.float32)
        clean[f"model.layers.{layer}.mlp.gate_proj.weight"] = gate
        deployment[f"model.layers.{layer}.mlp.gate_proj.weight"] = gate[rng.permutation(6), :]
        q = rng.normal(size=(HEADS * HEAD_DIM, D)).astype(np.float32)
        k = rng.normal(size=(HEADS * HEAD_DIM, D)).astype(np.float32)
        clean[f"model.layers.{layer}.self_attn.q_proj.weight"] = q
        clean[f"model.layers.{layer}.self_attn.k_proj.weight"] = k
        deployment[f"model.layers.{layer}.self_attn.q_proj.weight"] = 3.0 * q
        deployment[f"model.layers.{layer}.self_attn.k_proj.weight"] = 2.0 * k

    view = SpyPublic(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=L, layers=(0, 1), max_len=None,
        tokenizer_id="fixture", E_server=E_dep, E_clean=E,
        weight_source=WeightSource(clean, deployment),
        allow_clean_weights=True,
    )
    # SpyPublic deliberately carries a booby-trapped ``perm`` property so any
    # attack-side read fails; the real view is checked by assert_public in
    # tests/unit/test_public_view.py.
    return view, E, E_dep, perm


def test_gate_candidates_rank_the_full_deployment_table():
    view, E, E_dep, perm = _fixture()
    types = np.array([0, 3, 5, 7, 9, 11], dtype=np.int64)
    out = gate_public_candidates(view, types, top_k=3)
    assert out["n_candidates"] == V
    for token in types:
        candidates = out["candidates"][int(token)]
        assert len(candidates) == 3
        distances = out["candidate_distances"][int(token)]
        assert distances == sorted(distances)
        assert candidates[0] == int(perm[token])
        assert all(0 <= row < V for row in candidates)


def test_attn_candidates_rank_the_full_deployment_table_without_secret_reads():
    view, E, E_dep, perm = _fixture(seed=1)
    types = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    out = attn_public_candidates(view, types, layers=(0, 1), top_k=3,
                                 pairing="adjacent", block_size=2, chunk=5)
    assert out["n_candidates"] == V
    assert out["n_chunks"] >= 4
    for token in types:
        candidates = out["candidates"][int(token)]
        assert candidates[0] == int(perm[token])
        assert all(0 <= row < V for row in candidates)


def test_candidate_set_is_not_the_secret_image():
    """With a different permutation the public candidates must be built from rows, not images."""
    view, E, E_dep, perm = _fixture(seed=2)
    types = np.array([1, 4], dtype=np.int64)
    out = gate_public_candidates(view, types, top_k=V)
    # A secret-image implementation would return exactly one candidate per type.
    for token in types:
        assert len(out["candidates"][int(token)]) == V


def test_gate_reverse_mapping_preserves_collisions_and_secondary_candidates(monkeypatch):
    from obfuscation_attacks.evaluation import real_weight_attacks

    view, _, _, _ = _fixture()
    def fake_candidates(*args, **kwargs):
        return {"candidates": {0: [5, 6], 1: [5, 7]},
                "candidate_distances": {0: [0.1, 0.3], 1: [0.2, 0.4]},
                "n_candidates": V, "invariant_dims": 4, "layers": [0], "missing_weight_layers": []}
    monkeypatch.setattr(real_weight_attacks, "gate_public_candidates", fake_candidates)
    out = real_weight_attacks.recover_mapping_gate_ia(view, n_types=2, top_k=2)
    assert out["mapping"][5] == 0
    assert [item["token_id"] for item in out["mapping_candidates"][5]] == [0, 1]
    assert out["mapping"][6] == 0 and out["mapping"][7] == 1
    assert out["collision_rows"] == 1


def test_full_table_match_rejects_missing_overlapping_or_nonfinite_chunks():
    from obfuscation_attacks.evaluation.real_weight_attacks import _topk_against_full_table

    query = np.zeros((1, 2))
    with pytest.raises(ValueError, match="do not cover"):
        _topk_against_full_table(query, iter([(0, np.zeros((2, 2)))]), 1, 3)
    with pytest.raises(ValueError, match="exactly once"):
        _topk_against_full_table(query, iter([(0, np.zeros((2, 2))), (1, np.zeros((2, 2)))]), 1, 4)
    with pytest.raises(ValueError, match="finite"):
        _topk_against_full_table(query, iter([(0, np.array([[np.nan, 1.0]]))]), 1, 1)


@pytest.mark.parametrize("attack", [gate_public_candidates, attn_public_candidates])
def test_ia_decodes_deployment_rows_against_full_clean_table(attack):
    view, _, _, perm = _fixture(seed=3)
    rows = np.array([0, 3, 11])
    out = attack(view, rows, reverse=True, top_k=V, chunk=3)
    assert out["n_candidates"] == V
    for row in rows:
        candidates = out["candidates"][int(row)]
        assert candidates[0] == np.argsort(perm)[row]
        assert set(candidates) == set(range(V))


def test_table9_recovers_from_public_wire_with_all_six_groups():
    from obfuscation_attacks.evaluation.real_table9 import _feature, recover_table9_from_wire

    view, clean_table, _, perm = _fixture(seed=7)
    rng = np.random.default_rng(19)
    source = view.weight_source
    # Covariant hidden-axis transform and independent token/FFN permutations.
    embed_transform = np.diag(np.linspace(0.5, 2, D))
    head_transform = np.diag(np.linspace(2, 0.5, D))
    view.E_server[perm] = clean_table @ embed_transform
    head = rng.normal(size=(V, D))
    source.clean["lm_head.weight"] = head
    source.deployment["lm_head.weight"] = (head @ np.linalg.inv(embed_transform))[np.argsort(perm)]
    for layer in range(L):
        prefix = f"model.layers.{layer}."
        ffn = rng.permutation(6)
        for kind in ("gate_proj", "up_proj"):
            name = prefix + "mlp." + kind + ".weight"
            weight = rng.normal(size=(6, D))
            source.clean[name] = weight
            source.deployment[name] = (weight @ np.linalg.inv(embed_transform))[ffn]
        name = prefix + "mlp.down_proj.weight"
        weight = rng.normal(size=(D, 6))
        source.clean[name] = weight
        source.deployment[name] = (embed_transform @ weight)[:, ffn]
        name = prefix + "mlp.gate.weight"
        weight = rng.normal(size=(4, D))
        source.clean[name] = weight
        router = weight / np.linalg.norm(weight, axis=1, keepdims=True)
        source.deployment[name] = (router @ np.linalg.inv(embed_transform))[rng.permutation(4)]
        for kind, scale in (("q_proj", 2.0), ("k_proj", 0.5)):
            name = prefix + "self_attn." + kind + ".weight"
            source.deployment[name] = scale * source.clean[name] @ np.linalg.inv(embed_transform)

    expected = np.array([1, 9, 17])
    wire = view.E_server[perm[expected]]
    out = recover_table9_from_wire(view, wire, top_k=3, chunk=4)
    np.testing.assert_array_equal(out["stage1_candidates"][:, 0], perm[expected])
    np.testing.assert_array_equal(out["composed_candidates"][:, 0], expected)
    assert out["n_candidates"] == V
    assert not out["missing_weight_groups"]
    assert len(out["used_weight_groups"]) == 1 + 5 * L
    for spec in out["used_weight_groups"]:
        group, layer = spec["group"], spec["layer"]
        clean = _feature(view, "clean", group, layer)
        deployed = _feature(view, "deployment", group, layer)
        np.testing.assert_allclose(
            np.sort(clean(expected), axis=1),
            np.sort(deployed(perm[expected]), axis=1),
            rtol=1e-5, atol=1e-5,
            err_msg=f"Table 9 group {group}, layer {layer}",
        )
