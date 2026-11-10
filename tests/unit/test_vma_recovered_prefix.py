"""Acceptance tests for F04: VMA-H recovered-prefix is its own attack."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from obfuscation_attacks.evaluation.real_vma import (
    score_vma,
    vma_h_observed,
    vma_h_recovered,
)
from obfuscation_attacks.schemes.ab_deployment import WirePublic
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D = 8, 4


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(V, D)).astype(np.float32)
    W = rng.normal(size=(D, D)).astype(np.float32) + 2.0 * np.eye(D, dtype=np.float32)

    def forward(inputs_embeds):
        x = np.asarray(inputs_embeds, dtype=np.float32)
        single = x.ndim == 2
        if single:
            x = x[None]
        batch, length, _ = x.shape
        hidden = np.zeros((batch, length, D), dtype=np.float32)
        previous = np.zeros((batch, D), dtype=np.float32)
        for t in range(length):
            previous = x[:, t, :] @ W + 0.5 * previous
            hidden[:, t, :] = previous
        return SimpleNamespace(hidden_states=[x, hidden])

    public = PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E, forward_fn=forward,
        allow_clean_weights=True,
    )
    ids = np.array([2, 5, 1, 7], dtype=np.int64)
    hidden = forward(E[ids]).hidden_states[1]
    wire = WirePublic(
        arm="ab", request_id=0, seq_len=len(ids), wire_vectors=E[ids],
        hidden_states=hidden[None], layers=(1,),
    )
    return public, wire, ids, E


def test_recovered_prefix_recovers_the_sequence_and_includes_the_first_token():
    public, wire, ids, E = _fixture()
    out = vma_h_recovered(public, [wire], layers=(1,), full_vocab=True,
                          max_positions=len(ids), verbose=False)
    layer = out["layers"]["layer1"]
    assert out["prefix_mode"] == "recovered"
    assert out["first_token_included"] is True
    assert layer["recovered"][0] == ids.tolist()
    assert len(layer["candidates"][0][0]) == V
    scored = score_vma(out, ids)
    assert scored["layer1"]["total_top1"]["numerator"] == len(ids)
    assert scored["layer1"]["total_top1"]["denominator"] == len(ids)


def test_observed_prefix_is_a_different_named_variant():
    public, wire, ids, E = _fixture()
    out = vma_h_observed(public, [wire], layers=(1,), full_vocab=True,
                         max_positions=len(ids), verbose=False)
    assert out["prefix_mode"] == "observed"
    assert out["attack_id"] == "vma_h_observed_prefix"


def test_attacks_take_only_public_inputs():
    for fn in (vma_h_recovered, vma_h_observed):
        params = inspect.signature(fn).parameters
        assert "public" in params and "wires" in params
        assert "dep" not in params
        assert "truth" not in params
        assert "perm" not in params


def test_bos_pose_prepends_the_public_bos_row():
    public, wire, ids, E = _fixture()
    lengths: list[int] = []
    original = public.forward_fn

    def recording(inputs_embeds):
        array = np.asarray(inputs_embeds)
        lengths.append(int(array.shape[-2]) if array.ndim == 3 else int(array.shape[0]))
        return original(inputs_embeds)

    public.forward_fn = recording
    vma_h_recovered(public, [wire], layers=(1,), full_vocab=True,
                    max_positions=1, pose="empty", verbose=False)
    assert lengths[-1] == 1
    lengths.clear()
    out = vma_h_recovered(public, [wire], layers=(1,), full_vocab=True,
                          max_positions=1, pose="bos", bos_id=0, verbose=False)
    assert out["pose"] == "bos" and out["bos_id"] == 0
    assert lengths[-1] == 2


def test_bos_pose_without_a_public_id_is_refused():
    public, wire, ids, E = _fixture()
    with pytest.raises(ValueError, match="bos_id"):
        vma_h_recovered(public, [wire], layers=(1,), pose="bos", verbose=False)


def test_score_vma_rejects_a_single_truth_for_multiple_texts():
    public, wire, ids, E = _fixture()
    out = vma_h_recovered(public, [wire, wire], layers=(1,), full_vocab=True,
                          max_positions=2, verbose=False)
    with pytest.raises(ValueError, match="1-D truth"):
        score_vma(out, ids)
    scored = score_vma(out, np.stack([ids, ids]))
    # The attack decoded only ``max_positions=2`` per text, but the caller passed
    # the full 4-token truth rows: uncovered positions remain in the denominator.
    assert scored["layer1"]["total_top1"]["denominator"] == 8


def test_corrupting_the_wire_changes_the_recovery_not_a_secret_read():
    public, wire, ids, E = _fixture(seed=3)
    other = np.array([4, 0, 6, 3], dtype=np.int64)
    hidden = public.forward(E[other]).hidden_states[1]
    corrupted = WirePublic(
        arm="ab", request_id=0, seq_len=len(other), wire_vectors=E[other],
        hidden_states=hidden[None], layers=(1,),
    )
    out = vma_h_recovered(public, [corrupted], layers=(1,), full_vocab=True,
                          max_positions=len(other), verbose=False)
    assert out["layers"]["layer1"]["recovered"][0] == other.tolist()


def test_score_vma_rejects_a_2d_truth_row_count_mismatch():
    result = {
        "layers": {
            "layer1": {
                "candidates": [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])],
                "recovered": [[1, 3], [5]],
            }
        }
    }
    with pytest.raises(ValueError, match="truth rows"):
        score_vma(result, np.array([[1, 3], [5, 6], [7, 8]]))
    with pytest.raises(ValueError, match="truth rows"):
        score_vma(result, np.array([[1, 3]]))


def test_score_vma_keeps_uncovered_truth_positions_in_the_denominator():
    result = {
        "layers": {
            "layer1": {
                "candidates": [np.array([[1, 2], [3, 4]])],  # only 2 positions produced
                "recovered": [[1, 3]],
            }
        }
    }
    truth = [np.array([1, 3, 5, 0])]
    scored = score_vma(result, truth)
    block = scored["layer1"]["total_top1"]
    assert block["denominator"] == 4
    assert block["numerator"] == 2
    assert scored["layer1"]["first_error_position"] == 2


def test_score_vma_accepts_ragged_truth_and_per_text_clean_masks():
    result = {
        "layers": {
            "layer1": {
                "candidates": [
                    np.array([[1, 2], [3, 4]]),
                    np.array([[5, 6]]),
                ],
                "recovered": [[1, 3], [5]],
            }
        }
    }
    truth = [np.array([1, 3]), np.array([5])]
    masks = [np.array([True, False]), np.array([False])]
    scored = score_vma(result, truth, clean_mask=masks)
    assert scored["layer1"]["total_top1"]["denominator"] == 3
    assert scored["layer1"]["total_top1"]["numerator"] == 3
    assert scored["layer1"]["clean_top1"]["denominator"] == 1
    assert scored["layer1"]["clean_top1"]["numerator"] == 1
