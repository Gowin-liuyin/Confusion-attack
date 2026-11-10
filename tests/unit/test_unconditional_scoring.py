"""Acceptance tests for the unconditional denominator fix (F06, protocol §4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.evaluation.real_inversion import asr_at, ridge_inverter

pytestmark = pytest.mark.unit


def test_asr_at_keeps_oov_positions_in_the_denominator():
    candidates = np.array([[5, -1], [-1, -1]], dtype=np.int64)
    truth = np.array([5, 999], dtype=np.int64)
    result = asr_at(candidates, truth)
    assert result["top1"]["numerator"] == 1
    assert result["top1"]["denominator"] == 2
    assert result["top1"]["value"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["conditional_top1"]["numerator"] == 1
    assert result["conditional_top1"]["denominator"] == 1
    assert result["conditional_top1"]["value"] == pytest.approx(1.0)


def test_ridge_proposal_keeps_uncovered_test_labels():
    split = {
        "X_train": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "y_train": np.array([5, 5], dtype=np.int64),
        "X_test": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "y_test": np.array([5, 999], dtype=np.int64),
    }
    result = ridge_inverter(split, lam=0.01, n_classes=1, seed=0)
    assert result["top1"]["denominator"] == 2
    assert result["top1"]["numerator"] == 1
    assert result["top1"]["value"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["conditional_top1"]["denominator"] == 1
    assert result["candidate_mode"] == "proposal"
    assert result["n_classes"] == 1


def test_ridge_full_vocab_scores_all_classes_without_a_huge_matrix():
    split = {
        "X_train": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "y_train": np.array([1, 2], dtype=np.int64),
        "X_test": np.array([[1.0, 0.0]], dtype=np.float32),
        "y_test": np.array([1], dtype=np.int64),
    }
    result = ridge_inverter(split, lam=0.01, full_vocab=True, vocab_size=6, chunk=2, seed=0)
    assert result["candidate_mode"] == "full"
    assert result["top1"]["denominator"] == 1
    assert result["top1"]["value"] == pytest.approx(1.0)
    assert result["n_classes"] == 6


def test_compute_metrics_pads_short_candidate_lists():
    from obfuscation_attacks.data.contracts import AttackResult, AttackSample
    from obfuscation_attacks.evaluation.scoring import compute_metrics

    sample = AttackSample(
        tokenizer_id="t", model_id="m", seq_len=5, vocab_size=20,
        plaintext_token_ids=np.array([1, 2, 3, 4, 5]),
    )
    result = AttackResult(
        recovered_token_ids=np.array([1, 2, 9]),
        candidates=[[1], [2], [9]],
    )
    metrics = compute_metrics(result, sample)["token_recovery"]
    assert metrics["asr@1"] == pytest.approx(2 / 5)
    assert metrics["ttrsr"] == pytest.approx(2 / 5)


def test_mapping_denominator_uses_the_observed_ciphertext():
    from obfuscation_attacks.data.contracts import AttackResult, AttackSample
    from obfuscation_attacks.evaluation.scoring import compute_metrics

    sample = AttackSample(
        tokenizer_id="t", model_id="m", seq_len=3, vocab_size=20,
        ciphertext_token_ids=np.array([10, 10, 11]),
        meta={"true_mapping": {10: 1, 11: 2}},
    )
    result = AttackResult(
        recovered_mapping={10: 1},
        mapping_candidates={10: [1]},
    )
    metrics = compute_metrics(result, sample)["mapping_recovery"]
    assert metrics["num_observed_symbols"] == 2
    assert metrics["mapping_top1_accuracy"] == pytest.approx(0.5)
    assert metrics["mapping_asr@1"] == pytest.approx(0.5)
