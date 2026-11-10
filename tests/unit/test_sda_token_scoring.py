import pytest
import numpy as np

from obfuscation_attacks.evaluation.sda_public import (
    decode_sda_observations, decoded_sda_tokens, prepare_sda_observations,
)
from obfuscation_attacks.evaluation.sda_token_scoring import compare_sda_token_predictions, score_sda_token_rows
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


def test_public_observations_to_isolated_token_scoring():
    uploads = [Upload(surface=UploadSurface.IDS, request_id=9, seq_len=2, full_len=2,
                      payload=np.asarray([2, 3]))]
    public = prepare_sda_observations(uploads)
    decoded = decode_sda_observations({}, public, visible_plaintext=True)
    predictions = decoded_sda_tokens(decoded)
    scored = score_sda_token_rows({9: [2, 3]}, predictions)
    assert scored["token_accuracy"]["value"] == 1
    assert scored["exact_sequence"]["value"] == 1
    assert decoded["processed_observations_sha256"] == public["processed_observations_sha256"]


def test_public_predictions_are_scored_by_request_id_with_missing_in_denominator():
    decoded = {"mode": "neural", "rows": [
        {"request_id": 9, "decode": {"tokens": [2, 3, 4]}},
        {"request_id": 4, "decode": {"tokens": [1]}},
    ]}
    predictions = decoded_sda_tokens(decoded)
    scored = score_sda_token_rows({4: [1, 1], 9: [2, 3], 7: [5]}, predictions)
    assert scored["token_accuracy"]["numerator"] == 3
    assert scored["token_accuracy"]["denominator"] == 5
    assert scored["exact_sequence"]["numerator"] == 0  # short and long predictions are not exact
    assert scored["missing_predictions"] == 1
    assert scored["confidence_interval"]["status"] == "not_evaluated"
    assert "[1, 1]" not in str(scored)


def test_token_scoring_rejects_unmatched_and_malformed_ids():
    with pytest.raises(ValueError, match="absent from references"):
        score_sda_token_rows({4: [1]}, {5: [1]})
    with pytest.raises(ValueError, match="integers"):
        score_sda_token_rows({True: [1]}, {})
    with pytest.raises(ValueError, match="empty"):
        score_sda_token_rows({4: []}, {})
    with pytest.raises(ValueError, match="duplicate"):
        decoded_sda_tokens({"mode": "visible_plaintext", "rows": [
            {"request_id": 1, "tokens": [1]}, {"request_id": 1, "tokens": [2]}]})


def test_dependency_components_preserve_shared_session_and_key():
    references = {1: [1, 1, 1], 2: [2, 2, 2], 3: [3, 3, 3], 4: [4]}
    predictions = {1: [1], 4: [4]}
    groups = {
        1: {"document": "d1", "session": "s1", "key_group": "k1"},
        2: {"document": "d2", "session": "s1", "key_group": "k2"},
        3: {"document": "d3", "session": "s2", "key_group": "k2"},
        4: {"document": "d4", "session": "s3", "key_group": "k4"},
    }
    result = score_sda_token_rows(references, predictions, clusters=groups, seed=5)
    ci = result["confidence_interval"]
    assert result["token_accuracy"]["value"] == pytest.approx(0.2)
    assert ci["micro"] == pytest.approx(0.2)
    assert ci["n_clusters"] == 2 and ci["n_boot"] == 2000
    assert ci["lo"] <= 0.2 <= ci["hi"]
    assert result["macro_token_accuracy"]["macro"] == pytest.approx((1 / 3 + 1) / 4)
    assert result["macro_token_accuracy"]["n_clusters"] == 2
    assert result["exact_sequence"]["confidence_interval"]["micro"] == pytest.approx(0.25)
    assert result["exact_sequence"]["confidence_interval"]["n_boot"] == 2000
    assert "k1" not in str(result) and "s1" not in str(result)
    same_key = {key: {**value, "key_group": "common"} for key, value in groups.items()}
    unidentifiable = score_sda_token_rows(references, predictions, clusters=same_key)
    assert unidentifiable["confidence_interval"]["n_clusters"] == 1
    assert unidentifiable["confidence_interval"]["n_boot"] == 0
    assert unidentifiable["confidence_interval"]["lo"] is None


def test_cluster_metadata_must_be_complete_and_well_typed():
    references = {1: [2], 2: [3]}
    with pytest.raises(ValueError, match="exactly match"):
        score_sda_token_rows(references, {}, clusters={1: {}})
    with pytest.raises(ValueError, match="key_group"):
        score_sda_token_rows(references, {}, clusters={
            1: {"document": "a", "session": "s", "key_group": None},
            2: {"document": "b", "session": "t", "key_group": "k"}})


def test_paired_comparison_uses_same_micro_denominator_and_keeps_missing():
    truth = {1: [1] * 9, 2: [2]}
    groups = {1: {"document": "d1", "session": "s1", "key_group": "k1"},
              2: {"document": "d2", "session": "s2", "key_group": "k2"}}
    result = compare_sda_token_predictions(truth, {1: [1] * 9}, {2: [2]}, clusters=groups, seed=4)
    assert result["value"] == pytest.approx(0.8)
    assert result["denominator"] == 10
    assert result["attack_missing_predictions"] == result["baseline_missing_predictions"] == 1
    assert result["n_clusters"] == 2 and result["n_boot"] == 2000
    assert result["lo"] <= result["value"] <= result["hi"]
    with pytest.raises(ValueError, match="absent from references"):
        compare_sda_token_predictions(truth, {3: [1]}, {}, clusters=groups)
