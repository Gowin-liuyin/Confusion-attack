import numpy as np

from obfuscation_attacks.data.contracts import AttackResult, AttackSample
from obfuscation_attacks.evaluation.scoring import compute_metrics
from obfuscation_attacks.evaluation.pii_spans import score_entity_recovery
from obfuscation_attacks.attacks.vma.weight_rowsort import recover_permutations


def test_sequence_metrics_include_extra_and_missing_tokens():
    sample = AttackSample(tokenizer_id="t", model_id="m", seq_len=4, vocab_size=10,
                          plaintext_token_ids=np.array([1, 2, 3, 4]))
    for prediction in ([1, 2, 3, 4, 9], [1, 2, 3]):
        metrics = compute_metrics(AttackResult(recovered_token_ids=np.array(prediction)), sample)["token_recovery"]
        assert not metrics["exact_sequence_match"]
        assert metrics["token_f1"] < 1


def test_single_hypothesis_never_claims_unsupported_top_k():
    sample = AttackSample(tokenizer_id="t", model_id="m", seq_len=2, vocab_size=10,
                          plaintext_token_ids=np.array([1, 2]), content_mask=np.array([True, True]))
    result = AttackResult(recovered_token_ids=np.array([1, 3]), candidates=[[1], [3]],
                          diagnostics={"candidate_semantics": "one selected hypothesis per position; not top-k"})
    scores = compute_metrics(result, sample, ks=[1, 5, 10])["token_recovery"]
    assert scores["asr@1"] == 0.5
    assert scores["candidate_coverage"]["unavailable_requested_k"] == [5, 10]
    assert "asr@5" not in scores and "asr@10" not in scores
    assert "clean_asr@5" not in scores and "total_asr@10" not in scores


def test_entity_requires_contiguous_character_coverage():
    entity = [{"start": 0, "end": 4}]
    for offsets in ([(0, 2)], [(0, 1), (2, 4)]):
        assert score_entity_recovery([1, 2], [1, 2], entity, offsets)["numerator"] == 0
    assert score_entity_recovery([1, 2], [1, 2], entity, [(0, 2), (2, 4)])["numerator"] == 1


def test_rowsort_checks_joint_reconstruction():
    x = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
    y = x.copy()
    y[1] = np.roll(y[1], 1)
    result = recover_permutations(x, y)
    assert not result["exact_reconstruction"]
    assert result["reconstruction_residual"] > 0
