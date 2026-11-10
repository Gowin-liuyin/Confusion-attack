import pytest

from obfuscation_attacks.evaluation.sda_text_scoring import score_text_reconstruction, score_sda_text_rows
from obfuscation_attacks.evaluation.sda_public import decoded_sda_text


def test_empty_prediction_cannot_become_text_via_decoder():
    decoded = {"mode": "neural", "rows": [{"request_id": 1, "decode": {"tokens": []}}]}
    assert decoded_sda_text(decoded, lambda tokens: "fabricated output") == {1: ""}


def test_character_alignment_handles_insertions_and_entity_spans():
    truth = "Hello Alice, visit Paris!"
    entity = [{"start": 6, "end": 11, "type": "person"},
              {"start": 19, "end": 24, "type": "place"}]
    score = score_text_reconstruction(truth, "Hello Alice, please visit Paris!", entities=entity)
    assert score["word_recovery"]["numerator"] == score["word_recovery"]["denominator"] == 4
    assert score["entity_recovery"]["numerator"] == score["entity_recovery"]["denominator"] == 2
    assert score["exact_sequence"]["numerator"] == 0
    assert "Alice" not in str(score) and "Paris" not in str(score)


def test_deleted_or_ambiguous_repeated_entities_never_gain_credit():
    score = score_text_reconstruction("Alice Alice", "Alice", entities=[
        {"start": 0, "end": 5, "type": "person"},
        {"start": 6, "end": 11, "type": "person"}])
    assert score["entity_recovery"]["denominator"] == 2
    assert score["entity_recovery"]["numerator"] == 0
    assert score["word_recovery"]["numerator"] == 0


def test_missing_predictions_and_unicode_offsets_stay_in_denominator():
    score = score_sda_text_rows({3: "你好 Alice", 4: "Bob"}, {3: "你好 Alice"},
        entities={3: [{"start": 3, "end": 8, "type": "person"}],
                  4: [{"start": 0, "end": 3, "type": "person"}]})
    assert score["missing_predictions"] == 1
    assert score["exact_sequence"]["numerator"] == 1
    assert score["entity_recovery"]["numerator"] == 1
    assert score["entity_recovery"]["denominator"] == 2
    assert score["entity_recovery"]["by_type"]["person"]["denominator"] == 2
    assert score["reference_token_topk"]["status"] == "not_evaluated"
    assert score["sda2021_character_ter"]["edits"] == 3
    assert score["sda2021_character_ter"]["reference_characters"] == 11
    assert score["sda2021_character_ter"]["shift_operations"] is False


def test_sda2021_character_ter_includes_spaces_and_no_shift_operation():
    score = score_text_reconstruction("ab cd", "cd ab")
    assert score["sda2021_character_ter"]["edits"] == 4
    assert score["sda2021_character_ter"]["reference_characters"] == 5
    assert score["sda2021_character_ter"]["value"] == pytest.approx(0.8)
    assert score_text_reconstruction("", "x")["sda2021_character_ter"]["value"] is None


def test_scorer_refuses_unbounded_or_bad_input_without_truncation():
    with pytest.raises(ValueError, match="max_cells"):
        score_text_reconstruction("abcd", "abcd", max_cells=10)
    with pytest.raises(ValueError, match="entity offsets"):
        score_text_reconstruction("abc", "abc", entities=[{"start": 1, "end": 9}])
    with pytest.raises(ValueError, match="absent"):
        score_sda_text_rows({1: "abc"}, {2: "abc"})
    assert score_text_reconstruction("", "")["exact_sequence"]["value"] is None


def test_public_token_decoding_is_separate_from_reference_scoring():
    decoded = {"mode": "neural", "rows": [{"request_id": 9, "decode": {"tokens": [1, 2]}}]}
    predictions = decoded_sda_text(decoded, lambda ids: "Alice" if list(ids) == [1, 2] else "")
    score = score_sda_text_rows({9: "Alice"}, predictions)
    assert score["word_recovery"]["value"] == 1.0


def test_unambiguous_alignment_matches_all_optimal_paths_on_small_strings():
    from functools import lru_cache
    from itertools import product
    from obfuscation_attacks.evaluation.sda_text_scoring import _unambiguous_character_alignment
    strings = ["".join(items) for size in range(4) for items in product("ab", repeat=size)]
    for reference in strings:
        for predicted in strings:
            @lru_cache(None)
            def paths(i, j):
                if i == len(reference) and j == len(predicted):
                    return [(0, ())]
                choices = []
                if i < len(reference):
                    choices += [(1 + cost, (None,) + mapping) for cost, mapping in paths(i + 1, j)]
                if j < len(predicted):
                    choices += [(1 + cost, mapping) for cost, mapping in paths(i, j + 1)]
                if i < len(reference) and j < len(predicted):
                    same = reference[i] == predicted[j]
                    choices += [(int(not same) + cost, ((j if same else None),) + mapping)
                                for cost, mapping in paths(i + 1, j + 1)]
                best = min(cost for cost, _ in choices)
                return [(cost, mapping) for cost, mapping in choices if cost == best]
            optimal = paths(0, 0)
            expected = []
            for i in range(len(reference)):
                values = {mapping[i] for _, mapping in optimal}
                expected.append(next(iter(values)) if len(values) == 1 and None not in values else None)
            actual, distance = _unambiguous_character_alignment(reference, predicted, max_cells=100)
            assert distance == optimal[0][0]
            assert actual == expected, (reference, predicted)


def test_extended_word_or_entity_is_not_exact_recovery():
    result = score_text_reconstruction("cat", "cats", entities=[{"start": 0, "end": 3, "type": "name"}])
    assert result["word_recovery"]["numerator"] == 0
    assert result["entity_recovery"]["numerator"] == 0
