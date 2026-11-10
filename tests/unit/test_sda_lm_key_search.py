import itertools

import numpy as np
import pytest
import torch

from obfuscation_attacks.attacks.sda.lm_key_search import lm_key_beam_search, TorchCausalNextLogProbs


def _lm(prefix):
    return np.log([0.8, 0.2] if not prefix or prefix[-1] == 0 else [0.1, 0.9])


def test_final_beam_scores_match_exhaustive_injective_keys():
    cipher = [7, 9, 7, 9]
    result = lm_key_beam_search(cipher, [0, 1], _lm, beam_width=4, seed=3)
    exhaustive = []
    for key in itertools.permutations([0, 1]):
        text = [key[0], key[1], key[0], key[1]]
        score = sum(_lm(text[:i])[token] for i, token in enumerate(text))
        exhaustive.append((score, text))
    exhaustive.sort(reverse=True)
    assert [row["tokens"] for row in result["candidates"]] == [row[1] for row in exhaustive]
    assert [row["log_probability"] for row in result["candidates"]] == pytest.approx([row[0] for row in exhaustive])
    assert result["variant"].endswith("probe")
    assert result["frequency_heuristic"] is False


def test_homophonic_keys_and_reproducible_partial_sampling():
    args = ([7, 9, 7, 9], [0, 1], _lm)
    result = lm_key_beam_search(*args, beam_width=4, seed=44, homophonic=True)
    assert result == lm_key_beam_search(*args, beam_width=4, seed=44, homophonic=True)
    assert len(result["candidates"]) == 4
    assert any(len(set(item["mapping"].values())) == 1 for item in result["candidates"])
    assert all(item["tokens"] == [item["mapping"][x] for x in args[0]] for item in result["candidates"])


def test_joint_key_scores_reset_lm_context_for_each_sequence():
    cipher = [7, 9, 9, 7]
    result = lm_key_beam_search(cipher, [0, 1], _lm, beam_width=4, seed=3, sequence_lengths=[2, 2])
    for row in result["candidates"]:
        tokens = row["tokens"]
        expected = sum(_lm(part[:i])[token] for part in (tokens[:2], tokens[2:])
                       for i, token in enumerate(part))
        assert row["log_probability"] == pytest.approx(expected)
        assert tokens[0] == tokens[3] and tokens[1] == tokens[2]
    assert result["sequence_lengths"] == [2, 2]
    with pytest.raises(ValueError, match="sequence_lengths"):
        lm_key_beam_search(cipher, [0, 1], _lm, beam_width=4, seed=3, sequence_lengths=[3])


def test_fail_closed_on_invalid_key_and_lm_inputs():
    with pytest.raises(ValueError, match="injective"):
        lm_key_beam_search([1, 2, 3], [0, 1], _lm, beam_width=2, seed=1)
    with pytest.raises(ValueError, match="symbol_order"):
        lm_key_beam_search([1, 2], [0, 1], _lm, beam_width=2, seed=1, symbol_order=[1, 1])
    with pytest.raises(ValueError, match="normalized"):
        lm_key_beam_search([1], [0, 1], lambda _: [0.0, 0.0], beam_width=2, seed=1)


def test_torch_adapter_uses_bos_and_refuses_truncation():
    class FakeModel(torch.nn.Module):
        config = type("Config", (), {"max_position_embeddings": 2})()

        def forward(self, input_ids):
            assert input_ids[0, 0].item() == 2
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 3))})()

    adapter = TorchCausalNextLogProbs(FakeModel(), bos_token_id=2)
    assert np.exp(adapter([1])).tolist() == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError, match="positional capacity"):
        adapter([1, 1])
