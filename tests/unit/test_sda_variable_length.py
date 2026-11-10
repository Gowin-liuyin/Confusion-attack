import pytest
import numpy as np
import torch

from obfuscation_attacks.attacks.sda._neural import variable_length_beam_search
from obfuscation_attacks.evaluation.sda_stream import fit_sda, decode_sda, load_sda_checkpoint, _token_edit_error


def test_variable_beam_can_extend_and_score_eos_exactly():
    # Desired output 1, 2, 3, EOS, independent of ciphertext length.
    def score(batch):
        rows = []
        for prefix in batch.tolist():
            step = len(prefix)
            logits = torch.full((5,), -20.0)
            logits[[1, 2, 3, 4][min(step, 3)]] = 0.0
            rows.append(torch.log_softmax(logits, dim=0))
        return torch.stack(rows)
    tokens, score_value, beams = variable_length_beam_search(
        score, 3, eos_id=4, allowed_ids=torch.arange(4), max_output_len=5)
    assert tokens == [1, 2, 3]
    expected = sum(float(score(torch.tensor([tokens[:i]], dtype=torch.long))[0, token])
                   for i, token in enumerate(tokens + [4]))
    assert score_value == pytest.approx(expected)
    assert beams[0][0] == tokens


def test_variable_beam_can_end_early_and_rejects_impossible_eos():
    def early(batch):
        logits = torch.full((len(batch), 3), -10.0)
        logits[:, 2] = 0
        return torch.log_softmax(logits, -1)
    tokens, _, _ = variable_length_beam_search(
        early, 2, eos_id=2, allowed_ids=torch.arange(2), max_output_len=8)
    assert tokens == []
    def no_eos(batch):
        return torch.tensor([[0.0, -1.0, float("-inf")]]).repeat(len(batch), 1)
    with pytest.raises(ValueError, match="EOS-terminated"):
        variable_length_beam_search(no_eos, 2, eos_id=2, allowed_ids=torch.arange(2), max_output_len=2)


def test_edit_error_includes_insertions_and_deletions():
    assert _token_edit_error([1, 9, 2, 3], [1, 2, 3])["edits"] == 1
    assert _token_edit_error([1, 3], [1, 2, 3])["edits"] == 1
    assert _token_edit_error([1], [1, 2, 3])["reference_tokens"] == 3


@pytest.mark.parametrize("kind", ["ins", "del"])
def test_variable_noise_train_and_checkpoint_decode(tmp_path, kind):
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4,
              "max_len": 4, "noise_rate": 0.25, "noise_kinds": [kind],
              "length_mode": "variable", "max_output_len": 7, "max_cipher_len": 8}
    path = tmp_path / f"{kind}.pt"
    bundle = fit_sda([[1, 2, 1, 2]], [[3, 4, 3]], vocab_size=8,
                     params=params, seed=2, checkpoint_path=str(path))
    first = decode_sda(bundle, [3, 4, 3], beam_width=2)
    loaded = load_sda_checkpoint(path)
    assert loaded["length_mode"] == "variable"
    assert first["tokens"] == decode_sda(loaded, [3, 4, 3], beam_width=2)["tokens"]
    assert first["raw"]["score_includes_eos"] is True
    assert first["constrained"]["status"] == "not_applicable_variable_length"


def test_fixed_output_mode_uses_requested_length_and_scores_without_eos(tmp_path):
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4,
              "max_len": 4, "noise_rate": 0.25, "noise_kinds": ["del"],
              "length_mode": "fixed_output", "max_output_len": 6, "max_cipher_len": 8}
    path = tmp_path / "fixed_output.pt"
    bundle = fit_sda([[1, 2, 1, 2]], [[3, 4, 3]], vocab_size=8,
                     params=params, seed=2, checkpoint_path=str(path))
    result = decode_sda(load_sda_checkpoint(path), [3, 4, 3], beam_width=2)
    assert len(result["tokens"]) == 6 and result["mode"] == "fixed_output"
    assert result["raw"]["score_includes_eos"] is False
    assert result["constrained"]["status"] == "not_applicable_fixed_output"
    model, cfg = bundle["model"], bundle["cfg"]
    source = torch.tensor([[1, 2, 1]])
    with torch.no_grad():
        logits = model(source, torch.tensor([[cfg.bos] + result["tokens"][:-1]]), source == cfg.src_pad)
        expected = sum(torch.log_softmax(logits[0, i], -1)[token].item()
                       for i, token in enumerate(result["tokens"]))
    assert result["raw"]["score"] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("synthesis_mode", ["legacy", "invariant"])
def test_space_removal_preserves_public_training_target(synthesis_mode):
    from obfuscation_attacks.attacks.sda.sda2021_seq2seq import synthesize_pairs
    pairs = synthesize_pairs([np.array([1, 0, 2, 0, 1])], 4, 1, np.random.default_rng(7),
                             min_len=5, max_len=5, space_token_id=0, synthesis_mode=synthesis_mode)
    assert pairs == [([1, 2, 1], [1, 0, 2, 0, 1])]
    with pytest.raises(ValueError, match="empty ciphertext"):
        synthesize_pairs([np.array([0, 0])], 4, 1, np.random.default_rng(7),
                         min_len=2, max_len=2, space_token_id=0, synthesis_mode=synthesis_mode)


def test_space_restoration_training_and_checkpoint_are_consistent(tmp_path):
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 5, "max_len": 5,
              "length_mode": "fixed_output", "max_output_len": 5,
              "max_cipher_len": 5, "space_token_id": 0}
    path = tmp_path / "space.pt"
    bundle = fit_sda([[1, 0, 2, 0, 1]], [[4, 5, 4]], vocab_size=8, params=params,
                     validation_ids=[[3, 0, 4, 0, 3]], checkpoint_path=str(path), seed=2)
    loaded = load_sda_checkpoint(path)
    assert loaded["space_token_id"] == bundle["space_token_id"] == 0
    assert loaded["validation_checkpoint"]["status"] == "selected"
    assert len(decode_sda(loaded, [4, 5, 4], beam_width=2)["tokens"]) == 5
    with pytest.raises(ValueError, match="space removal requires"):
        fit_sda([[1, 0, 2]], [[3, 4]], vocab_size=8,
                params={"num_train": 1, "epochs": 1, "space_token_id": 0})
