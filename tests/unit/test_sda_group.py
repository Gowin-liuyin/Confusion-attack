import pytest
import numpy as np
import torch
import torch.nn.functional as F

from obfuscation_attacks.attacks.sda import encodings as enc
from obfuscation_attacks.evaluation.sda_group import decode_sda_group
from obfuscation_attacks.evaluation.sda_stream import fit_sda, decode_sda


@pytest.mark.parametrize("sequences", [[], [[]], [[True]], [[1.5]], [[-1]], [[[1, 2]]],
                                        [np.array([2**63], dtype=np.uint64)]])
def test_group_rejects_invalid_symbol_ids_without_coercion(sequences):
    bundle = {"method": "sda2023", "max_cipher_len": 4}
    with pytest.raises(ValueError, match="non-negative int64 IDs"):
        decode_sda_group(bundle, sequences, beam_width=1)


def test_homophonic_group_allows_several_cipher_symbols_per_plaintext():
    bundle = fit_sda([[0, 1, 0, 1]], [[4, 5, 6, 4]], vocab_size=2, method="sda2023",
                     params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                             "nhead": 2, "num_layers": 1, "dim_ff": 16, "homophonic": True,
                             "max_cipher_len": 4}, seed=3)
    result = decode_sda_group(bundle, [[4, 5], [6, 4]], beam_width=2)
    assert result["status"] == "ok" and result["injective"] is False
    assert result["tokens"][0] == result["tokens"][3]
    assert len(set(result["mapping"].values())) <= 2 < len(result["mapping"])


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_joint_beam_resets_context_and_scores_full_requests(method):
    torch.set_num_threads(1)
    bundle = fit_sda([[1, 2, 1, 2], [2, 1, 2, 1]], [[7, 8, 7]], vocab_size=4, method=method,
                     params={"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
                             "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0,
                             "max_cipher_len": 4}, seed=3)
    sequences = [[7, 8, 7], [8, 7, 8]]
    joint = decode_sda_group(bundle, sequences, beam_width=4)
    assert joint["status"] == "ok" and sum(joint["sequence_lengths"]) > bundle["max_cipher_len"]
    assert joint["tokens"] == [joint["mapping"][symbol] for row in sequences for symbol in row]
    model, cfg = bundle["model"], bundle["cfg"]
    with torch.no_grad():
        for beam_tokens, beam_score in joint["beams"]:
            expected = 0.0
            for cipher, tokens in zip(sequences, (beam_tokens[:3], beam_tokens[3:])):
                if method == "sda2021":
                    source = torch.tensor([enc.frequency_rank_encode(cipher)])
                    padding = source == cfg.src_pad
                    memory = model.encode(source, padding)
                    inputs = torch.tensor([[cfg.bos] + tokens[:-1]])
                    logits = model.decode(inputs, memory, memory_kpm=padding)[0]
                else:
                    prompt = [cfg.rec_tag] + [cfg.rec_id(x) for x in enc.recurrence_encode(cipher)] + [cfg.plain_tag]
                    inputs = torch.tensor([prompt + tokens[:-1]])
                    logits = model(inputs)[0, len(prompt) - 1:]
                expected += sum(F.log_softmax(logits, dim=-1)[i, token].item() for i, token in enumerate(tokens))
            assert beam_score == pytest.approx(expected, abs=1e-5)
    single = decode_sda_group(bundle, [sequences[0]], beam_width=4)
    original = decode_sda(bundle, sequences[0], beam_width=4)["constrained"]
    assert single["tokens"] == original["tokens"]
    assert single["score"] == pytest.approx(original["score"], abs=1e-5)
    if method == "sda2023":
        cached = decode_sda_group(bundle, sequences, beam_width=4, use_cache=True)
        assert cached["tokens"] == joint["tokens"]
        assert cached["score"] == pytest.approx(joint["score"], abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_training_and_joint_cache_agree():
    bundle = fit_sda([[1, 2, 1, 2], [2, 1, 2, 1]], [[7, 8, 7]], vocab_size=4,
                     method="sda2023", device="cuda", seed=3,
                     params={"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
                             "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0,
                             "max_cipher_len": 4})
    assert next(bundle["model"].parameters()).device.type == "cuda"
    sequences = [[7, 8, 7], [8, 7, 8]]
    full = decode_sda_group(bundle, sequences, beam_width=4)
    cached = decode_sda_group(bundle, sequences, beam_width=4, use_cache=True)
    assert [row[0] for row in cached["beams"]] == [row[0] for row in full["beams"]]
    assert [row[1] for row in cached["beams"]] == pytest.approx([row[1] for row in full["beams"]], abs=2e-5)
