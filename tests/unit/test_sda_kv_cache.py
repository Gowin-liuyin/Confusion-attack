import numpy as np
import pytest
import torch
import torch.nn.functional as F

from obfuscation_attacks.attacks.sda import encodings as enc
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import (
    CachedCausalStep, CausalCipherLM, CausalLMConfig, SDA2023,
)


@pytest.mark.parametrize("representation", ["lookup", "decimal_char_cnn"])
def test_cached_causal_scores_match_full_prefix_for_branching_beams(representation):
    torch.manual_seed(19)
    cfg = CausalLMConfig(vocab_size=9, max_rec=20, d_model=12, nhead=3,
                         num_layers=2, dim_ff=24, dropout=0.2, max_len=64,
                         representation=representation)
    model = CausalCipherLM(cfg).eval()
    prompt = torch.tensor([[cfg.rec_tag, cfg.rec_id(1), cfg.rec_id(2), cfg.rec_id(1), cfg.plain_tag]])
    cached = CachedCausalStep(model, prompt)
    batches = [torch.empty((1, 0), dtype=torch.long), torch.tensor([[1], [2], [3]]),
               torch.tensor([[1, 4], [1, 5], [2, 6], [3, 7]]),
               torch.tensor([[1, 4, 8], [2, 6, 0], [3, 7, 1]])]
    for batch in batches:
        with torch.no_grad():
            actual = cached(batch)
            full = torch.cat((prompt.expand(len(batch), -1), batch), dim=1)
            expected = F.log_softmax(model(full)[:, -1, :], dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="advance one token"):
        cached(torch.tensor([[1]]))


@pytest.mark.parametrize("representation", ["lookup", "decimal_char_cnn"])
def test_cached_beam_sequences_and_scores_match_full_model(representation):
    torch.manual_seed(31)
    cfg = CausalLMConfig(vocab_size=9, max_rec=20, d_model=12, nhead=3,
                         num_layers=2, dim_ff=24, dropout=0.0, max_len=64,
                         representation=representation)
    model = CausalCipherLM(cfg).eval()
    cipher = np.array([5, 9, 5, 2, 9, 7])
    rec = enc.recurrence_encode(cipher)
    full = SDA2023()._decode(model, cfg, cipher, rec, beam_width=4)
    cached = SDA2023()._decode(model, cfg, cipher, rec, beam_width=4, use_cache=True)
    for kind in ("raw", "constrained"):
        assert cached[kind]["tokens"] == full[kind]["tokens"]
        assert [b[0] for b in cached[kind]["beams"]] == [b[0] for b in full[kind]["beams"]]
        assert cached[kind]["score"] == pytest.approx(full[kind]["score"], abs=1e-5)
        for actual, expected in zip(cached[kind]["beams"], full[kind]["beams"]):
            assert actual[1] == pytest.approx(expected[1], abs=1e-5)


def test_cache_refuses_training_mode():
    cfg = CausalLMConfig(vocab_size=3, max_rec=4, d_model=8, nhead=2,
                         num_layers=1, dim_ff=16)
    with pytest.raises(ValueError, match="eval"):
        CachedCausalStep(CausalCipherLM(cfg).train(), torch.tensor([[cfg.rec_tag]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
@pytest.mark.parametrize("representation", ["lookup", "decimal_char_cnn"])
def test_cuda_cached_scores_and_beams_match_full_prefix(representation):
    torch.manual_seed(29)
    cfg = CausalLMConfig(vocab_size=9, max_rec=20, d_model=12, nhead=3,
                         num_layers=2, dim_ff=24, dropout=0, max_len=64,
                         representation=representation)
    model = CausalCipherLM(cfg).cuda().eval()
    prompt = torch.tensor([[cfg.rec_tag, cfg.rec_id(1), cfg.rec_id(2), cfg.plain_tag]], device="cuda")
    cached = CachedCausalStep(model, prompt)
    for prefixes in (torch.empty((1, 0), dtype=torch.long, device="cuda"),
                     torch.tensor([[1], [2], [3]], device="cuda"),
                     torch.tensor([[1, 4], [2, 5], [3, 6]], device="cuda")):
        with torch.no_grad():
            actual = cached(prefixes)
            full = torch.cat((prompt.expand(len(prefixes), -1), prefixes), dim=1)
            expected = F.log_softmax(model(full)[:, -1, :], dim=-1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    cipher = np.array([5, 9, 5, 2, 9, 7])
    rec = enc.recurrence_encode(cipher)
    full = SDA2023()._decode(model, cfg, cipher, rec, beam_width=4)
    cached_result = SDA2023()._decode(model, cfg, cipher, rec, beam_width=4, use_cache=True)
    for kind in ("raw", "constrained"):
        assert [row[0] for row in cached_result[kind]["beams"]] == [row[0] for row in full[kind]["beams"]]
        assert [row[1] for row in cached_result[kind]["beams"]] == pytest.approx(
            [row[1] for row in full[kind]["beams"]], abs=2e-5)
