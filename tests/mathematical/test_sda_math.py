"""Mathematical tests pinning SDA paper formulas to code.

* Encoding exactness (recurrence vs frequency-rank).
* 2023 two-term objective: L_CLM = -log P(X) - log P(Y|X). We prove BOTH terms
  are present by recomputing each from the model's per-token log-probs and
  showing the full loss strictly exceeds the target-only (2021-style) loss.
* 2021 objective: L_Seq2Seq = -log P(Y|X) equals the token-level cross-entropy
  of the target given the source.

The loss tests build tiny models, so they require the torch backend (``neural``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from obfuscation_attacks.attacks.sda.encodings import (
    frequency_rank_encode,
    recurrence_encode,
)
from obfuscation_attacks.attacks.sda.sda2021_seq2seq import (
    Seq2SeqCipherModel,
    Seq2SeqConfig,
    seq2seq_target_nll,
)
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import (
    CausalCipherLM,
    CausalLMConfig,
    build_full_sequence,
    clm_losses,
    make_clm_example,
)
from obfuscation_attacks.utils import seed_everything

pytestmark = [pytest.mark.mathematical, pytest.mark.neural]

TOL = 1e-4


# ---------------------------------------------------------------------------
# (a) encoding exactness
# ---------------------------------------------------------------------------
def test_encoding_exactness():
    tokens = [5, 9, 5, 2, 5, 7]  # "ABACAD" pattern with arbitrary ids
    assert recurrence_encode(tokens) == [1, 2, 1, 3, 1, 4]

    clear = [0, 0, 1, 1, 1, 2]  # A=2, B=3, C=1 (tie-free)
    assert recurrence_encode(clear) == [1, 1, 2, 2, 2, 3]
    assert frequency_rank_encode(clear) == [2, 2, 1, 1, 1, 3]
    assert recurrence_encode(clear) != frequency_rank_encode(clear)

    # SDA-2021-ENC is INJECTIVE on symbols even when counts tie (Fig 5 gives
    # distinct codes to distinct symbols): 'ab' and 'aa' must NOT collapse, and
    # two equally-frequent symbols (4 and 7) get DISTINCT dense codes.
    assert frequency_rank_encode([0, 1]) != frequency_rank_encode([0, 0])
    tie = frequency_rank_encode([4, 4, 7, 7, 9])  # 4 and 7 both count 2
    assert tie == [1, 1, 2, 2, 3] and len(set(tie)) == 3


# ---------------------------------------------------------------------------
# (b) 2023 two-term CLM decomposition
# ---------------------------------------------------------------------------
def _pad_clm_batch(examples, cfg):
    length = max(len(inp) for inp, _ in examples)
    inputs = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
    labels = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
    for r, (inp, lab) in enumerate(examples):
        inputs[r, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        labels[r, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return inputs, labels


def test_clm_loss_is_sum_of_two_terms_from_log_probs():
    seed_everything(0)
    cfg = CausalLMConfig(vocab_size=6, max_rec=5, d_model=16, nhead=2, num_layers=1, dim_ff=32)
    model = CausalCipherLM(cfg).eval()

    # two examples of different length -> also exercises padding / value masks
    rec_a, plain_a = [1, 2, 1, 3], [2, 4, 0, 5]
    rec_b, plain_b = [1, 2, 3], [1, 1, 0]
    examples = [make_clm_example(rec_a, plain_a, cfg), make_clm_example(rec_b, plain_b, cfg)]
    inputs, labels = _pad_clm_batch(examples, cfg)

    with torch.no_grad():
        logits = model(inputs)
        total, l_src, l_tgt, n_src, n_tgt = clm_losses(logits, labels, cfg)

        # X-code labels feed L_SRC, plaintext labels feed L_TGT; counts equal N.
        assert n_src == len(rec_a) + len(rec_b)
        assert n_tgt == len(plain_a) + len(plain_b)

        # manual recomputation of each term from per-token log-probs
        logp = F.log_softmax(logits, dim=-1)
        man_src = 0.0
        man_tgt = 0.0
        for r in range(labels.shape[0]):
            for i in range(labels.shape[1]):
                lab = int(labels[r, i])
                if cfg.rec_lo <= lab < cfg.rec_hi:        # a recurrence code -> P(X)
                    man_src += -float(logp[r, i, lab])
                elif 0 <= lab < cfg.vocab_size:           # a plaintext token -> P(Y|X)
                    man_tgt += -float(logp[r, i, lab])

    assert float(l_src) == pytest.approx(man_src, abs=TOL)
    assert float(l_tgt) == pytest.approx(man_tgt, abs=TOL)
    # the implemented loss is EXACTLY (-log P(X)) + (-log P(Y|X))
    assert float(total) == pytest.approx(man_src + man_tgt, abs=TOL)
    assert float(total) == pytest.approx(float(l_src) + float(l_tgt), abs=TOL)

    # BOTH terms are genuinely present: neither is zero, and the full objective
    # strictly exceeds the target-only (2021-style) objective by the source term.
    assert float(l_src) > 0.0
    assert float(l_tgt) > 0.0
    assert float(total) > float(l_tgt)
    assert float(total) - float(l_tgt) == pytest.approx(float(l_src), abs=TOL)


def test_clm_sequence_layout_is_rec_x_plain_y():
    cfg = CausalLMConfig(vocab_size=6, max_rec=5)
    full = build_full_sequence([1, 2, 1], [3, 0, 5], cfg)
    assert full[0] == cfg.rec_tag
    assert full[1:4] == [cfg.rec_id(1), cfg.rec_id(2), cfg.rec_id(1)]
    assert full[4] == cfg.plain_tag
    assert full[5:] == [3, 0, 5]


# ---------------------------------------------------------------------------
# (c) 2021 objective == token-level CE of P(Y|X)
# ---------------------------------------------------------------------------
def test_seq2seq_objective_is_target_cross_entropy():
    seed_everything(0)
    cfg = Seq2SeqConfig(vocab_size=6, src_vocab_size=7, d_model=16, nhead=2, num_layers=1, dim_ff=32)
    model = Seq2SeqCipherModel(cfg).eval()

    src = torch.tensor([[1, 2, 3, 1]])
    tgt_in = torch.tensor([[cfg.bos, 2, 4, 0]])
    tgt_out = torch.tensor([[2, 4, 0, cfg.eos]])

    with torch.no_grad():
        logits = model(src, tgt_in)
        loss_sum = seq2seq_target_nll(logits, tgt_out, tgt_pad_id=cfg.tgt_pad, reduction="sum")
        loss_mean = seq2seq_target_nll(logits, tgt_out, tgt_pad_id=cfg.tgt_pad, reduction="mean")

        # manual -log P(Y|X) = -sum_t log P(y_t | y_<t, X)
        logp = F.log_softmax(logits, dim=-1)[0]
        manual = -sum(float(logp[i, int(tgt_out[0, i])]) for i in range(tgt_out.shape[1]))

    assert float(loss_sum) == pytest.approx(manual, abs=TOL)
    # mean reduction is the same NLL averaged over the non-pad target tokens
    num_tokens = int((tgt_out != cfg.tgt_pad).sum())
    assert float(loss_mean) == pytest.approx(manual / num_tokens, abs=TOL)
