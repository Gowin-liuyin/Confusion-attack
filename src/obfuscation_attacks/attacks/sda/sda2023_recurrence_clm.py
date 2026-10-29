"""SDA 2023 - Kambhatla et al., *Decipherment as Regression* (symbol recurrence).

This is the 2023 lineage and must stay separate from the 2021 seq2seq method.

Pipeline (FORMULA_LEDGER SDA-2023):

1. Recurrence-encode the ciphertext (``encodings.recurrence_encode``): the source
   ``X`` is the sequence of first-appearance indices, which normalises arbitrary
   keys (and arbitrary homophonic alphabets) into one integer stream
   (SDA-2023-ENC). This is NOT frequency rank.
2. Train a single CAUSAL language model over the concatenation
   ``[REC] X [PLAIN] Y`` with causal masking over *both* the source and the
   target (paper Eq. 1).
3. The objective is the two-term joint negative log-likelihood (paper Eq. 2):

       L_CLM(X, Y) = L_SRC + L_TGT = -log P(X) - log P(Y | X)   (SDA-2023-OBJ)

   BOTH terms are optimised: the model must reproduce the recurrence-encoded
   ciphertext (``-log P(X)``) and then produce the plaintext (``-log P(Y|X)``).
   Dropping the first term is the 2021 "target-only" baseline, which this method
   explicitly is not.
4. Decode with beam search over the plaintext vocabulary, length-matched to the
   ciphertext (1:1 and homophonic substitutions both preserve length), which
   keeps the recovered sequence structurally consistent with the input.

Homophonic support (``homophonic=True``) synthesises training ciphers with the
2023 Algorithm 1 allocation (``encodings.allocate_homophones``), where frequent
plaintext symbols receive several cipher symbols. The declared required
capabilities stay ``{TOKEN_CIPHERTEXT, DETERMINISTIC_PERMUTATION}`` because the
AloePri target is a deterministic token permutation; homophonic enciphering is a
training-data option of the method itself.

Shared-vocabulary layout (one causal LM over source + target)::

    plaintext ids  : 0 .. V-1
    recurrence code k (1..R) -> id  V + (k-1)       (ids V .. V+R-1)
    REC   tag      : V + R
    PLAIN tag      : V + R + 1
    PAD            : V + R + 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import encodings as enc
from ._neural import SinusoidalPositionalEncoding, causal_mask, length_matched_beam_search

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# configuration / token layout
# ---------------------------------------------------------------------------
@dataclass
class CausalLMConfig:
    vocab_size: int          # plaintext vocabulary V
    max_rec: int             # largest recurrence code R to be represented
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 96
    dropout: float = 0.0
    max_len: int = 512

    def rec_id(self, k: int) -> int:
        """Shared-vocab id for recurrence code ``k`` (k >= 1)."""
        if not (1 <= k <= self.max_rec):
            raise ValueError(f"recurrence code {k} out of range [1, {self.max_rec}]")
        return self.vocab_size + (k - 1)

    @property
    def rec_lo(self) -> int:
        return self.vocab_size

    @property
    def rec_hi(self) -> int:  # exclusive
        return self.vocab_size + self.max_rec

    @property
    def rec_tag(self) -> int:
        return self.vocab_size + self.max_rec

    @property
    def plain_tag(self) -> int:
        return self.vocab_size + self.max_rec + 1

    @property
    def pad(self) -> int:
        return self.vocab_size + self.max_rec + 2

    @property
    def total_vocab(self) -> int:
        return self.vocab_size + self.max_rec + 3


class CausalCipherLM(nn.Module):
    """Tiny decoder-only (causal) Transformer LM over the shared vocabulary."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.total_vocab, cfg.d_model, padding_idx=cfg.pad)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(
            layer, cfg.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.total_vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        mask = causal_mask(length, tokens.device)
        h = self.pos(self.emb(tokens))
        h = self.blocks(h, mask=mask)
        return self.head(self.norm(h))


# ---------------------------------------------------------------------------
# sequence construction and the two-term CLM loss
# ---------------------------------------------------------------------------
def build_full_sequence(rec_codes: Sequence[int], plain_ids: Sequence[int], cfg: CausalLMConfig) -> list[int]:
    """Return ``[REC] X [PLAIN] Y`` as shared-vocab ids."""
    return (
        [cfg.rec_tag]
        + [cfg.rec_id(int(k)) for k in rec_codes]
        + [cfg.plain_tag]
        + [int(y) for y in plain_ids]
    )


def make_clm_example(
    rec_codes: Sequence[int], plain_ids: Sequence[int], cfg: CausalLMConfig
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` for one training example.

    ``input_ids = full[:-1]`` and ``labels = full[1:]`` (raw, with the PLAIN tag
    still present; it is masked out by value in :func:`clm_losses`).
    """
    full = build_full_sequence(rec_codes, plain_ids, cfg)
    return full[:-1], full[1:]


def clm_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: CausalLMConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Decompose the causal-LM NLL into the two paper terms (sums).

    Returns ``(total, l_src, l_tgt, n_src, n_tgt)`` where

        l_src = -sum log P(X_i | REC, X_<i)          = -log P(X)
        l_tgt = -sum log P(Y_j | X, PLAIN, Y_<j)     = -log P(Y | X)
        total = l_src + l_tgt                        = L_CLM

    Which label positions belong to each term is decided purely by value range:
    recurrence-code ids feed ``l_src``, plaintext ids feed ``l_tgt``; the PLAIN
    tag and any padding fall in neither and contribute nothing.
    """
    logp = F.log_softmax(logits, dim=-1)
    safe = labels.clamp(min=0)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [B, L]

    src_mask = (labels >= cfg.rec_lo) & (labels < cfg.rec_hi)
    tgt_mask = (labels >= 0) & (labels < cfg.vocab_size)
    valid = src_mask | tgt_mask

    nll = nll * valid  # kill PLAIN / PAD / REC contributions before summing
    l_src = (nll * src_mask).sum()
    l_tgt = (nll * tgt_mask).sum()
    total = l_src + l_tgt
    return total, l_src, l_tgt, int(src_mask.sum()), int(tgt_mask.sum())


# ---------------------------------------------------------------------------
# training-data synthesis
# ---------------------------------------------------------------------------
def synthesize_pairs(
    corpus: Sequence[np.ndarray],
    vocab_size: int,
    num_samples: int,
    rng: np.random.Generator,
    *,
    homophonic: bool = False,
    min_len: int = 4,
    max_len: int = 32,
    approx_num_symbols: Optional[int] = None,
) -> list[tuple[list[int], list[int]]]:
    """Make ``(recurrence(cipher), plaintext)`` pairs from public plaintext.

    For 1:1 keys the recurrence encoding is key-invariant; for homophonic keys
    each occurrence of a plaintext symbol may map to a different cipher symbol,
    so the random key genuinely diversifies ``X`` and the model must learn which
    recurrence codes are homophones of one another.
    """
    pairs: list[tuple[list[int], list[int]]] = []
    corpus = [np.asarray(s, dtype=int) for s in corpus if len(s) > 0]
    if not corpus:
        return pairs
    full_symbols = list(range(vocab_size))
    for _ in range(num_samples):
        seq = corpus[int(rng.integers(len(corpus)))]
        n = len(seq)
        want = int(rng.integers(min_len, max_len + 1)) if max_len > min_len else min_len
        want = min(want, n)
        if want < 1:
            want = n
        start = int(rng.integers(0, n - want + 1)) if n > want else 0
        plain = seq[start : start + want].astype(int)

        if homophonic:
            counts = {int(t): int(c) for t, c in zip(*np.unique(plain, return_counts=True))}
            key = enc.make_substitution_key(
                full_symbols,
                rng=rng,
                homophonic=True,
                plain_counts={s: counts.get(s, 1) for s in full_symbols},
                approx_num_symbols=approx_num_symbols or max(vocab_size + 2, 2 * len(counts)),
            )
            cipher = enc.apply_substitution(plain, key, rng=rng)
        else:
            key = enc.make_substitution_key(
                full_symbols, rng=rng, homophonic=False, cipher_alphabet_size=vocab_size
            )
            cipher = enc.apply_substitution(plain, key)

        rec = enc.recurrence_encode(cipher)
        pairs.append((rec, plain.tolist()))
    return pairs


def _make_batches(
    pairs: list[tuple[list[int], list[int]]],
    cfg: CausalLMConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    order = rng.permutation(len(pairs)).tolist()
    batches = []
    for b in range(0, len(order), batch_size):
        idxs = order[b : b + batch_size]
        examples = [make_clm_example(pairs[i][0], pairs[i][1], cfg) for i in idxs]
        length = max(len(inp) for inp, _ in examples)
        inputs = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
        labels = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
        for r, (inp, lab) in enumerate(examples):
            inputs[r, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            labels[r, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        batches.append((inputs, labels))
    return batches


# ---------------------------------------------------------------------------
# attack
# ---------------------------------------------------------------------------
@register
class SDA2023(Attack):
    name = "sda.sda2023"
    family = "sda"
    variant = "original_2023"
    required_capabilities = frozenset(
        {Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}
    )

    def _run(self, ctx: AttackContext) -> AttackResult:
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "SDA2023 needs a reference corpus to synthesise (cipher -> plaintext) pairs"
            )
        vocab_size = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab_size <= 0:
            return AttackResult.failed("vocab_size unknown; cannot build the shared vocabulary")

        cipher = np.asarray(ctx.token_ciphertext(), dtype=int)
        if cipher.size == 0:
            return AttackResult.failed("empty ciphertext")

        p = self.params
        seed = int(p.get("seed", 0))
        seed_everything(seed)
        rng = np.random.default_rng(seed)

        homophonic = bool(p.get("homophonic", False))
        num_train = int(p.get("num_train", 256))
        min_len = int(p.get("min_len", 4))
        max_len = int(p.get("max_len", 32))
        approx_num_symbols = p.get("homophone_symbols", None)
        pairs = synthesize_pairs(
            [np.asarray(s, dtype=int) for s in ctx.reference_corpus],
            vocab_size,
            num_train,
            rng,
            homophonic=homophonic,
            min_len=min_len,
            max_len=max_len,
            approx_num_symbols=approx_num_symbols,
        )
        if not pairs:
            return AttackResult.not_applicable("reference corpus had no usable sequences")

        cipher_rec = enc.recurrence_encode(cipher)
        max_rec = max(
            max((max(r) if r else 1) for r, _ in pairs),
            max(cipher_rec, default=1),
        )
        cfg = CausalLMConfig(
            vocab_size=vocab_size,
            max_rec=max_rec,
            d_model=int(p.get("d_model", 48)),
            nhead=int(p.get("nhead", 4)),
            num_layers=int(p.get("num_layers", 2)),
            dim_ff=int(p.get("dim_ff", 96)),
            dropout=float(p.get("dropout", 0.0)),
            max_len=max(512, 2 * cipher.size + 4),
        )
        model = CausalCipherLM(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=float(p.get("lr", 5e-4)))
        epochs = int(p.get("epochs", 20))
        batch_size = int(p.get("batch_size", 32))

        model.train()
        final = {"total": float("nan"), "src": float("nan"), "tgt": float("nan")}
        for _ in range(epochs):
            for inputs, labels in _make_batches(pairs, cfg, batch_size, rng):
                opt.zero_grad()
                logits = model(inputs)
                total, l_src, l_tgt, n_src, n_tgt = clm_losses(logits, labels, cfg)
                loss = total / max(n_src + n_tgt, 1)
                loss.backward()
                opt.step()
                final = {
                    "total": float(total.detach()),
                    "src": float(l_src.detach()),
                    "tgt": float(l_tgt.detach()),
                }

        recovered, mapping = self._decode(
            model, cfg, cipher, cipher_rec, beam_width=int(p.get("beam_width", 5))
        )
        return AttackResult(
            recovered_token_ids=np.asarray(recovered, dtype=int),
            candidates=[[int(t)] for t in recovered],
            recovered_text=" ".join(str(int(t)) for t in recovered),
            recovered_mapping={int(c): int(v) for c, v in mapping.items()},
            diagnostics={
                "method": "causal_lm_recurrence_2023",
                "objective": "L_CLM = -log P(X) - log P(Y|X)",
                "encoding": "recurrence",
                "homophonic": homophonic,
                "num_train_pairs": len(pairs),
                "epochs": epochs,
                "beam_width": int(p.get("beam_width", 5)),
                "max_rec_code": int(max_rec),
                "final_train_loss_total": final["total"],
                "final_train_loss_src": final["src"],
                "final_train_loss_tgt": final["tgt"],
                "cipher_len": int(cipher.size),
                "structural_consistency": "majority-vote key (equal cipher symbol -> equal plaintext)",
            },
        )

    @torch.no_grad()
    def _decode(
        self,
        model: CausalCipherLM,
        cfg: CausalLMConfig,
        cipher: np.ndarray,
        cipher_rec: Sequence[int],
        *,
        beam_width: int,
    ) -> tuple[list[int], dict[int, int]]:
        model.eval()
        n = len(cipher_rec)
        prompt_ids = [cfg.rec_tag] + [cfg.rec_id(int(k)) for k in cipher_rec] + [cfg.plain_tag]
        prompt = torch.tensor([prompt_ids], dtype=torch.long)

        def step(batch_gen: torch.Tensor) -> torch.Tensor:
            b = batch_gen.shape[0]
            pref = prompt.expand(b, -1)
            full = torch.cat([pref, batch_gen], dim=1)
            logits = model(full)
            return F.log_softmax(logits[:, -1, :], dim=-1)

        allowed = torch.arange(cfg.vocab_size, dtype=torch.long)  # plaintext ids only
        tokens, _, _ = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed
        )
        # Recover a deterministic cipher->plain key by majority vote. A homophonic
        # cipher maps many cipher symbols to one plaintext, so grouping by the
        # ORIGINAL cipher symbol keeps the map deterministic and makes equal
        # cipher symbols decode to equal plaintext (structural consistency).
        consistent, mapping = enc.enforce_substitution_consistency(cipher, tokens)
        return consistent, mapping


__all__ = [
    "SDA2023",
    "CausalCipherLM",
    "CausalLMConfig",
    "clm_losses",
    "make_clm_example",
    "build_full_sequence",
    "synthesize_pairs",
]
