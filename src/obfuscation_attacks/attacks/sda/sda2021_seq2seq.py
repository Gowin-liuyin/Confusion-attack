"""SDA 2021 - Aldarrab & May, *Can Seq2Seq Models Crack Substitution Ciphers?*

This is the 2021 lineage and must stay separate from the 2023 causal-LM method.

Pipeline (FORMULA_LEDGER SDA-2021):

1. Frequency-rank encode the ciphertext (``encodings.frequency_rank_encode``):
   the source ``X`` is the sequence of INJECTIVE frequency-rank codes (distinct
   symbols -> distinct codes, count desc, first-appearance tiebreak), which erase
   the key while preserving the frequency structure (SDA-2021-ENC, Fig 5).
2. Train a character-level Transformer encoder-decoder on synthesised pairs
   ``(freq-rank(cipher) -> plaintext)``, where each training plaintext is drawn
   from the public reference corpus and enciphered with a fresh random 1:1 key.
   Optional transcription-noise augmentation (2021 Sec 5.4) corrupts the training
   cipher before encoding, mirroring the paper's noise-robust models.
3. The training objective is the target-side cross-entropy only:

       L_Seq2Seq = L_TGT = -log P(Y | X)          (SDA-2021-OBJ)

   i.e. there is NO source-reconstruction term (that is the 2023 method).
4. Decode with beam search. Because the paper solves 1:1 ciphers where the
   plaintext and ciphertext have equal length, we length-match the decode to the
   ciphertext (``force the model to generate tokens to match source length``).
   The injective encoding makes equal cipher symbols share one source code, and
   :func:`encodings.enforce_substitution_consistency` then recovers a
   deterministic cipher->plain key by majority vote so equal cipher symbols
   decode to equal plaintext (structural consistency).

Token layout
------------
Source: injective frequency codes ``1..R`` map to ids ``1..R``; ``SRC_PAD = 0``.
Target: plaintext ids ``0..V-1``; ``BOS = V``, ``EOS = V+1``, ``PAD = V+2``.
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


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class Seq2SeqConfig:
    vocab_size: int          # plaintext vocabulary V
    src_vocab_size: int      # rank vocabulary (incl. SRC_PAD=0)
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 64
    dropout: float = 0.0
    max_len: int = 256

    @property
    def src_pad(self) -> int:
        return 0

    @property
    def bos(self) -> int:
        return self.vocab_size

    @property
    def eos(self) -> int:
        return self.vocab_size + 1

    @property
    def tgt_pad(self) -> int:
        return self.vocab_size + 2

    @property
    def tgt_vocab_size(self) -> int:
        return self.vocab_size + 3


class Seq2SeqCipherModel(nn.Module):
    """Tiny Transformer encoder-decoder for frequency-rank decipherment."""

    def __init__(self, cfg: Seq2SeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.src_pad)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.tgt_pad)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        enc_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, cfg.num_layers, enable_nested_tensor=False
        )
        dec_layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.num_layers)
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_kpm: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(self.pos(self.src_emb(src)), src_key_padding_mask=src_kpm)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_kpm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        length = tgt.size(1)
        tgt_mask = causal_mask(length, tgt.device)
        h = self.decoder(
            self.pos(self.tgt_emb(tgt)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_kpm,
        )
        return self.generator(h)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_kpm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_kpm)
        return self.decode(tgt_in, memory, memory_kpm=src_kpm)


def seq2seq_target_nll(
    logits: torch.Tensor,
    tgt_out: torch.Tensor,
    *,
    tgt_pad_id: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Target-side cross-entropy ``-log P(Y|X)`` = ``L_Seq2Seq`` (SDA-2021-OBJ).

    ``reduction='sum'`` returns ``-sum_t log P(y_t | y_<t, X)`` exactly, i.e. the
    negative log-likelihood of the whole target; ``'mean'`` averages over the
    non-pad target tokens (used for optimisation stability).
    """
    v = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, v),
        tgt_out.reshape(-1),
        ignore_index=tgt_pad_id,
        reduction=reduction,
    )


# ---------------------------------------------------------------------------
# training-data synthesis
# ---------------------------------------------------------------------------
def synthesize_pairs(
    corpus: Sequence[np.ndarray],
    vocab_size: int,
    num_samples: int,
    rng: np.random.Generator,
    *,
    min_len: int = 4,
    max_len: int = 32,
    noise_rate: float = 0.0,
    noise_kinds: Sequence[str] = ("sub",),
) -> list[tuple[list[int], list[int]]]:
    """Make ``(freq-rank(cipher), plaintext)`` pairs from public plaintext.

    Each sample draws a reference sequence, crops it to ``[min_len, max_len]``,
    enciphers with a fresh random 1:1 key over the whole vocabulary, and
    frequency-rank encodes the cipher. (The 1:1 encoding is key-invariant, so the
    random key does not change ``X``; it is applied anyway to mirror the paper's
    "encipher public text with random keys" data-generation faithfully.)

    When ``noise_rate > 0`` the cipher is corrupted with transcription noise
    (2021 Sec 5.4) BEFORE frequency encoding, so the model learns to be robust to
    it. The default ``noise_kinds=("sub",)`` keeps source and target the same
    length; adding ``"ins"``/``"del"`` makes the source length diverge from the
    target (the paper's space/noise-restoration setting, scored with TER).
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
        key = enc.make_substitution_key(
            full_symbols, rng=rng, homophonic=False, cipher_alphabet_size=vocab_size
        )
        cipher = enc.apply_substitution(plain, key)
        if noise_rate > 0.0:
            cipher = enc.apply_transcription_noise(
                cipher, rate=noise_rate, rng=rng, kinds=noise_kinds,
                alphabet=full_symbols,
            )
        src = enc.frequency_rank_encode(cipher)  # injective ranks 1..R used as ids
        pairs.append((src, plain.tolist()))
    return pairs


def _make_batches(
    pairs: list[tuple[list[int], list[int]]],
    cfg: Seq2SeqConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    order = rng.permutation(len(pairs)).tolist()
    batches = []
    for b in range(0, len(order), batch_size):
        idxs = order[b : b + batch_size]
        chunk = [pairs[i] for i in idxs]
        ls = max(len(s) for s, _ in chunk)
        lt = max(len(t) for _, t in chunk) + 1  # +1 for BOS/EOS extension
        src = torch.full((len(chunk), ls), cfg.src_pad, dtype=torch.long)
        tgt_in = torch.full((len(chunk), lt), cfg.tgt_pad, dtype=torch.long)
        tgt_out = torch.full((len(chunk), lt), cfg.tgt_pad, dtype=torch.long)
        for r, (s, t) in enumerate(chunk):
            src[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_in[r, 0] = cfg.bos
            tgt_in[r, 1 : 1 + len(t)] = torch.tensor(t, dtype=torch.long)
            tgt_out[r, : len(t)] = torch.tensor(t, dtype=torch.long)
            tgt_out[r, len(t)] = cfg.eos
        src_kpm = src == cfg.src_pad
        batches.append((src, tgt_in, tgt_out, src_kpm))
    return batches


# ---------------------------------------------------------------------------
# attack
# ---------------------------------------------------------------------------
@register
class SDA2021(Attack):
    name = "sda.sda2021"
    family = "sda"
    variant = "original_2021"
    required_capabilities = frozenset(
        {Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}
    )

    def _run(self, ctx: AttackContext) -> AttackResult:
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "SDA2021 needs a reference corpus to synthesise (cipher -> plaintext) pairs"
            )
        vocab_size = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab_size <= 0:
            return AttackResult.failed("vocab_size unknown; cannot build the decoder vocabulary")

        cipher = np.asarray(ctx.token_ciphertext(), dtype=int)
        if cipher.size == 0:
            return AttackResult.failed("empty ciphertext")

        p = self.params
        seed = int(p.get("seed", 0))
        seed_everything(seed)
        rng = np.random.default_rng(seed)

        num_train = int(p.get("num_train", 256))
        min_len = int(p.get("min_len", 4))
        max_len = int(p.get("max_len", 32))
        noise_rate = float(p.get("noise_rate", 0.0))
        noise_kinds = tuple(p.get("noise_kinds", ("sub",)))
        pairs = synthesize_pairs(
            [np.asarray(s, dtype=int) for s in ctx.reference_corpus],
            vocab_size,
            num_train,
            rng,
            min_len=min_len,
            max_len=max_len,
            noise_rate=noise_rate,
            noise_kinds=noise_kinds,
        )
        if not pairs:
            return AttackResult.not_applicable("reference corpus had no usable sequences")

        max_rank = max((max(s) if s else 0) for s, _ in pairs)
        max_rank = max(max_rank, max(enc.frequency_rank_encode(cipher), default=1))
        cfg = Seq2SeqConfig(
            vocab_size=vocab_size,
            src_vocab_size=max_rank + 1,  # ids 0..max_rank (0 == SRC_PAD)
            d_model=int(p.get("d_model", 32)),
            nhead=int(p.get("nhead", 4)),
            num_layers=int(p.get("num_layers", 2)),
            dim_ff=int(p.get("dim_ff", 64)),
            dropout=float(p.get("dropout", 0.0)),
            max_len=max(256, cipher.size + 2),
        )
        model = Seq2SeqCipherModel(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=float(p.get("lr", 1e-3)))
        epochs = int(p.get("epochs", 15))
        batch_size = int(p.get("batch_size", 32))

        model.train()
        final_loss = float("nan")
        for _ in range(epochs):
            for src, tgt_in, tgt_out, src_kpm in _make_batches(pairs, cfg, batch_size, rng):
                opt.zero_grad()
                logits = model(src, tgt_in, src_kpm=src_kpm)
                loss = seq2seq_target_nll(
                    logits, tgt_out, tgt_pad_id=cfg.tgt_pad, reduction="mean"
                )
                loss.backward()
                opt.step()
                final_loss = float(loss.detach())

        recovered, mapping = self._decode(
            model, cfg, cipher, beam_width=int(p.get("beam_width", 5))
        )
        return AttackResult(
            recovered_token_ids=np.asarray(recovered, dtype=int),
            candidates=[[int(t)] for t in recovered],
            recovered_text=" ".join(str(int(t)) for t in recovered),
            recovered_mapping={int(c): int(v) for c, v in mapping.items()},
            diagnostics={
                "method": "seq2seq_frequency_rank_2021",
                "objective": "L_Seq2Seq = -log P(Y|X)",
                "encoding": "frequency_rank",
                "num_train_pairs": len(pairs),
                "epochs": epochs,
                "beam_width": int(p.get("beam_width", 5)),
                "final_train_loss": final_loss,
                "cipher_len": int(cipher.size),
                "src_vocab_size": cfg.src_vocab_size,
                "noise_rate": noise_rate,
                "structural_consistency": "majority-vote key (equal cipher symbol -> equal plaintext)",
            },
        )

    @torch.no_grad()
    def _decode(
        self,
        model: Seq2SeqCipherModel,
        cfg: Seq2SeqConfig,
        cipher: np.ndarray,
        *,
        beam_width: int,
    ) -> tuple[list[int], dict[int, int]]:
        model.eval()
        src_ids = enc.frequency_rank_encode(cipher)
        n = len(src_ids)
        src = torch.tensor([src_ids], dtype=torch.long)
        src_kpm = src == cfg.src_pad
        memory = model.encode(src, src_kpm)  # [1, Ls, d]

        def step(batch_gen: torch.Tensor) -> torch.Tensor:
            b = batch_gen.shape[0]
            bos = torch.full((b, 1), cfg.bos, dtype=torch.long)
            dec_in = torch.cat([bos, batch_gen], dim=1)
            mem = memory.expand(b, -1, -1)
            mem_kpm = src_kpm.expand(b, -1)
            logits = model.decode(dec_in, mem, memory_kpm=mem_kpm)
            return F.log_softmax(logits[:, -1, :], dim=-1)

        allowed = torch.arange(cfg.vocab_size, dtype=torch.long)  # plaintext ids only
        tokens, _, _ = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed
        )
        # Recover a deterministic cipher->plain key so equal cipher symbols decode
        # to equal plaintext (structural consistency; the key is 1:1 here).
        consistent, mapping = enc.enforce_substitution_consistency(cipher, tokens)
        return consistent, mapping


__all__ = [
    "SDA2021",
    "Seq2SeqCipherModel",
    "Seq2SeqConfig",
    "seq2seq_target_nll",
    "synthesize_pairs",
]
