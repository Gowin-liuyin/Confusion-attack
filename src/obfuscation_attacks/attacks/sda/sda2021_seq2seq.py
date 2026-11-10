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
   The default output preserves the raw hypothesis. A separate engineering
   variant applies one-to-one constraints during beam search and reports its
   own sequence, mapping, score, and n-best list. An explicit variable-length
   engineering mode uses EOS and does not impose a positional substitution map.
   This implementation uses
   lookup embeddings and supports token sequences as an engineering variant;
   character-paper fidelity and strength require independent verification.


Token layout
------------
Source: injective frequency codes ``1..R`` map to ids ``1..R``; ``SRC_PAD = 0``.
Target: plaintext ids ``0..V-1``; ``BOS = V``, ``EOS = V+1``, ``PAD = V+2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import encodings as enc
from ._neural import (SinusoidalPositionalEncoding, causal_mask, length_matched_beam_search,
                      variable_length_beam_search)


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
    record_window: Optional[Callable[[int, int, np.ndarray], None]] = None,
    noise_rate: float = 0.0,
    noise_kinds: Sequence[str] = ("sub",),
    synthesis_mode: str = "legacy",
    pair_store=None,
    window_plan=None,
    space_token_id: int | None = None,
):
    """Make ``(freq-rank(cipher), plaintext)`` pairs from public plaintext.

    Each sample draws a reference sequence, crops it to ``[min_len, max_len]``,
    enciphers with a fresh random 1:1 key over the whole vocabulary, and
    frequency-rank encodes the cipher. (The 1:1 encoding is key-invariant, so the
    random key does not change ``X``). ``invariant`` skips materialising this
    key; it consumes fewer random numbers, so sampled windows differ from legacy
    for the same seed. ``legacy`` retains the former random draw sequence.

    When ``noise_rate > 0`` the cipher is corrupted with transcription noise
    (2021 Sec 5.4) BEFORE frequency encoding, so the model learns to be robust to
    it. The default ``noise_kinds=("sub",)`` keeps source and target the same
    length; adding ``"ins"``/``"del"`` makes the source length diverge from the
    target (the paper's space/noise-restoration setting, scored with TER).
    """
    pairs = [] if pair_store is None else pair_store
    corpus = [np.asarray(s, dtype=int) for s in corpus if len(s) > 0]
    if not corpus:
        return pairs
    if vocab_size <= 0 or any(np.any((seq < 0) | (seq >= vocab_size)) for seq in corpus):
        raise ValueError("public training tokens must be in the configured vocabulary")
    if synthesis_mode not in {"legacy", "invariant"}:
        raise ValueError("synthesis_mode must be legacy or invariant")
    if synthesis_mode == "invariant" and noise_rate != 0:
        raise ValueError("invariant synthesis requires zero noise")
    if space_token_id is not None and (isinstance(space_token_id, bool) or not isinstance(space_token_id, int)
                                       or not 0 <= space_token_id < vocab_size):
        raise ValueError("space_token_id must be an ID in the public vocabulary")
    full_symbols = list(range(vocab_size)) if synthesis_mode == "legacy" else None
    if window_plan is not None and len(window_plan) != num_samples:
        raise ValueError("window plan length must equal num_samples")
    for index in range(num_samples):
        doc_index, start, plain = enc.select_training_window(corpus, rng, min_len, max_len, window_plan, index)
        if record_window is not None:
            record_window(doc_index, start, plain)
        source_plain = plain if space_token_id is None else plain[plain != space_token_id]
        if not len(source_plain):
            raise ValueError("space removal produced an empty ciphertext")
        if synthesis_mode == "invariant":
            cipher = source_plain
        else:
            key = enc.make_substitution_key(
                full_symbols, rng=rng, homophonic=False, cipher_alphabet_size=vocab_size
            )
            cipher = enc.apply_substitution(source_plain, key)
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
):
    order = rng.permutation(len(pairs))
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
        yield src, tgt_in, tgt_out, src_kpm


# ---------------------------------------------------------------------------
# attack
# ---------------------------------------------------------------------------
@register
class SDA2021(Attack):
    name = "sda.sda2021"
    family = "sda"
    variant = "frequency_rank_lookup_variant"
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

        cipher = np.asarray(ctx.token_ciphertext())
        if cipher.size == 0:
            return AttackResult.failed("empty ciphertext")

        from ...evaluation.sda_stream import _reference_arrays, decode_sda, fit_sda

        cipher = _reference_arrays([cipher])[0]

        p = dict(self.params)
        p.setdefault("epochs", 15)
        seed = p.get("seed", 0)
        bundle = fit_sda(
            ctx.reference_corpus, [cipher], vocab_size=vocab_size, method="sda2021",
            params=p, seed=seed, device=str(p.get("device", "cpu")),
            checkpoint_path=p.get("checkpoint_path"), resume=p.get("resume", False),
        )
        model, cfg = bundle["model"], bundle["cfg"]
        epochs = bundle["epochs"]
        noise_rate = float(p.get("noise_rate", 0.0))
        final_loss = bundle["final_loss"]
        decoded = decode_sda(bundle, cipher, beam_width=p.get("beam_width", 5),
                             mode=p.get("decode_mode", "raw"))
        return AttackResult(
            recovered_token_ids=np.asarray(decoded["tokens"], dtype=int),
            candidates=[[int(t)] for t in decoded["tokens"]],
            recovered_text=" ".join(str(int(t)) for t in decoded["tokens"]),
            recovered_mapping=decoded["mapping"],
            diagnostics={
                "method": "seq2seq_frequency_rank_2021",
                "objective": "L_Seq2Seq = -log P(Y|X)",
                "encoding": "frequency_rank",
                "num_train_pairs": bundle["pairs"],
                "synthesis_mode": bundle["synthesis_mode"],
                "optimizer_steps": bundle["optimizer_steps"],
                "trained_max_code": bundle["trained_max_code"],
                "epochs": epochs,
                "beam_width": int(p.get("beam_width", 5)),
                "final_train_loss": final_loss,
                "final_train_loss_scope": "last_batch_mean_target_nll",
                "training_history": bundle["training_history"],
                "cipher_len": int(cipher.size),
                "src_vocab_size": cfg.src_vocab_size,
                "noise_rate": noise_rate,
                "decode": decoded,
                "candidate_semantics": "one selected hypothesis per position; not top-k",
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
        mode: str = "raw",
        length_mode: str = "fixed",
        max_output_len: int | None = None,
    ) -> dict:
        model.eval()
        device = next(model.parameters()).device
        src_ids = enc.frequency_rank_encode(cipher)
        n = len(src_ids)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_kpm = src == cfg.src_pad
        memory = model.encode(src, src_kpm)  # [1, Ls, d]

        def step(batch_gen: torch.Tensor) -> torch.Tensor:
            b = batch_gen.shape[0]
            bos = torch.full((b, 1), cfg.bos, dtype=torch.long, device=batch_gen.device)
            dec_in = torch.cat([bos, batch_gen], dim=1)
            mem = memory.expand(b, -1, -1)
            mem_kpm = src_kpm.expand(b, -1)
            logits = model.decode(dec_in, mem, memory_kpm=mem_kpm)
            return F.log_softmax(logits[:, -1, :], dim=-1)

        # plaintext ids only
        allowed = torch.arange(cfg.vocab_size, dtype=torch.long, device=device)
        if length_mode == "variable":
            if mode != "raw":
                raise ValueError("variable-length decoding does not support substitution constraints")
            if max_output_len is None:
                raise ValueError("variable-length decoding requires max_output_len")
            tokens, score, beams = variable_length_beam_search(
                step, beam_width, eos_id=cfg.eos, allowed_ids=allowed,
                max_output_len=max_output_len, device=device,
            )
            return {"mode": "variable", "tokens": tokens, "mapping": None,
                    "raw": {"tokens": tokens, "score": score, "beams": beams,
                            "score_includes_eos": True},
                    "constrained": {"status": "not_applicable_variable_length"},
                    "max_output_len": max_output_len}
        if length_mode == "fixed_output":
            if mode != "raw":
                raise ValueError("fixed-output decoding does not support substitution constraints")
            if isinstance(max_output_len, bool) or not isinstance(max_output_len, int) or max_output_len <= 0:
                raise ValueError("fixed-output decoding requires a positive max_output_len")
            tokens, score, beams = length_matched_beam_search(
                step, num_steps=max_output_len, beam_width=beam_width, allowed_ids=allowed, device=device)
            return {"mode": "fixed_output", "tokens": tokens, "mapping": None,
                    "raw": {"tokens": tokens, "score": score, "beams": beams,
                            "score_includes_eos": False},
                    "constrained": {"status": "not_applicable_fixed_output"},
                    "max_output_len": max_output_len}
        if length_mode != "fixed":
            raise ValueError("length_mode must be fixed, variable, or fixed_output")
        if mode not in {"raw", "constrained"}:
            raise ValueError("decode mode must be raw or constrained")
        raw, raw_score, raw_beams = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed, device=device
        )
        injective = True
        if injective and len(set(cipher.tolist())) > cfg.vocab_size:
            if mode == "constrained":
                raise ValueError("injective decoding requires at least one plaintext candidate per cipher symbol")
            return {"mode": mode, "tokens": raw, "mapping": None,
                    "raw": {"tokens": raw, "score": raw_score, "beams": raw_beams},
                    "constrained": {"status": "infeasible", "injective": True}}
        constrained, constrained_score, constrained_beams = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed,
            cipher_symbols=cipher.tolist(), injective=injective, device=device,
        )
        mapping = {int(c): int(t) for c, t in zip(cipher, constrained)}
        return {
            "mode": mode,
            "tokens": raw if mode == "raw" else constrained,
            "mapping": mapping if mode == "constrained" else None,
            "raw": {"tokens": raw, "score": raw_score, "beams": raw_beams},
            "constrained": {"status": "ok", "tokens": constrained, "score": constrained_score,
                            "beams": constrained_beams, "mapping": mapping,
                            "injective": injective},
        }



__all__ = [
    "SDA2021",
    "Seq2SeqCipherModel",
    "Seq2SeqConfig",
    "seq2seq_target_nll",
    "synthesize_pairs",
]
