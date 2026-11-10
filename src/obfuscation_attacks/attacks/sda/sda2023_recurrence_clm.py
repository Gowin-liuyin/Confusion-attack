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
   ciphertext (1:1 and homophonic substitutions both preserve length). The
   raw hypothesis is retained; an engineering constrained beam is reported
   separately, using injectivity only for 1:1 ciphers. Recurrence values use
   lookup embeddings, not the paper-described character CNN/highway encoder;
   this is an engineering variant, not a verified faithful reproduction.

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
from typing import Callable, Optional, Sequence

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
    representation: str = "lookup"

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


class DecimalRecurrenceEncoder(nn.Module):
    """Experimental decimal-digit CNN/highway encoder for recurrence IDs.

    Digit tokenization, padding, projection and mixed plaintext embeddings are
    engineering choices; they are not verified against the unavailable author code.
    """

    def __init__(self, max_rec: int, d_model: int) -> None:
        super().__init__()
        if max_rec <= 0:
            raise ValueError("max_rec must be positive")
        width = max(4, len(str(max_rec)))
        digits = torch.zeros(max_rec, width, dtype=torch.long)
        for number in range(1, max_rec + 1):
            values = [int(ch) + 1 for ch in str(number)]
            digits[number - 1, :len(values)] = torch.tensor(values)
        self.register_buffer("digits", digits)
        self.characters = nn.Embedding(11, 4, padding_idx=0)
        filters = ((1, 64), (2, 128), (3, 192), (4, 256))
        self.convs = nn.ModuleList(nn.Conv1d(4, channels, kernel) for kernel, channels in filters)
        width_out = sum(channels for _, channels in filters)
        self.highway_transform = nn.ModuleList(nn.Linear(width_out, width_out) for _ in range(2))
        self.highway_gate = nn.ModuleList(nn.Linear(width_out, width_out) for _ in range(2))
        self.projection = nn.Linear(width_out, d_model)

    def forward(self, rec_codes: torch.Tensor) -> torch.Tensor:
        if rec_codes.numel() and (int(rec_codes.min()) < 1 or int(rec_codes.max()) > len(self.digits)):
            raise ValueError("recurrence code outside configured range")
        unique, inverse = torch.unique(rec_codes.reshape(-1), sorted=True, return_inverse=True)
        chars = self.characters(self.digits[unique - 1]).transpose(1, 2)
        features = torch.cat([F.relu(conv(chars)).amax(dim=-1) for conv in self.convs], dim=-1)
        for transform, gate in zip(self.highway_transform, self.highway_gate):
            weight = torch.sigmoid(gate(features))
            features = weight * F.relu(transform(features)) + (1 - weight) * features
        return self.projection(features)[inverse].reshape(*rec_codes.shape, -1)


class CausalCipherLM(nn.Module):
    """Tiny decoder-only (causal) Transformer LM over the shared vocabulary."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.representation not in {"lookup", "decimal_char_cnn"}:
            raise ValueError("unknown recurrence representation")
        self.emb = nn.Embedding(cfg.total_vocab, cfg.d_model, padding_idx=cfg.pad)
        self.recurrence_encoder = (DecimalRecurrenceEncoder(cfg.max_rec, cfg.d_model)
                                   if cfg.representation == "decimal_char_cnn" else None)
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
        h = self.emb(tokens)
        if self.recurrence_encoder is not None:
            is_recurrence = (tokens >= self.cfg.rec_lo) & (tokens < self.cfg.rec_hi)
            if is_recurrence.any():
                h = h.clone()
                codes = tokens[is_recurrence] - self.cfg.rec_lo + 1
                h[is_recurrence] = self.recurrence_encoder(codes)
        h = self.pos(h)
        h = self.blocks(h, mask=mask)
        return self.head(self.norm(h))


class CachedCausalStep:
    """Inference-only KV caching for this module's post-norm causal Transformer.

    A beam callback advances only selected prefixes and discards previous-layer
    generations. It caches no scoring truth and does not mutate the model.
    """

    def __init__(self, model: CausalCipherLM, prompt: torch.Tensor) -> None:
        if model.training or prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError("KV cache requires an eval model and one nonempty prompt")
        if model.blocks.norm is not None:
            raise ValueError("KV cache does not support an encoder-level norm")
        for layer in model.blocks.layers:
            attention = layer.self_attn
            if (layer.training or layer.norm_first or not attention.batch_first
                    or not attention._qkv_same_embed_dim or attention.bias_k is not None
                    or attention.bias_v is not None or attention.add_zero_attn):
                raise ValueError("unsupported Transformer layout for incremental SDA decoding")
        self.model = model
        self.prompt_length = int(prompt.shape[1])
        self.root_logits, self.root_state = self._prefill(prompt)
        self.previous = {}
        self.previous_length = None

    def _embed(self, tokens: torch.Tensor, position: int) -> torch.Tensor:
        model = self.model
        length = tokens.shape[1]
        if position + length > model.pos.pe.shape[1]:
            raise ValueError("sequence length exceeds positional capacity")
        result = model.emb(tokens)
        if model.recurrence_encoder is not None:
            mask = (tokens >= model.cfg.rec_lo) & (tokens < model.cfg.rec_hi)
            if mask.any():
                result = result.clone()
                codes = tokens[mask] - model.cfg.rec_lo + 1
                result[mask] = model.recurrence_encoder(codes)
        return result + model.pos.pe[:, position:position + length]

    @staticmethod
    def _project_qkv(attention, hidden):
        projected = F.linear(hidden, attention.in_proj_weight, attention.in_proj_bias)
        batch, length, _ = hidden.shape
        heads = attention.num_heads
        depth = attention.embed_dim // heads
        return tuple(value.reshape(batch, length, heads, depth).transpose(1, 2)
                     for value in projected.chunk(3, dim=-1))

    @torch.no_grad()
    def _prefill(self, prompt):
        hidden = self._embed(prompt, 0)
        mask = causal_mask(prompt.shape[1], prompt.device)
        state = []
        for layer in self.model.blocks.layers:
            _, key, value = self._project_qkv(layer.self_attn, hidden)
            state.append((key, value))
            hidden = layer(hidden, src_mask=mask, is_causal=True)
        logits = self.model.head(self.model.norm(hidden[:, -1, :]))
        return F.log_softmax(logits, dim=-1), state

    @torch.no_grad()
    def _advance(self, token, state, position):
        hidden = self._embed(token, position)
        next_state = []
        for layer, (old_key, old_value) in zip(self.model.blocks.layers, state):
            attention = layer.self_attn
            query, key, value = self._project_qkv(attention, hidden)
            key = torch.cat((old_key, key), dim=2)
            value = torch.cat((old_value, value), dim=2)
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            attended = attended.transpose(1, 2).contiguous().view(hidden.shape[0], 1, attention.embed_dim)
            attended = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
            hidden = layer.norm1(hidden + layer.dropout1(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
            hidden = layer.norm2(hidden + layer.dropout2(feedforward))
            next_state.append((key, value))
        logits = self.model.head(self.model.norm(hidden[:, -1, :]))
        return F.log_softmax(logits, dim=-1), next_state

    @torch.no_grad()
    def __call__(self, batch_gen: torch.Tensor) -> torch.Tensor:
        if batch_gen.ndim != 2:
            raise ValueError("beam prefixes must be a matrix")
        length = int(batch_gen.shape[1])
        if length == 0:
            self.previous = {(): self.root_state}
            self.previous_length = 0
            return self.root_logits.expand(batch_gen.shape[0], -1)
        if self.previous_length != length - 1:
            raise ValueError("cached beam prefixes must advance one token per step")
        updated, scores = {}, []
        for prefix in batch_gen.tolist():
            parent = tuple(prefix[:-1])
            if parent not in self.previous:
                raise ValueError("cached beam prefix has no parent state")
            token = torch.tensor([[prefix[-1]]], dtype=torch.long, device=batch_gen.device)
            score, state = self._advance(token, self.previous[parent], self.prompt_length + length - 1)
            scores.append(score)
            updated[tuple(prefix)] = state
        self.previous, self.previous_length = updated, length
        return torch.cat(scores, dim=0)


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
    record_window: Optional[Callable[[int, int, np.ndarray], None]] = None,
    approx_num_symbols: Optional[int] = None,
    synthesis_mode: str = "legacy",
    pair_store=None,
    window_plan=None,
):
    """Make ``(recurrence(cipher), plaintext)`` pairs from public plaintext.

    For 1:1 keys the recurrence encoding is key-invariant; for homophonic keys
    each occurrence of a plaintext symbol may map to a different cipher symbol,
    so the random key genuinely diversifies ``X`` and the model must learn which
    recurrence codes are homophones of one another. The invariant mode avoids
    key materialisation but changes random-number consumption and hence the
    sampled windows for a fixed seed. Homophonic allocation uses only symbols
    observed in each window; its default alphabet size is twice that count.
    """
    pairs = [] if pair_store is None else pair_store
    corpus = [np.asarray(s, dtype=int) for s in corpus if len(s) > 0]
    if not corpus:
        return pairs
    if vocab_size <= 0 or any(np.any((seq < 0) | (seq >= vocab_size)) for seq in corpus):
        raise ValueError("public training tokens must be in the configured vocabulary")
    if synthesis_mode not in {"legacy", "invariant"}:
        raise ValueError("synthesis_mode must be legacy or invariant")
    if synthesis_mode == "invariant" and homophonic:
        raise ValueError("invariant synthesis is not valid for homophonic ciphers")
    if approx_num_symbols is not None and (isinstance(approx_num_symbols, bool) or
            not isinstance(approx_num_symbols, (int, np.integer)) or approx_num_symbols <= 0):
        raise ValueError("approx_num_symbols must be a positive integer")
    full_symbols = list(range(vocab_size)) if synthesis_mode == "legacy" and not homophonic else None
    if window_plan is not None and len(window_plan) != num_samples:
        raise ValueError("window plan length must equal num_samples")
    for index in range(num_samples):
        doc_index, start, plain = enc.select_training_window(corpus, rng, min_len, max_len, window_plan, index)
        if record_window is not None:
            record_window(doc_index, start, plain)

        if synthesis_mode == "invariant":
            cipher = plain
        elif homophonic:
            counts = {int(t): int(c) for t, c in zip(*np.unique(plain, return_counts=True))}
            key = enc.make_substitution_key(
                counts,
                rng=rng,
                homophonic=True,
                plain_counts=counts,
                approx_num_symbols=approx_num_symbols if approx_num_symbols is not None else 2 * len(counts),
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
):
    order = rng.permutation(len(pairs))
    for b in range(0, len(order), batch_size):
        idxs = order[b : b + batch_size]
        examples = [make_clm_example(pairs[i][0], pairs[i][1], cfg) for i in idxs]
        length = max(len(inp) for inp, _ in examples)
        inputs = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
        labels = torch.full((len(examples), length), cfg.pad, dtype=torch.long)
        for r, (inp, lab) in enumerate(examples):
            inputs[r, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            labels[r, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        yield inputs, labels


# ---------------------------------------------------------------------------
# attack
# ---------------------------------------------------------------------------
@register
class SDA2023(Attack):
    name = "sda.sda2023"
    family = "sda"
    variant = "recurrence_lookup_variant"
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

        cipher = np.asarray(ctx.token_ciphertext())
        if cipher.size == 0:
            return AttackResult.failed("empty ciphertext")

        from ...evaluation.sda_stream import _reference_arrays, decode_sda, fit_sda

        cipher = _reference_arrays([cipher])[0]

        p = dict(self.params)
        p.setdefault("epochs", 20)
        seed = p.get("seed", 0)
        bundle = fit_sda(
            ctx.reference_corpus, [cipher], vocab_size=vocab_size, method="sda2023",
            params=p, seed=seed, device=str(p.get("device", "cpu")),
            checkpoint_path=p.get("checkpoint_path"), resume=p.get("resume", False),
        )
        model, cfg = bundle["model"], bundle["cfg"]
        epochs = bundle["epochs"]
        homophonic = bool(p.get("homophonic", False))
        max_rec = cfg.max_rec
        final = bundle["final_loss"]
        decoded = decode_sda(bundle, cipher, beam_width=p.get("beam_width", 5),
                             mode=p.get("decode_mode", "raw"), use_cache=p.get("decode_cache", False))
        return AttackResult(
            recovered_token_ids=np.asarray(decoded["tokens"], dtype=int),
            candidates=[[int(t)] for t in decoded["tokens"]],
            recovered_text=" ".join(str(int(t)) for t in decoded["tokens"]),
            recovered_mapping=decoded["mapping"],
            diagnostics={
                "method": "causal_lm_recurrence_2023",
                "objective": "L_CLM = -log P(X) - log P(Y|X)",
                "encoding": "recurrence",
                "homophonic": homophonic,
                "num_train_pairs": bundle["pairs"],
                "synthesis_mode": bundle["synthesis_mode"],
                "optimizer_steps": bundle["optimizer_steps"],
                "trained_max_code": bundle["trained_max_code"],
                "epochs": epochs,
                "beam_width": int(p.get("beam_width", 5)),
                "max_rec_code": int(max_rec),
                "final_train_loss_total": final["total"],
                "final_train_loss_src": final["src"],
                "final_train_loss_tgt": final["tgt"],
                "final_train_loss_scope": "last_batch_summed_nll",
                "training_history": bundle["training_history"],
                "cipher_len": int(cipher.size),
                "decode": decoded,
                "candidate_semantics": "one selected hypothesis per position; not top-k",
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
        mode: str = "raw",
        homophonic: bool = False,
        use_cache: bool = False,
    ) -> dict:
        model.eval()
        device = next(model.parameters()).device
        n = len(cipher_rec)
        prompt_ids = [cfg.rec_tag] + [cfg.rec_id(int(k)) for k in cipher_rec] + [cfg.plain_tag]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        def step(batch_gen: torch.Tensor) -> torch.Tensor:
            b = batch_gen.shape[0]
            pref = prompt.expand(b, -1)
            full = torch.cat([pref, batch_gen], dim=1)
            logits = model(full)
            return F.log_softmax(logits[:, -1, :], dim=-1)

        if use_cache:
            step = CachedCausalStep(model, prompt)
        allowed = torch.arange(cfg.vocab_size, dtype=torch.long, device=device)
        if mode not in {"raw", "constrained"}:
            raise ValueError("decode mode must be raw or constrained")
        raw, raw_score, raw_beams = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed, device=device
        )
        injective = not homophonic
        if injective and len(set(cipher.tolist())) > cfg.vocab_size:
            if mode == "constrained":
                raise ValueError("injective decoding requires at least one plaintext candidate per cipher symbol")
            return {"mode": mode, "tokens": raw, "mapping": None,
                    "cache_mode": "kv_incremental" if use_cache else "full_prefix",
                    "raw": {"tokens": raw, "score": raw_score, "beams": raw_beams},
                    "constrained": {"status": "infeasible", "injective": True}}
        constrained, constrained_score, constrained_beams = length_matched_beam_search(
            step, num_steps=n, beam_width=beam_width, allowed_ids=allowed,
            cipher_symbols=cipher.tolist(), injective=injective, device=device,
        )
        mapping = {int(c): int(t) for c, t in zip(cipher, constrained)}
        return {
            "mode": mode,
            "cache_mode": "kv_incremental" if use_cache else "full_prefix",
            "tokens": raw if mode == "raw" else constrained,
            "mapping": mapping if mode == "constrained" else None,
            "raw": {"tokens": raw, "score": raw_score, "beams": raw_beams},
            "constrained": {"status": "ok", "tokens": constrained, "score": constrained_score,
                            "beams": constrained_beams, "mapping": mapping,
                            "injective": injective},
        }



__all__ = [
    "SDA2023",
    "CausalCipherLM",
    "CachedCausalStep",
    "DecimalRecurrenceEncoder",
    "CausalLMConfig",
    "clm_losses",
    "make_clm_example",
    "build_full_sequence",
    "synthesize_pairs",
]
