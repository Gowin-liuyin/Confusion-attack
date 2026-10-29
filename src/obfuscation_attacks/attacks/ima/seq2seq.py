"""IMA *Seq2Seq* inverter - InvBERT Seq2Seq line (Kugler et al., **BB** attack).

InvBERT Seq2Seq is the black-box attack: "We call our GB attack InvBert Classify
and our BB attack InvBert Seq2Seq" (InvBERT Sec. 3.2). In the black box only the
CTEs are available, so the one-to-one CTE<->id map is lost: "Without the
tokenizer, we lose the one-to-one mapping and cannot infer the token CTE ratio.
Thus, we have to train a custom tokenizer and optimize a transformer decoder
structure to predict our sequence of custom input IDs. The decoder utilizes
complete sentence CTEs as generator memory and predicts each token ID
sequentially" (InvBERT Sec. 4.2, InvBERT Seq2Seq (BB)).

Three consequences of the BB threat model are implemented faithfully here:

1. **An attacker-side vocabulary.** The decoder predicts over its *own* token
   vocabulary (InvBERT trains a BytePair tokenizer), not the target tokenizer's
   id space, so there is no assumed one-to-one CTE<->id correspondence. In this
   synthetic harness the attacker vocabulary shares the integer id space of the
   public corpus only so that recovery is measurable; the decoder itself never
   uses a CTE<->id lookup.
2. **Complete-sentence CTEs as memory, carrying position.** The whole sequence
   of obfuscated embeddings is the cross-attention memory. A positional encoding
   is added to the memory rows, so the memory is *order-aware*: permuting the
   memory rows changes the output (without it, cross-attention is a set
   operation and the decoder cannot order its predictions).
3. **Variable-length decoding until EOS.** The output length is produced by the
   decoder (it emits an end-of-sequence token), never read from the input
   length. Memory length and target length may therefore differ.

Formulae (docs/FORMULA_LEDGER.md: IMA-CE-1, IMA-S2S-1)::

    L = - sum_t log p_theta(y_t | context)              (IMA-CE-1)
    P(y | h) = prod_{t=1..N} p_theta(y_t | y_<t, h)      (IMA-S2S-1)

``h`` is the full sequence of obfuscated embeddings (decoder memory) and
``y_<t`` the previously generated tokens. Teacher forcing feeds the shifted gold
prefix ``[BOS, y_0, ..., y_{N-1}]`` and trains against ``[y_0, ..., y_{N-1},
EOS]`` under a causal mask, so the logit at step ``t`` depends only on ``y_<t``
(and all of ``h``). At attack time we decode greedily until EOS.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from .classify import cross_entropy_objective


class Seq2SeqDecoderInverter(nn.Module):
    """Tiny Transformer-decoder inverter implementing IMA-S2S-1.

    Token id layout over the attacker vocabulary of ``vocab_size`` content ids:

    * content ids ``0 .. vocab_size-1`` are the predictable tokens;
    * ``eos_id = vocab_size`` is the (predictable) end-of-sequence marker;
    * ``bos_id = vocab_size + 1`` is embedded as the decoder start token but is
      never produced (it is not among the head's output classes).

    The obfuscated embeddings are linearly projected to ``d_model`` and a
    positional encoding is added, giving an *order-aware* cross-attention memory
    (the "complete sentence CTEs"). Target tokens are embedded, positionally
    encoded, decoded under a causal mask, and projected to logits over the
    ``vocab_size + 1`` emittable ids (content + EOS).
    """

    def __init__(
        self,
        obf_dim: int,
        vocab_size: int,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 64,
        max_mem_len: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        self.obf_dim = int(obf_dim)
        self.vocab_size = int(vocab_size)
        self.eos_id = self.vocab_size  # predictable end-of-sequence token
        self.bos_id = self.vocab_size + 1  # embedded start token, never predicted
        self.num_emittable = self.vocab_size + 1  # content ids + EOS
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.max_mem_len = int(max_mem_len)

        self.tok_embed = nn.Embedding(self.vocab_size + 2, self.d_model)  # content + EOS + BOS
        self.pos_embed = nn.Embedding(self.max_len, self.d_model)  # decoder-side positions
        self.mem_proj = nn.Linear(self.obf_dim, self.d_model)
        self.mem_pos_embed = nn.Embedding(self.max_mem_len, self.d_model)  # memory-side positions
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=0.0,  # deterministic forward (no train/eval divergence)
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.head = nn.Linear(self.d_model, self.num_emittable)  # content + EOS logits

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device), diagonal=1
        )

    def encode_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Project the obfuscated embeddings and add a positional encoding.

        The positional term makes the memory order-aware: two memories that are
        row-permutations of each other encode to different tensors, so the
        decoder can use CTE order (InvBERT's "complete sentence CTEs").
        """
        if not torch.is_tensor(memory):
            memory = torch.as_tensor(np.asarray(memory, dtype=np.float32))
        memory = memory.float()
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        M = memory.shape[1]
        if M > self.max_mem_len:
            raise ValueError(f"memory length {M} exceeds max_mem_len {self.max_mem_len}")
        mem_pos = torch.arange(M, device=memory.device)
        return self.mem_proj(memory) + self.mem_pos_embed(mem_pos).unsqueeze(0)

    def forward(self, memory, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward.

        ``memory``: (B, M, obf_dim) obfuscated embeddings (M independent of T).
        ``tgt_in``: (B, T) decoder input ids (position 0 is BOS). Returns logits
        of shape (B, T, vocab_size + 1); the logit at step ``t`` depends only on
        ``tgt_in[:, :t+1]`` thanks to the causal mask.
        """
        if tgt_in.ndim == 1:
            tgt_in = tgt_in.unsqueeze(0)
        _, T = tgt_in.shape
        if T > self.max_len:
            raise ValueError(f"target length {T} exceeds max_len {self.max_len}")
        mem = self.encode_memory(memory)
        pos = torch.arange(T, device=tgt_in.device)
        x = self.tok_embed(tgt_in) + self.pos_embed(pos).unsqueeze(0)
        mask = self._causal_mask(T, tgt_in.device)
        out = self.decoder(tgt=x, memory=mem, tgt_mask=mask)
        return self.head(out)

    def teacher_forcing_inputs(self, targets: torch.Tensor) -> torch.Tensor:
        """Build the decoder input ``[BOS, y_0, ..., y_{N-1}]`` (length ``N+1``).

        Aligned with :meth:`teacher_forcing_targets` so that output step ``t``
        predicts that method's entry ``t`` from ``[BOS, y_<t]`` (shifted right).
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        b = targets.shape[0]
        bos = torch.full((b, 1), self.bos_id, dtype=torch.long, device=targets.device)
        return torch.cat([bos, targets], dim=1)

    def teacher_forcing_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Build the gold output ``[y_0, ..., y_{N-1}, EOS]`` (length ``N+1``).

        The trailing EOS teaches the decoder to *terminate*, so decoding length
        is learned rather than copied from the input.
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        b = targets.shape[0]
        eos = torch.full((b, 1), self.eos_id, dtype=torch.long, device=targets.device)
        return torch.cat([targets, eos], dim=1)


def train_seq2seq(
    model: Seq2SeqDecoderInverter,
    memories: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """Teacher-forced cross-entropy training (per-sample; variable lengths).

    Each target is trained against ``[y_0, ..., y_{N-1}, EOS]`` so the decoder
    learns to emit the end-of-sequence token; the memory length need not equal
    the target length.
    """
    seed_everything(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mem_t = [torch.as_tensor(np.asarray(m, dtype=np.float32)) for m in memories]
    tgt_t = [torch.as_tensor(np.asarray(t)).long() for t in targets]
    history: list[float] = []
    model.train()
    n = len(tgt_t)
    for _ in range(int(epochs)):
        order = torch.randperm(n)
        running = 0.0
        for i in order.tolist():
            y = tgt_t[i].unsqueeze(0)  # (1, N)
            tgt_in = model.teacher_forcing_inputs(y)  # (1, N+1)
            tgt_out = model.teacher_forcing_targets(y)  # (1, N+1)
            logits = model(mem_t[i].unsqueeze(0), tgt_in)  # (1, N+1, V+1)
            loss = cross_entropy_objective(
                logits.reshape(-1, model.num_emittable), tgt_out.reshape(-1), reduction="mean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        history.append(running / max(n, 1))
    return history


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqDecoderInverter, memory, max_len: int
) -> np.ndarray:
    """Greedy autoregressive decode from ``memory`` until EOS or ``max_len``.

    Starts from BOS and, at each step, appends ``argmax p_theta(y_t | y_<t, h)``
    - the greedy realisation of the IMA-S2S-1 factorisation. Decoding stops when
    EOS is emitted, so the output length is produced by the model and is *not*
    read from the memory length; ``max_len`` is only a hard cap. Content ids
    (``< vocab_size``) are returned; EOS/BOS are never part of the output.
    """
    model.eval()
    mem = torch.as_tensor(np.asarray(memory, dtype=np.float32))
    if mem.ndim == 2:
        mem = mem.unsqueeze(0)
    cap = min(int(max_len), model.max_len - 1)
    ids = [model.bos_id]
    out: list[int] = []
    for _ in range(max(cap, 0)):
        tgt_in = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        logits = model(mem, tgt_in)  # (1, t+1, V+1)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == model.eos_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return np.asarray(out, dtype=int)


@register
class Seq2SeqInverter(Attack):
    """InvBERT Seq2Seq (BB): order-aware decoder over CTEs, variable-length decode."""

    name = "ima.seq2seq"
    family = "ima"
    variant = "original"
    required_capabilities = frozenset(
        {Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}
    )

    @staticmethod
    def build_decoder(
        obf_dim: int, vocab_size: int, *, seed: int = 0, **kw
    ) -> Seq2SeqDecoderInverter:
        return Seq2SeqDecoderInverter(obf_dim, vocab_size, seed=seed, **kw)

    def _build_pairs(
        self, ctx: AttackContext
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        memories: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for s in ctx.reference_corpus:
            seq = np.asarray(s).astype(int).ravel()
            if seq.size == 0:
                continue
            obf = np.asarray(ctx.obfuscation_fn(seq), dtype=float)
            if obf.ndim != 2 or obf.shape[0] != seq.size:
                raise ValueError(
                    f"obfuscation_fn must map (len,) -> (len, d); got {obf.shape} for "
                    f"sequence length {seq.size}"
                )
            memories.append(obf)
            targets.append(seq)
        return memories, targets

    def _run(self, ctx: AttackContext) -> AttackResult:
        if ctx.obfuscation_fn is None:
            return AttackResult.not_applicable(
                "Seq2Seq-IMA needs obfuscation_fn (KNOWN_OBFUSCATION_ALGORITHM)"
            )
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "Seq2Seq-IMA needs a public reference_corpus to build training pairs"
            )
        vocab = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot size the decoder head")

        H = np.asarray(ctx.obfuscated_embeddings(), dtype=float)
        if H.ndim != 2:
            return AttackResult.failed(f"obfuscated embeddings must be 2-D; got {H.shape}")

        try:
            memories, targets = self._build_pairs(ctx)
        except ValueError as exc:
            return AttackResult.failed(str(exc))
        if not targets:
            return AttackResult.failed("no training pairs could be generated from the corpus")
        if memories[0].shape[1] != H.shape[1]:
            return AttackResult.failed(
                f"obf embedding dim mismatch: train d={memories[0].shape[1]}, eval d={H.shape[1]}"
            )

        seed = int(self.params.get("seed", 0))
        max_tgt = max(int(t.size) for t in targets)
        max_mem = max(max(int(m.shape[0]) for m in memories), int(H.shape[0]))
        # Decode length is produced by the model (EOS), so it is bounded by an
        # explicit cap rather than by the evaluated memory length.
        max_decode_len = int(self.params.get("max_decode_len", max_tgt + 5))
        # +2 head-room for the shifted BOS and the trailing EOS.
        max_len = int(self.params.get("max_len", max(max_tgt, max_decode_len) + 2))
        max_mem_len = int(self.params.get("max_mem_len", max_mem + 1))

        model = Seq2SeqDecoderInverter(
            memories[0].shape[1],
            vocab,
            d_model=int(self.params.get("d_model", 64)),
            nhead=int(self.params.get("nhead", 4)),
            num_layers=int(self.params.get("num_layers", 2)),
            dim_feedforward=int(self.params.get("dim_feedforward", 128)),
            max_len=max_len,
            max_mem_len=max_mem_len,
            seed=seed,
        )
        history = train_seq2seq(
            model,
            memories,
            targets,
            epochs=int(self.params.get("epochs", 200)),
            lr=float(self.params.get("lr", 1e-2)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
            seed=seed,
        )
        recovered = greedy_decode(model, H, max_decode_len)

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=[[int(t)] for t in recovered],
            diagnostics={
                "num_train_pairs": len(targets),
                "train_final_loss": float(history[-1]) if history else None,
                "vocab_size": vocab,
                "embed_dim": int(memories[0].shape[1]),
                "num_positions": int(H.shape[0]),
                "recovered_length": int(recovered.size),
                "epochs": int(self.params.get("epochs", 200)),
                "optimizer": "AdamW",
                "loss": "cross_entropy",
                "decode": "greedy_until_eos",
                "attacker_vocabulary": True,
                "memory_position_encoding": True,
            },
        )
