"""IMA *Seq2Seq* inverter - InvBERT Seq2Seq line (Kugler et al., BB attack).

In the black-box scenario only the contextualized token embeddings (CTEs) are
available: "the decoder utilizes complete sentence CTEs as generator memory and
predicts each token ID sequentially" (InvBERT 4.2). We reproduce that with a
tiny Transformer decoder whose cross-attention memory is the (projected) whole
sequence of obfuscated embeddings and whose self-attention is causal, trained
by teacher forcing with the cross-entropy objective.

Formulae (docs/FORMULA_LEDGER.md: IMA-CE-1, IMA-S2S-1)::

    L = - sum_t log p_theta(y_t | context)              (IMA-CE-1)
    P(y | h) = prod_{t=1..N} p_theta(y_t | y_<t, h)      (IMA-S2S-1)

``h`` is the full sequence of obfuscated embeddings (decoder memory) and
``y_<t`` the previously generated tokens. Teacher forcing feeds the shifted
gold prefix ``[BOS, y_0, ..., y_{t-1}]`` while a causal mask guarantees the
logit at step ``t`` depends only on ``y_<t`` (and all of ``h``). At attack time
we decode greedily.
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

    The obfuscated embeddings are linearly projected to ``d_model`` and used as
    cross-attention *memory* (the "complete sentence CTEs"); target tokens are
    embedded (with a dedicated BOS id) and positionally encoded, decoded under a
    causal mask, and projected to vocabulary logits.
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
        seed: int = 0,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        self.obf_dim = int(obf_dim)
        self.vocab_size = int(vocab_size)
        self.bos_id = self.vocab_size  # dedicated BOS, embedded but never predicted
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        self.tok_embed = nn.Embedding(self.vocab_size + 1, self.d_model)
        self.pos_embed = nn.Embedding(self.max_len, self.d_model)
        self.mem_proj = nn.Linear(self.obf_dim, self.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=0.0,  # deterministic forward (no train/eval divergence)
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.head = nn.Linear(self.d_model, self.vocab_size)  # vocab logits (no BOS)

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device), diagonal=1
        )

    def forward(self, memory, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward.

        ``memory``: (B, N, obf_dim) obfuscated embeddings.
        ``tgt_in``: (B, T) decoder input ids (position 0 is BOS). Returns logits
        of shape (B, T, vocab); the logit at step ``t`` depends only on
        ``tgt_in[:, :t+1]`` thanks to the causal mask.
        """
        if not torch.is_tensor(memory):
            memory = torch.as_tensor(np.asarray(memory, dtype=np.float32))
        memory = memory.float()
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if tgt_in.ndim == 1:
            tgt_in = tgt_in.unsqueeze(0)
        _, T = tgt_in.shape
        if T > self.max_len:
            raise ValueError(f"target length {T} exceeds max_len {self.max_len}")
        mem = self.mem_proj(memory)
        pos = torch.arange(T, device=tgt_in.device)
        x = self.tok_embed(tgt_in) + self.pos_embed(pos).unsqueeze(0)
        mask = self._causal_mask(T, tgt_in.device)
        out = self.decoder(tgt=x, memory=mem, tgt_mask=mask)
        return self.head(out)

    def teacher_forcing_inputs(self, targets: torch.Tensor) -> torch.Tensor:
        """Build ``[BOS, y_0, ..., y_{N-1}]`` from gold ``[y_0, ..., y_{N-1}]``.

        The returned length-``N`` decoder input, aligned with ``targets`` so that
        output step ``t`` predicts ``y_t`` from ``[BOS, y_<t]`` (shifted right).
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        b = targets.shape[0]
        bos = torch.full((b, 1), self.bos_id, dtype=torch.long, device=targets.device)
        return torch.cat([bos, targets[:, :-1]], dim=1)


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
    """Teacher-forced cross-entropy training (per-sample; variable lengths)."""
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
            tgt_in = model.teacher_forcing_inputs(y)
            logits = model(mem_t[i].unsqueeze(0), tgt_in)  # (1, N, V)
            loss = cross_entropy_objective(
                logits.reshape(-1, model.vocab_size), y.reshape(-1), reduction="mean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        history.append(running / max(n, 1))
    return history


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqDecoderInverter, memory, length: int
) -> np.ndarray:
    """Greedy autoregressive decode of exactly ``length`` tokens from ``memory``.

    Starts from BOS and, at each step, appends ``argmax p_theta(y_t | y_<t, h)``
    - the greedy realisation of the IMA-S2S-1 factorisation.
    """
    model.eval()
    mem = torch.as_tensor(np.asarray(memory, dtype=np.float32))
    if mem.ndim == 2:
        mem = mem.unsqueeze(0)
    ids = [model.bos_id]
    for _ in range(int(length)):
        tgt_in = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        logits = model(mem, tgt_in)  # (1, t+1, V)
        nxt = int(logits[0, -1].argmax().item())
        ids.append(nxt)
    return np.asarray(ids[1:], dtype=int)


@register
class Seq2SeqInverter(Attack):
    """InvBERT Seq2Seq: decoder over obfuscated CTEs, greedy decode at attack time."""

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
        max_len = int(self.params.get("max_len", max(H.shape[0], *(m.shape[0] for m in memories)) + 1))
        model = Seq2SeqDecoderInverter(
            memories[0].shape[1],
            vocab,
            d_model=int(self.params.get("d_model", 64)),
            nhead=int(self.params.get("nhead", 4)),
            num_layers=int(self.params.get("num_layers", 2)),
            dim_feedforward=int(self.params.get("dim_feedforward", 128)),
            max_len=max_len,
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
        recovered = greedy_decode(model, H, H.shape[0])

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=[[int(t)] for t in recovered],
            diagnostics={
                "num_train_pairs": len(targets),
                "train_final_loss": float(history[-1]) if history else None,
                "vocab_size": vocab,
                "embed_dim": int(memories[0].shape[1]),
                "num_positions": int(H.shape[0]),
                "epochs": int(self.params.get("epochs", 200)),
                "optimizer": "AdamW",
                "loss": "cross_entropy",
                "decode": "greedy",
            },
        )
