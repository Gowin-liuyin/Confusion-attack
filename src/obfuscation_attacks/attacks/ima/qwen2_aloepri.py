"""IMA AloePri Qwen2 inverter (AloePri App. D.1, IMA paragraph).

AloePri's inversion-model attack (App. D.1): "With knowledge of the obfuscation
mechanism, the attacker can train a model for embedding inversion. During the
training process, the attacker iterates over a public training dataset and
generates obfuscated embeddings using the target obfuscation mechanism. With
these obfuscated embeddings as inputs, the attacker trains the model to generate
raw plaintext embeddings or token indices. In the experiments, we train a Qwen2
model with 2 decoder layers and 8 attention heads to invert obfuscated
embeddings to plaintext token embeddings."

Two output paths follow directly from that paragraph:

* **embedding-regression (default - the paper experiment).** The paper's
  *experiment* inverts "to plaintext token embeddings", so the default target is
  the clean plaintext token embedding ``e_t`` at each position. The paper does
  NOT state a loss function; this module trains the path with a mean-squared
  error ``L_MSE = (1/N) sum ||e_hat_t - e_t||^2`` purely as an implementation
  choice. AloePri is *not* claimed to use MSE - only the inversion *target*
  (embeddings) is the paper's. At attack time a predicted embedding is mapped
  back to a token id by nearest neighbour in the (public, attacker-known) clean
  embedding table.
* **token-index (clearly-labelled alternative).** App. D.1 also lists "or token
  indices" as a possible target, so a cross-entropy token-index head
  ``L = - sum_t log p_theta(y_t | context)`` is provided as an alternative. It
  is NOT the path used in the paper's experiment and is labelled as such.

The backbone is a HuggingFace ``Qwen2Model`` configured with EXACTLY
``num_hidden_layers=2`` and ``num_attention_heads=8`` (App. D.1). The obfuscated
embeddings are projected to the backbone hidden size and fed as ``inputs_embeds``;
the backbone's last hidden state feeds whichever output head is selected.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from transformers import Qwen2Config, Qwen2Model

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from .classify import cross_entropy_objective

# AloePri's fixed inverter architecture (App. D.1 "2 decoder layers and 8
# attention heads"): these are paper-specified and not tunable.
REQUIRED_NUM_HIDDEN_LAYERS = 2
REQUIRED_NUM_ATTENTION_HEADS = 8
# ASSUMPTION (UNSPECIFIED): App. D.1 gives the attention-head count but says
# nothing about the number of key/value heads (Qwen2 supports grouped-query
# attention with num_key_value_heads <= num_attention_heads). We assume standard
# multi-head attention, i.e. num_key_value_heads == num_attention_heads == 8.
# This is an assumption, not a paper value (recorded in configs/aloepri/ima_qwen2.yaml).
REQUIRED_NUM_KEY_VALUE_HEADS = 8

# Objective aliases: the embedding-regression path is the paper-experiment
# default; "mse" is accepted as a legacy alias for it (MSE is only the loss used
# to train that path, not the paper's stated objective). The token-index path is
# the cross-entropy alternative; "ce" is its legacy alias.
_OBJECTIVE_ALIASES = {
    "embedding": "embedding",
    "embeddings": "embedding",
    "mse": "embedding",
    "token_index": "token_index",
    "token": "token_index",
    "ce": "token_index",
    "cross_entropy": "token_index",
}


class Qwen2InversionModel(nn.Module):
    """Qwen2 backbone (2 layers / 8 heads) with embedding- and token-index heads.

    The layer/head counts are hard-set to the AloePri specification and cannot be
    overridden through this constructor.
    """

    def __init__(
        self,
        obf_dim: int,
        vocab_size: int,
        *,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        max_position_embeddings: int = 64,
        plain_embed_dim: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        if hidden_size % REQUIRED_NUM_ATTENTION_HEADS != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({REQUIRED_NUM_ATTENTION_HEADS})"
            )
        self.obf_dim = int(obf_dim)
        self.vocab_size = int(vocab_size)
        self.config = Qwen2Config(
            vocab_size=int(vocab_size),
            hidden_size=int(hidden_size),
            num_hidden_layers=REQUIRED_NUM_HIDDEN_LAYERS,
            num_attention_heads=REQUIRED_NUM_ATTENTION_HEADS,
            num_key_value_heads=REQUIRED_NUM_KEY_VALUE_HEADS,  # UNSPECIFIED assumption (see above)
            intermediate_size=int(intermediate_size),
            max_position_embeddings=int(max_position_embeddings),
        )
        self.backbone = Qwen2Model(self.config)
        self.in_proj = nn.Linear(self.obf_dim, int(hidden_size))
        self.token_head = nn.Linear(int(hidden_size), int(vocab_size))  # token-index (CE) path
        self.plain_embed_dim = int(plain_embed_dim) if plain_embed_dim else None
        self.embed_head = (  # embedding-regression (paper-experiment) path
            nn.Linear(int(hidden_size), self.plain_embed_dim)
            if self.plain_embed_dim
            else None
        )

    def encode(self, obf: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obf):
            obf = torch.as_tensor(np.asarray(obf, dtype=np.float32))
        obf = obf.float()
        if obf.ndim == 2:
            obf = obf.unsqueeze(0)
        inputs_embeds = self.in_proj(obf)
        out = self.backbone(inputs_embeds=inputs_embeds)
        return out.last_hidden_state  # (B, N, hidden)

    def predict_token_logits(self, obf: torch.Tensor) -> torch.Tensor:
        """Token-index (CE alternative) path: vocabulary logits per position."""
        return self.token_head(self.encode(obf))

    # ``forward`` keeps the token-index head for backward compatibility with any
    # caller that treats the module as a plain logits producer.
    def forward(self, obf: torch.Tensor) -> torch.Tensor:
        return self.predict_token_logits(obf)

    def predict_embeddings(self, obf: torch.Tensor) -> torch.Tensor:
        """Embedding-regression (paper-experiment) path: plaintext token embeddings."""
        if self.embed_head is None:
            raise RuntimeError(
                "embedding-regression head not enabled; construct with plain_embed_dim"
            )
        return self.embed_head(self.encode(obf))


def mse_regression_objective(e_hat: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Loss for the embedding path ``L_MSE = (1/N) sum ||e_hat_t - e_t||^2``.

    NOTE: this MSE is the loss used to train the (paper-experiment) embedding
    path; it is an implementation choice. AloePri states the inversion *target*
    (plaintext token embeddings) but does not specify the loss, so MSE is NOT
    claimed to be AloePri's objective.
    """
    return ((e_hat - e) ** 2).sum(dim=-1).mean()


@register
class Qwen2AloePriInverter(Attack):
    """AloePri Qwen2 inversion-model attack (2 decoder layers, 8 attention heads)."""

    name = "ima.qwen2_aloepri"
    family = "ima"
    variant = "aloepri"
    required_capabilities = frozenset(
        {Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)
        # Assert the AloePri architecture. Layers/heads are paper-specified;
        # num_key_value_heads is an UNSPECIFIED assumption (= attention heads).
        # All three are pinned and MUST NOT be overridden through params.
        overrides = {
            "num_hidden_layers": REQUIRED_NUM_HIDDEN_LAYERS,
            "num_attention_heads": REQUIRED_NUM_ATTENTION_HEADS,
            "num_key_value_heads": REQUIRED_NUM_KEY_VALUE_HEADS,
        }
        for key, required in overrides.items():
            if key in params and int(params[key]) != required:
                raise ValueError(
                    f"{self.name} fixes {key}={required} (AloePri App. D.1); "
                    f"cannot override to {params[key]}"
                )
        raw = str(params.get("objective", "embedding")).lower()
        if raw not in _OBJECTIVE_ALIASES:
            raise ValueError(
                f"objective must be one of {sorted(set(_OBJECTIVE_ALIASES))}; got {raw!r}"
            )
        # Canonical objective: 'embedding' (paper experiment) or 'token_index' (alt).
        self.objective = _OBJECTIVE_ALIASES[raw]

    def build_model(
        self, obf_dim: int, vocab_size: int, *, plain_embed_dim: Optional[int] = None
    ) -> Qwen2InversionModel:
        """Construct the fixed 2-layer / 8-head Qwen2 inverter."""
        return Qwen2InversionModel(
            obf_dim,
            vocab_size,
            hidden_size=int(self.params.get("hidden_size", 64)),
            intermediate_size=int(self.params.get("intermediate_size", 128)),
            max_position_embeddings=int(self.params.get("max_position_embeddings", 64)),
            plain_embed_dim=plain_embed_dim,
            seed=int(self.params.get("seed", 0)),
        )

    def _build_pairs(self, ctx: AttackContext) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
                "Qwen2-IMA needs obfuscation_fn (KNOWN_OBFUSCATION_ALGORITHM)"
            )
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "Qwen2-IMA needs a public reference_corpus to build training pairs"
            )
        vocab = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot size the token head")

        H = np.asarray(ctx.obfuscated_embeddings(), dtype=float)
        if H.ndim != 2:
            return AttackResult.failed(f"obfuscated embeddings must be 2-D; got {H.shape}")

        try:
            memories, targets = self._build_pairs(ctx)
        except ValueError as exc:
            return AttackResult.failed(str(exc))
        if not targets:
            return AttackResult.failed("no training pairs could be generated from the corpus")
        obf_dim = memories[0].shape[1]
        if obf_dim != H.shape[1]:
            return AttackResult.failed(
                f"obf embedding dim mismatch: train d={obf_dim}, eval d={H.shape[1]}"
            )

        if self.objective == "embedding":
            return self._run_embedding_regression(ctx, memories, targets, H, vocab, obf_dim)
        return self._run_token_index(memories, targets, H, vocab, obf_dim)

    # --- embedding-regression path (default, paper experiment) ---------------
    def _run_embedding_regression(
        self,
        ctx: AttackContext,
        memories: list[np.ndarray],
        targets: list[np.ndarray],
        H: np.ndarray,
        vocab: int,
        obf_dim: int,
    ) -> AttackResult:
        E = ctx.reference_embeddings
        if E is None:
            return AttackResult.not_applicable(
                "embedding-regression (paper-experiment) path needs a clean reference "
                "embedding matrix for the plaintext-embedding targets"
            )
        E = np.asarray(E, dtype=float)
        plain_embed_dim = E.shape[1]
        seed = int(self.params.get("seed", 0))
        model = self.build_model(obf_dim, vocab, plain_embed_dim=plain_embed_dim)
        # Regression targets: clean plaintext token embeddings e_t = E[token_id].
        target_embeds = [E[t] for t in targets]
        history = self._train(
            model, memories, targets, objective="embedding", seed=seed, target_embeds=target_embeds
        )

        with torch.no_grad():
            model.eval()
            e_hat = model.predict_embeddings(H)[0].cpu().numpy()  # (N, plain_embed_dim)
        # Map predicted embeddings back to token ids by nearest neighbour in E.
        d2 = (
            np.sum(e_hat * e_hat, axis=1, keepdims=True)
            + np.sum(E * E, axis=1, keepdims=True).T
            - 2.0 * (e_hat @ E.T)
        )
        order = np.argsort(np.maximum(d2, 0.0), axis=1, kind="stable")
        top_k = min(int(self.params.get("top_k", 5)), vocab)
        recovered = order[:, 0].astype(int)
        candidates = [order[i, :top_k].tolist() for i in range(order.shape[0])]

        diag = self._diag(history, vocab, obf_dim, H, len(targets), "embedding")
        diag["output_path"] = (
            "embedding_regression (AloePri App. D.1 experiment: invert to plaintext token "
            "embeddings); recovery by nearest neighbour in the clean embedding table"
        )
        return AttackResult(
            recovered_token_ids=recovered, candidates=candidates, diagnostics=diag
        )

    # --- token-index path (cross-entropy, labelled alternative) --------------
    def _run_token_index(
        self,
        memories: list[np.ndarray],
        targets: list[np.ndarray],
        H: np.ndarray,
        vocab: int,
        obf_dim: int,
    ) -> AttackResult:
        seed = int(self.params.get("seed", 0))
        model = self.build_model(obf_dim, vocab)
        history = self._train(model, memories, targets, objective="token_index", seed=seed)

        with torch.no_grad():
            model.eval()
            logits = model.predict_token_logits(H)[0]  # (N, vocab)
            top_k = min(int(self.params.get("top_k", 5)), vocab)
            recovered = logits.argmax(dim=-1).cpu().numpy().astype(int)
            candidates = [row.tolist() for row in torch.topk(logits, top_k, dim=-1).indices.numpy()]

        diag = self._diag(history, vocab, obf_dim, H, len(targets), "token_index")
        diag["output_path"] = (
            "token_index cross-entropy (App. D.1 lists 'or token indices'); this is an "
            "alternative, NOT the path used in the paper's experiment"
        )
        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics=diag,
        )

    # --- shared training loop ------------------------------------------------
    def _train(
        self,
        model: Qwen2InversionModel,
        memories: list[np.ndarray],
        targets: list[np.ndarray],
        *,
        objective: str,
        seed: int,
        target_embeds: Optional[list[np.ndarray]] = None,
    ) -> list[float]:
        seed_everything(seed)
        lr = float(self.params.get("lr", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 0.0))
        epochs = int(self.params.get("epochs", 40))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        mem_t = [torch.as_tensor(np.asarray(m, dtype=np.float32)) for m in memories]
        tgt_t = [torch.as_tensor(np.asarray(t)).long() for t in targets]
        emb_t = (
            [torch.as_tensor(np.asarray(e, dtype=np.float32)) for e in target_embeds]
            if target_embeds is not None
            else None
        )
        history: list[float] = []
        model.train()
        n = len(tgt_t)
        for _ in range(epochs):
            order = torch.randperm(n)
            running = 0.0
            for i in order.tolist():
                mem = mem_t[i].unsqueeze(0)
                if objective == "token_index":
                    logits = model.predict_token_logits(mem)[0]  # (N, vocab)
                    loss = cross_entropy_objective(logits, tgt_t[i], reduction="mean")
                else:  # embedding regression
                    e_hat = model.predict_embeddings(mem)[0]  # (N, plain_embed_dim)
                    loss = mse_regression_objective(e_hat, emb_t[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
            history.append(running / max(n, 1))
        return history

    def _diag(
        self,
        history: list[float],
        vocab: int,
        obf_dim: int,
        H: np.ndarray,
        num_pairs: int,
        objective: str,
    ) -> dict:
        return {
            "variant": "aloepri",
            "backbone": "Qwen2Model",
            "num_hidden_layers": REQUIRED_NUM_HIDDEN_LAYERS,
            "num_attention_heads": REQUIRED_NUM_ATTENTION_HEADS,
            "num_key_value_heads": REQUIRED_NUM_KEY_VALUE_HEADS,  # UNSPECIFIED assumption
            "objective": objective,
            "loss": (
                "mse (implementation choice; AloePri specifies the embedding target, not the loss)"
                if objective == "embedding"
                else "cross_entropy"
            ),
            "optimizer": "AdamW",
            "num_train_pairs": int(num_pairs),
            "train_final_loss": float(history[-1]) if history else None,
            "vocab_size": int(vocab),
            "embed_dim": int(obf_dim),
            "num_positions": int(H.shape[0]),
            "epochs": int(self.params.get("epochs", 40)),
        }
