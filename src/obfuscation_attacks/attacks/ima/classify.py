"""IMA *Classify* inverter - InvBERT Classify line (Kugler et al., **GB** attack).

InvBERT draws the lines of attack explicitly: "We call our GB attack InvBert
Classify and our BB attack InvBert Seq2Seq" (InvBERT Sec. 3.2). Classify is
therefore a **Gray-Box** attack: the tokenizer ``tok()`` is known, so the
contextualized token embeddings (CTEs) and vocabulary ids have a one-to-one
mapping and the inversion collapses to a *high-dimensional per-position token
classification* task - "As we use the given tokenizer, CTEs and IDs have a
one-to-one mapping, and our attack boils down to a high-dimensional token
classification task" (InvBERT Sec. 4.2, InvBERT Classify (GB)). It is trained
with AdamW and "the basic cross-entropy loss" (InvBERT Sec. 4.2).

Because Classify is the GB attack, this module supports the **WB** and **GB**
knowledge settings only. The **BB** setting (no tokenizer, no one-to-one map) is
InvBERT *Seq2Seq* and is implemented in :mod:`.seq2seq`; asking Classify for BB
returns ``not_applicable`` rather than pretending a per-position classifier is a
black-box attack.

Formulae (docs/FORMULA_LEDGER.md: IMA-CE-1, IMA-CLS-1)::

    L = - sum_t log p_theta(y_t | context)                         (IMA-CE-1)
    p_theta(y_t | context) = softmax(W * MLP_theta(h_t) + b)_{y_t}  (IMA-CLS-1)

Here ``h_t`` is the obfuscated contextual embedding at position ``t``,
``MLP_theta`` is a shared multi-layer perceptron, and ``(W, b)`` is the final
linear projection to vocabulary logits. The same parameters are applied
independently at every position (weight sharing), so ``context`` for Classify is
exactly the target position's representation ``h_t``.

Knowledge settings (InvBERT Sec. 3.1) differ only in *how the training pairs /
knowledge are obtained*; the model maths are identical and the setting is
recorded in diagnostics:

* ``GB`` (gray box, the InvBERT Classify setting) - the tokenizer is known (so
  the one-to-one CTE<->id map and per-position classification are valid); the
  attacker pairs public plaintext with its obfuscated embeddings using unlimited
  oracle queries over the public corpus, with no vocabulary probing.
* ``WB`` (white box) - additionally the encoder architecture + parameters are
  known. Because a successful GB attack also works in WB (InvBERT Sec. 3.2), the
  model is unchanged; the extra power is used only to *probe every vocabulary id
  once* (each id queried on its own, see ``_build_pairs``) to guarantee full
  class coverage of the training set.
* ``BB`` (black box) -> ``not_applicable`` here: BB is InvBERT Seq2Seq.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register


def cross_entropy_objective(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """The InvBERT training objective ``L = - sum_t log p_theta(y_t | context)``.

    A thin, faithful wrapper over the standard cross entropy, which is exactly
    ``-log softmax(logits)`` evaluated at the true class. ``reduction='sum'``
    gives the summed form written in the paper; ``'mean'`` divides by the number
    of counted positions (used during optimisation for scale stability).
    """
    return F.cross_entropy(logits, targets, reduction=reduction)


def classify_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """``p_theta(y_t | context) = softmax(W MLP_theta(h_t) + b)`` over the vocab.

    ``logits`` is ``W MLP_theta(h_t) + b``; the softmax is taken over the final
    (vocabulary) dimension.
    """
    return torch.softmax(logits, dim=-1)


class PerPositionClassifier(nn.Module):
    """Shared per-position MLP classifier implementing IMA-CLS-1.

    ``MLP_theta`` maps the contextual embedding ``h_t`` to a feature vector and
    the final linear layer ``(W, b)`` produces vocabulary logits. ``forward``
    returns the logits ``W MLP_theta(h_t) + b``; apply :func:`classify_probabilities`
    for the softmax probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dims: Sequence[int] = (128,),
        seed: int = 0,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        dims = [self.input_dim, *[int(h) for h in hidden_dims]]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)  # MLP_theta (identity when hidden_dims empty)
        self.head = nn.Linear(dims[-1], self.vocab_size)  # (W, b)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(h))  # W MLP_theta(h) + b


def _to_float_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float()
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def train_classifier(
    model: PerPositionClassifier,
    X,
    y,
    *,
    epochs: int = 150,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    batch_size: Optional[int] = None,
    seed: int = 0,
) -> list[float]:
    """Minimise the cross-entropy objective with AdamW (InvBERT Sec. 4.2).

    Returns the per-epoch mean training loss so callers can assert convergence.
    """
    seed_everything(seed)
    X = _to_float_tensor(X)
    y = torch.as_tensor(np.asarray(y)).long()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (num_pairs, input_dim); got {tuple(X.shape)}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    bs = int(batch_size) if batch_size else n
    history: list[float] = []
    model.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logits = model(X[idx])
            loss = cross_entropy_objective(logits, y[idx], reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * int(idx.numel())
        history.append(running / n)
    return history


@torch.no_grad()
def classifier_accuracy(model: PerPositionClassifier, X, y) -> float:
    """Fraction of pairs whose argmax prediction matches the target id."""
    model.eval()
    logits = model(_to_float_tensor(X))
    pred = logits.argmax(dim=-1).cpu().numpy()
    return float((pred == np.asarray(y)).mean())


@torch.no_grad()
def classifier_predict(
    model: PerPositionClassifier, X, top_k: int = 5
) -> tuple[np.ndarray, list[list[int]]]:
    """Return (argmax ids, ranked top-k candidate ids) for each row of ``X``."""
    model.eval()
    logits = model(_to_float_tensor(X))
    k = min(int(top_k), logits.shape[-1])
    recovered = logits.argmax(dim=-1).cpu().numpy().astype(int)
    topk = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()
    candidates = [row.tolist() for row in topk]
    return recovered, candidates


@register
class ClassifyInverter(Attack):
    """InvBERT Classify (GB): per-position MLP token classifier over obfuscated CTEs."""

    name = "ima.classify"
    family = "ima"
    variant = "original"
    required_capabilities = frozenset(
        {Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}
    )

    # --- convenience API so the class itself can be driven from pairs ---------
    @staticmethod
    def build_classifier(
        input_dim: int, vocab_size: int, hidden_dims: Sequence[int] = (128,), seed: int = 0
    ) -> PerPositionClassifier:
        return PerPositionClassifier(input_dim, vocab_size, hidden_dims=hidden_dims, seed=seed)

    @staticmethod
    def train_on_pairs(
        X, y, *, vocab_size: int, hidden_dims: Sequence[int] = (128,), seed: int = 0, **kw
    ) -> tuple[PerPositionClassifier, list[float]]:
        X = np.asarray(X, dtype=float)
        model = PerPositionClassifier(X.shape[1], vocab_size, hidden_dims=hidden_dims, seed=seed)
        history = train_classifier(model, X, y, seed=seed, **kw)
        return model, history

    train_accuracy = staticmethod(classifier_accuracy)

    # --- training-pair synthesis --------------------------------------------
    def _build_pairs(
        self, ctx: AttackContext, knowledge: str, vocab: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Synthesise (obfuscated embedding, token id) pairs from public data.

        The attack consumes only ``ctx.reference_corpus`` (public plaintext) and
        ``ctx.obfuscation_fn`` (the known *algorithm*, keyed by the attacker, not
        the client). It never sees a key or the evaluated sample's ground truth.
        """
        corpus = [np.asarray(s).astype(int).ravel() for s in ctx.reference_corpus]
        info: dict = {"query_budget": None}
        if knowledge == "GB":
            info["knowledge_note"] = (
                "gray-box (InvBERT Classify): tokenizer known (one-to-one CTE<->id); "
                "unlimited oracle queries over the public corpus, no vocabulary probing"
            )
        else:  # WB
            info["knowledge_note"] = (
                "white-box: GB model plus full pipeline knowledge; unlimited oracle "
                "queries over the public corpus plus one probe per vocabulary id for "
                "full class coverage"
            )

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for seq in corpus:
            if seq.size == 0:
                continue
            obf = np.asarray(ctx.obfuscation_fn(seq), dtype=float)
            if obf.ndim != 2 or obf.shape[0] != seq.size:
                raise ValueError(
                    f"obfuscation_fn must map (len,) -> (len, d); got {obf.shape} for "
                    f"sequence length {seq.size}"
                )
            X_list.append(obf)
            y_list.append(seq)

        info["queries_used"] = len(X_list)

        # WB full-vocabulary probing. The white-box attacker has unlimited query
        # access, so it obtains one obfuscated embedding per vocabulary id to
        # guarantee every class is represented in the training set. Each id is
        # probed *on its own* - i.e. ``obfuscation_fn([id])`` in a length-1
        # context - rather than feeding the entire vocabulary as one long
        # sequence; the latter would entangle positions for a context-dependent
        # encoder and misrepresent the per-id probe. (For token-local schemes the
        # two coincide, but the per-id form is the faithful one.)
        if knowledge == "WB" and bool(self.params.get("wb_probe_vocab", True)):
            probe_rows: list[np.ndarray] = []
            probe_ids: list[int] = []
            for tid in range(int(vocab)):
                o = np.asarray(ctx.obfuscation_fn(np.array([tid], dtype=int)), dtype=float)
                if o.ndim == 2 and o.shape[0] == 1:
                    probe_rows.append(o[0])
                    probe_ids.append(tid)
            if probe_rows:
                X_list.append(np.stack(probe_rows, axis=0))
                y_list.append(np.asarray(probe_ids, dtype=int))
                info["wb_vocab_probed"] = len(probe_ids)

        if not X_list:
            return np.zeros((0, 0)), np.zeros((0,), dtype=int), info
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0).astype(int)
        return X, y, info

    def _run(self, ctx: AttackContext) -> AttackResult:
        # Classify is the InvBERT GB attack -> WB/GB only. BB is InvBERT Seq2Seq.
        knowledge = str(self.params.get("knowledge", "GB")).upper()
        if knowledge == "BB":
            return AttackResult.not_applicable(
                "InvBERT Classify is the GB attack (Sec. 3.2 'We call our GB attack InvBert "
                "Classify and our BB attack InvBert Seq2Seq'); the black-box setting is "
                "ima.seq2seq (InvBERT Seq2Seq), which trains a custom tokenizer instead of "
                "assuming a one-to-one CTE<->id map. Use ima.seq2seq for BB."
            )
        if knowledge not in {"WB", "GB"}:
            return AttackResult.failed(
                f"unknown knowledge setting {knowledge!r} (Classify supports WB or GB)"
            )
        if ctx.obfuscation_fn is None:
            return AttackResult.not_applicable(
                "Classify-IMA needs obfuscation_fn (KNOWN_OBFUSCATION_ALGORITHM)"
            )
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "Classify-IMA needs a public reference_corpus to build training pairs"
            )
        vocab = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot size the classifier head")

        H = np.asarray(ctx.obfuscated_embeddings(), dtype=float)
        if H.ndim != 2:
            return AttackResult.failed(f"obfuscated embeddings must be 2-D; got {H.shape}")

        try:
            X, y, pairs_info = self._build_pairs(ctx, knowledge, vocab)
        except ValueError as exc:
            return AttackResult.failed(str(exc))
        if y.size == 0:
            return AttackResult.failed("no training pairs could be generated from the corpus")
        if X.shape[1] != H.shape[1]:
            return AttackResult.failed(
                f"obf embedding dim mismatch: train d={X.shape[1]}, eval d={H.shape[1]}"
            )

        seed = int(self.params.get("seed", 0))
        hidden_dims = tuple(self.params.get("hidden_dims", (128,)))
        epochs = int(self.params.get("epochs", 150))
        lr = float(self.params.get("lr", 1e-2))
        weight_decay = float(self.params.get("weight_decay", 0.0))
        batch_size = self.params.get("batch_size", None)
        top_k = int(self.params.get("top_k", 5))

        model = PerPositionClassifier(X.shape[1], vocab, hidden_dims=hidden_dims, seed=seed)
        history = train_classifier(
            model,
            X,
            y,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
        )
        recovered, candidates = classifier_predict(model, H, top_k=top_k)

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "knowledge": knowledge,
                "num_train_pairs": int(y.size),
                "train_final_loss": float(history[-1]) if history else None,
                "train_accuracy": classifier_accuracy(model, X, y),
                "vocab_size": vocab,
                "embed_dim": int(X.shape[1]),
                "num_positions": int(H.shape[0]),
                "epochs": epochs,
                "optimizer": "AdamW",
                "loss": "cross_entropy",
                **pairs_info,
            },
        )
