"""Shared building blocks for the ISA (Internal-State Attack) family.

Everything the three ISA attacks have in common lives here so that each attack
module reads as a faithful transcription of its paper formula rather than a
re-implementation of plumbing. The pieces are:

* :func:`resolve_model` - validate that the attacker owns a differentiable
  ``TorchToyModel`` (``ctx.attacker_model``); ISA is an optimization attack and
  cannot run without a model to back-propagate through.
* :func:`state_distance` - the distance ``d`` of the inversion objective
  ``xhat = argmin_{x'} d(h_l^{fA}(x'), h_l^{fV}(x))`` (ISA paper Eq. 1). Cosine
  distance by default; squared-error ("l2", the paper's MSE) optional.
* :class:`RandomFeatureNet` + :func:`distribution_matching_loss` - the ER
  distribution-matching penalty ``L_dm`` (ISA paper Eq. 3): a *fixed* random
  Gaussian network used as a universal feature extractor, matching batch
  statistics (mean and variance across the sequence positions) of the dummy
  embeddings against real token embeddings. It is deliberately **not** a plain
  L2/norm regulariser.
* :func:`recover_tokens_by_cosine` - the second phase shared by ER/TBS/AloePri:
  pick, per row of the optimized embedding, the vocabulary token of highest
  cosine similarity with ``fA.embedding_matrix()`` (ISA Algorithm 1 line 11).

All distances and the feature network are differentiable so the optimization
genuinely minimises the state distance; the victim/attacker model is frozen.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.contracts import AttackContext


def resolve_model(ctx: AttackContext):
    """Return ``(model, "")`` when a usable attacker model is present.

    ISA back-propagates through the attacker model ``fA``; without it the attack
    is honestly *not applicable* rather than silently returning zeros. We accept
    any object exposing the differentiable ``TorchToyModel`` interface used by
    the family (duck-typed to avoid a hard import coupling).
    """
    model = getattr(ctx, "attacker_model", None)
    if model is None:
        return None, "ISA requires an attacker model (ctx.attacker_model is None)"
    needed = ("embedding_matrix", "forward_from_embeddings")
    missing = [m for m in needed if not hasattr(model, m)]
    if missing:
        return None, (
            "ISA attacker_model lacks the differentiable TorchToyModel interface: "
            f"missing {missing}"
        )
    return model, ""


def freeze_model(model) -> None:
    """Fix the model weights for the whole optimization (ISA: weights are fixed).

    Only toggles ``requires_grad`` (no value change), so the model's parameters
    are identical before and after an attack.
    """
    if hasattr(model, "freeze"):
        model.freeze()
    else:  # pragma: no cover - defensive
        for p in model.parameters():
            p.requires_grad_(False)


def as_target_tensor(target_np: np.ndarray) -> torch.Tensor:
    """Observed victim state ``h_l^{fV}`` as a float32 tensor (no grad)."""
    arr = np.asarray(target_np, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"target hidden state must be 2-D (N, d); got {arr.shape}")
    return torch.from_numpy(arr)


def state_distance(a: torch.Tensor, b: torch.Tensor, kind: str = "cos") -> torch.Tensor:
    """Distance ``d`` between two internal states (ISA Eq. 1 / L_im).

    * ``"cos"`` - cosine distance averaged over sequence positions:
      ``mean_i (1 - cos(a_i, b_i))``. This is the default the paper reports as
      the more reliable metric for self-attention states.
    * ``"l2"`` - mean squared error ``mean((a - b) ** 2)`` (the paper's MSE
      default; a monotone function of the squared L2 norm of the difference).
    """
    if kind == "cos":
        cos = F.cosine_similarity(a, b, dim=-1)  # per-position cosine
        return (1.0 - cos).mean()
    if kind == "l2":
        return ((a - b) ** 2).mean()
    raise ValueError(f"unknown distance {kind!r}; use 'cos' or 'l2'")


class RandomFeatureNet(nn.Module):
    """Fixed random Gaussian network used as a universal feature extractor.

    This is the ``phi`` of the ER distribution-matching penalty ``L_dm`` (ISA
    paper Eq. 3). Weights and biases are drawn once from a Gaussian with a
    dedicated generator and then frozen (``requires_grad_(False)``), so the
    extractor is a fixed, data-independent random projection - exactly the
    "random Gaussian neural network as universal feature extractor" of the
    paper, not a learned module.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 64,
        depth: int = 2,
        seed: int = 0,
        nonlinearity: str = "relu",
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed))
        dims = [int(d_in)] + [int(hidden)] * int(depth)
        self.layers = nn.ModuleList()
        for i in range(int(depth)):
            fan_in = dims[i]
            # Gaussian weights with standard 1/sqrt(fan_in) scaling so the random
            # features stay O(1); without it the stacked randn layers blow up and
            # the statistics-matching signal is swamped by numerical scale.
            scale = 1.0 / (fan_in ** 0.5)
            lin = nn.Linear(dims[i], dims[i + 1])
            with torch.no_grad():
                lin.weight.copy_(
                    scale * torch.randn(dims[i + 1], dims[i], generator=gen)
                )
                lin.bias.copy_(scale * torch.randn(dims[i + 1], generator=gen))
            self.layers.append(lin)
        self.act = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = self.act(lin(x))
        return x


def build_feature_nets(
    d_in: int, num_nets: int, hidden: int, depth: int, seed: int
) -> list[RandomFeatureNet]:
    """Sample ``num_nets`` independent fixed feature extractors (approx E_{phi})."""
    return [
        RandomFeatureNet(d_in, hidden=hidden, depth=depth, seed=seed + 1000 * i)
        for i in range(int(num_nets))
    ]


def distribution_matching_loss(
    what: torch.Tensor,
    embedding_matrix: torch.Tensor,
    feature_nets: list[RandomFeatureNet],
    generator: torch.Generator,
) -> torch.Tensor:
    r"""ER distribution-matching penalty ``L_dm`` (ISA paper Eq. 3).

    ``L_dm = E_{phi ~ P_phi} || stats(phi(what)) - stats(phi(E_batch)) ||^2``

    where ``phi`` is a fixed random Gaussian network (:class:`RandomFeatureNet`),
    ``E_batch`` is a batch of real token embeddings sampled from
    ``fA.embedding_matrix()`` of the *same size* as ``what`` (one real row per
    dummy row), and ``stats(.)`` collects the **batch mean and variance across
    the sequence positions** - the "averaging" (overline) of the paper applied
    to the random features. The expectation over ``P_phi`` is estimated by
    averaging over the sampled feature networks.

    Matching the first two batch moments of a random non-linear feature keeps
    ``what`` distributed like real embeddings (preventing the high-norm
    overfitting the paper observed) without ever being a plain L2/norm penalty.
    """
    n = what.shape[0]
    vocab = embedding_matrix.shape[0]
    idx = torch.randint(0, vocab, (n,), generator=generator)
    real = embedding_matrix[idx]  # (n, d), constant w.r.t. the optimization
    total = what.new_zeros(())
    for net in feature_nets:
        fw = net(what)  # (n, hidden) - depends on what
        fr = net(real)  # (n, hidden) - constant
        mean_w, mean_r = fw.mean(dim=0), fr.mean(dim=0)
        var_w, var_r = fw.var(dim=0, unbiased=False), fr.var(dim=0, unbiased=False)
        total = total + ((mean_w - mean_r) ** 2).mean() + ((var_w - var_r) ** 2).mean()
    return total / max(1, len(feature_nets))


def recover_tokens_by_cosine(
    what_np: np.ndarray, embedding_np: np.ndarray, top_k: int
) -> tuple[np.ndarray, list[list[int]]]:
    """Second phase: tokens of highest cosine similarity with ``w_in`` rows.

    ISA Algorithm 1 line 11:
    ``t = argmax_row [ what . w_in^T / (||what|| ||w_in||) ]``, with ranked
    top-k candidates per position (descending cosine similarity).
    """
    W = np.asarray(what_np, dtype=float)
    E = np.asarray(embedding_np, dtype=float)
    wn = np.linalg.norm(W, axis=1, keepdims=True)
    en = np.linalg.norm(E, axis=1, keepdims=True)
    wn = np.where(wn == 0.0, 1.0, wn)
    en = np.where(en == 0.0, 1.0, en)
    cos = (W @ E.T) / (wn @ en.T)  # (N, vocab)
    order = np.argsort(-cos, axis=1, kind="stable")  # descending similarity
    k = min(int(top_k), E.shape[0])
    candidates = [order[i, :k].tolist() for i in range(W.shape[0])]
    recovered = order[:, 0].astype(int)
    return recovered, candidates
