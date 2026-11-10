"""Shared building blocks for the ISA (Internal-State Attack) family.

Everything the three ISA attacks have in common lives here so that each attack
module reads as a faithful transcription of its paper formula rather than a
re-implementation of plumbing. The pieces are:

* :func:`resolve_model` - validate that the attacker owns a differentiable
  ``TorchToyModel`` (``ctx.attacker_model``); ISA is an optimization attack and
  cannot run without a model to back-propagate through.
* :func:`state_distance` - the distance ``d`` of the inversion objective
  ``xhat = argmin_{x'} d(h_l^{fA}(x'), h_l^{fV}(x))`` (ISA paper Eq. 1). The
  paper's *default* is the Mean Squared Error (MSE); cosine distance is the
  documented alternative ("Common choices of d include Lp norm and cosine
  distance", Sec. 4.2; "the default setting uses MSE as the distance metric",
  Sec. 5.1).
* :class:`RandomFeatureNet` + :func:`distribution_matching_loss` - the ER
  distribution-matching penalty ``L_dm`` (ISA paper Eq. 3):
  ``L_dm = E_{phi ~ P_phi} || phi_bar(what) - phi_bar(w_in) ||_2`` where ``phi``
  is a random Gaussian neural network used as a universal feature extractor and
  ``phi_bar`` is the *average* (overline) of its features over an embedding
  batch. It is deliberately **not** a plain L2/norm regulariser, and it is
  **mean**-matching only (no variance term), with the random net ``phi`` and the
  real-embedding batch resampled every step.
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
    are identical before and after an attack and never receive a gradient.
    """
    if hasattr(model, "freeze"):
        model.freeze()
    else:  # pragma: no cover - defensive
        for p in model.parameters():
            p.requires_grad_(False)


def model_device(model) -> torch.device:
    """Device of the attacker model's embedding matrix (CPU fallback)."""
    try:
        return model.embedding_matrix().device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cpu")


def as_target_tensor(target_np: np.ndarray, device=None) -> torch.Tensor:
    """Observed victim state ``h_l^{fV}`` as a float32 tensor (no grad)."""
    arr = np.asarray(target_np, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"target hidden state must be 2-D (N, d); got {arr.shape}")
    t = torch.from_numpy(arr)
    return t if device is None else t.to(device)


def state_distance(a: torch.Tensor, b: torch.Tensor, kind: str = "l2") -> torch.Tensor:
    """Distance ``d`` between two internal states (ISA Eq. 1 / ``L_im``).

    The paper's **default** distance is the Mean Squared Error (MSE), with cosine
    distance as the documented alternative ("Common choices of d include Lp norm
    and cosine distance", Sec. 4.2; "For simplicity, the default setting uses MSE
    as the distance metric", Sec. 5.1). Cosine is reported as more reliable for
    self-attention states (Sec. 5.2), hence it stays available.

    * ``"l2"`` - mean squared error ``mean((a - b) ** 2)`` (the paper's MSE
      default; a monotone function of the squared L2 norm of the difference).
    * ``"cos"`` - cosine distance averaged over sequence positions:
      ``mean_i (1 - cos(a_i, b_i))``.
    """
    if a.shape != b.shape:
        raise ValueError(f"state shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    if kind == "l2":
        return ((a - b) ** 2).mean()
    if kind == "cos":
        cos = F.cosine_similarity(a, b, dim=-1)  # per-position cosine
        return (1.0 - cos).mean()
    raise ValueError(f"unknown distance {kind!r}; use 'l2' (MSE, default) or 'cos'")


class RandomFeatureNet(nn.Module):
    """Fixed random Gaussian network used as a universal feature extractor.

    This is the ``phi`` of the ER distribution-matching penalty ``L_dm`` (ISA
    paper Eq. 3, following the distribution-matching idea of Zhao & Bilen [39]).
    Weights and biases are drawn once from a Gaussian with a supplied
    ``generator`` and then frozen (``requires_grad_(False)``), so the extractor
    is a fixed, data-independent random projection - exactly the "random Gaussian
    neural network as universal feature extractor" of the paper, not a learned
    module.

    EXPERIMENTAL_COMPLETION: the paper names the extractor ("random Gaussian
    neural network") but does **not** specify its architecture (depth, width,
    nonlinearity) or the Gaussian weight scale. We use a small MLP with
    ``1/sqrt(fan_in)`` Gaussian weights (so the random features stay O(1) rather
    than exploding through the stacked layers) and ReLU by default. These
    specifics are our reproduction choice, not the paper's.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 64,
        depth: int = 2,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        nonlinearity: str = "relu",
    ) -> None:
        super().__init__()
        if generator is None:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(int(seed))
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
                    scale * torch.randn(dims[i + 1], dims[i], generator=generator)
                )
                lin.bias.copy_(scale * torch.randn(dims[i + 1], generator=generator))
            self.layers.append(lin)
        self.act = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = self.act(lin(x))
        return x


def distribution_matching_loss(
    what: torch.Tensor,
    embedding_matrix: torch.Tensor,
    generator: torch.Generator,
    *,
    num_nets: int = 1,
    hidden: int = 64,
    depth: int = 2,
    nonlinearity: str = "relu",
) -> torch.Tensor:
    r"""ER distribution-matching penalty ``L_dm`` (ISA paper Eq. 3).

    .. math::
        L_{dm} = \mathbb{E}_{\phi \sim P_\phi}\;
                 \big\| \overline{\phi}(\hat{w}) - \overline{\phi}(w_{in}) \big\|_2

    where ``phi`` is a *random Gaussian neural network* universal feature
    extractor (:class:`RandomFeatureNet`), the overline denotes **averaging**
    (the batch mean over embedding rows), and ``w_in`` is a batch of real token
    embeddings sampled from ``fA.embedding_matrix()`` of the *same size* as
    ``what`` ("we sample random embedding batches of equal size from ``what`` and
    ``w_in`` before averaging in each step"). The expectation over ``P_phi`` is
    estimated by resampling ``phi`` (``num_nets`` draws per call), and the whole
    penalty is resampled *every optimization step* because ``_run`` passes the
    same advancing ``generator`` each step.

    This is faithfully Eq. 3 and nothing more:

    * **mean**-matching only - the Euclidean norm is of the difference between
      the two batch *means* ``phi_bar``; there is no variance / second-moment term;
    * the **Euclidean norm** of the mean-difference vector;
    * ``phi`` and the real-embedding batch are **resampled** from ``generator``
      on every call, matching "random ... in each step".

    The ``what`` batch is the full set of dummy rows (``|h_l|`` of them) and we
    draw an equal-size (one-real-row-per-dummy-row) real batch each call; this is
    the toy-scale reading of "batches of equal size from ``what`` and ``w_in``".

    Matching the batch mean of a random non-linear feature keeps ``what``
    distributed like real embeddings (preventing the high-norm overfitting the
    paper observed in Sec. 4.2) without ever being a plain L2/norm penalty.
    """
    n, d = int(what.shape[0]), int(what.shape[1])
    vocab = int(embedding_matrix.shape[0])
    num_nets = max(1, int(num_nets))
    total = what.new_zeros(())
    for _ in range(num_nets):
        # Resample phi first, then the equal-size real batch, both from the same
        # advancing generator (so the test can replay the exact draws).
        net = RandomFeatureNet(
            d, hidden=hidden, depth=depth, generator=generator, nonlinearity=nonlinearity
        )
        # The net is drawn on CPU (so a CPU generator replays the exact draws),
        # then moved to the state's device: a CPU net applied to CUDA hidden
        # states raises "Expected all tensors to be on the same device".
        net = net.to(device=what.device)
        idx = torch.randint(0, vocab, (n,), generator=generator)
        real = embedding_matrix[idx]  # (n, d), constant w.r.t. the optimization
        fw = net(what)  # (n, hidden) - depends on what
        fr = net(real)  # (n, hidden) - constant
        mean_w = fw.mean(dim=0)  # overline = averaging over the batch rows
        mean_r = fr.mean(dim=0)
        total = total + (mean_w - mean_r).norm()
    return total / num_nets


def attention_token_features(attn: torch.Tensor) -> torch.Tensor:
    """Per-token AttnScore features for AloePri Table 4.

    ``attn`` is ``[heads, seq, seq]`` or ``[batch, heads, seq, seq]``. Batch
    size must be 1. HuggingFace stores the attention of decoder block ``k``
    (0-indexed) at ``attentions[k]``, which is the block whose output is
    ``hidden_states[k + 1]``. Each token feature is that token's attention row
    concatenated over heads, shape ``[seq, heads * seq]``. Gradients are kept.
    """
    if attn.ndim == 4:
        if int(attn.shape[0]) != 1:
            raise ValueError(f"attention batch must be 1, got {tuple(attn.shape)}")
        attn = attn[0]
    if attn.ndim != 3:
        raise ValueError(f"attention must be [heads, seq, seq], got {tuple(attn.shape)}")
    heads, seq, key = (int(attn.shape[0]), int(attn.shape[1]), int(attn.shape[2]))
    if key != seq:
        raise ValueError(f"attention is not square: {tuple(attn.shape)}")
    return attn.permute(1, 0, 2).reshape(seq, heads * key)


def recover_tokens_by_cosine(
    what_np, embedding_np, top_k: int
) -> tuple[np.ndarray, list[list[int]]]:
    """Second phase: tokens of highest cosine similarity with ``w_in`` rows.

    ISA Algorithm 1 line 11:
    ``t = argmax_row [ what . w_in^T / (||what|| ||w_in||) ]``, with ranked
    top-k candidates per position (descending cosine similarity).

    Accepts numpy arrays or torch tensors (any device); the cosine product stays
    on the tensor device and only the ranked ids are copied back to numpy, so the
    full 128k vocabulary does not round-trip through host memory.
    """
    if torch.is_tensor(what_np) or torch.is_tensor(embedding_np):
        W = torch.as_tensor(what_np, dtype=torch.float32)
        E = torch.as_tensor(embedding_np, dtype=torch.float32, device=W.device)
        wn = W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        en = E.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos = (W @ E.T) / (wn @ en.T)
        k = min(int(top_k), int(E.shape[0]))
        order = torch.argsort(-cos, dim=1, stable=True)
        candidates = order[:, :k].detach().cpu().tolist()
        recovered = order[:, 0].detach().cpu().numpy().astype(int)
        return recovered, candidates

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
