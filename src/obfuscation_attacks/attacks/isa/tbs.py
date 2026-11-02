"""TBS - Token Basis Selection (ISA original, Dong et al. 2025).

Algorithm 1 (red blocks) + App. A. For deep layers the gradients on a free
dummy embedding explode, so TBS optimizes the projection coefficients of an
orthogonal basis of the input-embedding space instead of the embedding itself:

    W_in = U diag(S) V^T              (SVD of fA.embedding_matrix())
    B    = V  (unbiased basis, default) | V^T    (Algorithm 1 lines 2-3)
    what = phi_z(z @ B)               (Algorithm 1 line 5)
    phi_z(.) = alpha * arctan(.)      (App. A; alpha = 5/pi)

Crucially the projection ``z`` spans the **full** ``d_in`` basis
(``z in R^{|h_l| x d_in}``, Algorithm 1), *not* a truncated top-``r`` subspace;
there is no low-rank approximation. ``r`` is kept as an optional knob that
defaults to the full ``d_in``; setting ``r < d_in`` is a genuine truncation and
is reported as such (``truncation`` / ``full_basis`` diagnostics).

Two basis strategies are implemented (Algorithm 1 lines 2-3: ``B <- V^T``, and
``if ApplyUnbiasedBasis then B <- V``; "we use unbiased basis in default",
Sec. 5.1): ``basis="unbiased"`` uses ``B = V`` (the default) and ``basis="vt"``
uses ``B = V^T``. Note ``torch.linalg.svd`` returns ``U, S, Vh`` with ``Vh = V^T``,
so ``V = Vh^T``.

``phi_z = alpha * arctan`` (App. A, Eq. 11: ``d phi_z / dz = alpha/(1+(zB)^2) B^T``)
bounds every entry of ``what`` to ``(-alpha*pi/2, alpha*pi/2)``; with the paper's
``alpha = 5/pi`` that is ``(-2.5, 2.5)`` (the ``alpha`` factor "accelerates
convergence", Sec. 5.1). The historical best-loss state is kept (Algorithm 1
line 9) and tokens are recovered by cosine similarity with ``w_in`` (Algorithm 1
line 11). The model weights are fixed throughout, and ``z`` is optimized with
AdamW (Sec. 5.1).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import core

_HALF_PI = math.pi / 2.0
_ALPHA_DEFAULT = 5.0 / math.pi  # App. A / Sec. 5.1: phi_z = alpha * arctan, alpha = 5/pi


def tbs_svd(w_in: torch.Tensor):
    """Return ``(U, S, Vh)`` with ``w_in = U diag(S) Vh`` and ``Vh = V^T`` (reduced SVD)."""
    return torch.linalg.svd(w_in, full_matrices=False)


def tbs_basis(w_in: torch.Tensor, basis: str = "unbiased", r: Optional[int] = None) -> torch.Tensor:
    """Orthogonal basis ``B`` for TBS (Algorithm 1 lines 2-3).

    * ``basis="unbiased"`` (default) -> ``B = V`` (``ApplyUnbiasedBasis``), where
      ``V = Vh^T`` since ``torch.linalg.svd`` returns ``Vh = V^T``;
    * ``basis="vt"`` -> ``B = V^T = Vh``.

    ``r`` selects the first ``r`` basis rows; ``r=None`` keeps the **full**
    ``d_in`` basis (the paper default, no truncation).
    """
    _, _, vh = tbs_svd(w_in)  # vh: (k, d), rows are right singular vectors v_i^T
    if basis == "unbiased":
        B = vh.transpose(-2, -1).contiguous()  # V = (V^T)^T
    elif basis in ("vt", "transpose", "svd"):
        B = vh  # V^T
    else:
        raise ValueError(
            f"unknown basis {basis!r}; use 'unbiased' (B=V, default) or 'vt' (B=V^T)"
        )
    k = B.shape[0]
    if r is None:
        r = k
    r = max(1, min(int(r), k))
    return B[:r]


def tbs_reparam(z: torch.Tensor, basis: torch.Tensor, alpha: float = _ALPHA_DEFAULT) -> torch.Tensor:
    """``what = phi_z(z @ B) = alpha * arctan(z @ B)`` (Algorithm 1 line 5, App. A).

    Output entries are bounded to ``(-alpha*pi/2, alpha*pi/2)`` by ``arctan``.
    """
    return alpha * torch.arctan(z @ basis)


@register
class TBSAttack(Attack):
    name = "isa.tbs"
    family = "isa"
    variant = "original"
    required_capabilities = frozenset({Capability.HIDDEN_STATES})

    DEFAULTS = {
        "distance": "l2",         # paper default MSE; "cos" available
        "lr": 0.1,
        "steps": 300,
        "basis": "unbiased",      # B = V (paper default); "vt" -> B = V^T
        "alpha": _ALPHA_DEFAULT,  # phi_z = alpha * arctan, alpha = 5/pi (App. A)
        "r": None,                # None -> full d_in basis (no truncation)
        "lambda_dm": 0.0,         # TBS core is pure L_im; >0 adds the shared L_dm (Eq. 3)
        "dm_feature_nets": 1,
        "dm_hidden": 64,
        "dm_depth": 2,
        "weight_decay": 0.0,      # UNSPECIFIED in paper (AdamW, no decay value given)
        "top_k": 5,
        "seed": 0,
    }

    def _p(self, key):
        return self.params.get(key, self.DEFAULTS[key])

    def _run(self, ctx: AttackContext) -> AttackResult:
        model, reason = core.resolve_model(ctx)
        if model is None:
            return AttackResult.not_applicable(reason)

        target = core.as_target_tensor(ctx.obfuscated_hidden_states())
        layer = ctx.descriptor.get("layer")

        distance = str(self._p("distance"))
        lr = float(self._p("lr"))
        steps = int(self._p("steps"))
        basis_name = str(self._p("basis"))
        alpha = float(self._p("alpha"))
        lam = float(self._p("lambda_dm"))
        weight_decay = float(self._p("weight_decay"))
        top_k = int(self._p("top_k"))
        seed = int(self._p("seed"))

        seed_everything(seed)
        core.freeze_model(model)

        W_in = model.embedding_matrix()  # (vocab, d)
        d = int(W_in.shape[1])
        n = int(target.shape[0])
        if int(target.shape[1]) != d:
            return AttackResult.failed(
                f"state dim {target.shape[1]} != attacker embedding dim {d}"
            )

        # Full d_in basis by default; r < d_in is an honest (flagged) truncation.
        r_param = self._p("r")
        r = d if r_param is None else max(1, min(int(r_param), d))
        truncation = r < d

        U, S, Vh = tbs_svd(W_in)
        basis = tbs_basis(W_in, basis_name, r).detach()  # (r, d) fixed orthonormal rows

        # SVD reconstruction error (diagnostic / pinned by the math test).
        recon = (U * S) @ Vh
        recon_err = float(torch.max(torch.abs(recon - W_in)))

        # Paper initialises the projection weights at 1/d_in (Algorithm 1 line 3).
        z = torch.full((n, r), 1.0 / d, dtype=torch.float32, requires_grad=True)

        dm_gen = torch.Generator().manual_seed(seed + 7)

        # AdamW (ISA Sec. 5.1). weight_decay is UNSPECIFIED in the paper.
        opt = torch.optim.AdamW([z], lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        best_z = z.detach().clone()
        best_im = float("inf")
        loss_history: list[float] = []

        for _ in range(steps):
            opt.zero_grad()
            what = tbs_reparam(z, basis, alpha)  # alpha*arctan(z @ B), bounded
            psi = model.forward_from_embeddings(what, layer)
            l_im = core.state_distance(psi, target, distance)
            if lam > 0.0:
                l_dm = core.distribution_matching_loss(
                    what,
                    W_in,
                    dm_gen,
                    num_nets=int(self._p("dm_feature_nets")),
                    hidden=int(self._p("dm_hidden")),
                    depth=int(self._p("dm_depth")),
                )
                loss = l_im + lam * l_dm
            else:
                loss = l_im
            loss.backward()
            opt.step()

            lv = float(loss.detach())
            loss_history.append(lv)
            if lv < best_loss:  # Algorithm 1 line 9: restore argmin_i L[i]
                best_loss = lv
                best_im = float(l_im.detach())
                best_z = z.detach().clone()

        best_what = tbs_reparam(best_z, basis, alpha).detach()
        arctan_bound = alpha * _HALF_PI
        arctan_bounded = bool(
            torch.all(best_what > -arctan_bound) and torch.all(best_what < arctan_bound)
        )
        recovered, candidates = core.recover_tokens_by_cosine(
            best_what.numpy(), W_in.numpy(), top_k
        )

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "attack": "ISA-TBS (original)",
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "distance": distance,
                "basis": basis_name,
                "alpha": alpha,
                "r": r,
                "embed_dim": d,
                "full_basis": not truncation,
                "truncation": truncation,
                "svd_reconstruction_error": recon_err,
                "arctan_bound": arctan_bound,
                "arctan_bounded": arctan_bounded,
                "steps": steps,
                "lr": lr,
                "final_loss": best_loss,
                "final_state_distance": best_im,
                "num_positions": n,
                "lambda_dm": lam,
                "uses_distribution_matching": lam > 0.0,
            },
        )
