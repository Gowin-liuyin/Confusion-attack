"""TBS - Token Basis Selection (ISA original, Dong et al. 2025).

Algorithm 1 (red blocks) + App. A. For deep layers the gradients on a free
dummy embedding explode, so TBS instead optimizes a *low-dimensional* coefficient
vector over an orthogonal basis of the input-embedding space:

    W_in = U diag(S) V^T              (SVD of fA.embedding_matrix())
    B    = V^T  (rows of Vt)          (SVD basis; we take the first r rows)
    what = phi_z(z @ B_r)             (phi_z = arctan, Algorithm 1 line 5)

``z`` has shape ``seq_len x r`` with ``r <= d``, so the search space is a few
singular directions instead of the full dictionary. ``phi_z`` is the variable
change that bounds the scale; the paper uses ``arctan`` (App. A), which we keep
exactly - every entry of ``what`` therefore lies in ``(-pi/2, pi/2)``. (The
paper multiplies by ``alpha = 5/pi`` only to accelerate convergence; we keep the
plain ``arctan`` so the ``(-pi/2, pi/2)`` scale bound holds as stated.)

The historical best-loss state is kept (Algorithm 1 line 9) and tokens are
recovered by cosine similarity with ``w_in`` (Algorithm 1 line 11). The model
weights are fixed throughout.
"""

from __future__ import annotations

import math

import torch

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import core

_HALF_PI = math.pi / 2.0


def tbs_svd(w_in: torch.Tensor):
    """Return ``(U, S, Vt)`` with ``w_in = U diag(S) Vt`` (reduced SVD)."""
    return torch.linalg.svd(w_in, full_matrices=False)


def tbs_basis(w_in: torch.Tensor, r: int) -> torch.Tensor:
    """SVD basis ``B_r`` = first ``r`` right-singular vectors (rows of ``Vt``)."""
    _, _, vt = tbs_svd(w_in)
    r = max(1, min(int(r), vt.shape[0]))
    return vt[:r]


def tbs_reparam(z: torch.Tensor, basis_r: torch.Tensor) -> torch.Tensor:
    """``what = phi_z(z @ B_r)`` with ``phi_z = arctan`` (Algorithm 1 line 5).

    Output entries are bounded to ``(-pi/2, pi/2)`` by ``arctan``.
    """
    return torch.arctan(z @ basis_r)


@register
class TBSAttack(Attack):
    name = "isa.tbs"
    family = "isa"
    variant = "original"
    required_capabilities = frozenset({Capability.HIDDEN_STATES})

    DEFAULTS = {
        "distance": "cos",
        "lr": 0.1,
        "steps": 300,
        "r": 8,            # number of singular directions; capped to min(d, r)
        "lambda_dm": 0.0,  # TBS core is pure L_im; >0 adds the shared L_dm (Eq. 3)
        "dm_feature_nets": 3,
        "dm_hidden": 64,
        "dm_depth": 2,
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
        lam = float(self._p("lambda_dm"))
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

        r = max(1, min(int(self._p("r")), d))
        U, S, Vt = tbs_svd(W_in)
        basis_r = Vt[:r].detach()  # (r, d) fixed orthonormal basis rows

        # SVD reconstruction error (diagnostic / pinned by the math test).
        recon = (U * S) @ Vt
        recon_err = float(torch.max(torch.abs(recon - W_in)))

        # Paper initialises the projection weights at 1/d (baseline/TBS init 1/din).
        z = torch.full((n, r), 1.0 / d, dtype=torch.float32, requires_grad=True)

        feat_nets = (
            core.build_feature_nets(
                d,
                num_nets=int(self._p("dm_feature_nets")),
                hidden=int(self._p("dm_hidden")),
                depth=int(self._p("dm_depth")),
                seed=seed,
            )
            if lam > 0.0
            else []
        )
        dm_gen = torch.Generator().manual_seed(seed + 7)

        opt = torch.optim.Adam([z], lr=lr)
        best_loss = float("inf")
        best_z = z.detach().clone()
        best_im = float("inf")
        loss_history: list[float] = []

        for _ in range(steps):
            opt.zero_grad()
            what = tbs_reparam(z, basis_r)  # phi_z(z @ B_r), bounded by arctan
            psi = model.forward_from_embeddings(what, layer)
            l_im = core.state_distance(psi, target, distance)
            if lam > 0.0:
                l_dm = core.distribution_matching_loss(what, W_in, feat_nets, dm_gen)
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

        best_what = tbs_reparam(best_z, basis_r).detach()
        arctan_bounded = bool(
            torch.all(best_what > -_HALF_PI) and torch.all(best_what < _HALF_PI)
        )
        recovered, candidates = core.recover_tokens_by_cosine(
            best_what.numpy(), W_in.numpy(), top_k
        )

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "attack": "ISA-TBS (original)",
                "distance": distance,
                "r": r,
                "embed_dim": d,
                "svd_reconstruction_error": recon_err,
                "arctan_bound": _HALF_PI,
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
