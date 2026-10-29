"""ER - Embedding Recovery (ISA original, Dong et al. 2025, USENIX Security '25).

Algorithm 1 (blue blocks) + Eq. 3. For shallow layers the gradients on the dummy
embedding are small and stable, so ER directly optimizes a continuous
input-embedding matrix ``what`` (shape ``seq_len x d``) to match the observed
victim state:

    L_im  = d( psi_l^{fA}(what), h_l^{fV} )            (state-matching loss)
    L_inv = L_im + lambda * L_dm                        (ISA Eq. 3)

``L_dm`` is the distribution-matching penalty computed with a fixed random
Gaussian network as a universal feature extractor, matching the batch
statistics (mean and variance across sequence positions) of ``what`` against
real token embeddings sampled from ``fA.embedding_matrix()`` (see
:func:`..isa.core.distribution_matching_loss`). It keeps ``what`` distributed
like genuine embeddings so the cosine token-recovery phase is not thrown off by
an over-large norm.

The model weights are fixed during optimization; only ``what`` is optimized
(Adam). The best-loss iterate is kept (Algorithm 1 line 8). Tokens are then
recovered as those of highest cosine similarity with ``w_in`` (Algorithm 1
line 11).
"""

from __future__ import annotations

import torch

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import core


@register
class ERAttack(Attack):
    name = "isa.er"
    family = "isa"
    variant = "original"
    required_capabilities = frozenset({Capability.HIDDEN_STATES})

    # Sensible small defaults for the toy/CPU setting (the paper uses AdamW with
    # mu in {1e-3, 5e-4}, lambda in {0, 1e-3}, E = 5e4 steps on real LLMs; those
    # scales are unnecessary and slow for a tiny deterministic model).
    DEFAULTS = {
        "distance": "cos",   # d in Eq. 1 / L_im; "cos" default, "l2" (=MSE) optional
        "lr": 0.1,
        "steps": 300,
        # lambda in Eq. 3. The paper's default is lambda=0 (distribution matching
        # is the optional "improved setting", lambda in {0, 1e-3}); set >0 to turn
        # on the L_dm penalty built in core.distribution_matching_loss.
        "lambda_dm": 0.0,
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

        E = model.embedding_matrix()  # (vocab, d), detached
        d = int(E.shape[1])
        n = int(target.shape[0])
        if int(target.shape[1]) != d:
            return AttackResult.failed(
                f"state dim {target.shape[1]} != attacker embedding dim {d}"
            )

        # ER initialises the dummy embeddings at zero (paper 5.1).
        what = torch.zeros(n, d, dtype=torch.float32, requires_grad=True)

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

        opt = torch.optim.Adam([what], lr=lr)
        best_loss = float("inf")
        best_what = what.detach().clone()
        best_im = float("inf")
        loss_history: list[float] = []

        for _ in range(steps):
            opt.zero_grad()
            psi = model.forward_from_embeddings(what, layer)  # psi_l^{fA}(what)
            l_im = core.state_distance(psi, target, distance)
            if lam > 0.0:
                l_dm = core.distribution_matching_loss(what, E, feat_nets, dm_gen)
                loss = l_im + lam * l_dm
            else:
                loss = l_im
            loss.backward()
            opt.step()

            lv = float(loss.detach())
            loss_history.append(lv)
            if lv < best_loss:  # Algorithm 1 line 8: keep argmin_i L[i]
                best_loss = lv
                best_im = float(l_im.detach())
                best_what = what.detach().clone()

        recovered, candidates = core.recover_tokens_by_cosine(
            best_what.numpy(), E.numpy(), top_k
        )

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "attack": "ISA-ER (original)",
                "distance": distance,
                "lambda_dm": lam,
                "steps": steps,
                "lr": lr,
                "final_loss": best_loss,
                "final_state_distance": best_im,
                "num_positions": n,
                "embed_dim": d,
                "uses_distribution_matching": lam > 0.0,
                "dm_feature_nets": len(feat_nets),
            },
        )
