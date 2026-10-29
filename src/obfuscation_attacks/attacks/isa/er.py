"""ER - Embedding Recovery (ISA original, Dong et al. 2025, USENIX Security '25).

Algorithm 1 (blue blocks) + Eq. 3. For shallow layers the gradients on the dummy
embedding are small and stable, so ER directly optimizes a continuous
input-embedding matrix ``what`` (shape ``|h_l| x din``) to match the observed
victim state:

    L_im  = d( psi_l^{fA}(what), h_l^{fV} )            (state-matching loss)
    L_inv = L_im + lambda * L_dm                        (ISA Eq. 3)

``L_dm`` is the distribution-matching penalty computed with a *random* Gaussian
network as a universal feature extractor, **mean**-matching the batch average of
its random features on ``what`` against real token embeddings sampled from
``fA.embedding_matrix()`` (see :func:`..isa.core.distribution_matching_loss`).
The random net and the real batch are resampled every step. It keeps ``what``
distributed like genuine embeddings so the cosine token-recovery phase is not
thrown off by an over-large norm.

The model weights are fixed during optimization; only ``what`` is optimized with
AdamW (ISA Sec. 5.1: "We use the optimizer AdamW"). The best-loss iterate is
kept (Algorithm 1 line 8). Tokens are then recovered as those of highest cosine
similarity with ``w_in`` (Algorithm 1 line 11).
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

    # Sensible small defaults for the toy/CPU setting. The paper uses AdamW with
    # mu in {1e-3, 5e-4, 1e-4} (mu=1e-3 for layers < 8, 1e-2 deeper; Table 3),
    # lambda in {0, 1e-3}, E = 5e4 steps on real LLMs; those scales are
    # unnecessary and slow for a tiny deterministic model.
    DEFAULTS = {
        # d in Eq. 1 / L_im. Paper default is MSE ("l2"); "cos" stays available.
        "distance": "l2",
        "lr": 0.1,
        "steps": 300,
        # lambda in Eq. 3. The paper's default is lambda=0 (distribution matching
        # is the optional "improved setting", lambda in {0, 1e-3}); set >0 to turn
        # on the L_dm penalty built in core.distribution_matching_loss.
        "lambda_dm": 0.0,
        "dm_feature_nets": 1,  # phi draws per step (E_{phi} estimate); resampled each step
        "dm_hidden": 64,
        "dm_depth": 2,
        # UNSPECIFIED: the paper states AdamW but gives no weight-decay value, so
        # we default to 0.0 (no unstated regulariser; AdamW with wd=0 == Adam).
        "weight_decay": 0.0,
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
        weight_decay = float(self._p("weight_decay"))
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

        # ER initialises the dummy embeddings at zero (Algorithm 1 line 2; Sec 5.1).
        what = torch.zeros(n, d, dtype=torch.float32, requires_grad=True)

        dm_gen = torch.Generator().manual_seed(seed + 7)

        # AdamW (ISA Sec. 5.1). weight_decay is UNSPECIFIED in the paper.
        opt = torch.optim.AdamW([what], lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        best_what = what.detach().clone()
        best_im = float("inf")
        loss_history: list[float] = []

        for _ in range(steps):
            opt.zero_grad()
            psi = model.forward_from_embeddings(what, layer)  # psi_l^{fA}(what)
            l_im = core.state_distance(psi, target, distance)
            if lam > 0.0:
                # L_dm resampled (fresh phi + real batch) every step via dm_gen.
                l_dm = core.distribution_matching_loss(
                    what,
                    E,
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
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "distance": distance,
                "lambda_dm": lam,
                "steps": steps,
                "lr": lr,
                "final_loss": best_loss,
                "final_state_distance": best_im,
                "num_positions": n,
                "embed_dim": d,
                "uses_distribution_matching": lam > 0.0,
                "dm_feature_nets": int(self._p("dm_feature_nets")) if lam > 0.0 else 0,
            },
        )
