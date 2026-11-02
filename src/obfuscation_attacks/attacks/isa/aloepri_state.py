"""AloePri State1/State2 adaptation (AloePri App. D.1, "Internal State Attack").

The AloePri appendix describes a *generic* state-matching optimization, not the
full ISA pipeline (App. D.1): "The attack optimizes input embeddings by
leveraging the loss derived from hidden states. ... the attacker first records
the hidden states State1 when clients request inference using their private
input X1. Subsequently, attackers randomly initialize X2 and feeds it into the
pretrained model to evaluate State2. The attacker can use State1 and State2 to
evaluate the loss, thereby optimizing X2 to recover X1." The model weights stay
fixed (in AloePri they are additionally obfuscated with noise/transformation).

This attack reproduces exactly that description and nothing more. It uses only
the state-matching loss ``L_im = d(psi_l^{fA}(X2), State1)`` - **not** ER's
distribution-matching penalty (Eq. 3) and **not** TBS's SVD basis + arctan
reparameterization. It therefore does **not** claim to reproduce ER or TBS in
full; those remain the original-ISA attacks in ``er.py`` / ``tbs.py``. The
difference from ER is also concrete: ER initialises the dummy embeddings at zero
and adds ``L_dm``, whereas this adaptation random-initialises X2 and optimizes
``L_im`` only. X2 is optimized with AdamW, consistent with the ISA optimizer.

AttnScore vs HiddenState. AloePri Table 4 (Sec. 7.5) ablates ISA against *two*
internal-state targets: attention scores ``AttnScore`` and decoder-layer
``HiddenState``. This adaptation targets **HiddenState** only - the layer-``l``
hidden state produced by the toy model. An attention-score target is **NOT
IMPLEMENTED** here (the toy model exposes no attention map, and the scenario
plumbing only carries hidden states), so we report that status honestly rather
than claiming AttnScore coverage.

Tokens are recovered, as in the ISA second phase, by cosine similarity with
``w_in`` so the result is comparable to the other ISA attacks.
"""

from __future__ import annotations

import torch

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import core


@register
class AloePriStateAttack(Attack):
    name = "isa.aloepri_state"
    family = "isa"
    variant = "aloepri"
    required_capabilities = frozenset({Capability.HIDDEN_STATES})

    DEFAULTS = {
        "distance": "l2",   # ISA default MSE (AloePri does not specify); "cos" available
        "lr": 0.1,
        "steps": 300,
        "init_scale": 0.1,  # std of the random X2 initialisation
        "weight_decay": 0.0,  # UNSPECIFIED (AdamW, no decay value given by the paper)
        "top_k": 5,
        "seed": 0,
    }

    def _p(self, key):
        return self.params.get(key, self.DEFAULTS[key])

    def _run(self, ctx: AttackContext) -> AttackResult:
        model, reason = core.resolve_model(ctx)
        if model is None:
            return AttackResult.not_applicable(reason)

        # State1 = the observed target state for the private input X1.
        state1 = core.as_target_tensor(ctx.obfuscated_hidden_states())
        layer = ctx.descriptor.get("layer")

        distance = str(self._p("distance"))
        lr = float(self._p("lr"))
        steps = int(self._p("steps"))
        init_scale = float(self._p("init_scale"))
        weight_decay = float(self._p("weight_decay"))
        top_k = int(self._p("top_k"))
        seed = int(self._p("seed"))

        seed_everything(seed)
        core.freeze_model(model)

        E = model.embedding_matrix()  # (vocab, d)
        d = int(E.shape[1])
        n = int(state1.shape[0])
        if int(state1.shape[1]) != d:
            return AttackResult.failed(
                f"state dim {state1.shape[1]} != attacker embedding dim {d}"
            )

        # Step 2: randomly initialise X2's continuous input embeddings.
        init_gen = torch.Generator().manual_seed(seed + 3)
        x2 = (init_scale * torch.randn(n, d, generator=init_gen)).requires_grad_(True)

        # AdamW (consistent with the ISA optimizer). weight_decay UNSPECIFIED.
        opt = torch.optim.AdamW([x2], lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        best_x2 = x2.detach().clone()

        # Step 3: optimize X2 by the state loss L_im only.
        for _ in range(steps):
            opt.zero_grad()
            state2 = model.forward_from_embeddings(x2, layer)
            l_im = core.state_distance(state2, state1, distance)
            l_im.backward()
            opt.step()
            lv = float(l_im.detach())
            if lv < best_loss:
                best_loss = lv
                best_x2 = x2.detach().clone()

        recovered, candidates = core.recover_tokens_by_cosine(
            best_x2.numpy(), E.numpy(), top_k
        )

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "attack": "ISA AloePri State1/State2 adaptation (App. D.1)",
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "adaptation_note": (
                    "generic state-matching optimization; L_im only, random X2 "
                    "init; does NOT reproduce ER (no L_dm) or TBS (no SVD/arctan)"
                ),
                # Table 4 targets AttnScore AND HiddenState; we cover HiddenState.
                "target_state": "HiddenState",
                "attnscore_implemented": False,
                "attnscore_status": (
                    "NOT IMPLEMENTED: AloePri Table 4 also targets AttnScore, but "
                    "the toy model exposes no attention map and the scenario only "
                    "carries hidden states; AttnScore is not claimed"
                ),
                "distance": distance,
                "steps": steps,
                "lr": lr,
                "init_scale": init_scale,
                "final_state_distance": best_loss,
                "num_positions": n,
                "embed_dim": d,
            },
        )
