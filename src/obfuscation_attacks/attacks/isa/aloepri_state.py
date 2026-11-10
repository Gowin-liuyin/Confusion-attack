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
``HiddenState``. ``target="hidden"`` (the default, and the only target the
appendix text names) matches layer-``l`` hidden states. ``target="attention"``
matches the same layer's attention map, flattened per token to
``[seq, heads * seq]`` (:func:`core.attention_token_features`). That target
needs ``attacker_model.forward_attention_scores``; a model with no attention
map returns ``not_applicable`` instead of a hidden-state result relabelled as
AttnScore. Neither target uses ER's ``L_dm`` or TBS's SVD basis.

Tokens are recovered, as in the ISA second phase, by cosine similarity with
``w_in`` so the result is comparable to the other ISA attacks.
"""

from __future__ import annotations

import torch

from ...data.contracts import AttackContext, AttackResult, Capability
from ...utils import seed_everything
from ..base import Attack, register
from . import core


def _init_generator(device, seed: int) -> "torch.Generator":
    """Torch generator on the SAME device as the tensor it will fill.

    A CPU generator passed to ``torch.randn(device="cuda")`` raises
    ``RuntimeError``; the generator device must match the target tensor device.
    CPU behaviour is unchanged.
    """
    return torch.Generator(device=device).manual_seed(int(seed))


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
        # "hidden" = Table 4 HiddenState; "attention" = Table 4 AttnScore.
        "target": "hidden",
    }

    def _p(self, key):
        return self.params.get(key, self.DEFAULTS[key])

    def _predict_state(self, model, x2, layer, target_kind: str):
        """State2 for the selected Table 4 target. Weights stay frozen."""
        if target_kind == "hidden":
            return model.forward_from_embeddings(x2, layer)
        if target_kind != "attention":
            raise ValueError(f"unknown target {target_kind!r}")
        return model.forward_attention_scores(x2, layer)

    def _run(self, ctx: AttackContext) -> AttackResult:
        model, reason = core.resolve_model(ctx)
        if model is None:
            return AttackResult.not_applicable(reason)

        layer = ctx.descriptor.get("layer")
        # State1 = the observed target state for the private input X1, on the
        # attacker model's device (the optimization runs there).
        state1 = core.as_target_tensor(ctx.obfuscated_hidden_states(),
                                       core.model_device(model))

        distance = str(self._p("distance"))
        lr = float(self._p("lr"))
        steps = int(self._p("steps"))
        init_scale = float(self._p("init_scale"))
        weight_decay = float(self._p("weight_decay"))
        top_k = int(self._p("top_k"))
        seed = int(self._p("seed"))
        target_kind = str(self._p("target"))
        if target_kind not in {"hidden", "attention"}:
            return AttackResult.failed(
                f"target must be 'hidden' or 'attention', got {target_kind!r}"
            )

        if steps <= 0:
            return AttackResult.failed("steps must be positive")

        seed_everything(seed)
        core.freeze_model(model)

        E = model.embedding_matrix()  # (vocab, d)
        d = int(E.shape[1])
        n = int(state1.shape[0])
        if target_kind == "attention" and not hasattr(model, "forward_attention_scores"):
            return AttackResult.not_applicable(
                "AttnScore requires attacker_model.forward_attention_scores; "
                "this model exposes no attention map"
            )
        if target_kind == "hidden" and int(state1.shape[1]) != d:
            return AttackResult.failed(
                f"state dim {state1.shape[1]} != attacker embedding dim {d}"
            )

        # Step 2: randomly initialise X2's continuous input embeddings.
        dev = state1.device
        init_gen = _init_generator(dev, seed + 3)
        x2 = (init_scale * torch.randn(n, d, generator=init_gen,
                                    device=dev)).requires_grad_(True)

        # AdamW (consistent with the ISA optimizer). weight_decay UNSPECIFIED.
        opt = torch.optim.AdamW([x2], lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        best_x2 = x2.detach().clone()

        callback = self.params.get("checkpoint_callback")
        checkpoints = set(self.params.get("checkpoints", ()))

        def snapshot(step):
            recovered, candidates = core.recover_tokens_by_cosine(best_x2, E, top_k)
            return AttackResult(
                recovered_token_ids=recovered,
                candidates=candidates,
                diagnostics={
                    "attack": "ISA AloePri State1/State2 adaptation (App. D.1)",
                    "optimizer": "AdamW", "weight_decay": weight_decay,
                    "adaptation_note": (
                        "attention-score matching (L_im only); does not "
                        "reproduce ISA ER or TBS"
                        if target_kind == "attention"
                        else "hidden-state matching (L_im only); does not "
                             "reproduce ISA ER or TBS"
                    ),
                    "target_state": "AttnScore" if target_kind == "attention" else "HiddenState",
                    "attnscore_implemented": True,
                    "attnscore_status": (
                        "this run targets AttnScore"
                        if target_kind == "attention"
                        else "AttnScore is a separate target='attention' run; this run is HiddenState"
                    ),
                    "distance": distance, "steps": step, "lr": lr,
                    "init_scale": init_scale, "final_loss": best_loss,
                    "final_state_distance": best_loss, "num_positions": n, "embed_dim": d,
                },
            )

        # Step 3: optimize X2 by the state loss L_im only.
        for step in range(1, steps + 1):
            opt.zero_grad()
            try:
                state2 = self._predict_state(model, x2, layer, target_kind)
            except (RuntimeError, ValueError) as exc:
                failed = AttackResult.failed(f"state forward failed at step {step}: {exc}")
                failed.diagnostics["executed_steps"] = step - 1
                failed.diagnostics["target_state"] = (
                    "AttnScore" if target_kind == "attention" else "HiddenState"
                )
                return failed
            l_im = core.state_distance(state2, state1, distance)
            if not torch.isfinite(l_im):
                failed = AttackResult.failed(f"non-finite loss at step {step}")
                failed.diagnostics["executed_steps"] = step - 1
                return failed
            l_im.backward()
            if x2.grad is None or not torch.isfinite(x2.grad).all():
                failed = AttackResult.failed(f"non-finite gradient at step {step}")
                failed.diagnostics["executed_steps"] = step - 1
                return failed
            # Snapshot x2_i (the state that produced l_im) before the update.
            current_x2 = x2.detach().clone()
            opt.step()
            if not torch.isfinite(x2).all():
                failed = AttackResult.failed(f"non-finite optimizer state at step {step}")
                failed.diagnostics["executed_steps"] = step
                return failed
            lv = float(l_im.detach())
            if lv < best_loss:
                best_loss = lv
                best_x2 = current_x2
            if callback is not None and step in checkpoints:
                callback(step, snapshot(step))

        return snapshot(steps)
