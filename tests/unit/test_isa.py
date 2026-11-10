"""Unit tests for the ISA family (small-model recovery sanity).

These run a genuine gradient-based inversion against a tiny, frozen
``TorchToyModel`` (white-box ``fA == fV``) and check that the attack recovers
the planted tokens. Everything is sized to finish in well under a second.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability
from obfuscation_attacks.metrics import ttrsr
from obfuscation_attacks.models import TorchToyModel
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.isa.tbs import TBSAttack
from obfuscation_attacks.attacks.isa.aloepri_state import AloePriStateAttack

pytestmark = [pytest.mark.unit, pytest.mark.neural]

LAYER = 1
TOKENS = [1, 5, 9, 13]


def _ctx(model, tokens=TOKENS, layer=LAYER):
    target = model.hidden_states(tokens, layer=layer).numpy()
    return AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": model.vocab_size, "seq_len": len(tokens), "layer": layer},
        _obfuscated_hidden_states=target,
        attacker_model=model,
    )


@pytest.fixture()
def victim():
    """Frozen toy model + a context exposing only the layer-1 target state."""
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    return model, _ctx(model)


def test_er_recovers_tokens(victim):
    _, ctx = victim
    # ER with the MSE state loss (the paper's default d) and lambda=0 recovers
    # the shallow-layer tokens exactly in the toy; >= 0.75 is the task bar.
    res = ERAttack(distance="l2", lr=0.1, steps=400, lambda_dm=0.0, seed=0).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.recovered_token_ids is not None
    assert res.recovered_token_ids.shape == (len(TOKENS),)
    assert np.issubdtype(res.recovered_token_ids.dtype, np.integer)
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75


def test_er_default_distance_is_mse_and_recovers(victim):
    # The paper's default distance is MSE; the default path must exercise it.
    _, ctx = victim
    res = ERAttack(seed=0).run(ctx)  # distance defaults to "l2" (MSE), lambda_dm=0
    assert res.status is AttackStatus.OK
    assert res.diagnostics["distance"] == "l2"
    assert res.diagnostics["optimizer"] == "AdamW"
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75


def test_er_cosine_distance_available(victim):
    # Cosine stays available as the documented alternative distance.
    _, ctx = victim
    res = ERAttack(distance="cos", lr=0.1, steps=400, seed=0).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["distance"] == "cos"
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75


def test_er_distribution_matching_penalty_runs(victim):
    # lambda > 0 turns on the Eq. 3 L_dm penalty; it must still produce a valid
    # recovery and advertise that distribution matching was used.
    _, ctx = victim
    res = ERAttack(distance="l2", lr=0.1, steps=400, lambda_dm=1e-3, seed=0).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["uses_distribution_matching"] is True
    assert res.diagnostics["dm_feature_nets"] >= 1
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75


def test_tbs_recovers_tokens(victim):
    _, ctx = victim
    res = TBSAttack(lr=0.1, steps=300, seed=0).run(ctx)
    assert res.status is AttackStatus.OK
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75
    # TBS-specific diagnostics required by the deliverable.
    assert res.diagnostics["svd_reconstruction_error"] < 1e-4
    assert res.diagnostics["arctan_bounded"] is True
    # Full d_in basis by default (no low-rank truncation).
    assert res.diagnostics["r"] == 16
    assert res.diagnostics["full_basis"] is True
    assert res.diagnostics["truncation"] is False
    assert res.diagnostics["basis"] == "unbiased"
    assert res.diagnostics["alpha"] == pytest.approx(5.0 / math.pi)


def test_tbs_recovers_on_square_toy_full_basis():
    # Square embedding matrix (vocab == d): the full d_in basis is exercised.
    model = TorchToyModel(vocab_size=12, hidden_size=12, num_layers=2, seed=1).freeze()
    tokens = [0, 3, 6, 9]
    res = TBSAttack(lr=0.1, steps=400, seed=0).run(_ctx(model, tokens))
    assert res.status is AttackStatus.OK
    assert res.diagnostics["r"] == 12          # full basis == d_in == vocab
    assert res.diagnostics["full_basis"] is True
    assert ttrsr(res.recovered_token_ids, tokens) >= 0.75


def test_tbs_supports_vocab_smaller_than_hidden_dimension():
    model = TorchToyModel(vocab_size=4, hidden_size=8, num_layers=2, seed=7).freeze()
    result = TBSAttack(lr=.1, steps=3, seed=0).run(_ctx(model, [0, 2]))
    assert result.status is AttackStatus.OK
    assert result.diagnostics["r"] == 8
    assert result.diagnostics["full_basis"] is True


@pytest.mark.parametrize("attack", [ERAttack, TBSAttack])
def test_invalid_isa_budget_and_nonfinite_target_fail(attack, victim):
    _, ctx = victim
    assert attack(steps=0).run(ctx).status is AttackStatus.FAILED
    ctx._obfuscated_hidden_states = np.full_like(ctx._obfuscated_hidden_states, np.nan)
    result = attack(steps=2).run(ctx)
    assert result.status is AttackStatus.FAILED
    assert "non-finite" in result.status_reason


def test_tbs_basis_strategies_both_recover(victim):
    _, ctx = victim
    res_unbiased = TBSAttack(lr=0.1, steps=300, basis="unbiased", seed=0).run(ctx)
    res_vt = TBSAttack(lr=0.1, steps=300, basis="vt", seed=0).run(ctx)
    assert res_unbiased.diagnostics["basis"] == "unbiased"
    assert res_vt.diagnostics["basis"] == "vt"
    # Both full-basis strategies recover the planted tokens on the toy.
    assert ttrsr(res_unbiased.recovered_token_ids, TOKENS) >= 0.75
    assert ttrsr(res_vt.recovered_token_ids, TOKENS) >= 0.75


def test_aloepri_state_recovers_tokens(victim):
    _, ctx = victim
    res = AloePriStateAttack(lr=0.1, steps=300, seed=0).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["optimizer"] == "AdamW"
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75
    # Must advertise itself as the adaptation, not a full ER/TBS reproduction.
    note = res.diagnostics["adaptation_note"].lower()
    assert "l_im only" in note or "l_im" in note
    assert "does not reproduce" in note or "not reproduce" in note
    # AttnScore is a separate target. This default run is HiddenState, and the
    # toy model has no attention map.
    assert res.diagnostics["target_state"] == "HiddenState"
    assert res.diagnostics["attnscore_implemented"] is True
    assert "hiddenstate" in res.diagnostics["attnscore_status"].lower()
    missing = AloePriStateAttack(target="attention", steps=2, seed=0).run(ctx)
    assert missing.status is AttackStatus.NOT_APPLICABLE
    assert "attention" in missing.status_reason.lower()


def test_aloepri_attention_target_recovers_when_the_model_exposes_maps(victim):
    model, ctx = victim

    class _WithAttention:
        def embedding_matrix(self):
            return model.embedding_matrix()

        def forward_from_embeddings(self, what, layer):
            return model.forward_from_embeddings(what, layer)

        def forward_attention_scores(self, what, layer):
            # Same differentiable map as the hidden state, so the Table 4
            # attention branch is the same optimization with a different probe.
            return model.forward_from_embeddings(what, layer)

        def freeze(self):
            return model.freeze()

    wrapped = _WithAttention()
    attn_ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor=dict(ctx.descriptor),
        _obfuscated_hidden_states=ctx.obfuscated_hidden_states(),
        attacker_model=wrapped,
    )
    res = AloePriStateAttack(target="attention", lr=0.1, steps=300, seed=0).run(attn_ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["target_state"] == "AttnScore"
    assert ttrsr(res.recovered_token_ids, TOKENS) >= 0.75


def test_candidates_are_ranked_topk(victim):
    _, ctx = victim
    top_k = 4
    res = ERAttack(distance="l2", lr=0.1, steps=400, top_k=top_k, seed=0).run(ctx)
    assert res.candidates is not None
    assert len(res.candidates) == len(TOKENS)
    for i, cand in enumerate(res.candidates):
        assert len(cand) == top_k  # vocab (16) >= top_k
        # The top-1 candidate is exactly the recovered token at that position.
        assert cand[0] == int(res.recovered_token_ids[i])


def test_missing_attacker_model_is_not_applicable():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    target = model.hidden_states(TOKENS, layer=LAYER).numpy()
    ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": 16, "seq_len": len(TOKENS), "layer": LAYER},
        _obfuscated_hidden_states=target,
        attacker_model=None,  # ISA cannot back-propagate without a model
    )
    for attack in (ERAttack(), TBSAttack(), AloePriStateAttack()):
        res = attack.run(ctx)
        assert res.status is AttackStatus.NOT_APPLICABLE


def test_missing_hidden_states_capability_is_not_applicable(victim):
    # Capability gate: without HIDDEN_STATES the base run() short-circuits.
    model, _ = victim
    target = model.hidden_states(TOKENS, layer=LAYER).numpy()
    ctx = AttackContext(
        granted=frozenset(),  # nothing granted
        descriptor={"vocab_size": 16, "seq_len": len(TOKENS), "layer": LAYER},
        _obfuscated_hidden_states=target,
        attacker_model=model,
    )
    res = ERAttack().run(ctx)
    assert res.status is AttackStatus.NOT_APPLICABLE
