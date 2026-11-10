"""Acceptance tests for F05: the real ISA entry runs the original ER/TBS classes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.isa.tbs import TBSAttack
from obfuscation_attacks.attacks.isa.aloepri_state import AloePriStateAttack
from obfuscation_attacks.data.contracts import AttackContext, Capability
from obfuscation_attacks.attacks.isa.core import attention_token_features
from obfuscation_attacks.evaluation.real_isa import (
    invert_permutation,
    isa_attack,
    map_prediction_lists,
    score_recovery,
    wire_from_states,
    _target,
)
from obfuscation_attacks.schemes.ab_deployment import WirePublic
from obfuscation_attacks.schemes.public_view import PublicDeployment, assert_public

pytestmark = pytest.mark.unit

V, D = 10, 6


class TinyModel(torch.nn.Module):
    def __init__(self, E: np.ndarray, W: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("E", torch.tensor(E, dtype=torch.float32))
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))

    def embedding_matrix(self) -> torch.Tensor:
        return self.E

    def forward_from_embeddings(self, what: torch.Tensor, layer: int) -> torch.Tensor:
        return what @ self.W

    def parameters(self):
        return []

    def freeze(self) -> None:
        return None


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(V, D)).astype(np.float32)
    W = rng.normal(size=(D, D)).astype(np.float32) + 2.0 * np.eye(D, dtype=np.float32)
    model = TinyModel(E, W)
    ids = np.array([1, 3, 5], dtype=np.int64)
    target = (torch.tensor(E[ids]) @ model.W).detach().numpy()
    public = PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E,
        forward_fn=lambda embeds: SimpleNamespace(
            hidden_states=[torch.as_tensor(embeds), torch.as_tensor(embeds) @ model.W]
        ),
        allow_clean_weights=True,
    )
    wire = WirePublic(
        arm="ab", request_id=0, seq_len=len(ids), hidden_states=target[None], layers=(1,),
    )
    return public, wire, model, target, ids


def _direct(attack_cls, model, target, **params):
    ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"layer": 1},
        _obfuscated_hidden_states=target,
        attacker_model=model,
    )
    return attack_cls(**params).run(ctx)


def test_wire_from_states_is_the_contract_isa_attack_reads():
    states = [np.ones((4, 3), np.float32), np.zeros((4, 5), np.float32)]
    wire = wire_from_states("aloepri", 7, (1, 5), states)
    assert _target(wire, 1).shape == (4, 3)
    assert _target(wire, 5).shape == (4, 5)
    assert wire.seq_len == 4 and wire.request_id == 7
    bare = SimpleNamespace(hidden_states=tuple(states))
    with pytest.raises(AttributeError):
        _target(bare, 1)


def test_plaintext_score_inverts_permutation_without_dropping_positions():
    inverse = invert_permutation(np.array([2, 0, 1, 3]))
    recovered = [[2, 0], []]
    candidates = [[[2, 0, 1], [0, 2]], []]
    plain_rec, plain_cand = map_prediction_lists(recovered, candidates, inverse)
    # server id 2 is plaintext 0; server id 0 is plaintext 1.
    assert plain_rec[0] == [0, 1]
    truth = [[0, 1, 9], [4]]
    scored = score_recovery(plain_rec, plain_cand, truth)
    assert scored["denominator"] == 4
    assert scored["numerator"]["top1"] == 2
    assert scored["top1"] == pytest.approx(0.5)


def test_attention_features_match_hidden_state_layer_index_and_keep_grad():
    attn = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    attn.requires_grad_(True)
    feat = attention_token_features(attn)
    assert feat.shape == (3, 6)
    # token 0 concatenates head 0's row 0 and head 1's row 0.
    assert feat[0].detach().tolist() == [0, 1, 2, 9, 10, 11]
    feat.sum().backward()
    assert attn.grad is not None
    public, wire, model, target, ids = _fixture()
    assert_public(public)
    direct = _direct(
        ERAttack, model, target,
        distance="l2", lr=0.1, steps=20, lambda_dm=0.0, top_k=5, seed=0,
    )
    out = isa_attack(
        public, [wire], method="er", layers=(1,), steps=[20], lr=0.1,
        restarts=1, seed=0, lambda_dm=0.0, top_k=5, verbose=False,
    )
    best = out["layers"]["layer1"]["best"]
    assert out["method"] == "ER"
    assert best["recovered_token_ids"][0] == direct.recovered_token_ids.tolist()
    assert best["loss"] == pytest.approx(direct.diagnostics["final_loss"], abs=1e-6)
    assert out["optimizer"] == "AdamW"


def test_isa_reads_layer_from_wire_order():
    public, wire, _, _, _ = _fixture()
    expected = isa_attack(public, [wire], steps=[3], restarts=1)
    wire.layers = (0, 1)
    wire.hidden_states = np.concatenate(
        [np.zeros_like(wire.hidden_states), wire.hidden_states], axis=0
    )
    actual = isa_attack(public, [wire], steps=[3], restarts=1)
    assert actual["layers"] == expected["layers"]


def test_tbs_entry_matches_the_original_class_exactly():
    public, wire, model, target, ids = _fixture(seed=4)
    direct = _direct(
        TBSAttack, model, target,
        distance="l2", lr=0.1, steps=20, basis="unbiased", alpha=5.0 / np.pi,
        r=None, lambda_dm=0.0, top_k=5, seed=0,
    )
    out = isa_attack(
        public, [wire], method="tbs", layers=(1,), steps=[20], lr=0.1,
        restarts=1, seed=0, lambda_dm=0.0, top_k=5, basis="unbiased", verbose=False,
    )
    best = out["layers"]["layer1"]["best"]
    assert out["method"] == "TBS"
    assert best["recovered_token_ids"][0] == direct.recovered_token_ids.tolist()
    assert best["loss"] == pytest.approx(direct.diagnostics["final_loss"], abs=1e-6)
    assert out["basis"] == "unbiased" and out["full_basis"] is True


def test_aloepri_state_entry_matches_direct_attack_at_continuous_checkpoints():
    public, wire, model, target, _ = _fixture(seed=8)
    direct = _direct(AloePriStateAttack, model, target,
                     distance="l2", lr=0.1, steps=4, top_k=5, seed=0)
    out = isa_attack(public, [wire], method="aloepri_state", layers=(1,),
                     steps=[2, 4], lr=0.1, restarts=1, seed=0, top_k=5)
    curve = out["layers"]["layer1"]["checkpoints"]
    assert out["method"] == "AloePri hidden-state adaptation"
    assert out["optimizer_steps_total"] == 4
    assert curve[4]["recovered_token_ids"][0] == direct.recovered_token_ids.tolist()
    assert curve[4]["loss"] == pytest.approx(direct.diagnostics["final_state_distance"])
    assert curve[2]["loss"] >= curve[4]["loss"]
    assert curve[4]["diagnostics"]["attnscore_implemented"] is True
    assert curve[4]["diagnostics"]["target_state"] == "HiddenState"
    with pytest.raises(ValueError, match="does not implement distribution matching"):
        isa_attack(public, [wire], method="aloepri_state", steps=[1], lambda_dm=0.1)


def test_aloepri_state_rejects_zero_steps_and_nonfinite_loss():
    _, _, model, target, _ = _fixture(seed=9)
    empty = _direct(AloePriStateAttack, model, target, steps=0)
    assert "steps must be positive" in empty.status_reason
    bad = _direct(AloePriStateAttack, model, target * np.nan, steps=2)
    assert "non-finite loss" in bad.status_reason
    assert bad.diagnostics["executed_steps"] == 0


def test_restart_priors_are_not_seed_only_duplicates():
    from obfuscation_attacks.evaluation.real_isa import _public_init

    public, wire, model, target, ids = _fixture(seed=5)
    first = _public_init(public, target.shape[0], 0)
    second = _public_init(public, target.shape[0], 1000)
    assert first.shape == target.shape
    assert not np.array_equal(first, second)


def test_restarts_keep_the_best_checkpoint_entry():
    public, wire, model, target, ids = _fixture(seed=6)
    out = isa_attack(
        public, [wire], method="er", layers=(1,), steps=[2], lr=0.1,
        restarts=2, seed=0, lambda_dm=0.0, top_k=5, verbose=False,
    )
    entry = out["layers"]["layer1"]["checkpoints"][2]
    assert entry["restart"] == [entry["per_wire"][0]["restart"]]
    assert entry["per_wire"][0]["restart"] in {0, 1}
    assert out["layers"]["layer1"]["best"]["loss"] <= entry["loss"] + 1e-9
