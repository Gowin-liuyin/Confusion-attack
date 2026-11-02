"""Regression tests: discretization re-forward verification.

For each ISA attack we run it end-to-end through ``.run(ctx)``, take the
recovered discrete tokens, re-embed and forward them through the frozen model to
layer ``l``, and confirm the resulting state is genuinely closer to the target
than the state of random token sequences. This checks the attack actually
reduced the inversion objective (Eq. 1), not merely that it returned something.
The random token sequences form a baseline; the recovered state must beat the
*best* of many random controls by a clear margin, so the threshold is meaningful
rather than trivially satisfied.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability
from obfuscation_attacks.models import TorchToyModel
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.isa.tbs import TBSAttack
from obfuscation_attacks.attacks.isa.aloepri_state import AloePriStateAttack

pytestmark = [pytest.mark.regression, pytest.mark.neural]

LAYER = 1
TOKENS = [1, 5, 9, 13]
N = len(TOKENS)
VOCAB = 16


def _cos_state_distance(a: np.ndarray, b: np.ndarray) -> float:
    at = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    bt = torch.as_tensor(np.asarray(b), dtype=torch.float32)
    return float((1.0 - torch.nn.functional.cosine_similarity(at, bt, dim=-1)).mean())


@pytest.fixture()
def victim():
    model = TorchToyModel(vocab_size=VOCAB, hidden_size=16, num_layers=2, seed=0).freeze()
    target = model.hidden_states(TOKENS, layer=LAYER).numpy()
    ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": VOCAB, "seq_len": N, "layer": LAYER},
        _obfuscated_hidden_states=target,
        attacker_model=model,
    )
    return model, target, ctx


def _random_baseline(model, target, trials=64):
    """Distances to target of many random token sequences (the control group)."""
    rng = np.random.default_rng(12345)
    dists = []
    for _ in range(trials):
        seq = rng.integers(0, VOCAB, size=N).tolist()
        state = model.hidden_states(seq, layer=LAYER).numpy()
        dists.append(_cos_state_distance(state, target))
    return np.asarray(dists)


@pytest.mark.parametrize(
    "make_attack",
    [
        pytest.param(lambda: ERAttack(seed=0), id="er-default-mse"),
        pytest.param(lambda: ERAttack(distance="cos", steps=400, seed=0), id="er-cos"),
        pytest.param(lambda: ERAttack(distance="l2", lambda_dm=1e-3, steps=400, seed=0), id="er-l2-dm"),
        pytest.param(lambda: TBSAttack(seed=0), id="tbs-default"),
        pytest.param(lambda: TBSAttack(basis="vt", seed=0), id="tbs-vt"),
        pytest.param(lambda: AloePriStateAttack(seed=0), id="aloepri-default"),
    ],
)
def test_recovered_tokens_reduce_state_distance(victim, make_attack):
    model, target, ctx = victim
    res = make_attack().run(ctx)
    assert res.status is AttackStatus.OK
    assert res.recovered_token_ids is not None

    recovered_ids = [int(t) for t in res.recovered_token_ids]
    # Re-embed + forward the DISCRETE tokens through the frozen model to layer l.
    recovered_state = model.hidden_states(recovered_ids, layer=LAYER).numpy()
    recovered_dist = _cos_state_distance(recovered_state, target)

    baseline = _random_baseline(model, target)
    # The random control group is genuinely far from the target (so the bar is
    # not trivially met): a non-degenerate spread well above zero.
    assert baseline.min() > 0.05
    # Genuinely reduced the objective: strictly closer than EVERY random control,
    # and by a clear margin below the best random draw (meaningful threshold).
    assert recovered_dist < baseline.min()
    assert recovered_dist < 0.5 * baseline.min()
    assert recovered_dist < 0.25 * baseline.mean()


def test_perfect_token_sequence_has_zero_distance(victim):
    # Sanity anchor for the metric: the true tokens re-forward to distance ~0.
    model, target, _ = victim
    exact = model.hidden_states(TOKENS, layer=LAYER).numpy()
    assert _cos_state_distance(exact, target) == pytest.approx(0.0, abs=1e-6)
