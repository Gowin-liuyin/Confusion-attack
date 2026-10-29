"""Mathematical tests pinning ISA paper formulas to code.

Covers: (a) the TBS SVD factorisation and shapes; (b) the two basis strategies
``B = V`` (unbiased, default) vs ``B = V^T`` (Algorithm 1 lines 2-3); (c) that
TBS uses the **full** ``d_in`` basis by default (no low-rank truncation); (d) the
``phi_z = alpha * arctan`` scale bound with ``alpha = 5/pi`` (App. A, Eq. 11);
(e) that the attacker model is genuinely frozen by the attack (an *unfrozen*
model is handed in; weights must end unchanged with no gradient); and (f) that
the distribution-matching penalty ``L_dm`` is exactly Eq. 3 - **mean**-matching,
a **sum of squares** over the feature dim, with ``phi`` resampled each call.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.data.contracts import AttackContext, Capability
from obfuscation_attacks.models import TorchToyModel
from obfuscation_attacks.attacks.isa import core
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.isa.tbs import (
    TBSAttack,
    tbs_svd,
    tbs_basis,
    tbs_reparam,
    _ALPHA_DEFAULT,
)

pytestmark = [pytest.mark.mathematical, pytest.mark.neural]

_HALF_PI = math.pi / 2.0
LAYER = 1
TOKENS = [1, 5, 9, 13]


def _ctx(model):
    target = model.hidden_states(TOKENS, layer=LAYER).numpy()
    return AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": model.vocab_size, "seq_len": len(TOKENS), "layer": LAYER},
        _obfuscated_hidden_states=target,
        attacker_model=model,
    )


# --- (a) SVD reconstruction ------------------------------------------------
def test_tbs_svd_reconstructs_embedding_matrix():
    model = TorchToyModel(vocab_size=20, hidden_size=16, num_layers=2, seed=1).freeze()
    W_in = model.embedding_matrix()  # (vocab=20, d=16)
    U, S, Vh = tbs_svd(W_in)
    # Shapes of the reduced SVD: U (V, d), S (d,), Vh = V^T (d, d).
    assert U.shape == (20, 16)
    assert S.shape == (16,)
    assert Vh.shape == (16, 16)
    recon = (U * S) @ Vh  # U diag(S) V^T
    assert torch.allclose(recon, W_in, atol=1e-5)
    # Right singular vectors are orthonormal: V^T V = I.
    assert torch.allclose(Vh @ Vh.T, torch.eye(16), atol=1e-5)


# --- (b) two basis strategies: B = V (unbiased, default) vs B = V^T --------
def test_tbs_basis_strategies_V_and_Vt_differ_and_are_orthonormal():
    model = TorchToyModel(vocab_size=20, hidden_size=16, num_layers=2, seed=3).freeze()
    W_in = model.embedding_matrix()
    _, _, Vh = tbs_svd(W_in)  # Vh = V^T

    B_unbiased = tbs_basis(W_in, "unbiased")  # default -> B = V = Vh^T
    B_vt = tbs_basis(W_in, "vt")              # B = V^T = Vh

    assert B_unbiased.shape == (16, 16)
    assert B_vt.shape == (16, 16)
    # Unbiased basis is V (= Vh^T); the other strategy is V^T (= Vh).
    assert torch.allclose(B_unbiased, Vh.T, atol=1e-6)
    assert torch.allclose(B_vt, Vh, atol=1e-6)
    # The two strategies are genuinely different bases for a non-symmetric W_in.
    assert not torch.allclose(B_unbiased, B_vt, atol=1e-4)
    # Both are orthonormal: B B^T = I.
    assert torch.allclose(B_unbiased @ B_unbiased.T, torch.eye(16), atol=1e-5)
    assert torch.allclose(B_vt @ B_vt.T, torch.eye(16), atol=1e-5)


# --- (c) full d_in basis by default (no low-rank truncation) ---------------
def test_tbs_uses_full_din_basis_by_default():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    # Default r is None -> full basis.
    assert TBSAttack.DEFAULTS["r"] is None
    full_basis = tbs_basis(model.embedding_matrix(), "unbiased", r=None)
    assert full_basis.shape == (16, 16)  # all d_in rows, not a top-r slice

    res = TBSAttack(steps=60, seed=0).run(_ctx(model))
    assert res.diagnostics["r"] == 16          # == d_in
    assert res.diagnostics["full_basis"] is True
    assert res.diagnostics["truncation"] is False
    assert res.diagnostics["basis"] == "unbiased"

    # An explicit r < d_in is an honest (flagged) truncation.
    res_trunc = TBSAttack(steps=60, r=8, seed=0).run(_ctx(model))
    assert res_trunc.diagnostics["r"] == 8
    assert res_trunc.diagnostics["full_basis"] is False
    assert res_trunc.diagnostics["truncation"] is True


# --- (d) phi_z = alpha * arctan, alpha = 5/pi (App. A, Eq. 11) --------------
def test_tbs_reparam_entries_within_alpha_arctan_bound():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=2).freeze()
    W_in = model.embedding_matrix()
    basis = tbs_basis(W_in, "unbiased")  # full (16, 16)
    assert basis.shape == (16, W_in.shape[1])
    gen = torch.Generator().manual_seed(0)
    # Even with huge coefficients, alpha*arctan keeps every entry in the open
    # interval (-alpha*pi/2, alpha*pi/2).
    z = 100.0 * torch.randn(len(TOKENS), 16, generator=gen)
    what = tbs_reparam(z, basis, alpha=_ALPHA_DEFAULT)
    bound = _ALPHA_DEFAULT * _HALF_PI
    assert bound == pytest.approx(2.5)  # (5/pi) * (pi/2) = 5/2
    assert torch.all(what > -bound)
    assert torch.all(what < bound)


def test_tbs_reparam_applies_alpha_scaling():
    # phi_z = alpha * arctan(z @ B): the output is exactly alpha times arctan.
    gen = torch.Generator().manual_seed(1)
    basis = torch.eye(4)
    z = torch.randn(3, 4, generator=gen)
    plain = torch.arctan(z @ basis)
    scaled = tbs_reparam(z, basis, alpha=_ALPHA_DEFAULT)
    assert torch.allclose(scaled, _ALPHA_DEFAULT * plain, atol=1e-6)
    # The default alpha is 5/pi.
    assert _ALPHA_DEFAULT == pytest.approx(5.0 / math.pi)
    assert TBSAttack.DEFAULTS["alpha"] == pytest.approx(5.0 / math.pi)


def test_tbs_attack_reports_bounded_what():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    res = TBSAttack(steps=120, seed=0).run(_ctx(model))
    assert res.diagnostics["arctan_bounded"] is True
    assert res.diagnostics["arctan_bound"] == pytest.approx(_ALPHA_DEFAULT * _HALF_PI)
    assert res.diagnostics["alpha"] == pytest.approx(5.0 / math.pi)


# --- (e) model is frozen BY the attack; weights unchanged, no gradient -----
@pytest.mark.parametrize(
    "attack", [ERAttack(distance="l2", steps=120), TBSAttack(steps=120)]
)
def test_attack_freezes_unfrozen_model_and_leaves_it_unchanged(attack):
    # Hand in an UNFROZEN model: its parameters start trainable.
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0)
    assert any(p.requires_grad for p in model.parameters())  # genuinely unfrozen
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    attack.run(_ctx(model))

    for name, p in model.named_parameters():
        # The attack must have frozen the model ...
        assert p.requires_grad is False
        # ... it must never have received a gradient ...
        assert p.grad is None, f"parameter {name} received a gradient"
        # ... and not a single weight value moved.
        assert torch.equal(p, before[name]), f"parameter {name} changed during optimization"


# --- (f) L_dm is exactly Eq. 3: mean-matching, sum-of-squares, phi resampled
def test_distribution_matching_is_eq3_mean_and_sum_of_squares():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    d = int(E.shape[1])
    n, vocab, H = 4, int(E.shape[0]), 8
    what = (0.1 * torch.arange(n * d, dtype=torch.float32)).reshape(n, d)

    loss = core.distribution_matching_loss(
        what, E, torch.Generator().manual_seed(123), num_nets=1, hidden=H, depth=1
    )

    # Replay the exact draws (phi then equal-size real batch) and recompute Eq. 3
    # by hand: mean over the batch rows (overline), squared L2 over feature dim.
    g = torch.Generator().manual_seed(123)
    net = core.RandomFeatureNet(d, hidden=H, depth=1, generator=g)
    idx = torch.randint(0, vocab, (n,), generator=g)
    real = E[idx]
    mean_w = net(what).mean(dim=0)
    mean_r = net(real).mean(dim=0)
    expected_sum = ((mean_w - mean_r) ** 2).sum()
    assert torch.isclose(loss, expected_sum, atol=1e-6)

    # It is a SUM over features, not a mean: sum == H * mean, so they differ.
    expected_mean = ((mean_w - mean_r) ** 2).mean()
    assert not torch.isclose(loss, expected_mean)


def test_distribution_matching_mean_matching_is_row_permutation_invariant():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    what = torch.randn(4, E.shape[1], generator=torch.Generator().manual_seed(2))
    perm = what[[3, 1, 0, 2]]

    # Fresh identical generators -> same phi and same real batch; only the order
    # of the dummy rows differs, and the batch MEAN is order-invariant.
    a = core.distribution_matching_loss(
        what, E, torch.Generator().manual_seed(9), num_nets=2, hidden=16, depth=2
    )
    b = core.distribution_matching_loss(
        perm, E, torch.Generator().manual_seed(9), num_nets=2, hidden=16, depth=2
    )
    assert torch.isclose(a, b, atol=1e-6)


def test_distribution_matching_resamples_phi_each_call():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    what = torch.randn(4, E.shape[1], generator=torch.Generator().manual_seed(2))
    gen = torch.Generator().manual_seed(5)
    # The same advancing generator yields a fresh phi + real batch each call, so
    # two consecutive evaluations differ (the expectation E_{phi} is estimated by
    # resampling, which is what ``_run`` relies on across steps).
    a = core.distribution_matching_loss(what, E, gen, num_nets=1, hidden=16, depth=2)
    b = core.distribution_matching_loss(what, E, gen, num_nets=1, hidden=16, depth=2)
    assert not torch.isclose(a, b)


def test_distribution_matching_is_differentiable_wrt_input():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    what = (0.1 * torch.randn(4, E.shape[1], generator=torch.Generator().manual_seed(4)))
    what.requires_grad_(True)
    loss = core.distribution_matching_loss(
        what, E, torch.Generator().manual_seed(1), num_nets=2, hidden=32, depth=2
    )
    loss.backward()
    assert what.grad is not None
    assert torch.isfinite(what.grad).all()
    # The feature nets carry no trainable parameters (fixed random extractor).
    net = core.RandomFeatureNet(E.shape[1], hidden=32, depth=2, seed=0)
    assert all(p.requires_grad is False for p in net.parameters())
