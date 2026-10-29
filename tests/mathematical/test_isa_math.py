"""Mathematical tests pinning ISA paper formulas to code.

Covers: (a) the TBS SVD factorisation and shapes, (b) the ``arctan`` scale
bound of the TBS reparameterisation, (c) that the attacker model is truly frozen
(weights unchanged, ``requires_grad`` False) across an optimization run, and the
distribution-matching penalty ``L_dm`` really is a random-feature batch-statistic
loss rather than a plain norm.
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
from obfuscation_attacks.attacks.isa.tbs import TBSAttack, tbs_svd, tbs_basis, tbs_reparam

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
    U, S, Vt = tbs_svd(W_in)
    # Shapes of the reduced SVD: U (V, d), S (d,), Vt (d, d).
    assert U.shape == (20, 16)
    assert S.shape == (16,)
    assert Vt.shape == (16, 16)
    recon = (U * S) @ Vt  # U diag(S) V^T
    assert torch.allclose(recon, W_in, atol=1e-5)
    # Right singular vectors are orthonormal: V^T V = I.
    assert torch.allclose(Vt @ Vt.T, torch.eye(16), atol=1e-5)


# --- (b) arctan bound ------------------------------------------------------
def test_tbs_reparam_entries_within_arctan_bound():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=2).freeze()
    W_in = model.embedding_matrix()
    r = 8
    basis_r = tbs_basis(W_in, r)
    assert basis_r.shape == (r, W_in.shape[1])
    gen = torch.Generator().manual_seed(0)
    # Even with huge coefficients, arctan keeps every entry inside (-pi/2, pi/2).
    z = 100.0 * torch.randn(len(TOKENS), r, generator=gen)
    what = tbs_reparam(z, basis_r)
    assert torch.all(what > -_HALF_PI)
    assert torch.all(what < _HALF_PI)


def test_tbs_attack_reports_bounded_what():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    res = TBSAttack(steps=200, seed=0).run(_ctx(model))
    assert res.diagnostics["arctan_bounded"] is True
    assert res.diagnostics["arctan_bound"] == pytest.approx(_HALF_PI)


# --- (c) model weights frozen & unchanged ----------------------------------
@pytest.mark.parametrize("attack", [ERAttack(distance="l2", steps=120), TBSAttack(steps=120)])
def test_model_weights_unchanged_during_optimization(attack):
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    attack.run(_ctx(model))

    for name, p in model.named_parameters():
        # No gradient tracking on the victim/attacker model ...
        assert p.requires_grad is False
        # ... and not a single weight value moved.
        assert torch.equal(p, before[name]), f"parameter {name} changed during optimization"


# --- L_dm is a random-feature batch-statistic loss, not a norm -------------
def test_distribution_matching_uses_batch_statistics_not_norm():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    nets = core.build_feature_nets(E.shape[1], num_nets=3, hidden=64, depth=2, seed=0)

    def dm(x):
        gen = torch.Generator().manual_seed(123)  # identical embedding sampling
        return float(core.distribution_matching_loss(x, E, nets, gen))

    real = E[[2, 4, 6, 8]].clone()        # on-distribution batch
    perm = real[[3, 2, 1, 0]].clone()     # same rows, reordered
    zeros = torch.zeros_like(real)        # a plain norm penalty's favourite
    shift = real + 6.0                    # mean-shifted, off-distribution
    scaled = real * 8.0                   # same directions, wrong scale

    # Batch statistic => invariant to the order of sequence positions.
    assert dm(real) == pytest.approx(dm(perm), abs=1e-6)
    # NOT a norm penalty: the zero matrix (smallest Frobenius norm) is penalised
    # *more* than real embeddings, because its zero batch-variance mismatches.
    assert dm(zeros) > dm(real)
    # Distribution-sensitive to a mean shift and to a pure rescale.
    assert dm(shift) > dm(real)
    assert dm(scaled) > dm(real)


def test_distribution_matching_is_differentiable_wrt_input():
    model = TorchToyModel(vocab_size=16, hidden_size=16, num_layers=2, seed=0).freeze()
    E = model.embedding_matrix()
    nets = core.build_feature_nets(E.shape[1], num_nets=2, hidden=32, depth=2, seed=0)
    what = torch.zeros(4, E.shape[1], requires_grad=True)
    gen = torch.Generator().manual_seed(1)
    loss = core.distribution_matching_loss(what, E, nets, gen)
    loss.backward()
    assert what.grad is not None
    assert torch.isfinite(what.grad).all()
    # The feature nets themselves carry no trainable parameters.
    for net in nets:
        assert all(p.requires_grad is False for p in net.parameters())
