"""Mathematical tests for the IA family (FORMULA_LEDGER IA-EDNN/GATE/ATTN)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.ia.attn import whitening_invariant
from obfuscation_attacks.attacks.ia.ednn import difference_representation, lshift
from obfuscation_attacks.attacks.ia.gate import gate_invariant

pytestmark = pytest.mark.mathematical


# --- EDNN: coordinate-difference invariant of the glide reflection --------
def _glide(e, a, b):
    ones = np.ones_like(e)
    l = ones * a
    t = ones * b
    return e - 2.0 * (e @ l) / (l @ l) * l + t


def test_glide_preserves_coordinate_differences():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(8)
    ep = _glide(e, a=rng.standard_normal() + 1.0, b=rng.standard_normal())
    # e'[k1]-e'[k2] == e[k1]-e[k2] for all pairs
    for k1 in range(8):
        for k2 in range(8):
            assert (ep[k1] - ep[k2]) == pytest.approx(e[k1] - e[k2], abs=1e-9)


def test_lshift_is_cyclic_keeps_last_coordinate():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(lshift(x), np.array([2.0, 3.0, 4.0, 1.0]))  # wrap, not drop


def test_difference_representation_invariant_under_glide():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(10)
    ep = _glide(e, a=2.3, b=-0.7)
    assert np.allclose(difference_representation(ep), difference_representation(e), atol=1e-9)


def test_non_all_equal_normal_breaks_the_invariant():
    # The invariant is specific to l = 1*a (all-equal coordinates). A general
    # reflection normal must NOT be claimed to preserve coordinate differences.
    rng = np.random.default_rng(2)
    e = rng.standard_normal(8)
    normal = rng.standard_normal(8)  # arbitrary direction, not all-equal
    ep = e - 2.0 * (e @ normal) / (normal @ normal) * normal
    assert not np.allclose(difference_representation(ep), difference_representation(e), atol=1e-6)


# --- Gate-IA: gate-weighted average invariant -----------------------------
def test_gate_average_invariant_under_column_permutation():
    rng = np.random.default_rng(3)
    vocab, d, hf = 5, 4, 7
    E = rng.standard_normal((vocab, d))
    W = rng.standard_normal((d, hf))
    zffn = rng.permutation(hf)
    W_obf = W[:, zffn]  # FFN gate column permutation
    inv_plain = gate_invariant(E, W)
    inv_obf = gate_invariant(E, W_obf)
    # Avg over the FFN dimension is invariant to the column permutation
    assert np.allclose(inv_plain, inv_obf, atol=1e-9)


# --- Attn-IA: whitening quadratic-form invariant --------------------------
def test_whitening_invariant_under_invertible_block_transform():
    rng = np.random.default_rng(4)
    n, block, nb = 12, 2, 3
    Q = rng.standard_normal((n, block * nb))
    # block-diagonal invertible transform
    M = np.zeros((block * nb, block * nb))
    for j in range(nb):
        A = rng.standard_normal((block, block)) + 2 * np.eye(block)
        M[j * block : (j + 1) * block, j * block : (j + 1) * block] = A
    Qt = Q @ M
    inv_a, blocks_a = whitening_invariant(Q, block)
    inv_b, blocks_b = whitening_invariant(Qt, block)
    assert blocks_a == blocks_b
    assert np.allclose(inv_a, inv_b, atol=1e-6)  # Mahalanobis form is invariant


def test_whitening_skips_singular_block_without_pseudo_inverse():
    rng = np.random.default_rng(5)
    n, block, nb = 8, 2, 3
    Q = rng.standard_normal((n, block * nb))
    Q[:, 0:block] = 0.0  # make block 0's Gram singular
    inv, used_blocks = whitening_invariant(Q, block)
    assert 0 not in used_blocks  # singular block skipped, never pseudo-inverted
    assert inv.shape[1] == len(used_blocks) == nb - 1
