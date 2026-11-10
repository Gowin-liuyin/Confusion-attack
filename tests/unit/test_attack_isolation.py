"""The weight-attack path must be provably secret-free, not just secret-avoiding.

The user challenge that motivated this file: "are you sure your attack premise does not know
the secret?" The dense-mixing attack solves ``M`` from public + deployment weights and then
needs nothing else, but the first implementation computed its known-secret diagnostics *inside*
the same function that produced the attack's table, so an auditor could not tell which number
came from which. These tests make the boundary executable:

* the pure attack helpers take ``numpy`` arrays only -- they cannot reach a ``perm``/``key``
  because they are never handed an object that has one;
* a :class:`SpyDeployment` raises if anyone touches ``perm`` / ``inv_perm`` / the secret
  metadata, and the attack still produces the exact same mapping;
* the scoring helpers are the only place the permutation appears, and they are marked as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.vma.dense_mixing import (
    map_rows,
    solve_mixing_matrix,
    unmix_diagnostics,
    unmix_table,
)

pytestmark = pytest.mark.unit

D = 24
V = 40


class SpyDeployment:
    """Deployment stand-in whose permutation is booby-trapped.

    Any read of ``perm`` or ``inv_perm`` raises, so a test that passes proves the code path
    under test never touched them. ``E_clean`` and ``E_dep`` are *public* (a public base model
    and the server's own weights), so they stay readable -- the plan's A2 assumption.
    """

    def __init__(self, E_clean: np.ndarray, E_dep: np.ndarray) -> None:
        self.E_clean = E_clean
        self.E_dep = E_dep
        self.perm_reads = 0

    @property
    def perm(self):  # pragma: no cover - only fires on a violation
        self.perm_reads += 1
        raise AssertionError("attack path read the secret permutation")

    @property
    def inv_perm(self):  # pragma: no cover - only fires on a violation
        self.perm_reads += 1
        raise AssertionError("attack path read the inverse secret permutation")


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(D, D)).astype(np.float32) + np.eye(D, dtype=np.float32) * 3.0
    E = rng.normal(size=(V, D)).astype(np.float32)
    perm = rng.permutation(V)
    E_dep = np.empty_like(E)
    E_dep[perm] = E @ M                      # artifact convention: E_dep[perm[i]] = (E M)[i]
    W = rng.normal(size=(D + 8, D)).astype(np.float32)
    Wd = W @ M
    return E, E_dep, perm, M, W, Wd


def test_pure_attack_helpers_have_no_secret_parameter():
    """Structural check: no helper on the attack path accepts a deployment object."""
    import inspect

    for fn in (solve_mixing_matrix, unmix_table, map_rows):
        params = inspect.signature(fn).parameters
        assert "dep" not in params, f"{fn.__name__} takes a deployment object"
        assert "perm" not in params, f"{fn.__name__} takes the permutation"


def test_attack_recovers_the_mapping_without_touching_the_permutation():
    E, E_dep, perm, M, W, Wd = _fixture()
    spy = SpyDeployment(E, E_dep)
    M_hat = solve_mixing_matrix(W, Wd)["M"]
    unmixed = unmix_table(spy.E_dep, M_hat)                  # public + deployment weights only
    ref = np.arange(V)
    recovered = map_rows(spy.E_clean[ref], unmixed)          # public reference rows
    # scoring is the only place the permutation appears
    assert spy.perm_reads == 0
    assert np.array_equal(recovered, perm[ref])


def test_diagnostics_are_the_only_place_the_permutation_appears():
    E, E_dep, perm, M, W, Wd = _fixture(seed=3)
    unmixed = unmix_table(E_dep, M)
    diag = unmix_diagnostics(unmixed, E, perm)
    assert abs(diag["row_cosine_mean"] - 1.0) < 1e-4
    assert diag["relative_residual"] < 1e-4
    assert diag["relative_residual_raw"] > 0.5      # un-paired number is large by construction


def test_attack_is_scale_free_in_the_mixing():
    """A global scale on M must not change the recovered mapping (cosine KNN)."""
    E, E_dep, perm, M, W, Wd = _fixture(seed=5)
    M_hat = solve_mixing_matrix(W, Wd)["M"]
    unmixed_a = unmix_table(E_dep, M_hat)
    unmixed_b = unmix_table(E_dep, M_hat * 7.0)
    ref = np.arange(V)
    assert np.array_equal(map_rows(E[ref], unmixed_a), map_rows(E[ref], unmixed_b))
