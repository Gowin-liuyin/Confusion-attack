"""The dense-mixing inversion must be exact on a linear model, and diagnosable otherwise.

``dense_mixing`` is the variant that measures the packaged artifact's weight surface, so its
maths has to be pinned independently of the artifact: if ``W' = W M`` and ``E' = E M`` with no
noise, solving ``M`` and inverting it must return ``E`` exactly (row order aside), and the token
nearest-neighbour must be perfect. A second case adds the artifact's ``alpha_h`` weight noise and
checks that the residual grows while the row directions survive -- the regime the real run
reports.

The API is split on purpose: ``unmix_table`` / ``map_rows`` are the *attack* (arrays only, no
secret reachable), ``unmix_diagnostics`` is the *scoring* side (takes the permutation explicitly).
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.vma.dense_mixing import (
    forward_match_stats,
    map_rows,
    recover_mixing,
    solve_mixing_matrix,
    unmix_diagnostics,
    unmix_table,
)

pytestmark = pytest.mark.unit


class _FakeDep:
    """Minimal stand-in for ``ABDeployment``: two weight dicts plus an embedding pair."""

    def __init__(self, clean_weights: dict, deployment_weights: dict,
                 E_clean: np.ndarray, E_dep: np.ndarray, perm: np.ndarray, n_layers: int) -> None:
        self._w = {"clean": clean_weights, "deployment": deployment_weights}
        self.E_clean = E_clean
        self.E_dep = E_dep
        self.perm = perm
        self.n_layers = n_layers

    def weights(self, side: str) -> dict:
        return self._w[side]


def _make(n_vocab: int = 64, d: int = 32, q_dim: int = 48, layers: int = 2, noise: float = 0.0,
          seed: int = 0):
    """``W' = W M`` with W [q_dim, d] (over-determined, as in a real model's k/v projections).

    The embedding follows the artifact's convention (``ab_runtime.py``: ``mapped = perm[ids]``
    and the server looks up ``E_dep[mapped]``), i.e. ``E_dep[perm[i]] = (E @ M)[i]``. Getting
    this direction wrong makes an otherwise correct inversion look broken, so it is pinned here
    rather than assumed.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(d, d)).astype(np.float32) + np.eye(d, dtype=np.float32) * 3.0
    E = rng.normal(size=(n_vocab, d)).astype(np.float32)
    perm = rng.permutation(n_vocab)
    E_dep = np.empty_like(E)
    E_dep[perm] = E @ M
    if noise:
        E_dep = E_dep + noise * rng.normal(size=E_dep.shape).astype(np.float32)
    clean, dep = {}, {}
    for layer in range(layers):
        W = rng.normal(size=(q_dim, d)).astype(np.float32)
        clean[f"model.layers.{layer}.self_attn.q_proj.weight"] = W
        Wd = W @ M
        if noise:
            Wd = Wd + noise * rng.normal(size=Wd.shape).astype(np.float32)
        dep[f"model.layers.{layer}.self_attn.q_proj.weight"] = Wd
    return _FakeDep(clean, dep, E, E_dep, perm, layers), M


def test_solve_mixing_is_exact_without_noise():
    dep, M = _make()
    solved = solve_mixing_matrix(
        dep.weights("clean")["model.layers.0.self_attn.q_proj.weight"],
        dep.weights("deployment")["model.layers.0.self_attn.q_proj.weight"],
    )
    assert solved["relative_residual"] < 1e-5
    assert np.allclose(solved["M"], M, atol=1e-3)


def test_unmix_restores_the_embedding_table_and_the_token_order():
    dep, _ = _make()
    mix = recover_mixing(dep, layers=(0, 1), groups=("self_attn.q_proj",))
    assert mix["residual_max"] < 1e-5
    unmixed = unmix_table(dep.E_dep, mix["M"])
    diag = unmix_diagnostics(unmixed, dep.E_clean, dep.perm)
    assert diag["relative_residual"] < 1e-4
    assert diag["row_cosine_mean"] > 0.999
    ref = np.arange(len(dep.perm))
    assert np.array_equal(map_rows(dep.E_clean[ref], unmixed), dep.perm[ref])


def test_noise_degrades_residual_but_keeps_row_directions():
    dep, _ = _make(noise=0.01, seed=7)
    mix = recover_mixing(dep, layers=(0, 1), groups=("self_attn.q_proj",))
    assert 1e-4 < mix["residual_max"] < 0.2
    unmixed = unmix_table(dep.E_dep, mix["M"])
    diag = unmix_diagnostics(unmixed, dep.E_clean, dep.perm)
    assert diag["row_cosine_mean"] > 0.95


def test_forward_match_confidence_distinguishes_alignment():
    rng = np.random.default_rng(9)
    E = rng.standard_normal((64, 8)).astype(np.float32)
    M = rng.standard_normal((8, 8)).astype(np.float32)
    aligned = E @ M
    ids, cosine, margin = forward_match_stats(aligned, aligned)
    assert np.array_equal(ids, np.arange(len(E)))
    assert float(np.mean(cosine)) > 0.999
    assert float(np.mean(margin)) > 1e-3
    unrelated = rng.standard_normal((64, 8)).astype(np.float32)
    _, cosine_bad, margin_bad = forward_match_stats(unrelated, aligned)
    assert float(np.mean(cosine_bad)) < 0.9
    assert float(np.mean(cosine_bad)) < float(np.mean(cosine))


def test_median_vote_survives_one_bad_matrix():
    """With three proposals (two good, one corrupted) the per-entry median keeps the geometry."""
    dep, _ = _make(layers=3)
    bad = dict(dep.weights("deployment"))
    key = "model.layers.2.self_attn.q_proj.weight"
    rng = np.random.default_rng(3)
    bad[key] = rng.normal(size=bad[key].shape).astype(np.float32)
    dep._w["deployment"] = bad
    mix = recover_mixing(dep, layers=(0, 1, 2), groups=("self_attn.q_proj",))
    assert mix["residual_max"] > 0.5           # the corrupted matrix is detected
    assert mix["proposal_spread_mean"] > 1e-2  # and the disagreement is visible
    unmixed = unmix_table(dep.E_dep, mix["M"])
    diag = unmix_diagnostics(unmixed, dep.E_clean, dep.perm)
    assert diag["row_cosine_mean"] > 0.8       # yet the vote still recovers the geometry
