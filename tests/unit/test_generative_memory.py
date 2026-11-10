"""Acceptance tests for F14: generative attacks use chunked exact Gaussian scoring."""

from __future__ import annotations

import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from obfuscation_attacks.attacks.beamclean.core import (
    _l1_distances,
    _sq_distances,
    nn_squared_distances,
    noise_loglik,
)
from obfuscation_attacks.evaluation.real_generative import beamclean_recover
from obfuscation_attacks.evaluation.real_neural import match_hidden_to_database
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D = 50_000, 512


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(V, D)).astype(np.float32)
    public = PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E,
        forward_fn=lambda embeds: SimpleNamespace(
            logits=np.zeros((1, np.asarray(embeds).shape[0], V), dtype=np.float32)
        ),
        allow_clean_weights=True,
    )
    observations = np.stack([E[3], E[7]])
    return public, observations, E


def test_gaussian_loglik_matches_the_closed_form():
    public, observations, E = _fixture(seed=1)
    sigma = 2.0
    out = beamclean_recover(
        public, observations, arm="ab", beam=2, n_candidates=4,
        max_len=2, sigma=sigma, chunk=10_000, verbose=False,
    )
    ids = np.asarray(out["candidate_ids"][0])
    loglik = out["candidate_loglik"][0]
    token = int(np.where(ids == 3)[0][0])
    expected = -0.5 * float(np.sum((observations[0] - E[3]) ** 2)) / (sigma ** 2)
    assert loglik[token] == pytest.approx(expected, rel=1e-5)


def test_top_candidates_include_the_true_tokens():
    public, observations, E = _fixture(seed=2)
    out = beamclean_recover(
        public, observations, arm="ab", beam=2, n_candidates=6,
        max_len=2, sigma=1.0, chunk=10_000, verbose=False,
    )
    assert 3 in out["candidate_ids"][0]
    assert 7 in out["candidate_ids"][1]
    assert out["n_chunks"] >= 2


def test_peak_allocation_stays_below_the_naive_n_v_d_tensor():
    public, observations, E = _fixture(seed=3)
    tracemalloc.start()
    beamclean_recover(
        public, observations, arm="ab", beam=2, n_candidates=4,
        max_len=2, sigma=1.0, chunk=10_000, verbose=False,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # A naive [positions, V, D] float32 tensor is 2 * 50k * 512 * 4 = 204,800,000
    # bytes (195.3 MiB). The guard must sit below that exact allocation, not at a
    # round 200 MiB that the forbidden tensor itself would still pass.
    assert peak < 200_000_000, f"peak {peak / 1024 / 1024:.1f} MiB"


def test_beamclean_core_distances_and_loglik_match_the_naive_definition():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(3, 8)).astype(np.float32)
    E = rng.normal(size=(17, 8)).astype(np.float32)
    naive_sq = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(axis=-1)
    naive_l1 = np.abs(Z[:, None, :] - E[None, :, :]).sum(axis=-1)
    assert np.allclose(_sq_distances(Z, E), naive_sq, atol=1e-5)
    assert np.allclose(_l1_distances(Z, E), naive_l1, atol=1e-5)
    assert np.allclose(nn_squared_distances(Z, E), naive_sq.min(axis=1), atol=1e-5)
    gaussian = noise_loglik(Z, E, 2.0, "gaussian")
    assert np.allclose(gaussian, -0.5 * naive_sq / 4.0, atol=1e-5)
    laplace = noise_loglik(Z, E, 2.0, "laplacian")
    assert np.allclose(laplace, -naive_l1 / 2.0, atol=1e-5)


def test_match_hidden_to_database_chunked_matches_the_naive_l1_search():
    rng = np.random.default_rng(5)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    database = rng.normal(size=(37, 8)).astype(np.float32)
    candidates = np.arange(len(database), dtype=np.int64)
    out = match_hidden_to_database(queries, database, candidates, k=4, chunk=2)
    distances = np.abs(queries[:, None, :] - database[None, :, :]).sum(axis=2)
    expected = np.argsort(distances, axis=1, kind="stable")[:, :4]
    assert np.array_equal(out, expected)


def test_hidden_database_small_blocks_ties_and_small_vocabulary():
    queries = np.array([[0., 0.], [1., 0.]])
    database = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., 1.], [1., 1.]])
    ids = np.array([9, 3, 5, 7, 1])
    got = match_hidden_to_database(queries, database, ids, k=100, chunk=1, database_chunk=2)
    distances = np.abs(queries[:, None, :] - database[None, :, :]).sum(axis=2)
    expected = np.array([ids[np.lexsort((ids, row))] for row in distances])
    assert np.array_equal(got, expected)
