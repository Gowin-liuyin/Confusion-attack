"""Regression tests for the IMA family (end-to-end attacker training).

The attacker trains only on a public reference corpus and recovers a held-out
obfuscated sample strictly better than random chance. A data-isolation
assertion verifies the evaluated plaintext sequence is absent from the training
corpus, so success reflects genuine per-token inversion, not sequence lookup.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability


def make_obfuscation(vocab: int, d: int, seed: int):
    """Deterministic, token-local obfuscation (reproducible training pairs)."""
    rng = np.random.default_rng(seed)
    E_clean = rng.standard_normal((vocab, d))
    A = rng.standard_normal((d, d))
    perm = rng.permutation(vocab)

    def fn(token_ids):
        ids = np.asarray(token_ids, dtype=int).ravel()
        return E_clean[perm[ids]] @ A

    return fn, E_clean, perm


def _as_tuples(seqs):
    return {tuple(int(x) for x in np.asarray(s).ravel()) for s in seqs}


def _build_ctx(V, N, obf, fn, corpus, reference_embeddings=None):
    return AttackContext(
        granted=frozenset(
            {Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}
        ),
        descriptor={"vocab_size": V, "seq_len": N},
        _obfuscated_embeddings=obf,
        obfuscation_fn=fn,
        reference_corpus=corpus,
        reference_embeddings=reference_embeddings,
    )


@pytest.mark.regression
@pytest.mark.neural
def test_classify_end_to_end_better_than_random():
    V, d, N = 10, 8, 6
    fn, E_clean, _perm = make_obfuscation(V, d, seed=0)

    rng = np.random.default_rng(123)
    corpus = [rng.integers(0, V, size=N) for _ in range(40)]
    corpus.append(np.arange(V))  # guarantee every vocabulary id is covered

    held = np.array([0, 2, 4, 6, 8, 9])
    # DATA ISOLATION: the evaluated plaintext sequence must not be in training.
    corpus = [s for s in corpus if tuple(int(x) for x in np.asarray(s)) != tuple(held.tolist())]
    assert tuple(held.tolist()) not in _as_tuples(corpus)

    obf = fn(held)
    ctx = _build_ctx(V, N, obf, fn, corpus)
    atk = ClassifyInverter(knowledge="GB", epochs=200, lr=2e-2, hidden_dims=(64,), seed=0)
    res = atk.run(ctx)

    assert res.status == AttackStatus.OK
    pred = res.recovered_token_ids
    assert pred.shape == held.shape
    acc = float((pred == held).mean())
    chance = 1.0 / V
    assert acc > 3 * chance  # strictly, clearly better than random
    assert acc >= 0.5
    assert res.diagnostics["knowledge"] == "GB"


@pytest.mark.regression
@pytest.mark.neural
def test_classify_wb_gb_bb_all_run_and_record_setting():
    V, d, N = 8, 6, 5
    fn, _E, _perm = make_obfuscation(V, d, seed=2)
    rng = np.random.default_rng(7)
    corpus = [rng.integers(0, V, size=N) for _ in range(20)]
    held = np.array([1, 3, 5, 7, 2])
    corpus = [s for s in corpus if tuple(int(x) for x in s) != tuple(held.tolist())]
    obf = fn(held)

    for knowledge in ("WB", "GB", "BB"):
        ctx = _build_ctx(V, N, obf, fn, corpus)
        atk = ClassifyInverter(knowledge=knowledge, epochs=120, lr=2e-2, hidden_dims=(32,), seed=0)
        res = atk.run(ctx)
        assert res.status == AttackStatus.OK
        assert res.diagnostics["knowledge"] == knowledge
        if knowledge == "BB":
            assert res.diagnostics["query_budget"] is not None
        if knowledge == "WB":
            assert res.diagnostics.get("wb_vocab_probed") == V


@pytest.mark.regression
@pytest.mark.neural
def test_qwen2_aloepri_ce_learns_better_than_random():
    """AloePri Qwen2 inverter (CE token-index path) learns a held-out mapping."""
    V, d, N = 8, 6, 5
    fn, E_clean, _perm = make_obfuscation(V, d, seed=3)
    rng = np.random.default_rng(11)
    corpus = [rng.integers(0, V, size=N) for _ in range(14)]
    corpus.append(np.arange(V))

    held = np.array([0, 7, 3, 5, 1])
    corpus = [s for s in corpus if tuple(int(x) for x in np.asarray(s)) != tuple(held.tolist())]
    assert tuple(held.tolist()) not in _as_tuples(corpus)

    obf = fn(held)
    ctx = _build_ctx(V, N, obf, fn, corpus)
    atk = Qwen2AloePriInverter(
        objective="ce", epochs=40, lr=3e-3, seed=0, hidden_size=32, intermediate_size=64
    )
    res = atk.run(ctx)

    assert res.status == AttackStatus.OK
    assert res.diagnostics["num_hidden_layers"] == 2
    assert res.diagnostics["num_attention_heads"] == 8
    acc = float((res.recovered_token_ids == held).mean())
    assert acc > 1.5 / V  # better than random chance


@pytest.mark.regression
@pytest.mark.neural
def test_qwen2_aloepri_optional_mse_path_runs():
    """The optional embedding-regression (MSE) path trains and recovers tokens."""
    V, d, N = 8, 6, 5
    fn, E_clean, _perm = make_obfuscation(V, d, seed=5)
    rng = np.random.default_rng(13)
    corpus = [rng.integers(0, V, size=N) for _ in range(14)]
    corpus.append(np.arange(V))

    held = np.array([2, 4, 6, 0, 7])
    corpus = [s for s in corpus if tuple(int(x) for x in np.asarray(s)) != tuple(held.tolist())]

    obf = fn(held)
    # MSE path regresses to the attacker's clean reference embeddings E_clean.
    ctx = _build_ctx(V, N, obf, fn, corpus, reference_embeddings=E_clean)
    atk = Qwen2AloePriInverter(
        objective="mse", epochs=40, lr=3e-3, seed=0, hidden_size=32, intermediate_size=64
    )
    res = atk.run(ctx)

    assert res.status == AttackStatus.OK
    assert res.diagnostics["objective"] == "mse"
    assert res.recovered_token_ids.shape == held.shape
    acc = float((res.recovered_token_ids == held).mean())
    assert acc > 1.5 / V  # better than random chance
