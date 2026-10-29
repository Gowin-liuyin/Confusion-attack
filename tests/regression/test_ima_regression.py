"""Regression tests for the IMA family (end-to-end attacker training).

These tests use the shared keyed mechanism (``schemes/mechanism.py``) to enforce
the faithful threat model: the attacker knows the obfuscation *algorithm* and
keys it itself to synthesise training pairs, but the *evaluated* sample is
obfuscated with a held-out client key.

* The faithful **unknown-key** attack (fresh attacker key per query) recovers at
  roughly chance - matching AloePri Table 2, which reports IMA TTRSR = 0.0
  against AloePri (paper line 806).
* A separate **known-key** run (the attacker is handed the client key - a
  white-box sanity setting) shows the very same inverter architecture *does*
  learn, proving the near-chance result is the key's doing, not a weak model.

A data-isolation assertion verifies the evaluated (sequence, key) pair is absent
from the attacker's training: the evaluated plaintext sequence is not in the
corpus, and no attacker key reproduces the evaluated embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter
from obfuscation_attacks.attacks.ima.seq2seq import Seq2SeqInverter
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability
from obfuscation_attacks.metrics import ttrsr
from obfuscation_attacks.schemes.mechanism import AloePriToyMechanism, attacker_oracle


def _build_ctx(V, N, obf, fn, corpus, reference_embeddings=None):
    return AttackContext(
        granted=frozenset({Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}),
        descriptor={"vocab_size": V, "seq_len": N},
        _obfuscated_embeddings=obf,
        obfuscation_fn=fn,
        reference_corpus=corpus,
        reference_embeddings=reference_embeddings,
    )


def _mechanism(V, d, seed):
    """Keyed AloePri-style toy mechanism (public E, A; secret permutation key)."""
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((V, d))
    A = rng.standard_normal((d, d))
    mech = AloePriToyMechanism(E, alpha_e=0.0, A=A, seed=seed)
    return mech, E


def _tup(s):
    return tuple(int(x) for x in np.asarray(s).ravel())


def _as_tuples(seqs):
    return {_tup(s) for s in seqs}


# ---------------------------------------------------------------------------
# Defect 1: faithful unknown-key recovery is near chance (key never leaks)
# ---------------------------------------------------------------------------
@pytest.mark.regression
@pytest.mark.neural
def test_classify_unknown_key_recovery_is_near_chance():
    """Fresh attacker key per query -> recovery ~ chance (AloePri Table 2 IMA TTRSR=0.0)."""
    V, d, N = 12, 8, 24
    mech, _E = _mechanism(V, d, seed=0)
    client_key = mech.sample_key(np.random.default_rng(777))  # held-out client key

    # The faithful oracle samples a FRESH key on every call, so training pairs
    # never use the client key.
    fn = attacker_oracle(mech, attacker_known_key=False, seed=0)

    rng = np.random.default_rng(123)
    corpus = [rng.integers(0, V, size=10) for _ in range(60)]
    corpus.append(np.arange(V))  # full-vocab coverage (still signal-free under fresh keys)
    held = np.resize(np.arange(10), N)  # length-N content sequence
    corpus = [s for s in corpus if _tup(s) != _tup(held)]

    # The EVALUATED embeddings use the held-out client key.
    eval_obf = mech.obfuscate_embeddings(held, client_key)

    # DATA ISOLATION. (a) the evaluated sequence is not in the training corpus;
    # (b) no attacker key reproduces the evaluated embeddings -> the (sequence,
    # key) pair is genuinely held out. Use an independent attacker oracle so the
    # attack's own oracle stream is not disturbed.
    assert _tup(held) not in _as_tuples(corpus)
    fn_other_attacker = attacker_oracle(mech, attacker_known_key=False, seed=999)
    assert not np.allclose(fn_other_attacker(held), eval_obf)

    ctx = _build_ctx(V, N, eval_obf, fn, corpus)
    res = ClassifyInverter(knowledge="GB", epochs=150, lr=2e-2, hidden_dims=(64,), seed=0).run(ctx)

    assert res.status == AttackStatus.OK
    assert res.recovered_token_ids.shape == held.shape
    acc = float((res.recovered_token_ids == held).mean())
    chance = 1.0 / V
    # Near chance (measured 0.0): clearly far below the known-key capacity (~1.0,
    # see test_classify_known_key_recovery_proves_capacity). Matches AloePri
    # Table 2, which reports IMA TTRSR = 0.0 against AloePri (paper line 806).
    assert acc <= 3 * chance


@pytest.mark.regression
@pytest.mark.neural
def test_classify_known_key_recovery_proves_capacity():
    """attacker_known_key=True: the same architecture clearly learns (capacity proof)."""
    V, d, N = 12, 8, 10
    mech, _E = _mechanism(V, d, seed=1)
    client_key = mech.sample_key(np.random.default_rng(888))

    # White-box sanity oracle: training pairs use the client key.
    fn = attacker_oracle(mech, client_key=client_key, attacker_known_key=True, seed=1)

    rng = np.random.default_rng(321)
    corpus = [rng.integers(0, V, size=N) for _ in range(40)]
    corpus.append(np.arange(V))  # full-vocab coverage under the client key
    held = np.arange(10)
    corpus = [s for s in corpus if _tup(s) != _tup(held)]
    assert _tup(held) not in _as_tuples(corpus)

    eval_obf = mech.obfuscate_embeddings(held, client_key)
    ctx = _build_ctx(V, N, eval_obf, fn, corpus)
    res = ClassifyInverter(knowledge="GB", epochs=150, lr=2e-2, hidden_dims=(64,), seed=0).run(ctx)

    assert res.status == AttackStatus.OK
    acc = float((res.recovered_token_ids == held).mean())
    assert acc >= 0.8  # clearly above chance (1/12); the inverter has capacity


# ---------------------------------------------------------------------------
# Defect 3: Classify is the GB attack -> WB/GB run; BB is not applicable
# ---------------------------------------------------------------------------
@pytest.mark.regression
@pytest.mark.neural
def test_classify_wb_gb_run_and_bb_not_applicable():
    V, d, N = 8, 6, 5
    mech, _E = _mechanism(V, d, seed=2)
    client_key = mech.sample_key(np.random.default_rng(55))
    fn = attacker_oracle(mech, client_key=client_key, attacker_known_key=True, seed=2)

    rng = np.random.default_rng(7)
    corpus = [rng.integers(0, V, size=N) for _ in range(20)]
    corpus.append(np.arange(V))
    held = np.array([1, 3, 5, 7, 2])
    corpus = [s for s in corpus if _tup(s) != _tup(held)]
    eval_obf = mech.obfuscate_embeddings(held, client_key)

    for knowledge in ("WB", "GB"):
        ctx = _build_ctx(V, N, eval_obf, fn, corpus)
        res = ClassifyInverter(
            knowledge=knowledge, epochs=120, lr=2e-2, hidden_dims=(32,), seed=0
        ).run(ctx)
        assert res.status == AttackStatus.OK
        assert res.diagnostics["knowledge"] == knowledge
        assert res.diagnostics["query_budget"] is None  # no invented BB query cap
        if knowledge == "WB":
            assert res.diagnostics.get("wb_vocab_probed") == V

    # BB belongs to InvBERT Seq2Seq, not Classify -> honest not_applicable.
    ctx = _build_ctx(V, N, eval_obf, fn, corpus)
    res_bb = ClassifyInverter(knowledge="BB", epochs=10, seed=0).run(ctx)
    assert res_bb.status == AttackStatus.NOT_APPLICABLE
    assert "seq2seq" in res_bb.status_reason.lower()


# ---------------------------------------------------------------------------
# Defect 6 + 4: Qwen2 trained vs untrained on the SAME known-key data
# ---------------------------------------------------------------------------
def _qwen2_known_key_scenario(seed_mech, seed_key, seed_corpus):
    V, d, N = 8, 6, 5
    mech, E = _mechanism(V, d, seed=seed_mech)
    client_key = mech.sample_key(np.random.default_rng(seed_key))
    fn = attacker_oracle(mech, client_key=client_key, attacker_known_key=True, seed=seed_mech)
    rng = np.random.default_rng(seed_corpus)
    corpus = [rng.integers(0, V, size=N) for _ in range(16)]
    corpus.append(np.arange(V))
    held = np.array([0, 7, 3, 5, 1])
    corpus = [s for s in corpus if _tup(s) != _tup(held)]
    eval_obf = mech.obfuscate_embeddings(held, client_key)
    return V, d, N, E, fn, corpus, held, eval_obf


@pytest.mark.regression
@pytest.mark.neural
def test_qwen2_embedding_default_trained_beats_untrained():
    """Paper-experiment embedding path (default): trained clearly beats 0-epoch baseline."""
    V, d, N, E, fn, corpus, held, eval_obf = _qwen2_known_key_scenario(3, 99, 11)

    common = dict(lr=3e-3, seed=0, hidden_size=32, intermediate_size=64)
    trained = Qwen2AloePriInverter(epochs=60, **common)  # default objective = embedding
    assert trained.objective == "embedding"
    res_t = trained.run(_build_ctx(V, N, eval_obf, fn, corpus, reference_embeddings=E))
    res_u = Qwen2AloePriInverter(epochs=0, **common).run(
        _build_ctx(V, N, eval_obf, fn, corpus, reference_embeddings=E)
    )

    assert res_t.status == AttackStatus.OK and res_u.status == AttackStatus.OK
    assert res_t.diagnostics["objective"] == "embedding"
    assert "mse" in res_t.diagnostics["loss"].lower()  # implementation-choice loss, labelled
    acc_t = float((res_t.recovered_token_ids == held).mean())
    acc_u = float((res_u.recovered_token_ids == held).mean())
    assert acc_t > acc_u  # training beats the untrained baseline on the SAME data
    assert acc_t >= 0.6


@pytest.mark.regression
@pytest.mark.neural
def test_qwen2_token_index_alternative_trained_beats_untrained():
    """Labelled CE alternative (token indices): trained clearly beats 0-epoch baseline."""
    V, d, N, E, fn, corpus, held, eval_obf = _qwen2_known_key_scenario(4, 100, 12)

    common = dict(objective="token_index", lr=3e-3, seed=0, hidden_size=32, intermediate_size=64)
    res_t = Qwen2AloePriInverter(epochs=60, **common).run(_build_ctx(V, N, eval_obf, fn, corpus))
    res_u = Qwen2AloePriInverter(epochs=0, **common).run(_build_ctx(V, N, eval_obf, fn, corpus))

    assert res_t.status == AttackStatus.OK and res_u.status == AttackStatus.OK
    assert res_t.diagnostics["objective"] == "token_index"
    assert res_t.diagnostics["loss"] == "cross_entropy"
    acc_t = float((res_t.recovered_token_ids == held).mean())
    acc_u = float((res_u.recovered_token_ids == held).mean())
    assert acc_t > acc_u
    assert acc_t >= 0.6


# ---------------------------------------------------------------------------
# Defect 2: Seq2Seq end-to-end - recovers an ORDERED sequence (known key)
# ---------------------------------------------------------------------------
@pytest.mark.regression
@pytest.mark.neural
def test_seq2seq_end_to_end_known_key_recovers_ordered_sequence():
    V, d = 6, 8
    mech, _E = _mechanism(V, d, seed=4)
    client_key = mech.sample_key(np.random.default_rng(202))
    fn = attacker_oracle(mech, client_key=client_key, attacker_known_key=True, seed=4)

    rng = np.random.default_rng(31)
    corpus = [rng.integers(0, V, size=5) for _ in range(24)]
    corpus += [np.roll(np.arange(5) % V, k) for k in range(5)]  # positional coverage
    held = np.array([1, 3, 0, 4, 2])
    corpus = [s for s in corpus if _tup(s) != _tup(held)]
    eval_obf = mech.obfuscate_embeddings(held, client_key)

    common = dict(seed=0, d_model=64, nhead=4, num_layers=2, lr=2e-3)
    res_t = Seq2SeqInverter(epochs=90, **common).run(_build_ctx(V, held.size, eval_obf, fn, corpus))
    res_u = Seq2SeqInverter(epochs=0, **common).run(_build_ctx(V, held.size, eval_obf, fn, corpus))

    assert res_t.status == AttackStatus.OK
    assert res_t.diagnostics["decode"] == "greedy_until_eos"
    rec = np.asarray(res_t.recovered_token_ids)
    assert rec.size == 0 or (rec.min() >= 0 and rec.max() < V)  # content ids only

    t_trained = ttrsr(rec.tolist(), held.tolist())
    t_untrained = ttrsr(np.asarray(res_u.recovered_token_ids).tolist(), held.tolist())
    assert t_trained > t_untrained  # ordering learned, not an artefact of init
    assert t_trained >= 0.6  # recovers a majority of positions IN ORDER
