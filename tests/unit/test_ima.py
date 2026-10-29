"""Unit tests for the IMA family.

(a) small-data overfit sanity for the Classify per-position MLP;
(b) teacher-forcing behaviour of the Seq2Seq decoder (shapes + causal
    conditioning on y_<t).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.ima.seq2seq import Seq2SeqDecoderInverter
from obfuscation_attacks.utils import seed_everything


def make_obfuscation(vocab: int, d: int, seed: int):
    """Deterministic, token-local obfuscation: plaintext id -> permuted clean
    embedding -> fixed linear map. No per-call randomness, so training pairs are
    reproducible and each id maps to a single point in feature space."""
    rng = np.random.default_rng(seed)
    E_clean = rng.standard_normal((vocab, d))
    A = rng.standard_normal((d, d))
    perm = rng.permutation(vocab)

    def fn(token_ids):
        ids = np.asarray(token_ids, dtype=int).ravel()
        return E_clean[perm[ids]] @ A

    return fn, E_clean, perm


@pytest.mark.unit
@pytest.mark.neural
def test_classify_overfit_small_data():
    """Train the Classify MLP on a handful of (obf, token) pairs -> ~100% train."""
    fn, _E, _perm = make_obfuscation(vocab=4, d=6, seed=1)
    ids = np.array([0, 1, 2, 3, 1, 2], dtype=int)  # a handful of pairs
    X = fn(ids)
    y = ids
    model, history = ClassifyInverter.train_on_pairs(
        X, y, vocab_size=4, hidden_dims=(32,), epochs=300, lr=5e-2, seed=0
    )
    acc = ClassifyInverter.train_accuracy(model, X, y)
    assert acc >= 0.99  # near-100% TRAIN accuracy
    assert history[-1] < history[0]  # loss actually decreased


@pytest.mark.unit
@pytest.mark.neural
def test_seq2seq_teacher_forcing_shapes_and_conditioning():
    """Teacher forcing: shifted-BOS input, (B,N,V) logits, causal y_<t conditioning."""
    seed_everything(0)
    V, d, N = 6, 5, 4
    model = Seq2SeqDecoderInverter(
        obf_dim=d, vocab_size=V, d_model=32, nhead=4, num_layers=2, max_len=16, seed=0
    )
    model.eval()

    rng = np.random.default_rng(0)
    memory = torch.as_tensor(rng.standard_normal((1, N, d)), dtype=torch.float32)
    y = torch.tensor([[1, 3, 2, 0]], dtype=torch.long)  # gold targets y_0..y_{N-1}

    # teacher-forcing input is the right-shifted gold prefix with BOS prepended
    tgt_in = model.teacher_forcing_inputs(y)
    assert tuple(tgt_in.shape) == (1, N)
    assert int(tgt_in[0, 0]) == model.bos_id
    assert torch.equal(tgt_in[0, 1:], y[0, :-1])

    logits = model(memory, tgt_in)
    assert tuple(logits.shape) == (1, N, V)  # per-step vocabulary logits

    # Conditions on y_<t: change the token at input position 1 (= y_0). Output
    # position 0 attends only to [BOS] and must be unchanged; output position 1
    # attends to [BOS, y_0] and must change.
    tgt_a = tgt_in.clone()
    tgt_a[0, 1] = (int(tgt_in[0, 1]) + 1) % V
    logits_a = model(memory, tgt_a)
    assert (logits[0, 0] - logits_a[0, 0]).abs().max() < 1e-6
    assert (logits[0, 1] - logits_a[0, 1]).abs().max() > 1e-4

    # Causal masking the other way: changing the LAST input token affects only
    # the last output position, never earlier ones.
    tgt_b = tgt_in.clone()
    tgt_b[0, N - 1] = (int(tgt_in[0, N - 1]) + 1) % V
    logits_b = model(memory, tgt_b)
    assert (logits[0, : N - 1] - logits_b[0, : N - 1]).abs().max() < 1e-6
    assert (logits[0, N - 1] - logits_b[0, N - 1]).abs().max() > 1e-4


@pytest.mark.unit
@pytest.mark.neural
def test_seq2seq_greedy_decode_length_and_range():
    """Greedy decode yields exactly N in-vocabulary token ids (no BOS leakage)."""
    from obfuscation_attacks.attacks.ima.seq2seq import greedy_decode

    seed_everything(0)
    V, d, N = 6, 5, 4
    model = Seq2SeqDecoderInverter(obf_dim=d, vocab_size=V, d_model=32, max_len=16, seed=0)
    rng = np.random.default_rng(1)
    memory = rng.standard_normal((N, d))
    out = greedy_decode(model, memory, N)
    assert out.shape == (N,)
    assert out.min() >= 0 and out.max() < V  # BOS id (== V) never emitted
