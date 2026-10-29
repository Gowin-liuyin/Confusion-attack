"""Unit tests for the IMA family.

(a) small-data overfit sanity for the Classify per-position MLP;
(b) teacher-forcing behaviour of the Seq2Seq decoder (EOS-terminated targets,
    shapes + causal conditioning on y_<t);
(c) the Seq2Seq memory carries position (permuting memory rows changes the
    output) and decoding length is produced by the model, not the input;
(d) the Qwen2 inverter defaults to the paper-experiment embedding path.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.ima.seq2seq import Seq2SeqDecoderInverter, greedy_decode
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
    """Teacher forcing: [BOS, y] input, [y, EOS] target, (B,N+1,V+1) logits, causal y_<t."""
    seed_everything(0)
    V, d, N = 6, 5, 4
    model = Seq2SeqDecoderInverter(
        obf_dim=d, vocab_size=V, d_model=32, nhead=4, num_layers=2, max_len=16, max_mem_len=16, seed=0
    )
    model.eval()

    rng = np.random.default_rng(0)
    memory = torch.as_tensor(rng.standard_normal((1, N, d)), dtype=torch.float32)
    y = torch.tensor([[1, 3, 2, 0]], dtype=torch.long)  # gold targets y_0..y_{N-1}

    # Decoder input is BOS followed by the full gold sequence (shifted right).
    tgt_in = model.teacher_forcing_inputs(y)
    assert tuple(tgt_in.shape) == (1, N + 1)
    assert int(tgt_in[0, 0]) == model.bos_id
    assert torch.equal(tgt_in[0, 1:], y[0])

    # Gold output is the sequence followed by EOS (teaches termination).
    tgt_out = model.teacher_forcing_targets(y)
    assert tuple(tgt_out.shape) == (1, N + 1)
    assert torch.equal(tgt_out[0, :N], y[0])
    assert int(tgt_out[0, N]) == model.eos_id

    logits = model(memory, tgt_in)
    assert tuple(logits.shape) == (1, N + 1, V + 1)  # content + EOS logits per step

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
    last = tgt_in.shape[1] - 1
    tgt_b = tgt_in.clone()
    tgt_b[0, last] = (int(tgt_in[0, last]) + 1) % V
    logits_b = model(memory, tgt_b)
    assert (logits[0, :last] - logits_b[0, :last]).abs().max() < 1e-6
    assert (logits[0, last] - logits_b[0, last]).abs().max() > 1e-4


@pytest.mark.unit
@pytest.mark.neural
def test_seq2seq_memory_carries_position_signal():
    """Permuting the memory ROWS changes the logits: the memory is order-aware.

    This is the fix for the defect where ``mem = mem_proj(memory)`` made
    cross-attention a set operation (permuting memory barely moved the logits),
    so the decoder could not order its predictions.
    """
    seed_everything(0)
    V, d, M = 6, 5, 4
    model = Seq2SeqDecoderInverter(
        obf_dim=d, vocab_size=V, d_model=32, nhead=4, num_layers=2, max_len=16, max_mem_len=16, seed=0
    ).eval()
    rng = np.random.default_rng(1)
    memory = torch.as_tensor(rng.standard_normal((1, M, d)), dtype=torch.float32)
    tgt_in = torch.tensor([[model.bos_id, 1, 2]], dtype=torch.long)

    logits = model(memory, tgt_in)
    perm = [M - 1 - i for i in range(M)]  # reverse the memory rows (non-trivial)
    logits_perm = model(memory[:, perm, :], tgt_in)
    assert (logits - logits_perm).abs().max() > 1e-3  # order matters now


@pytest.mark.unit
@pytest.mark.neural
def test_seq2seq_greedy_decode_variable_length_and_range():
    """Greedy decode: content-only ids, bounded by the cap not the memory length."""
    seed_everything(0)
    V, d = 6, 5
    model = Seq2SeqDecoderInverter(
        obf_dim=d, vocab_size=V, d_model=32, max_len=16, max_mem_len=16, seed=0
    )
    rng = np.random.default_rng(1)

    out = greedy_decode(model, rng.standard_normal((4, d)), max_len=4)
    assert out.ndim == 1
    assert out.size <= 4  # never longer than the cap
    if out.size:
        assert out.min() >= 0 and out.max() < V  # BOS/EOS (>= V) never emitted

    # Output length is produced by the model (EOS), not read from the input: a
    # length-7 memory with cap 5 yields at most 5 tokens, i.e. fewer than the
    # memory length.
    out_long = greedy_decode(model, rng.standard_normal((7, d)), max_len=5)
    assert out_long.size <= 5


@pytest.mark.unit
@pytest.mark.neural
def test_qwen2_default_objective_is_embedding_regression():
    """Default objective is the paper-experiment embedding path; CE is the alt."""
    from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter

    assert Qwen2AloePriInverter().objective == "embedding"
    assert Qwen2AloePriInverter(objective="embedding").objective == "embedding"
    assert Qwen2AloePriInverter(objective="token_index").objective == "token_index"
    # legacy aliases keep working and map to the right canonical path
    assert Qwen2AloePriInverter(objective="mse").objective == "embedding"
    assert Qwen2AloePriInverter(objective="ce").objective == "token_index"
    with pytest.raises(ValueError):
        Qwen2AloePriInverter(objective="bogus")
