"""Synthetic scenario builders for smoke runs and integration tests.

Each builder returns a :class:`Scenario` = (scheme, sample, attacker_model) that
reproduces *exactly* the structure the corresponding attack consumes, with the
ground truth kept on the sample for scoring. These are genuine instances of the
leakage each attack exploits (so the attack really recovers the secret), not
mocks of the result. They are used by the integration tests and by the smoke
experiment configs; the full paper configs instead point at real models/data
(see ``docs/RESOURCE_REQUIREMENTS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .contracts import AttackSample, Capability
from ..schemes.generic import GenericScheme


@dataclass
class Scenario:
    name: str
    scheme: Any
    sample: AttackSample
    attacker_model: Optional[Any] = None
    note: str = ""


# ---------------------------------------------------------------------------
# NN / KNN
# ---------------------------------------------------------------------------
def build_nn_scenario(
    *, vocab: int = 40, d: int = 12, n: int = 8, noise: float = 0.05, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    ids = rng.integers(0, vocab, size=n)
    obf = E[ids] + noise * rng.standard_normal((n, d))
    scheme = GenericScheme("nn_noise", frozenset({Capability.EMBEDDINGS}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=vocab,
        obfuscated_embeddings=obf, reference_embeddings=E,
        plaintext_token_ids=ids, text_source="synthetic", data_split="eval",
        meta={"noise": noise},
    )
    return Scenario("nn", scheme, sample)


# ---------------------------------------------------------------------------
# VMA B2 - weight RowSort (Y = Z1 X Z2)
# ---------------------------------------------------------------------------
def build_vma_weight_scenario(
    *, n: int = 10, m: int = 7, layers: int = 1, seed: int = 0, duplicate_rows: bool = False
) -> Scenario:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((layers, n, m))
    if duplicate_rows:
        X[:, 1] = X[:, 0]  # force a duplicate row to exercise disambiguation
    z1 = rng.permutation(n)  # output row i <- input row z1[i]
    Y = np.empty_like(X)
    z2_layers = []
    for l in range(layers):
        z2 = rng.permutation(m)
        z2_layers.append(z2)
        Y[l] = X[l][np.ix_(z1, z2)]
    Xw = X[0] if layers == 1 else X
    Yw = Y[0] if layers == 1 else Y
    scheme = GenericScheme("aloepri_weight", frozenset({Capability.WHITE_BOX_WEIGHTS}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=n,
        known_weights={"X": Xw, "Y": Yw},
        secret_matrices={"Z1": z1, **{f"Z2_layer{l}": z for l, z in enumerate(z2_layers)}},
        text_source="synthetic", data_split="eval",
        meta={"true_mapping": {int(i): int(z1[i]) for i in range(n)}, "layers": layers},
    )
    return Scenario("vma_weight", scheme, sample)


# ---------------------------------------------------------------------------
# VMA B1 - autoregressive (model replay)
# ---------------------------------------------------------------------------
def build_vma_autoregressive_scenario(
    *, vocab: int = 16, d: int = 8, n: int = 5, layers: int = 2,
    permute_hidden: bool = False, seed: int = 0,
) -> Scenario:
    from ..models import CausalToyModel

    model = CausalToyModel(vocab_size=vocab, hidden_size=d, num_layers=layers, seed=seed)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=n)
    layer = layers
    target = model.causal_hidden_states(ids, layer)
    perm = None
    if permute_hidden:
        perm = rng.permutation(d)
        target = target[:, perm]

    def replay_fn(token_ids):
        return model.causal_hidden_states(token_ids, layer)

    scheme = GenericScheme(
        "aloepri_autoregressive",
        frozenset({Capability.MODEL_REPLAY, Capability.HIDDEN_STATES}),
        replay_fn=replay_fn,
    )
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=vocab, layer=layer,
        obfuscated_hidden_states=target, plaintext_token_ids=ids,
        text_source="synthetic", data_split="eval",
        meta={"permute_hidden": permute_hidden},
    )
    return Scenario("vma_autoregressive", scheme, sample)


# ---------------------------------------------------------------------------
# VMA B3 - OSNIP vocabulary-matching (autoregressive L1 replay)
# ---------------------------------------------------------------------------
def build_vma_osnip_scenario(
    *, vocab: int = 50, d: int = 12, n: int = 10, layers: int = 2, noise: float = 0.02,
    stopword_frac: float = 0.3, seed: int = 0,
) -> Scenario:
    # OSNIP's vocabulary-matching attack (App. C.2 / Sec 4.1) reconstructs the
    # sequence AUTOREGRESSIVELY: at each step it replays the clean model on
    # [xhat_<i, v] for every token v, takes the layer-l hidden state, and picks
    # the L1-nearest to the observed (perturbed) state. The scenario therefore
    # exposes a real causal model to replay (MODEL_REPLAY) plus the observed,
    # slightly perturbed layer-l states (HIDDEN_STATES) - not a static pool.
    from ..models import CausalToyModel

    model = CausalToyModel(vocab_size=vocab, hidden_size=d, num_layers=layers, seed=seed)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=n)
    layer = layers
    target = model.causal_hidden_states(ids, layer)
    if noise:
        target = target + noise * rng.standard_normal(target.shape)

    def replay_fn(token_ids):
        return model.causal_hidden_states(token_ids, layer)

    # Mark a subset of the vocabulary as stop words -> content_mask for Clean ASR.
    num_stop = max(1, int(stopword_frac * vocab))
    stop_ids = set(rng.choice(vocab, size=num_stop, replace=False).tolist())
    content_mask = np.array([int(t) not in stop_ids for t in ids], dtype=bool)
    scheme = GenericScheme(
        "osnip_projection",
        frozenset({Capability.HIDDEN_STATES, Capability.MODEL_REPLAY}),
        replay_fn=replay_fn,
    )
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=vocab, layer=layer,
        obfuscated_hidden_states=target,
        plaintext_token_ids=ids, content_mask=content_mask,
        text_source="synthetic", data_split="eval",
        meta={"noise": noise, "stop_ids": sorted(stop_ids)},
    )
    return Scenario("vma_osnip", scheme, sample)


# ---------------------------------------------------------------------------
# IA - EDNN (coordinate-difference invariant of the glide reflection)
# ---------------------------------------------------------------------------
def build_ia_ednn_scenario(
    *, vocab: int = 40, d: int = 12, n: int = 8, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    ids = rng.integers(0, vocab, size=n)
    ones = np.ones(d)
    obf = np.empty((n, d))
    for i, t in enumerate(ids):
        e = E[t]
        a = rng.standard_normal()
        while abs(a) < 1e-3:
            a = rng.standard_normal()
        b = rng.standard_normal()
        l = ones * a
        tvec = ones * b
        obf[i] = e - 2.0 * (e @ l) / (l @ l) * l + tvec
    scheme = GenericScheme("ia_ednn_glide", frozenset({Capability.EMBEDDINGS}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=vocab,
        obfuscated_embeddings=obf, reference_embeddings=E,
        plaintext_token_ids=ids, text_source="synthetic", data_split="eval",
    )
    return Scenario("ia_ednn", scheme, sample)


# ---------------------------------------------------------------------------
# IA - Gate (gate-weighted average invariant)
# ---------------------------------------------------------------------------
def build_ia_gate_scenario(
    *, vocab: int = 24, d: int = 10, hf: int = 16, layers: int = 4, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    W_gate = rng.standard_normal((layers, d, hf))
    perm = rng.permutation(vocab)  # obf token i corresponds to plain token perm[i]
    E_obf = E[perm]
    # FFN gate obfuscation permutes the hidden (hf) columns; mean over hf is
    # invariant to this permutation (alpha_e = 0: no embedding noise).
    W_gate_obf = np.empty_like(W_gate)
    for l in range(layers):
        zffn = rng.permutation(hf)
        W_gate_obf[l] = W_gate[l][:, zffn]
    scheme = GenericScheme("ia_gate", frozenset({Capability.WHITE_BOX_WEIGHTS}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=vocab, vocab_size=vocab,
        known_weights={"E": E, "E_obf": E_obf, "W_gate": W_gate, "W_gate_obf": W_gate_obf},
        secret_permutation=perm, text_source="synthetic", data_split="eval",
        meta={"true_mapping": {int(i): int(perm[i]) for i in range(vocab)}, "alpha_e": 0.0},
    )
    return Scenario("ia_gate", scheme, sample)


# ---------------------------------------------------------------------------
# IA - Attn (whitening quadratic-form invariant)
# ---------------------------------------------------------------------------
def build_ia_attn_scenario(
    *, n_tokens: int = 24, block: int = 2, num_blocks: int = 4, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    dq = block * num_blocks
    Q = rng.standard_normal((n_tokens, dq))
    perm = rng.permutation(n_tokens)
    # Block-diagonal invertible transform on the query columns.
    blocks = []
    for _ in range(num_blocks):
        while True:
            A = rng.standard_normal((block, block))
            if abs(np.linalg.det(A)) > 0.3:
                break
        blocks.append(A)
    M = np.zeros((dq, dq))
    for j, A in enumerate(blocks):
        M[j * block : (j + 1) * block, j * block : (j + 1) * block] = A
    Q_obf = (Q @ M)[perm]
    scheme = GenericScheme("ia_attn", frozenset({Capability.WHITE_BOX_WEIGHTS}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n_tokens, vocab_size=n_tokens,
        known_weights={"Q": Q, "Q_obf": Q_obf},
        secret_permutation=perm, text_source="synthetic", data_split="eval",
        meta={"true_mapping": {int(i): int(perm[i]) for i in range(n_tokens)},
              "block_size": block},
    )
    return Scenario("ia_attn", scheme, sample)


# ---------------------------------------------------------------------------
# TFMA - deterministic token permutation + frequency prior
# ---------------------------------------------------------------------------
def _skewed_tokens(rng: np.random.Generator, vocab: int, length: int) -> np.ndarray:
    # Zipf-like skew so frequency structure is informative.
    ranks = np.arange(1, vocab + 1)
    p = 1.0 / ranks
    p /= p.sum()
    return rng.choice(vocab, size=length, p=p)


def build_tfma_scenario(
    *, vocab: int = 30, length: int = 4000, ref_length: int = 20000, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    plain = _skewed_tokens(rng, vocab, length)
    perm = rng.permutation(vocab)  # plaintext id -> cipher id (deterministic)
    cipher = perm[plain]
    # Reference prior from an independent draw of the same distribution.
    ref = _skewed_tokens(np.random.default_rng(seed + 1), vocab, ref_length)
    ref_counts = {int(t): int(c) for t, c in zip(*np.unique(ref, return_counts=True))}
    # true mapping cipher -> plain is the inverse of perm
    inv = np.empty(vocab, dtype=int)
    inv[perm] = np.arange(vocab)
    scheme = GenericScheme(
        "tfma_permutation",
        frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}),
    )
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=length, vocab_size=vocab,
        ciphertext_token_ids=cipher, reference_token_counts=ref_counts,
        secret_permutation=perm, text_source="synthetic", data_split="eval",
        meta={"true_mapping": {int(cipher_id): int(inv[cipher_id]) for cipher_id in range(vocab)}},
    )
    return Scenario("tfma", scheme, sample)


# ---------------------------------------------------------------------------
# SDA - small structured language, unseen key
# ---------------------------------------------------------------------------
def build_sda_scenario(
    *, seed: int = 0, num_corpus: int = 120, eval_words: int = 3
) -> Scenario:
    from ..attacks.sda import encodings as enc

    vocab = 10
    words = [[1, 2, 1], [3, 1], [1, 4], [5, 6, 5], [2, 1], [7, 8], [1, 9]]
    weights = np.array([0.34, 0.22, 0.16, 0.10, 0.08, 0.06, 0.04])
    weights = weights / weights.sum()

    def sentence(r, nwords):
        out = []
        for _ in range(nwords):
            out.append(0)
            out += words[r.choice(len(words), p=weights)]
        return np.asarray(out[1:], dtype=int)

    rng = np.random.default_rng(seed)
    corpus = [sentence(rng, int(rng.integers(4, 9))) for _ in range(num_corpus)]
    eval_plain = sentence(np.random.default_rng(seed + 500), eval_words)
    key = enc.make_substitution_key(
        range(vocab), rng=np.random.default_rng(seed + 7),
        homophonic=False, cipher_alphabet_size=vocab,
    )
    cipher = enc.apply_substitution(eval_plain, key)
    scheme = GenericScheme(
        "sda_permutation",
        frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}),
    )
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=int(cipher.size), vocab_size=vocab,
        ciphertext_token_ids=cipher, reference_corpus=corpus,
        plaintext_token_ids=eval_plain, text_source="synthetic", data_split="eval",
    )
    return Scenario("sda", scheme, sample)


# ---------------------------------------------------------------------------
# ISA - optimization against a frozen toy model
# ---------------------------------------------------------------------------
def build_isa_scenario(
    *, vocab: int = 16, d: int = 16, n: int = 4, layers: int = 1, layer: int = 1, seed: int = 0
) -> Scenario:
    from ..models import TorchToyModel

    model = TorchToyModel(vocab_size=vocab, hidden_size=d, num_layers=layers, seed=seed).freeze()
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=n)
    target = model.hidden_states(ids, layer).numpy()
    scheme = GenericScheme("isa_state", frozenset({Capability.HIDDEN_STATES}))
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=n, vocab_size=vocab, layer=layer,
        obfuscated_hidden_states=target, plaintext_token_ids=ids,
        text_source="synthetic", data_split="eval",
    )
    return Scenario("isa", scheme, sample, attacker_model=model)


# ---------------------------------------------------------------------------
# IMA - learned inverter, known obfuscation
# ---------------------------------------------------------------------------
def build_ima_scenario(
    *, vocab: int = 10, d: int = 8, n: int = 6, num_corpus: int = 40, seed: int = 0
) -> Scenario:
    rng = np.random.default_rng(seed)
    E_clean = rng.standard_normal((vocab, d))
    A = rng.standard_normal((d, d))
    perm = rng.permutation(vocab)

    def obfuscation_fn(token_ids):
        ids = np.asarray(token_ids, dtype=int).ravel()
        return E_clean[perm[ids]] @ A

    corpus = [rng.integers(0, vocab, size=n) for _ in range(num_corpus)]
    corpus.append(np.arange(vocab))  # full-vocab coverage
    held = np.array([0, 2, 4, 6, 8, 9][:n])
    corpus = [s for s in corpus if tuple(int(x) for x in np.asarray(s)) != tuple(held.tolist())]
    obf = obfuscation_fn(held)
    scheme = GenericScheme(
        "ima_known_obf",
        frozenset({Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}),
        obfuscation_fn=obfuscation_fn,
    )
    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=int(held.size), vocab_size=vocab,
        obfuscated_embeddings=obf, reference_embeddings=E_clean, reference_corpus=corpus,
        plaintext_token_ids=held, text_source="synthetic", data_split="eval",
    )
    return Scenario("ima", scheme, sample)


# ---------------------------------------------------------------------------
# Custom-scheme example (Goal section 9) - demonstrates the plug-in interface
# ---------------------------------------------------------------------------
def build_custom_scheme_scenario(
    *, vocab: int = 50, d: int = 16, n: int = 15, epsilon: float = 0.1, seed: int = 0
) -> Scenario:
    from ..schemes.custom_example import CustomNullSpaceScheme

    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    ids = rng.integers(0, vocab, size=n)
    scheme = CustomNullSpaceScheme(epsilon=epsilon, seed=seed)
    sample = scheme.build_sample(ids, E)
    return Scenario("custom_nullspace", scheme, sample, note="custom-scheme interface demo")


BUILDERS: dict[str, Callable[..., Scenario]] = {
    "nn": build_nn_scenario,
    "vma_weight": build_vma_weight_scenario,
    "vma_autoregressive": build_vma_autoregressive_scenario,
    "vma_osnip": build_vma_osnip_scenario,
    "ia_ednn": build_ia_ednn_scenario,
    "ia_gate": build_ia_gate_scenario,
    "ia_attn": build_ia_attn_scenario,
    "tfma": build_tfma_scenario,
    "sda": build_sda_scenario,
    "isa": build_isa_scenario,
    "ima": build_ima_scenario,
    "custom_nullspace": build_custom_scheme_scenario,
}


def build_scenario(name: str, **params) -> Scenario:
    if name not in BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(BUILDERS)}")
    return BUILDERS[name](**params)
