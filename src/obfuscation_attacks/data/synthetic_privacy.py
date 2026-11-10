"""Synthetic scenario builders for the families added in the C02/C04/C07/C12/C13/C14 pass.

Kept in a separate module from :mod:`.synthetic` so the original seven-family
builders stay untouched and reviewable; both register into the same
``BUILDERS`` map through :func:`register_builders`.

Every builder produces a *genuine instance* of the leakage its attack exploits
(with the ground truth kept for scoring only), exactly like the existing ones:

* ``linkage``        - a request log under key rotation (S-C), with per-epoch
  symbol re-mapping so exact linking has real signal and the SH-P / SH-B
  difference is measurable.
* ``beamclean``      - Gaussian/Laplacian obfuscated wire embeddings (S1) plus a
  public reference corpus for the LM prior.
* ``pii`` / ``lm_recon`` - an obfuscated output token stream (S5) with (for PII)
  annotated entity spans.
* ``ima_cross_instance`` / ``ima_cross_checkpoint`` - a deployment instance and a
  *different* attacker instance of the same mechanism (KI), never the deployment
  secret.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .contracts import AttackSample, Capability
from .streams import RequestRecord, RequestStream
from ..schemes.generic import GenericScheme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .synthetic import Scenario


def _scenario(*args, **kwargs):
    """Build a :class:`Scenario` without importing ``synthetic`` at module load.

    ``synthetic`` imports this module at its bottom (to extend ``BUILDERS``), so
    a module-level ``from .synthetic import Scenario`` would make the pair
    unimportable in either order. Deferring the import keeps both directions
    valid: ``import synthetic_privacy`` first now works, and
    ``import synthetic`` still registers the new builders.
    """
    from .synthetic import Scenario

    return Scenario(*args, **kwargs)


def _zipf_tokens(rng: np.random.Generator, vocab: int, length: int, s: float = 1.1) -> np.ndarray:
    ranks = np.arange(1, vocab + 1, dtype=float)
    p = 1.0 / ranks**s
    p /= p.sum()
    return rng.choice(vocab, size=length, p=p)


def _markov_tokens(rng: np.random.Generator, vocab: int, length: int, n_states: int = 4) -> np.ndarray:
    """Small block-structured sequence so an LM prior carries real information."""
    out = [int(rng.integers(0, n_states))]
    for _ in range(max(0, length - 1)):
        prev = out[-1]
        if rng.random() < 0.75:
            nxt = (prev + 1) % n_states
        else:
            nxt = int(rng.integers(0, n_states))
        out.append(nxt if nxt < vocab else int(rng.integers(0, vocab)))
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# C12 - cross-request linkage under rotation (S-C)
# ---------------------------------------------------------------------------
def build_linkage_scenario(
    *,
    vocab: int = 24,
    d: int = 10,
    n_requests: int = 24,
    per_request: int = 6,
    key_size: int = 4,
    rotation_n: int = 8,
    noise: float = 0.02,
    metadata_visible: bool = True,
    seed: int = 0,
) -> Scenario:
    """Request log with per-epoch keys (plan S-C / P0-9).

    Each epoch uses one key from a pool of ``key_size``: inside an epoch a token
    maps to one symbol, across epochs the mapping is re-randomised. The wire
    vector is a per-epoch random rotation of the symbol embedding plus noise, so
    cosine-based linkage degrades exactly like a rotation scheme would.

    ``metadata_visible=False`` produces the SH-B sensitivity variant (same
    attack code, no session/epoch metadata).
    """
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    keys = [rng.permutation(vocab) for _ in range(max(1, key_size))]
    rotations = []
    for _ in range(max(1, key_size)):
        A = rng.standard_normal((d, d))
        q, _ = np.linalg.qr(A)  # orthonormal: cosine is preserved within an epoch
        rotations.append(q)

    records: list[RequestRecord] = []
    truth: list[np.ndarray] = []
    for i in range(n_requests):
        epoch = i // max(1, rotation_n)
        k = epoch % len(keys)
        ids = _zipf_tokens(rng, vocab, per_request)
        symbols = keys[k][ids]
        vectors = (E[symbols] @ rotations[k]) + noise * rng.standard_normal((per_request, d))
        records.append(
            RequestRecord(
                request_id=i,
                symbols=symbols,
                vectors=vectors,
                session_id=f"s{i % max(1, rotation_n)}",
                epoch=epoch,
                key_id=f"k{k}",
                arrival=i,
            )
        )
        truth.append(ids)

    caps = {Capability.EMBEDDINGS, Capability.TOKEN_CIPHERTEXT}
    if metadata_visible:
        caps.add(Capability.REQUEST_METADATA)
    scheme = GenericScheme("rotation_stream", frozenset(caps))
    stream = RequestStream(
        requests=records,
        plaintext_ids=truth,
        vocab_size=vocab,
        wire="embedding",
        meta={"key_size": key_size, "rotation_n": rotation_n, "metadata_visible": metadata_visible},
    )
    return _scenario(
        "linkage",
        scheme,
        sample=None,
        stream=stream,
        note="S-C request log; per-epoch key remapping; SH-P when metadata is exposed",
    )


# ---------------------------------------------------------------------------
# C07 - BeamClean on obfuscated wire embeddings (S1)
# ---------------------------------------------------------------------------
def build_beamclean_scenario(
    *,
    vocab: int = 24,
    d: int = 12,
    n: int = 8,
    noise: float = 0.25,
    noise_model: str = "gaussian",
    corpus_len: int = 60,
    corpus_max: int = 8,
    seed: int = 0,
) -> Scenario:
    """``Z = E[x] + eps`` on a recursive token sequence, plus a public corpus.

    ``noise=0`` is the positive-control setting: the naive distance decoder must
    recover the sequence exactly, otherwise the attack (not the defense) is broken.
    """
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    ids = _markov_tokens(rng, vocab, n)
    eps = rng.standard_normal((n, d)) if noise_model == "gaussian" else rng.laplace(size=(n, d))
    Z = E[ids] + float(noise) * eps
    corpus = [_markov_tokens(rng, vocab, int(rng.integers(3, corpus_max + 1))) for _ in range(corpus_len)]
    scheme = GenericScheme("wire_obfuscation", frozenset({Capability.EMBEDDINGS}))
    sample = AttackSample(
        tokenizer_id="toy",
        model_id="toy",
        seq_len=n,
        vocab_size=vocab,
        obfuscated_embeddings=Z,
        reference_embeddings=E,
        reference_corpus=corpus,
        plaintext_token_ids=ids,
        text_source="synthetic",
        data_split="eval",
        meta={"noise": noise, "noise_model": noise_model},
    )
    return _scenario(
        "beamclean",
        scheme,
        sample,
        note="obfuscated input embeddings + public embedding table + public corpus",
    )


# ---------------------------------------------------------------------------
# C13 / C14 - output-side PII + reconstruction (S5)
# ---------------------------------------------------------------------------
def build_pii_scenario(
    *,
    vocab: int = 40,
    length: int = 40,
    n_entities: int = 3,
    identity: bool = False,
    seed: int = 0,
) -> Scenario:
    """Obfuscated output stream with annotated PII entity spans.

    ``identity=True`` is the plan's positive control (§7.1): the "protection" is
    the identity map, so a working extractor must reach PIIRSR = 1.0. If it does
    not, the extractor is broken and no protected-arm number may be reported.
    """
    rng = np.random.default_rng(seed)
    ids = _zipf_tokens(rng, vocab, length)
    perm = np.arange(vocab) if identity else rng.permutation(vocab)

    # Entities: disjoint token patterns planted at distinct positions.
    entity_types = ["person", "phone", "email", "address"]
    entities: list[dict] = []
    cursor = 2
    for e in range(n_entities):
        span_len = int(rng.integers(2, 5))
        if cursor + span_len >= length:
            break
        pattern = rng.choice(vocab, size=span_len, replace=False)
        ids[cursor : cursor + span_len] = pattern
        entities.append(
            {
                "type": entity_types[e % len(entity_types)],
                "start": int(cursor),
                "end": int(cursor + span_len),
                "value": [int(t) for t in pattern],
            }
        )
        cursor += span_len + int(rng.integers(3, 7))

    cipher = perm[ids]
    ref = _zipf_tokens(np.random.default_rng(seed + 1), vocab, max(2000, 50 * length))
    ref_counts = {int(t): int(c) for t, c in zip(*np.unique(ref, return_counts=True))}
    # The plaintext control arm exposes no permutation capability: with
    # ``identity=True`` nothing is obfuscated, so the scheme must not tell the
    # attacker "this is a substitution cipher" (that would force a frequency
    # decode onto a stream that needs none, plan §7.1 control semantics).
    caps = (
        frozenset({Capability.TOKEN_CIPHERTEXT})
        if identity
        else frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION})
    )
    scheme = GenericScheme("output_permutation", caps)
    sample = AttackSample(
        tokenizer_id="toy",
        model_id="toy",
        seq_len=length,
        vocab_size=vocab,
        ciphertext_token_ids=cipher,
        reference_token_counts=ref_counts,
        plaintext_token_ids=ids,
        text_source="synthetic",
        data_split="eval",
        meta={
            "pii_entities": entities,
            "true_mapping": {int(perm[t]): int(t) for t in range(vocab)},
            "arm": "plaintext" if identity else "obfuscated",
        },
    )
    return _scenario(
        "pii",
        scheme,
        sample,
        note="S5 output token stream with annotated entities (Limitations/appendix only)",
    )


def build_lm_recon_scenario(
    *, vocab: int = 40, length: int = 32, identity: bool = False, seed: int = 0
) -> Scenario:
    """Obfuscated output stream for the InferDPT-style text metrics.

    ``identity=True`` is the plaintext positive control (reconstruction must be
    exact when nothing is obfuscated).
    """
    rng = np.random.default_rng(seed)
    ids = _markov_tokens(rng, vocab, length)
    perm = np.arange(vocab) if identity else rng.permutation(vocab)
    ref = _zipf_tokens(np.random.default_rng(seed + 1), vocab, max(2000, 50 * length))
    ref_counts = {int(t): int(c) for t, c in zip(*np.unique(ref, return_counts=True))}
    caps = (
        frozenset({Capability.TOKEN_CIPHERTEXT})
        if identity
        else frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION})
    )
    scheme = GenericScheme("output_permutation", caps)
    sample = AttackSample(
        tokenizer_id="toy",
        model_id="toy",
        seq_len=length,
        vocab_size=vocab,
        ciphertext_token_ids=perm[ids],
        reference_token_counts=ref_counts,
        plaintext_token_ids=ids,
        text_source="synthetic",
        data_split="eval",
        meta={
            "true_mapping": {int(perm[t]): int(t) for t in range(vocab)},
            "arm": "plaintext" if identity else "obfuscated",
        },
    )
    return _scenario("lm_recon", scheme, sample, note="S5 output stream, no PII annotation")


# ---------------------------------------------------------------------------
# C02 / C04 - KI inversion: attacker's own instance of the mechanism
# ---------------------------------------------------------------------------
def _mechanism_instance(E: np.ndarray, perm: np.ndarray, A: np.ndarray) -> Callable:
    def obfuscate(token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=int).ravel()
        return E[perm[ids]] @ A

    return obfuscate


def build_ima_cross_instance_scenario(
    *, vocab: int = 12, d: int = 8, n: int = 6, corpus_len: int = 60, seed: int = 0
) -> Scenario:
    """Deployment instance (secret A) vs attacker instance (secret B), both public-algorithm."""
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab, d))
    perm_dep, perm_atk = rng.permutation(vocab), rng.permutation(vocab)
    A_dep = rng.standard_normal((d, d))
    A_atk = rng.standard_normal((d, d))

    deploy = _mechanism_instance(E, perm_dep, A_dep)
    attacker = _mechanism_instance(E, perm_atk, A_atk)
    ids = rng.integers(0, vocab, size=n)
    corpus = [rng.integers(0, vocab, size=int(rng.integers(3, 10))) for _ in range(corpus_len)]
    corpus.append(np.arange(vocab))  # ensures every class appears in training
    scheme = GenericScheme("ki_mechanism", frozenset({Capability.EMBEDDINGS}))
    sample = AttackSample(
        tokenizer_id="toy",
        model_id="toy",
        seq_len=n,
        vocab_size=vocab,
        obfuscated_embeddings=deploy(ids),
        reference_embeddings=E,
        reference_corpus=corpus,
        plaintext_token_ids=ids,
        text_source="synthetic",
        data_split="eval",
    )
    return _scenario(
        "ima_cross_instance",
        scheme,
        sample,
        attacker_obfuscator=attacker,
        note="KI: attacker instance has its own secret; deployment key never used",
    )


def build_cross_checkpoint_scenario(
    *, vocab: int = 12, d: int = 8, n: int = 6, corpus_len: int = 60, seed: int = 0
) -> Scenario:
    """Transfer: attacker checkpoint != deployment checkpoint (C04).

    Two differences from the deployment instance, and neither of them may be
    skipped without breaking the plan's KI/KP boundary (D8 / A5):

    * the attacker's **checkpoint** differs - a different embedding table was
      trained for the same architecture (``E_atk`` vs ``E_dep``);
    * the attacker's **secret** differs - its own permutation and mechanism
      matrix (``perm_atk`` / ``A_atk``). Sharing the deployment ``perm``/``A``
      would mint training pairs under the deployment key, i.e. turn the arm into
      the KP appendix口径 and make the transfer number meaningless.
    """
    rng = np.random.default_rng(seed)
    E_dep = rng.standard_normal((vocab, d))
    E_atk = rng.standard_normal((vocab, d))  # different checkpoint's embedding table
    perm_dep, perm_atk = rng.permutation(vocab), rng.permutation(vocab)
    A_dep, A_atk = rng.standard_normal((d, d)), rng.standard_normal((d, d))

    deploy = _mechanism_instance(E_dep, perm_dep, A_dep)
    attacker = _mechanism_instance(E_atk, perm_atk, A_atk)
    ids = rng.integers(0, vocab, size=n)
    corpus = [rng.integers(0, vocab, size=int(rng.integers(3, 10))) for _ in range(corpus_len)]
    corpus.append(np.arange(vocab))
    scheme = GenericScheme("ki_transfer", frozenset({Capability.EMBEDDINGS}))
    sample = AttackSample(
        tokenizer_id="toy",
        model_id="toy",
        seq_len=n,
        vocab_size=vocab,
        obfuscated_embeddings=deploy(ids),
        reference_embeddings=E_dep,
        reference_corpus=corpus,
        plaintext_token_ids=ids,
        text_source="synthetic",
        data_split="eval",
        # Runner-only bookkeeping so a test can assert the KI/KP boundary: the
        # attacker instance must differ in table *and* secret from the
        # deployment instance. ``meta`` never reaches an attack context
        # (``_safe_descriptor`` does not copy it).
        meta={
            "instances": {
                "deployment": {"table": E_dep, "perm": perm_dep, "matrix": A_dep},
                "attacker": {"table": E_atk, "perm": perm_atk, "matrix": A_atk},
            }
        },
    )
    return _scenario(
        "cross_checkpoint",
        scheme,
        sample,
        attacker_obfuscator=attacker,
        note="C04 transfer: attacker checkpoint != deployment checkpoint",
    )


NEW_BUILDERS: dict[str, Callable[..., Scenario]] = {
    "linkage": build_linkage_scenario,
    "beamclean": build_beamclean_scenario,
    "pii": build_pii_scenario,
    "lm_recon": build_lm_recon_scenario,
    "ima_cross_instance": build_ima_cross_instance_scenario,
    "cross_checkpoint": build_cross_checkpoint_scenario,
}
