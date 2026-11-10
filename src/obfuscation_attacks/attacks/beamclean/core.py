"""BeamClean: language-aware embedding reconstruction (plan class C07, P1).

**Paper.** Kaan Kale, Kyle Mylonakis, Jay Roberts, Sidhartha Roy, *BeamClean:
Language Aware Embedding Reconstruction*, arXiv:2505.13758 (2025).

The paper's setting is exactly the plan's S1 surface for a continuous wire: the
adversary sees the **obfuscated input embeddings** ``z`` and the model's public
**embedding table** ``E``, but *not* the language model and *not* the obfuscation
mechanism. BeamClean then (a) jointly estimates the noise parameters and (b)
decodes token sequences by integrating a **language-model prior**, and it
"always surpasses naive distance-based attacks" against Laplacian and Gaussian
obfuscation.

Re-implementation (each formula pinned in ``docs/FORMULA_LEDGER.md``):

``Z = E[x] + eps`` (Gaussian ``eps ~ N(0, sigma^2 I)`` or Laplacian
``eps ~ Lap(0, b I)``), decoded by maximising

    score(w_1..w_n) = lambda * log P_LM(w_1..w_n) + sum_i log p_noise(z_i | E[w_i])

with beam search of width ``beam``. ``lambda = 0`` degrades BeamClean to the naive
distance-based attacker the paper compares against - that is the built-in
positive control (no noise => the distance term alone recovers the sequence
exactly, so any loss at ``lambda > 0`` is the LM prior's doing, not a bug).

Noise estimation is the paper's "joint" part: with the NN distance
``d_i = min_v ||z_i - E_v||`` and a d-dimensional isotropic noise,
``E||eps||^2 = sigma^2 d`` (Gaussian) or ``2 b^2 d`` (Laplacian), so
``sigma_hat^2 = median_i d_i^2 / d`` and ``b_hat^2 = median_i d_i^2 / (2d)``
give a robust closed-form start; :func:`estimate_noise_em` then alternates
decode / residual re-estimation, which is the joint estimate the paper describes.

**Deviation (recorded).** The LM prior is a *neural* LM in the paper. This
harness takes it from ``ctx.attacker_model`` when the handle exposes
``next_token_logprobs`` (neural path); otherwise it falls back to a bigram LM
estimated on the public reference corpus and records
``lm_prior="bigram_reference"`` in the result. A bigram prior is *weaker* than
the paper's neural prior, so a low ASR under the fallback cannot be read as
"privacy": the fidelity gate flags the fallback, and the full config requires the
neural handle (``resource_blocked`` otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register

NOISE_MODELS = ("gaussian", "laplacian")


# ---------------------------------------------------------------------------
# noise model
# ---------------------------------------------------------------------------
def _sq_distances(Z: np.ndarray, E: np.ndarray, *,
                  budget_bytes: int = 256 * 1024 ** 2) -> np.ndarray:
    """Exact ``||z_i - E_v||^2`` as ``[n, V]`` without a 3-D temporary.

    ``z^2 - 2 z E^T + E^2`` in float64, chunked over the vocabulary so the
    peak allocation stays bounded at the real 128k-vocabulary scale.
    """
    Z = np.asarray(Z, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n, d = Z.shape
    out = np.empty((n, E.shape[0]), dtype=np.float64)
    z2 = (Z * Z).sum(axis=1)[:, None]
    rows = max(1, int(budget_bytes / max(1, n * d * 8)))
    for start in range(0, E.shape[0], rows):
        block = E[start:start + rows]
        out[:, start:start + block.shape[0]] = (
            z2 - 2.0 * (Z @ block.T) + (block * block).sum(axis=1)[None, :]
        )
    return out


def _l1_distances(Z: np.ndarray, E: np.ndarray, *,
                  budget_bytes: int = 256 * 1024 ** 2) -> np.ndarray:
    """Exact ``||z_i - E_v||_1`` as ``[n, V]``, chunked over the vocabulary."""
    Z = np.asarray(Z, dtype=np.float32)
    E = np.asarray(E, dtype=np.float32)
    n, d = Z.shape
    out = np.empty((n, E.shape[0]), dtype=np.float64)
    rows = max(1, int(budget_bytes / max(1, n * d * 4)))
    for start in range(0, E.shape[0], rows):
        block = E[start:start + rows]
        out[:, start:start + block.shape[0]] = np.abs(
            Z[:, None, :] - block[None, :, :]
        ).sum(axis=2, dtype=np.float64)
    return out


def nn_squared_distances(Z: np.ndarray, E: np.ndarray) -> np.ndarray:
    """``d_i^2 = min_v ||z_i - E_v||^2`` for every position (naive NN reference)."""
    return _sq_distances(Z, E).min(axis=1)


def estimate_noise(Z: np.ndarray, E: np.ndarray, kind: str = "gaussian") -> float:
    """Robust closed-form noise-scale estimate from the NN distances.

    Gaussian: ``sigma^2 = median(d^2) / d``; Laplacian: ``b^2 = median(d^2) / (2d)``.
    The median keeps a handful of confidently-decoded positions from dragging the
    estimate (the paper's motivation for a joint/robust estimate).
    """
    if kind not in NOISE_MODELS:
        raise ValueError(f"unknown noise model {kind!r}")
    d2 = nn_squared_distances(Z, E)
    dim = float(np.asarray(E).shape[1])
    med = float(np.median(d2))
    scale2 = med / dim if kind == "gaussian" else med / (2.0 * dim)
    return float(np.sqrt(max(scale2, 1e-12)))


def noise_loglik(Z: np.ndarray, E: np.ndarray, scale: float, kind: str = "gaussian") -> np.ndarray:
    """``log p_noise(z_i | E_v)`` for every (position, vocabulary entry).

    Gaussian:   ``-0.5 ||z - e||^2 / sigma^2 - d log sigma + const``
    Laplacian:  ``-||z - e||_1 / b - d log(2b) + const``

    Constants independent of ``v`` are dropped (they do not change the argmax and
    the beam keeps only relative scores).
    """
    Z = np.asarray(Z, dtype=float)
    E = np.asarray(E, dtype=float)
    if kind == "gaussian":
        s2 = max(float(scale) ** 2, 1e-12)
        return -0.5 * _sq_distances(Z, E) / s2
    b = max(float(scale), 1e-12)
    return -_l1_distances(Z, E) / b


# ---------------------------------------------------------------------------
# language-model prior
# ---------------------------------------------------------------------------
def fit_bigram_lm(
    corpus: Sequence[Sequence[int]], vocab_size: int, *, alpha: float = 1.0,
    dense_threshold: int = 4096,
) -> dict:
    """Add-``alpha`` smoothed bigram LM over the public reference corpus.

    ``P(w_i | w_{i-1}) = (c(w_{i-1}, w_i) + alpha) / (c(w_{i-1}) + alpha V)`` with
    ``<s>`` folded into a virtual previous token ``-1`` (index ``vocab_size``).

    The counts are stored SPARSELY and a decoding row is built on demand (one
    ``O(V)`` array fill per distinct previous token): the dense ``(V+1) x V``
    table is 123 GiB at the real 128,256-entry vocabulary and can never be
    materialised. Below ``dense_threshold`` the dense log table is additionally
    precomputed (toy-scale convenience; 4096 keeps it under ~270 MiB).
    """
    from scipy.sparse import csr_matrix

    V = int(vocab_size)
    rows: list[int] = []
    cols: list[int] = []
    for seq in corpus:
        prev = V
        for tid in (int(t) for t in np.asarray(seq).ravel()):
            if 0 <= tid < V:
                rows.append(prev)
                cols.append(tid)
                prev = tid
    counts = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)), shape=(V + 1, V)
    )
    counts.sum_duplicates()
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    log_table = None
    if V <= int(dense_threshold):
        dense_counts = np.asarray(counts.todense(), dtype=float)
        probs = (dense_counts + float(alpha)) / (totals[:, None] + float(alpha) * V)
        log_table = np.log(np.maximum(probs, 1e-300))
    return {"counts": counts, "totals": totals, "log_probs": log_table,
            "order": 2, "alpha": float(alpha), "vocab_size": V}


def _bigram_row(lm: dict, prev: int) -> np.ndarray:
    """Smoothed ``log P(. | prev)`` row, dense table when available else on demand."""
    V = lm["vocab_size"]
    if lm.get("log_probs") is not None:
        return lm["log_probs"][prev]
    alpha = float(lm["alpha"])
    denom = float(lm["totals"][prev]) + alpha * V
    row = np.full(V, np.log(max(alpha / denom, 1e-300)))
    counts = lm["counts"]
    start, stop = int(counts.indptr[prev]), int(counts.indptr[prev + 1])
    if start < stop:
        idx = counts.indices[start:stop]
        row[idx] = np.log(np.maximum((counts.data[start:stop] + alpha) / denom, 1e-300))
    return row


def ngram_logprob(lm: dict, prefix: Sequence[int], token: int) -> float:
    """``log P(token | last prefix token)`` under the bigram LM (``<s>`` = V)."""
    V = lm["vocab_size"]
    prev = V if not prefix else int(prefix[-1])
    if not (0 <= prev <= V) or not (0 <= int(token) < V):
        return float("-inf")
    return float(_bigram_row(lm, prev)[int(token)])


def bigram_logprob_row(lm: dict, prefix: Sequence[int]) -> np.ndarray:
    """Full ``log P(. | last prefix token)`` row of the bigram LM (``<s>`` = V)."""
    V = lm["vocab_size"]
    prev = V if not prefix else int(prefix[-1])
    if not (0 <= prev <= V):
        return np.full(V, float("-inf"))
    return _bigram_row(lm, prev)


class NeuralLMPrior:
    """Adapter for a model handle that exposes ``next_token_logprobs``.

    ``logprobs(prefix)`` fetches the WHOLE next-token distribution once per
    prefix; the beam then indexes every candidate from that one array. Asking
    the handle per (prefix, candidate) pair re-runs the forward ``V`` times per
    step, which is what this adapter exists to avoid.
    """

    def __init__(self, handle) -> None:
        self.handle = handle

    def logprobs(self, prefix: Sequence[int]) -> np.ndarray:  # pragma: no cover
        return np.asarray(self.handle.next_token_logprobs(list(prefix)), dtype=float)

    def logprob(self, prefix: Sequence[int], token: int) -> float:  # pragma: no cover
        return float(self.logprobs(prefix)[int(token)])


def make_lm_prior(ctx: AttackContext, vocab: int, *, alpha: float = 1.0) -> tuple[Callable, str]:
    """Return ``(logprobs_row(prefix) -> (V,) array, kind)`` - neural if available, else bigram."""
    model = getattr(ctx, "attacker_model", None)
    if model is not None and hasattr(model, "next_token_logprobs"):
        prior = NeuralLMPrior(model)
        return prior.logprobs, "neural_attacker_model"
    corpus = ctx.reference_corpus or []
    if not corpus:
        raise ValueError(
            "BeamClean needs a language-model prior: pass an attacker model with "
            "next_token_logprobs, or a public reference corpus for the bigram fallback"
        )
    lm = fit_bigram_lm(corpus, vocab, alpha=alpha)
    return (lambda prefix: bigram_logprob_row(lm, prefix)), "bigram_reference"


# ---------------------------------------------------------------------------
# beam decoding
# ---------------------------------------------------------------------------
@dataclass
class BeamResult:
    tokens: list[int]
    score: float


def beam_decode(
    Z: np.ndarray,
    E: np.ndarray,
    logprobs_row: Callable[[Sequence[int]], np.ndarray],
    *,
    kind: str = "gaussian",
    scale: float = 1.0,
    beam: int = 8,
    lambda_lm: float = 1.0,
    candidate_topk: Optional[int] = None,
) -> tuple[BeamResult, list[list[int]]]:
    """Maximise ``lambda log P_LM(w) + sum_i log p_noise(z_i | E[w_i])`` by beam search.

    ``logprobs_row(prefix)`` returns the WHOLE next-token log-probability row
    ``(V,)``; the step scores every candidate from that one array, so an LM
    evaluation happens once per (beam, position), never once per candidate.
    Returns the best hypothesis and, per position, the ranked candidate ids of the
    winning hypothesis' beam states (so the scorer can compute ASR@k). When
    ``candidate_topk`` is set the per-position candidate set is pruned by the
    noise likelihood alone - that is an *approximation* and the caller must label
    the run ``candidate_mode="proposal"``; the default ``None`` keeps the full
    vocabulary as the plan requires for main-table numbers.
    """
    Z = np.asarray(Z, dtype=float)
    E = np.asarray(E, dtype=float)
    n, V = Z.shape[0], E.shape[0]
    ll = noise_loglik(Z, E, scale, kind)  # (n, V)
    pruned = candidate_topk is not None and int(candidate_topk) < V
    use_lm = bool(lambda_lm)

    def _pool(i: int) -> np.ndarray:
        if pruned:
            pool = np.argsort(-ll[i])[: int(candidate_topk)]
            pool.sort()
            return pool
        return np.arange(V)

    # beam entries: (score, tokens)
    beams: list[tuple[float, list[int]]] = [(0.0, [])]
    for i in range(n):
        pool = _pool(i)
        noise = ll[i] if not pruned else ll[i, pool]
        # (B, P) step scores; row-major flatten keeps the generation order
        # (beams in rank order, candidates ascending), so the stable argsort
        # below breaks score ties exactly like the former per-candidate loop.
        rows = np.empty((len(beams), pool.size), dtype=float)
        for b, (score, toks) in enumerate(beams):
            lm = (float(lambda_lm) * logprobs_row(toks)[pool]) if use_lm else 0.0
            rows[b] = score + noise + lm
        flat = rows.reshape(-1)
        order = np.argsort(-flat, kind="stable")[: max(1, int(beam))]
        beams = [
            (float(flat[int(index)]),
             beams[int(index) // pool.size][1] + [int(pool[int(index) % pool.size])])
            for index in order
        ]

    # Candidate lists for the winning hypothesis: rank the tokens the search actually
    # considered at that position, keeping the prefix fixed. With
    # ``candidate_topk`` the returned set is the same pruned pool the beam used (the
    # caller labels it ``candidate_mode="proposal"``); the default ranks the full
    # vocabulary, so candidate containment and the search never disagree.
    best_score, best_tokens = beams[0]
    candidates: list[list[int]] = []
    for i in range(n):
        prefix = best_tokens[:i]
        pool = _pool(i)
        scores = ll[i, pool] if pruned else ll[i]
        if use_lm:
            scores = scores + float(lambda_lm) * (
                logprobs_row(prefix)[pool] if pruned else logprobs_row(prefix)
            )
        order = pool[np.argsort(-scores, kind="stable")]
        candidates.append(order.tolist())
    return BeamResult(tokens=best_tokens, score=best_score), candidates


def estimate_noise_em(
    Z: np.ndarray,
    E: np.ndarray,
    logprobs_row: Callable,
    *,
    kind: str = "gaussian",
    beam: int = 8,
    lambda_lm: float = 1.0,
    iterations: int = 3,
) -> tuple[float, list[dict]]:
    """Alternate decode / residual re-estimation ("joint" noise estimate)."""
    scale = estimate_noise(Z, E, kind)
    history: list[dict] = []
    for it in range(max(1, int(iterations))):
        result, _ = beam_decode(
            Z, E, logprobs_row, kind=kind, scale=scale, beam=beam, lambda_lm=lambda_lm
        )
        pred = np.asarray(result.tokens, dtype=int)
        resid = Z - E[pred]
        dim = float(E.shape[1])
        if kind == "gaussian":
            new_scale = float(np.sqrt(max((resid ** 2).sum(axis=1).mean() / dim, 1e-12)))
        else:
            new_scale = float(np.mean(np.abs(resid).sum(axis=1)) / dim)
        history.append(
            {"iteration": it, "scale": scale, "reestimated": new_scale, "tokens": pred.tolist()}
        )
        if abs(new_scale - scale) <= 1e-6 * max(scale, 1.0):
            scale = new_scale
            break
        scale = new_scale
    return scale, history


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def _prepare(ctx: AttackContext):
    Z = np.asarray(ctx.obfuscated_embeddings(), dtype=float)
    if Z.ndim != 2:
        return None, "obfuscated embeddings must be 2-D (seq_len, d)"
    E = ctx.reference_embeddings
    if E is None:
        return None, "BeamClean needs the model's public embedding table (reference_embeddings)"
    E = np.asarray(E, dtype=float)
    if E.ndim != 2 or E.shape[1] != Z.shape[1]:
        return None, (
            f"embedding table dim {E.shape} does not match wire dim {Z.shape}; "
            "the scorer cannot compare in a different coordinate system"
        )
    return (Z, E), None


class _BeamCleanBase(Attack):
    required_capabilities = frozenset({Capability.EMBEDDINGS})
    variant = "original"
    noise_kind = "gaussian"

    def _run(self, ctx: AttackContext) -> AttackResult:
        prepared, err = _prepare(ctx)
        if err:
            return AttackResult.not_applicable(err)
        Z, E = prepared
        vocab = int(ctx.descriptor.get("vocab_size") or E.shape[0])
        try:
            logprobs_row, lm_kind = make_lm_prior(
                ctx, vocab, alpha=float(self.params.get("lm_alpha", 1.0))
            )
        except ValueError as exc:
            return AttackResult.not_applicable(str(exc))

        beam = int(self.params.get("beam", 8))
        lambda_lm = float(self.params.get("lambda_lm", 1.0))
        candidate_topk = self.params.get("candidate_topk", None)
        em_iterations = int(self.params.get("em_iterations", 3))
        kind = self.noise_kind

        if self.params.get("estimate_noise", "em") == "closed_form":
            scale = estimate_noise(Z, E, kind)
            history = []
        else:
            scale, history = estimate_noise_em(
                Z, E, logprobs_row, kind=kind, beam=beam, lambda_lm=lambda_lm,
                iterations=em_iterations,
            )
        result, candidates = beam_decode(
            Z, E, logprobs_row, kind=kind, scale=scale, beam=beam, lambda_lm=lambda_lm,
            candidate_topk=candidate_topk,
        )
        # Positive control: the naive NN decoder (lambda = 0, no LM) on the same
        # observation. The paper's baseline; it must be perfect on clean wire.
        naive, _ = beam_decode(Z, E, logprobs_row, kind=kind, scale=scale, beam=1,
                               lambda_lm=0.0)

        return AttackResult(
            recovered_token_ids=np.asarray(result.tokens, dtype=int),
            candidates=candidates,
            diagnostics={
                "noise_model": kind,
                "noise_scale": float(scale),
                "noise_scale_history": history,
                "beam": beam,
                "lambda_lm": lambda_lm,
                "lm_prior": lm_kind,
                "candidate_mode": "proposal" if candidate_topk else "full",
                "candidate_topk": candidate_topk,
                "score": float(result.score),
                "naive_nn_tokens": naive.tokens,
                "naive_nn_agreement": float(
                    np.mean(np.asarray(naive.tokens) == np.asarray(result.tokens))
                ),
                "positive_control": {
                    "name": "naive_nn_decoder_lambda0",
                    "tokens": naive.tokens,
                },
            },
        )


@register
class GaussianBeamClean(_BeamCleanBase):
    """BeamClean against Gaussian obfuscation (paper §3-4)."""

    name = "beamclean.gaussian"
    family = "beamclean"
    variant = "original"
    noise_kind = "gaussian"


@register
class LaplacianBeamClean(_BeamCleanBase):
    """BeamClean against Laplacian obfuscation (the paper's second noise model)."""

    name = "beamclean.laplacian"
    family = "beamclean"
    variant = "original"
    noise_kind = "laplacian"
