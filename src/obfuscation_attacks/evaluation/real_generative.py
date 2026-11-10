"""Generative reconstruction on the public surface (F14).

* BeamClean: beam search over ``log p_LM(token | prefix) + log p(obs | token)``
  with ``p(obs|token) = N(E[token]; obs, sigma^2 I)``. The Gaussian likelihood is
  computed from the squared **L2** distance in log space, in vocabulary chunks;
  an ``[n_positions, V, D]`` tensor is never materialised.
* Iterative prior refinement is a Vec2Text-shaped heuristic (no trained
  corrector); it is labelled ``engineering_variant``.
* LM rewrite is a local greedy decode on the public clean model.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..schemes.public_view import PublicDeployment


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _table(public: PublicDeployment, arm: str) -> np.ndarray:
    return np.asarray(
        public.table("plaintext") if arm == "plaintext" else public.table(public.arm),
        dtype=np.float32,
    )


def _next_token_logprobs(public: PublicDeployment, prefix_embeds: np.ndarray,
                         *, arm: str) -> np.ndarray:
    out = public.forward(prefix_embeds[None, :, :], clean=(arm == "plaintext"))
    logits = np.asarray(out.logits, dtype=np.float32)
    if logits.ndim == 3:
        logits = logits[0]
    return _log_softmax(logits[-1])


def _top_candidates(
    observations: np.ndarray,
    table: np.ndarray,
    *,
    n_candidates: int,
    sigma: float,
    chunk: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-position Top-N candidates and Gaussian log-likelihoods, chunked over V."""
    n_pos, dim = observations.shape
    v = table.shape[0]
    k = max(1, min(int(n_candidates), v))
    if chunk is None:
        chunk = max(1, int(2 ** 23 / max(1, n_pos * dim)))
    chunk = max(1, int(chunk))
    best_d = np.full((n_pos, k), np.inf, dtype=np.float64)
    best_ids = np.full((n_pos, k), -1, dtype=np.int64)
    n_chunks = 0
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        block = table[start:stop]
        d = ((observations[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)
        ids = np.broadcast_to(np.arange(start, stop, dtype=np.int64), d.shape)
        combined_d = np.concatenate([best_d, d], axis=1)
        combined_ids = np.concatenate([best_ids, ids], axis=1)
        take = np.argpartition(combined_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(combined_d, take, axis=1)
        best_ids = np.take_along_axis(combined_ids, take, axis=1)
        n_chunks += 1
    order = np.argsort(best_d, axis=1, kind="stable")
    distances = np.take_along_axis(best_d, order, axis=1)
    ids = np.take_along_axis(best_ids, order, axis=1)
    loglik = -0.5 * distances / float(sigma) ** 2
    return ids, loglik, distances, n_chunks


def beamclean_recover(
    public: PublicDeployment,
    observations: np.ndarray,
    *,
    arm: str,
    beam: int = 3,
    n_candidates: int = 6,
    max_len: int = 8,
    sigma: float = 1.0,
    chunk: int | None = None,
    verbose: bool = False,
) -> dict:
    """BeamClean over the public wire with an exact chunked Gaussian likelihood."""
    t0 = time.time()
    table = _table(public, arm)
    obs = np.asarray(observations, dtype=np.float32)[:max_len]
    cand_ids, cand_loglik, distances, n_chunks = _top_candidates(
        obs, table, n_candidates=n_candidates, sigma=sigma, chunk=chunk
    )
    hyps: list[tuple[list[int], float]] = [([], 0.0)]
    for t in range(len(obs)):
        new: list[tuple[list[int], float]] = []
        for prefix, score in hyps:
            if prefix:
                lp = _next_token_logprobs(public, table[prefix], arm=arm)
            else:
                lp = np.zeros(table.shape[0], dtype=np.float32)
            for c, ll in zip(cand_ids[t], cand_loglik[t]):
                new.append((prefix + [int(c)], score + float(lp[int(c)]) + float(ll)))
        new.sort(key=lambda item: -item[1])
        hyps = new[:beam]
    best_ids, best_score = hyps[0]
    if verbose:
        print(f"  BeamClean: ids={best_ids} score={best_score:.3f} ({time.time()-t0:.1f}s)",
              flush=True)
    return {
        "ids": best_ids,
        "score": best_score,
        "n_hypotheses": len(hyps),
        "candidate_set_size": int(n_candidates),
        "candidate_ids": cand_ids.tolist(),
        "candidate_loglik": cand_loglik.tolist(),
        "n_chunks": int(n_chunks),
        "sigma": float(sigma),
        "likelihood": "gaussian_l2_squared",
        "seconds": round(time.time() - t0, 1),
    }


def iterative_prior_refine(
    public: PublicDeployment,
    initial: Sequence[int],
    observations: np.ndarray,
    *,
    arm: str,
    rounds: int = 3,
    k_alt: int = 5,
) -> dict:
    """Vec2Text-shaped heuristic; no trained corrector."""
    table = _table(public, arm)
    ids = [int(x) for x in initial]
    obs = np.asarray(observations, dtype=np.float32)
    history = []
    for r in range(rounds):
        changed = 0
        for t in range(min(len(ids), len(obs))):
            prefix = ids[:t]
            if prefix:
                lp = _next_token_logprobs(public, table[prefix], arm=arm)
                alts = np.argsort(-lp)[:k_alt]
            else:
                alts = np.array([ids[t]])
            current = float(np.sum((table[ids[t]] - obs[t]) ** 2))
            best_id, best_d = ids[t], current
            for alt in alts:
                distance = float(np.sum((table[int(alt)] - obs[t]) ** 2))
                if distance < best_d:
                    best_id, best_d = int(alt), distance
            if best_id != ids[t]:
                ids[t] = best_id
                changed += 1
        history.append({"round": r, "changed": changed})
    return {
        "ids": ids,
        "history": history,
        "engineering_variant": "Vec2Text-shaped heuristic, no trained corrector",
    }


def vec2text_refine(*args, **kwargs) -> dict:
    """Deprecated alias; the output is explicitly an engineering heuristic."""
    return iterative_prior_refine(*args, **kwargs)


def lm_rewrite(public: PublicDeployment, text: str, *, max_new_tokens: int = 32) -> str:
    """Greedy local rewrite on the public clean model (never a secret decode)."""
    if public.encode_fn is None or public.decode_fn is None:
        raise RuntimeError("LM rewrite needs the public tokenizer")
    prompt = (
        "Rewrite the following note in fluent English, keeping all names and numbers:\n"
        f"{text}\nRewrite:"
    )
    ids = [int(x) for x in public.encode(prompt)]
    table = np.asarray(public.table("plaintext"), dtype=np.float32)
    for _ in range(int(max_new_tokens)):
        logits = _next_token_logprobs(public, table[np.asarray(ids[-2048:])], arm="plaintext")
        next_id = int(np.argmax(logits))
        ids.append(next_id)
    return public.decode(ids[len(public.encode(prompt)):])


def text_metrics(public: PublicDeployment, recovered: str, truth: str) -> dict:
    """Token overlap and a clearly labeled mean-embedding cosine proxy."""
    from ..metrics.recovery import bleu4, ngram_overlap_ratio

    r = [int(x) for x in public.encode(recovered)] if public.encode_fn else []
    t = [int(x) for x in public.encode(truth)] if public.encode_fn else []
    if not r or not t:
        return {"bleu4": 0.0, "ngram": 0.0, "mean_embedding_cosine_proxy": None}
    table = np.asarray(public.table("plaintext"), dtype=np.float32)
    e_r = table[np.asarray(r)].mean(axis=0)
    e_t = table[np.asarray(t)].mean(axis=0)
    cosine = float(e_r @ e_t / (np.linalg.norm(e_r) * np.linalg.norm(e_t) + 1e-12))
    return {
        "bleu4": float(bleu4(r, t)),
        "ngram": float(ngram_overlap_ratio(r, t, 2)),
        "mean_embedding_cosine_proxy": cosine,
    }
