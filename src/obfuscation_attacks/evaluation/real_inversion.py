"""Learned inversion on the real artifact: ridge and neural inverters (C02 / C09).

The unconditional denominator is the protocol rule (§4.3): every test position
stays in the Top-k denominator. A label outside the inverter's candidate set is
an explicit unrecoverable row (``-1``), never a dropped row. ``coverage`` and
``conditional_*`` are reported beside the unconditional number.

Formal IMA uses the full-vocabulary path (chunked over classes); a capped head is
a probe and is labelled ``candidate_mode="proposal"``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def collect_pairs(dep, texts: Sequence[str], *, arm: str = "ab",
                  layer: int | None = None) -> dict:
    """Observations and labels for a train/test split by request.

    Kept for the legacy probe path; the W1 public-view entry is the formal path.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    req_rows: list[np.ndarray] = []
    for i, text in enumerate(texts):
        wire = dep.observe(text, arm=arm, request_id=i, key_seed=i)
        if layer is None:
            X = np.asarray(wire.public.wire_vectors, dtype=np.float32)
        else:
            X = np.asarray(wire.public.hidden_states[dep.layers.index(layer)], dtype=np.float32)
        y = (wire.truth.plaintext_ids if arm == "plaintext" else wire.truth.permuted_ids)
        X_rows.append(X)
        y_rows.append(np.asarray(y, dtype=np.int64))
        req_rows.append(np.full(len(y), i, dtype=np.int64))
    return {
        "X": np.concatenate(X_rows, axis=0),
        "y": np.concatenate(y_rows),
        "request": np.concatenate(req_rows),
    }


def split_by_request(data: dict, *, test_fraction: float = 0.25, seed: int = 0) -> dict:
    """Split by request id so no token from a test request appears in training."""
    rng = np.random.default_rng(seed)
    reqs = np.unique(data["request"])
    shuffled = rng.permutation(reqs)
    n_test = max(1, int(round(len(reqs) * test_fraction)))
    test_reqs = set(shuffled[:n_test].tolist())
    is_test = np.array([int(r) in test_reqs for r in data["request"]])
    return {
        "X_train": data["X"][~is_test], "y_train": data["y"][~is_test],
        "X_test": data["X"][is_test], "y_test": data["y"][is_test],
        "n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
    }


def _topk_from_scores(scores: np.ndarray, classes: np.ndarray, k: int = 100) -> np.ndarray:
    k = max(1, min(k, scores.shape[1]))
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return classes[np.take_along_axis(idx, order, axis=1)]


def asr_at(cands: np.ndarray, truth: np.ndarray, ks=(1, 5, 10, 100)) -> dict:
    """Unconditional Top-k over every test position, plus conditional/coverage."""
    candidates = np.asarray(cands, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64).ravel()
    n = truth.size
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    covered = candidates[:, 0] >= 0
    out: dict = {}
    for k in ks:
        kk = min(k, candidates.shape[1])
        hit = (candidates[:, :kk] == truth[:, None]).any(axis=1)
        out[f"top{k}"] = {
            "value": float(hit.mean()) if n else None,
            "numerator": int(hit.sum()),
            "denominator": int(n),
        }
        if covered.any():
            out[f"conditional_top{k}"] = {
                "value": float(hit[covered].mean()),
                "numerator": int(hit[covered].sum()),
                "denominator": int(covered.sum()),
            }
        else:
            out[f"conditional_top{k}"] = {
                "value": None, "numerator": 0, "denominator": 0
            }
    out["coverage"] = float(covered.mean()) if n else None
    out["n_unrecoverable"] = int((~covered).sum())
    return out


def _mask_uncovered(cands: np.ndarray, y_test: np.ndarray, classes: np.ndarray) -> np.ndarray:
    allowed = set(int(c) for c in classes)
    mask = np.array([int(y) in allowed for y in y_test])
    out = np.array(cands, dtype=np.int64, copy=True)
    out[~mask] = -1
    return out


def _ridge_full_vocab(
    split: dict, *, lam: float, vocab_size: int, chunk: int, top_k: int
) -> dict:
    X_train = np.asarray(split["X_train"], dtype=np.float64)
    y_train = np.asarray(split["y_train"], dtype=np.int64)
    X_test = np.asarray(split["X_test"], dtype=np.float64)
    d = int(X_train.shape[1])
    XtX = X_train.T @ X_train + lam * np.eye(d)
    # X^T Y for all classes without materialising an n x V one-hot matrix.
    XtY = np.zeros((d, vocab_size), dtype=np.float64)
    keep = (y_train >= 0) & (y_train < vocab_size)
    for feature in range(d):
        XtY[feature] = np.bincount(
            y_train[keep], weights=X_train[keep, feature], minlength=vocab_size
        )[:vocab_size]
    k = max(1, min(top_k, vocab_size))
    n = X_test.shape[0]
    best_scores = np.full((n, k), -np.inf, dtype=np.float64)
    best_ids = np.full((n, k), -1, dtype=np.int64)
    for start in range(0, vocab_size, int(chunk)):
        stop = min(start + int(chunk), vocab_size)
        classes = np.arange(start, stop, dtype=np.int64)
        W = np.linalg.solve(XtX, XtY[:, start:stop])
        scores = X_test @ W
        combined_scores = np.concatenate([best_scores, scores], axis=1)
        combined_ids = np.concatenate(
            [best_ids, np.broadcast_to(classes, (n, len(classes)))], axis=1
        )
        keep_idx = np.argpartition(-combined_scores, k - 1, axis=1)[:, :k]
        best_scores = np.take_along_axis(combined_scores, keep_idx, axis=1)
        best_ids = np.take_along_axis(combined_ids, keep_idx, axis=1)
    order = np.argsort(-best_scores, axis=1)
    cands = np.take_along_axis(best_ids, order, axis=1)
    result = asr_at(cands, split["y_test"])
    result.update({
        "coverage": 1.0,
        "n_classes": int(vocab_size),
        "candidate_mode": "full",
        "full_vocab": True,
    })
    return result


def ridge_inverter(
    split: dict,
    *,
    lam: float = 1.0,
    n_classes: int = 20000,
    full_vocab: bool = False,
    vocab_size: int | None = None,
    chunk: int = 4096,
    top_k: int = 100,
    seed: int = 0,
) -> dict:
    """Closed-form ridge map; proposal head by default, full vocabulary on request."""
    if full_vocab:
        if vocab_size is None:
            raise ValueError("full_vocab=True requires vocab_size")
        return _ridge_full_vocab(
            split, lam=lam, vocab_size=int(vocab_size), chunk=chunk, top_k=top_k
        )

    rng = np.random.default_rng(seed)
    del rng
    y_train = np.asarray(split["y_train"], dtype=np.int64)
    uniq, counts = np.unique(y_train, return_counts=True)
    if len(uniq) > n_classes:
        uniq = uniq[np.argsort(-counts)[:n_classes]]
    cls_index = {int(c): i for i, c in enumerate(uniq)}
    keep = np.array([int(y) in cls_index for y in y_train])
    X = np.asarray(split["X_train"], dtype=np.float64)[keep]
    Y = np.zeros((int(keep.sum()), len(uniq)), dtype=np.float32)
    Y[np.arange(int(keep.sum())), [cls_index[int(y)] for y in y_train[keep]]] = 1.0
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    W = np.linalg.solve(XtX, X.T @ Y)
    scores = np.asarray(split["X_test"], dtype=np.float64) @ W
    cands = _topk_from_scores(scores, np.asarray(uniq, dtype=np.int64), k=top_k)
    cands = _mask_uncovered(cands, np.asarray(split["y_test"]), uniq)
    res = asr_at(cands, split["y_test"])
    res["n_classes"] = int(len(uniq))
    res["candidate_mode"] = "proposal"
    res["full_vocab"] = False
    return res


def mlp_inverter(
    split: dict,
    *,
    hidden: int = 512,
    epochs: int = 3,
    batch: int = 256,
    n_classes: int = 5000,
    full_vocab: bool = False,
    vocab_size: int | None = None,
    lr: float = 1e-3,
    seed: int = 0,
    top_k: int = 100,
) -> dict:
    """Neural inverter; a capped softmax head is proposal mode, full vocab on request."""
    import torch

    torch.manual_seed(seed)
    y_train = np.asarray(split["y_train"], dtype=np.int64)
    if full_vocab:
        if vocab_size is None:
            raise ValueError("full_vocab=True requires vocab_size")
        classes = np.arange(int(vocab_size), dtype=np.int64)
        cls_index = {int(c): int(c) for c in classes}
        keep = np.ones(len(y_train), dtype=bool)
        candidate_mode = "full"
    else:
        uniq, counts = np.unique(y_train, return_counts=True)
        if len(uniq) > n_classes:
            uniq = uniq[np.argsort(-counts)[:n_classes]]
        classes = np.asarray(uniq, dtype=np.int64)
        cls_index = {int(c): i for i, c in enumerate(classes)}
        keep = np.array([int(y) in cls_index for y in y_train])
        candidate_mode = "proposal"

    X = torch.from_numpy(np.asarray(split["X_train"], dtype=np.float32)[keep])
    y = torch.tensor([cls_index[int(t)] for t in y_train[keep]], dtype=torch.long)
    d = X.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, len(classes))
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad(set_to_none=True)
            out = net(X[idx])
            loss = lossf(out, y[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"    mlp epoch {epoch + 1}/{epochs} loss={total / max(1, n):.4f}", flush=True)
    with torch.no_grad():
        scores = net(torch.from_numpy(np.asarray(split["X_test"], dtype=np.float32))).numpy()
    cands = _topk_from_scores(scores, classes, k=top_k)
    if not full_vocab:
        cands = _mask_uncovered(cands, np.asarray(split["y_test"]), classes)
    res = asr_at(cands, split["y_test"])
    res["n_classes"] = int(len(classes))
    res["candidate_mode"] = candidate_mode
    res["full_vocab"] = bool(full_vocab)
    res["params"] = int(sum(p.numel() for p in net.parameters()))
    return res
