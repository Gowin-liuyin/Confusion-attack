"""Supervised known-pair substitution diagnostic, not a neural SDA implementation."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np


def make_substitution_pairs(
    sequences: Sequence[Sequence[int]], *, vocab_size: int, seed: int = 0, n_classes: int = 2000, held_out: int = 40,
) -> dict:
    """Random token substitution over the classes that actually occur (known ground truth)."""
    if held_out <= 0 or len(sequences) <= held_out or n_classes <= 0:
        raise ValueError("need positive n_classes and non-empty independent train and test splits")
    if any(not seq for seq in sequences):
        raise ValueError("empty sequences are not supported")
    cut = len(sequences) - held_out
    rng = np.random.default_rng(seed)
    counts: dict[int, int] = {}
    for seq in sequences[:cut]:
        for t in seq:
            counts[int(t)] = counts.get(int(t), 0) + 1
    classes = np.array(sorted(counts, key=lambda t: -counts[t])[:n_classes], dtype=np.int64)
    perm = rng.permutation(len(classes))
    # substitution only over the class set; tokens outside it are passed through (and masked out)
    cipher_of = {int(c): int(classes[perm[i]]) for i, c in enumerate(classes)}
    inverse = {v: k for k, v in cipher_of.items()}
    # compact vocabulary: the class set is remapped to 0..K-1 and everything else to K (OOV).
    # Without this the transformer carries a 128k-way softmax head and cannot learn from a few
    # hundred short sequences (measured: loss stuck at 10.2, BLEU 0).
    # ids are shifted by +1 so index 0 stays free for padding
    index_of = {int(c): i + 1 for i, c in enumerate(classes)}
    oov = len(classes) + 1
    def _compact(seq):
        return [index_of.get(int(t), oov) for t in seq]
    plain_compact = [_compact(seq) for seq in sequences]
    cipher_compact = [[index_of.get(int(cipher_of.get(int(t), int(t))), oov) for t in seq]
                      for seq in sequences]
    return {"plain": plain_compact, "cipher": cipher_compact,
            "cipher_of": cipher_of, "inverse": inverse, "classes": classes,
            "vocab_size": oov + 1, "oov_index": oov,
            "original_plain": [list(map(int, seq)) for seq in sequences], "train_count": cut}


def _batch(pairs: list[tuple[list[int], list[int]]], idx: np.ndarray, pad: int):
    import torch

    xs = [pairs[i][0] for i in idx]
    ys = [pairs[i][1] for i in idx]
    L = max(len(x) for x in xs)
    X = torch.full((len(xs), L), pad, dtype=torch.long)
    Y = torch.full((len(ys), L), pad, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


def train_sda_transformer(
    data: dict, *, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
    epochs: int = 10, batch: int = 32, lr: float = 3e-4, seed: int = 0,
    held_out: int = 40, verbose: bool = True,
) -> dict:
    """Train cipher->plain on all but ``held_out`` sequences; return BLEU-4 on the holdout."""
    import torch

    torch.manual_seed(seed)
    pairs = list(zip(data["cipher"], data["plain"]))
    n = len(pairs)
    cut = n - held_out
    if held_out <= 0 or cut <= 0 or data.get("train_count") != cut:
        raise ValueError("train/test split must match the split used to construct classes")
    if epochs <= 0 or batch <= 0:
        raise ValueError("epochs and batch must be positive")
    train, test = pairs[:cut], pairs[cut:]
    vocab = int(data.get("vocab_size", max(max(s) for s in data["cipher"] + data["plain"]) + 1))
    pad = 0            # ids from make_substitution_pairs start at 1, so 0 is free for padding

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d_model, padding_idx=pad)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                batch_first=True, dropout=0.0)
            self.enc = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = torch.nn.Linear(d_model, vocab)

        def forward(self, x):
            h = self.emb(x)
            L = x.shape[1]
            mask = torch.ones((L, L), dtype=torch.bool, device=x.device).triu(1)
            h = self.enc(h, mask=mask, src_key_padding_mask=(x == pad))
            return self.head(h)

    net = Tiny()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=pad)
    t0 = time.time()
    for epoch in range(epochs):
        total, seen = 0.0, 0
        perm = np.random.default_rng(seed + epoch).permutation(cut)
        for start in range(0, cut, batch):
            idx = perm[start:start + batch]
            X, Y = _batch(train, idx, pad)
            opt.zero_grad(set_to_none=True)
            logits = net(X)
            loss = lossf(logits.reshape(-1, vocab), Y.reshape(-1))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        if verbose:
            print(f"    sda epoch {epoch+1}/{epochs} loss={total/max(1,seen):.4f}", flush=True)
    net.eval()
    exact = hits = covered = count = 0
    classes = data["classes"]
    with torch.no_grad():
        for (x, _), truth in zip(test, data["original_plain"][cut:]):
            X = torch.tensor([x])
            pred = net(X).argmax(-1)[0].tolist()[:len(truth)]
            decoded = [int(classes[t - 1]) if 1 <= t <= len(classes) else None for t in pred]
            matches = [p == t for p, t in zip(decoded, truth)]
            hits += sum(matches)
            count += len(truth)
            exact += int(len(matches) == len(truth) and all(matches))
    covered = sum(t != data["oov_index"] for seq in data["plain"][cut:] for t in seq)
    return {"bleu4_mean": None, "bleu4_status": "not_evaluated",
            "token_accuracy": hits / count if count else None,
            "token_hits": hits, "token_count": count,
            "class_coverage": covered / count if count else None,
            "exact_match": exact / len(test) if test else None,
            "n_train": cut, "n_test": len(test), "params": int(sum(p.numel() for p in net.parameters())),
            "seconds": round(time.time() - t0, 1)}
