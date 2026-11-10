"""Real corpora for the plan's strength floors (``privacy_test_plan_20261106`` §6.9).

The plan forbids entering the main table on data whose provenance is not
recorded, so this module loads the *local* snapshots, applies the plan's filters
verbatim, and returns a manifest (source path, sha256, filter, counts, seed,
split) alongside the token ids.

Datasets wired here:

``alpaca``
    ``references/online/stanford_alpaca-alpaca_data.json`` (tatsu-lab/stanford_alpaca
    ``alpaca_data.json``, sha pinned in ``alpaca-api-20261017.json``). The plan's
    KNN protocol (§6.9(1)) is "instruction + input", 20-512 characters, 1,000
    random rows, disjoint from the VMA pool; the VMA pool (§6.9(2)) uses the same
    source. Both are produced from one deterministic split so that the disjointness
    the plan demands is structural rather than asserted.

``reference_corpus``
    The TFMA reference side (§6.6) wants >=4.45M tokens with zero document overlap
    with the private stream. Only Alpaca-class text is available offline here, so
    the reference corpus is built from the same snapshot with a *document-level*
    split from the private stream and every substitute source is recorded
    (``reference_substitute: true``). The plan explicitly allows substitutes when
    CCI3/MedDialog snapshots are unavailable, as long as they are declared and the
    token counts are reported with the target tokenizer.

Token counting is always done with the target tokenizer (plan §6.6: never
whitespace words), which is why the tokenizer is passed in rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALPACA = ROOT.parent / "OSNIP_package_20261101" / "references" / "online" / "stanford_alpaca-alpaca_data.json"
CLUSTER_ALGORITHM = "char5_containment80_cluster_v1"
CLUSTERED_SPLIT = "char5_containment80_representative_v2"


def cluster_near_duplicate_texts(texts: Sequence[str], *, max_posting_visits: int = 20_000_000) -> tuple[list[list[int]], dict]:
    """Connected components under the recorded character-shingle heuristic.

    This checks every pair that can satisfy the heuristic, rather than relying
    on approximate nearest-neighbour recall. It is not proof that all semantic
    paraphrases or other kinds of dependence have been detected.
    """
    if max_posting_visits < 0:
        raise ValueError("max_posting_visits must be non-negative")
    signatures = []
    for text in texts:
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if len(normalized) < 5:
            raise ValueError("near-duplicate isolation needs at least five normalized characters")
        signatures.append(frozenset(
            int.from_bytes(hashlib.blake2b(normalized[i:i + 5].encode("utf-8"), digest_size=16).digest(), "big")
            for i in range(len(normalized) - 4)
        ))
    frequencies = Counter(shingle for signature in signatures for shingle in signature)
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    postings: dict[int, list[int]] = {}
    visits = comparisons = links = 0
    for index in sorted(range(len(texts)), key=lambda i: (len(signatures[i]), i)):
        signature = signatures[index]
        candidates: set[int] = set()
        for shingle in signature:
            matches = postings.get(shingle, ())
            visits += len(matches)
            if visits > max_posting_visits:
                raise ValueError("near-duplicate scan exceeded its posting limit; split not produced")
            candidates.update(matches)
        for other in sorted(candidates):
            if root(index) == root(other):
                continue
            comparisons += 1
            # Prior documents have no more shingles than this one. A common
            # member of their rare prefix is necessary at 80% containment.
            intersection = len(signature & signatures[other])
            if 5 * intersection >= 4 * len(signatures[other]):
                a, b = root(index), root(other)
                parent[max(a, b)] = min(a, b)
                links += 1
        required = (4 * len(signature) + 4) // 5
        prefix_size = len(signature) - required + 1
        for shingle in sorted(signature, key=lambda value: (frequencies[value], value))[:prefix_size]:
            postings.setdefault(shingle, []).append(index)
    grouped: dict[int, list[int]] = {}
    for index in range(len(texts)):
        grouped.setdefault(root(index), []).append(index)
    clusters = sorted(grouped.values(), key=lambda group: group[0])
    return clusters, {
        "algorithm": CLUSTER_ALGORITHM,
        "normalization": "Unicode NFKC, casefold, collapse whitespace",
        "shingle": "distinct character 5-grams, blake2b-128",
        "rule": "intersection / smaller shingle set >= 0.8; transitive closure",
        "n_documents": len(texts), "n_clusters": len(clusters),
        "max_cluster_size": max(map(len, clusters), default=0),
        "candidate_comparisons": comparisons, "posting_visits": visits,
        "max_posting_visits": max_posting_visits, "matched_edges": links,
        "limitation": "heuristic detects this shingle relation, not all paraphrases or shared sources",
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _text_of(row: dict) -> str:
    """Alpaca text construction: instruction + newline + input (plan §6.9(1))."""
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return f"{instruction}\n{inp}" if inp else instruction


@dataclass
class TextSet:
    """A filtered text set plus the manifest the plan requires."""

    name: str
    rows: list[dict]                     # {"text", "source_index", "chars"}
    manifest: dict = field(default_factory=dict)

    def texts(self) -> list[str]:
        return [r["text"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


def load_alpaca(
    path: Path = DEFAULT_ALPACA,
    *,
    seed: int = 20261106,
    n_knn: int = 1000,
    n_vma: int = 200,
    n_robust: int = 0,
    min_chars: int = 20,
    max_chars: int = 512,
    pool_cap: Optional[int] = None,
    split_strategy: str = "exact_v1",
) -> dict[str, TextSet]:
    """Split Alpaca into the plan's KNN / VMA / key-robustness / TFMA roles.

    The split is index-based and seeded, so the roles are disjoint by
    construction (the plan's "与训练/VMA 池不相交" and "零文档重叠" checks).
    ``n_robust`` adds the independent 200-text key-robustness set after the VMA
    pool; the KNN and VMA prefixes are unchanged by that addition.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Alpaca snapshot not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("expected a JSON list of Alpaca rows")

    filtered: list[dict] = []
    dropped = {"short": 0, "long": 0, "duplicate": 0, "empty": 0}
    seen: set[str] = set()
    for idx, row in enumerate(raw):
        text = _text_of(row)
        if not text:
            dropped["empty"] += 1
            continue
        if len(text) < min_chars:
            dropped["short"] += 1
            continue
        if len(text) > max_chars:
            dropped["long"] += 1
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        filtered.append({"text": text, "source_index": idx, "chars": len(text), "sha256": key})

    if split_strategy not in {"exact_v1", CLUSTER_ALGORITHM, CLUSTERED_SPLIT}:
        raise ValueError(f"unknown corpus split strategy: {split_strategy}")
    rng = np.random.default_rng(seed)
    need = n_knn + n_vma + n_robust
    if min(n_knn, n_vma, n_robust) < 0:
        raise ValueError("role sizes must be non-negative")
    if need >= len(filtered):
        raise ValueError(f"need at least {need + 2} rows, only {len(filtered)} pass the 20-512 char filter")
    if split_strategy == "exact_v1":
        order = rng.permutation(len(filtered))
        knn_idx = order[:n_knn]
        vma_idx = order[n_knn:n_knn + n_vma]
        robust_idx = order[n_knn + n_vma:need]
        rest_idx = order[need:]
        split = int(len(rest_idx) * 0.6)
        ref_idx, private_idx = rest_idx[:split], rest_idx[split:]
        if pool_cap is not None:
            ref_idx = ref_idx[:pool_cap]
        isolation = {"algorithm": "exact_v1", "near_duplicate_status": "not_checked"}
    else:
        groups, isolation = cluster_near_duplicate_texts([row["text"] for row in filtered])
        shuffled = [groups[int(i)] for i in rng.permutation(len(groups))]
        if split_strategy == CLUSTERED_SPLIT:
            if len(groups) < need + 2:
                raise ValueError("insufficient distinct near-duplicate clusters to fill the requested roles")
            # Repeated templates cannot count as independent documents.
            order = [group[0] for group in shuffled]
            knn_idx = order[:n_knn]
            vma_idx = order[n_knn:n_knn + n_vma]
            robust_idx = order[n_knn + n_vma:need]
            rest = order[need:]
            target = int(len(rest) * 0.6)
            if pool_cap is not None:
                if pool_cap < 0:
                    raise ValueError("pool_cap must be non-negative")
                target = min(target, int(pool_cap))
            ref_idx, private_idx = rest[:target], rest[target:]
            isolation = {**isolation, "representative_rule": "first source-ordered document in each component",
                         "discarded_near_duplicates": len(filtered) - len(groups)}
        else:
            # Rebuild historical v1 artifacts without changing their grouping.
            def take_groups(available, desired):
                selected, remaining = [], []
                for group in available:
                    if len(group) <= desired:
                        selected.extend(group)
                        desired -= len(group)
                    else:
                        remaining.append(group)
                if desired:
                    raise ValueError("cannot fill requested role size without splitting a near-duplicate cluster")
                return selected, remaining

            knn_idx, shuffled = take_groups(shuffled, n_knn)
            vma_idx, shuffled = take_groups(shuffled, n_vma)
            robust_idx, shuffled = take_groups(shuffled, n_robust)
            target = int(sum(map(len, shuffled)) * 0.6)
            if pool_cap is not None:
                if pool_cap < 0:
                    raise ValueError("pool_cap must be non-negative")
                target = min(target, int(pool_cap))
            ref_idx, private_idx = [], []
            for group in shuffled:
                (ref_idx if len(ref_idx) + len(group) <= target else private_idx).extend(group)
        isolation = {**isolation, "near_duplicate_status": "clustered_by_recorded_heuristic"}

    def _take(idx: Iterable[int]) -> list[dict]:
        return [filtered[int(i)] for i in idx]

    role_indices = {"knn": knn_idx, "vma": vma_idx, "robust": robust_idx,
                    "reference": ref_idx, "private": private_idx}
    split_hash = hashlib.sha256(json.dumps(
        {name: [filtered[int(i)]["sha256"] for i in indices] for name, indices in role_indices.items()},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()

    manifest_common = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "source_snapshot": "tatsu-lab/stanford_alpaca alpaca_data.json",
        "filter": {"min_chars": min_chars, "max_chars": max_chars, "text_rule": "instruction + '\\n' + input"},
        "seed": seed,
        "n_raw": len(raw),
        "n_filtered": len(filtered),
        "dropped": dropped,
        "disjoint_by_construction": True,
        "split_strategy": split_strategy,
        "split_sha256": split_hash,
        "isolation": isolation,
    }
    sets = {
        "knn": TextSet(
            "alpaca_knn", _take(knn_idx),
            {**manifest_common, "role": "KNN (plan §6.9(1))", "n": len(knn_idx)},
        ),
        "vma": TextSet(
            "alpaca_vma", _take(vma_idx),
            {**manifest_common, "role": "VMA pool / ISA prompts (plan §6.9(2),(4))", "n": len(vma_idx)},
        ),
        "robust": TextSet(
            "alpaca_robust", _take(robust_idx),
            {**manifest_common,
             "role": "KNN key robustness (protocol §8.1: independent 200 texts x 5 fresh keys)",
             "n": len(robust_idx)},
        ),
        "reference": TextSet(
            "alpaca_reference", _take(ref_idx),
            {**manifest_common, "role": "TFMA reference corpus (plan §6.6)", "n": len(ref_idx)},
        ),
        "private": TextSet(
            "alpaca_private", _take(private_idx),
            {**manifest_common, "role": "TFMA / linkage private stream (plan §6.6, zero overlap)", "n": len(private_idx)},
        ),
    }
    return sets


def reference_counts(
    texts: Sequence[str],
    encode: Callable[[str], Sequence[int]],
    *,
    vocab_size: int,
) -> tuple[dict[int, int], int, dict[int, int]]:
    """Token-frequency table for the frequency attacks (TFMA/SDA/linkage).

    Returns ``(counts, n_tokens, doc_freq)`` where ``doc_freq`` is the number of
    documents containing a token (used for document-level disjointness checks).
    """
    counts: dict[int, int] = {}
    doc_freq: dict[int, int] = {}
    n_tokens = 0
    for text in texts:
        ids = list(encode(text))
        n_tokens += len(ids)
        for t in ids:
            counts[int(t)] = counts.get(int(t), 0) + 1
        for t in set(int(x) for x in ids):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    for t, c in counts.items():
        if not 0 <= t < vocab_size:
            raise ValueError(f"token id {t} outside vocab {vocab_size}")
    return counts, n_tokens, doc_freq


def overlap_report(a: Sequence[str], b: Sequence[str]) -> dict:
    """Exact-text, document overlap between the two roles (plan: must be 0)."""
    ha = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in a}
    hb = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in b}
    return {"n_a": len(ha), "n_b": len(hb), "intersection": len(ha & hb)}


def audit_corpus_isolation(corpus: dict[str, TextSet]) -> dict:
    """Recompute the frozen role hash and detect cross-role shingle clusters."""
    names = ("knn", "vma", "robust", "reference", "private")
    if any(name not in corpus for name in names):
        raise ValueError("all five corpus roles are required for isolation audit")
    roles, texts, hashes = [], [], {}
    for name in names:
        documents = corpus[name].texts()
        hashes[name] = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents]
        roles.extend([name] * len(documents))
        texts.extend(documents)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    clusters, details = cluster_near_duplicate_texts(texts)
    cross_role = sum(len({roles[index] for index in group}) > 1 for group in clusters)
    within_role = sum(len(group) > 1 and len({roles[index] for index in group}) == 1 for group in clusters)
    return {"split_sha256": digest, "cross_role_clusters": cross_role,
            "within_role_duplicate_clusters": within_role,
            "role_counts": {name: len(hashes[name]) for name in names},
            "role_distinct_hashes": {name: len(set(hashes[name])) for name in names},
            "passed": cross_role == 0 and within_role == 0, "heuristic": details}
