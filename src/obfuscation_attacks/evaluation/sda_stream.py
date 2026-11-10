"""SDA on the real deployment stream (protocol §8.2; W4/W6 bridge).

The original SDA-2021 (seq2seq, frequency-rank source) and SDA-2023 (causal LM,
recurrence source) classes already train on self-built substitution pairs from an
attacker-owned auxiliary plaintext corpus and never use the target key. This
module supplies the missing bridge:

* a continuous wire is quantised into stable symbols (the same symbol step as
  TFMA/linkage) and mapped to a compact cipher alphabet;
* the original model is fitted once per (seed, epochs, beam) setting and decodes
  every request in the stream (instead of training a new model per request);
* a known-substitution positive control and the real private stream are reported
  separately.

Document roles are frozen by hash BEFORE any training pair is synthesised: the
attacker-owned auxiliary corpus is split into disjoint training / validation /
held-out-control documents, and ``fit_sda`` only ever receives the training
role. The positive control therefore comes from a document the model never
trained on (defect S07: the control used to be ``reference_ids[-1]``, i.e. a
training document, which can pass by memorisation because the frequency-rank and
recurrence encodings preserve repetition structure). The evaluated private texts
form a fourth ``private`` role that is checked against every auxiliary role with
the same exact / shingle / short-document rules, so the runner refuses to fit on
a document that is also being scored.

No target plaintext-cipher pairs are used for training. The fitting loop is a
transcription of the original class; ``tests/unit/test_sda_stream.py`` checks
that its output equals a direct ``SDA2021/SDA2023`` run on one cipher with the
same seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from ..attacks.sda import encodings as enc
from ..attacks.sda.sda2021_seq2seq import (
    Seq2SeqCipherModel,
    Seq2SeqConfig,
    _make_batches as _make_batches_2021,
    seq2seq_target_nll,
    synthesize_pairs as synthesize_pairs_2021,
)
from ..attacks.sda.sda2023_recurrence_clm import (
    CausalCipherLM,
    CausalLMConfig,
    _make_batches as _make_batches_2023,
    clm_losses,
    synthesize_pairs as synthesize_pairs_2023,
)
from ..utils import seed_everything

METHODS = ("sda2021", "sda2023")


# ---------------------------------------------------------------------------
# S07: document-role isolation for the positive control
# ---------------------------------------------------------------------------
# The original bridge fitted every ``reference_ids`` entry and then used
# ``reference_ids[-1]`` as the known-substitution positive control. Regenerating
# the substitution key does not make such a control independent: both SDA source
# encodings preserve repetition structure, so a trained-on document can be
# deciphered from memorisation instead of from the cipher. Roles are frozen by
# hash BEFORE pair synthesis and ``fit_sda`` only receives the training role.

SPLIT_RULE = (
    "seeded hash-rank split over usable reference documents: rank each document "
    "by sha256(uint64_le(seed) || sha256(canonical int64_le token bytes)) "
    "ascending (ties by input order); the tail becomes the control role, the "
    "preceding slice the validation role, the remainder the training role. "
    "Explicit control_texts / validation_texts replace the role they name."
)
SHINGLE_SIZE = 5
NEAR_DUPLICATE_JACCARD = 0.8
NEAR_DUPLICATE_CONTAINMENT = 0.8
ROLE_LISTING_CAP = 64


def _document_bytes(token_ids: Sequence[int]) -> bytes:
    """Canonical little-endian int64 bytes for a token-id sequence."""
    array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    return np.ascontiguousarray(array.astype("<i8", copy=False)).tobytes()


def document_sha256(token_ids: Sequence[int]) -> str:
    """Exact-overlap identity: sha256 over the canonical int64 token bytes."""
    return hashlib.sha256(_document_bytes(token_ids)).hexdigest()


def _split_rank(token_ids: Sequence[int], seed: int) -> str:
    """Seeded rank key; the split is reproducible for a (seed, corpus) pair."""
    seed_bytes = int(seed % (1 << 64)).to_bytes(8, "little", signed=False)
    content = bytes.fromhex(document_sha256(token_ids))
    return hashlib.sha256(seed_bytes + content).hexdigest()


def shingle_hashes(
    token_ids: Sequence[int], *, size: int = SHINGLE_SIZE
) -> frozenset[int]:
    """Normalised near-duplicate signature: hashed overlapping token shingles."""
    array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    n = int(array.size)
    if n == 0:
        return frozenset()
    width = max(1, min(int(size), n))
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(
                _document_bytes(array[start : start + width]), digest_size=8
            ).digest(),
            "little",
        )
        for start in range(n - width + 1)
    )


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def assign_document_roles(
    documents: Sequence[Sequence[int]],
    *,
    seed: int = 0,
    control_fraction: float = 0.1,
    validation_fraction: float = 0.1,
    n_control: int | None = None,
    n_validation: int | None = None,
    require_control: bool = True,
) -> dict[str, list[int]]:
    """Deterministically split document indices into disjoint roles.

    ``n_control`` / ``n_validation`` override the fraction-derived counts (pass
    ``0`` when an explicit corpus already fills that role). The split is a pure
    function of ``(documents, seed, fractions)``: ranked by
    ``sha256(seed || document_sha256)`` ascending with the input index as a
    stable tiebreak, the tail is control, the preceding slice validation and the
    remainder training. See :data:`SPLIT_RULE`.
    """
    n = len(documents)
    if not 0.0 <= float(control_fraction) < 1.0:
        raise ValueError("control_fraction must be in [0, 1)")
    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    want_control = (
        max(1, _round_half_up(float(control_fraction) * n))
        if n_control is None
        else int(n_control)
    )
    want_validation = (
        _round_half_up(float(validation_fraction) * n)
        if n_validation is None
        else int(n_validation)
    )
    want_control = min(max(want_control, 0), max(n - 1, 0))
    want_validation = min(max(want_validation, 0), max(n - want_control - 1, 0))
    n_train = n - want_control - want_validation
    if n_train < 1 or (require_control and want_control < 1):
        raise ValueError(
            "cannot build an isolated SDA positive control: "
            f"{n} usable reference document(s) yield {n_train} training / "
            f"{want_validation} validation / {want_control} held-out control "
            "document(s). An isolated control needs >= 1 training document AND "
            ">= 1 disjoint control document; pass more reference_texts or "
            "explicit disjoint control_texts. Refusing to fall back to training "
            "documents."
        )
    ordered = sorted(range(n), key=lambda i: (_split_rank(documents[i], seed), i))
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train : n_train + want_validation],
        "control": ordered[n_train + want_validation :],
    }


def _materialise_documents(dep, texts: Sequence[str], source: str) -> tuple[list[dict], int]:
    """Encode texts into non-empty token-id documents; count dropped empties."""
    documents: list[dict] = []
    n_empty = 0
    for index, text in enumerate(texts):
        ids = np.asarray(dep.encode(text), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            n_empty += 1
            continue
        documents.append({"source": source, "index": index, "tokens": ids})
    return documents, n_empty


def _document_record(document: dict, *, role: str) -> dict:
    tokens = np.asarray(document["tokens"], dtype=np.int64).reshape(-1)
    return {
        "role": role,
        "source": str(document["source"]),
        "index": int(document["index"]),
        "tokens": tokens,
        "length": int(tokens.size),
        "sha256": document_sha256(tokens),
        "shingles": shingle_hashes(tokens),
    }


def check_document_isolation(
    roles: dict[str, Sequence[dict]],
    *,
    jaccard_threshold: float = NEAR_DUPLICATE_JACCARD,
    containment_threshold: float = NEAR_DUPLICATE_CONTAINMENT,
    max_examples: int = 10,
) -> dict:
    """Cross-role exact and near-duplicate check over frozen role records.

    Exact overlap: identical :func:`document_sha256`. Near-duplicate: 5-token
    shingle sets where Jaccard ``|A&B| / |A|B|`` >= threshold, or containment
    ``|A&B| / min(|A|, |B|)`` >= threshold (a short control document wholly
    contained in a long training document is a memorisable window even when the
    Jaccard value is diluted). Exact hits are reported separately and excluded
    from the near-duplicate count. Candidate pairs are generated through an
    inverted shingle index, so the scan is linear in the total shingle count
    plus the cross-role posting hits rather than quadratic in document count.

    Short-document pairs with disjoint token sets are provably disjoint;
    pairs sharing tokens remain unverifiable and fail closed unless exact.

    The evaluated private texts form a fourth role (``private``) and are checked
    against every auxiliary role (train / validation / control) with the same
    exact, near-duplicate and short-document rules: the evaluated document must
    never be an auxiliary-corpus document.
    """
    role_pairs = (
        ("train", "validation"),
        ("train", "control"),
        ("validation", "control"),
        ("train", "private"),
        ("validation", "private"),
        ("control", "private"),
    )

    def _label(document: dict) -> dict:
        return {
            "role": document["role"],
            "source": document["source"],
            "index": int(document["index"]),
            "sha256": document["sha256"],
        }

    exact_examples: list[dict] = []
    near_examples: list[dict] = []
    short_examples: list[dict] = []
    exact_count = 0
    near_count = 0
    short_pair_count = 0
    checked_pairs = 0
    max_jaccard = 0.0
    max_containment = 0.0
    by_role_pair: dict[str, dict] = {}
    for left_role, right_role in role_pairs:
        left_docs = list(roles.get(left_role) or ())
        right_docs = list(roles.get(right_role) or ())
        checked_pairs += len(left_docs) * len(right_docs)

        by_hash: dict[str, list[dict]] = {}
        for document in left_docs:
            by_hash.setdefault(document["sha256"], []).append(document)
        exact_here = 0
        exact_short_here = 0
        for document in right_docs:
            for other in by_hash.get(document["sha256"], ()):
                exact_here += 1
                if (
                    other["length"] < SHINGLE_SIZE
                    or document["length"] < SHINGLE_SIZE
                ):
                    exact_short_here += 1
                if len(exact_examples) < max_examples:
                    exact_examples.append(
                        {
                            "left": _label(other),
                            "right": _label(document),
                            "kind": "exact",
                        }
                    )
        exact_count += exact_here

        unverifiable_here = 0
        short_left = [d for d in left_docs if d["length"] < SHINGLE_SIZE]
        long_left = [d for d in left_docs if d["length"] >= SHINGLE_SIZE]
        short_right = [d for d in right_docs if d["length"] < SHINGLE_SIZE]
        short_pairs = ((left, right) for left in short_left for right in right_docs)
        short_pairs_right = ((left, right) for left in long_left for right in short_right)
        from itertools import chain
        for left, right in chain(short_pairs, short_pairs_right):
            if left["sha256"] == right["sha256"]:
                continue  # already counted as exact
            # No shared token proves there is no shared non-empty token span.
            if set(map(int, left["tokens"])).isdisjoint(map(int, right["tokens"])):
                continue
            unverifiable_here += 1
            if len(short_examples) < max_examples:
                short_doc, other = (left, right) if left["length"] < SHINGLE_SIZE else (right, left)
                short_examples.append({
                    "short_document": {**_label(short_doc), "length": int(short_doc["length"])},
                    "compared_with": _label(other),
                    "reason": f"document shorter than SHINGLE_SIZE={SHINGLE_SIZE} shares tokens; near-duplicate status cannot be established",
                })
        short_pair_count += unverifiable_here

        # Near-duplicate scan over documents that can carry a full-width
        # shingle; pairs involving shorter documents are unverifiable above.
        long_left = [
            document for document in left_docs if document["length"] >= SHINGLE_SIZE
        ]
        long_right = [
            document for document in right_docs if document["length"] >= SHINGLE_SIZE
        ]
        if len(long_right) > len(long_left):
            index_docs, query_docs = long_right, long_left
        else:
            index_docs, query_docs = long_left, long_right
        postings: dict[int, list[int]] = {}
        for position, document in enumerate(index_docs):
            for shingle in document["shingles"]:
                postings.setdefault(int(shingle), []).append(position)
        near_here = 0
        for query in query_docs:
            counts: Counter = Counter()
            query_shingles = query["shingles"]
            for shingle in query_shingles:
                hits = postings.get(int(shingle))
                if hits:
                    for position in hits:
                        counts[position] += 1
            for position, intersection in counts.items():
                other = index_docs[position]
                if other["sha256"] == query["sha256"]:
                    continue  # exact duplicate, reported above
                union = len(query_shingles) + len(other["shingles"]) - intersection
                jaccard = intersection / union if union else 1.0
                smaller = min(len(query_shingles), len(other["shingles"]))
                containment = intersection / smaller if smaller else 1.0
                max_jaccard = max(max_jaccard, jaccard)
                max_containment = max(max_containment, containment)
                if jaccard >= jaccard_threshold or containment >= containment_threshold:
                    near_count += 1
                    near_here += 1
                    if len(near_examples) < max_examples:
                        near_examples.append(
                            {
                                "left": _label(other),
                                "right": _label(query),
                                "jaccard": round(jaccard, 6),
                                "containment": round(containment, 6),
                                "kind": "near_duplicate",
                            }
                        )
        by_role_pair[f"{left_role}~{right_role}"] = {
            "checked_pairs": len(left_docs) * len(right_docs),
            "exact": int(exact_here),
            "near_duplicate": int(near_here),
            "unverifiable_short": int(unverifiable_here),
        }
    return {
        "role_pairs_checked": [list(pair) for pair in role_pairs],
        "checked_pairs": int(checked_pairs),
        "by_role_pair": by_role_pair,
        "exact": {
            "definition": "identical sha256 over canonical little-endian int64 token bytes",
            "collisions": int(exact_count),
            "examples": exact_examples[:max_examples],
            "passed": exact_count == 0,
        },
        "near_duplicate": {
            "definition": (
                f"{SHINGLE_SIZE}-token shingle sets; violation when Jaccard >= "
                "jaccard_threshold or containment (intersection / smaller set) "
                ">= containment_threshold"
            ),
            "shingle_size": SHINGLE_SIZE,
            "jaccard_threshold": float(jaccard_threshold),
            "containment_threshold": float(containment_threshold),
            "collisions": int(near_count),
            "max_jaccard_observed": round(max_jaccard, 6),
            "max_containment_observed": round(max_containment, 6),
            "examples": near_examples[:max_examples],
            "passed": near_count == 0,
        },
        "unverifiable_short_documents": {
            "definition": (
                f"cross-role pairs involving a document shorter than "
                f"SHINGLE_SIZE={SHINGLE_SIZE} tokens and sharing at least one token; "
                "near-duplicate status cannot be established, so the check fails closed"
            ),
            "shingle_size": SHINGLE_SIZE,
            "pairs": int(short_pair_count),
            "examples": short_examples[:max_examples],
            "passed": short_pair_count == 0,
        },
        "passed": exact_count == 0 and near_count == 0 and short_pair_count == 0,
    }


def cluster_control_documents(documents: Sequence[dict]) -> tuple[list[str], dict]:
    """Group exact and near-duplicate controls before statistical resampling.

    Short documents sharing any token are conservatively grouped because the
    five-token test cannot verify them. Connected components prevent transitive
    near duplicates from being counted as separate clusters. This heuristic
    does not prove statistical independence of the resulting clusters.
    """
    docs = list(documents)
    parent = list(range(len(docs)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    by_hash: dict[str, int] = {}
    shingles: dict[int, list[int]] = {}
    has_short = any(doc["length"] < SHINGLE_SIZE for doc in docs)
    all_tokens: dict[int, list[int]] = {}
    short_tokens: dict[int, list[int]] = {}
    for index, doc in enumerate(docs):
        prior = by_hash.setdefault(doc["sha256"], index)
        union(index, prior)
        if doc["length"] >= SHINGLE_SIZE:
            matches = Counter()
            for shingle in doc["shingles"]:
                matches.update(shingles.get(int(shingle), ()))
            for other_index, intersection in matches.items():
                other = docs[other_index]
                union_size = len(doc["shingles"]) + len(other["shingles"]) - intersection
                smaller = min(len(doc["shingles"]), len(other["shingles"]))
                if ((intersection / union_size if union_size else 1.0) >= NEAR_DUPLICATE_JACCARD
                        or (intersection / smaller if smaller else 1.0) >= NEAR_DUPLICATE_CONTAINMENT):
                    union(index, other_index)
            for shingle in doc["shingles"]:
                shingles.setdefault(int(shingle), []).append(index)
        if has_short:
            token_set = set(map(int, doc["tokens"]))
            compare = all_tokens if doc["length"] < SHINGLE_SIZE else short_tokens
            for token in token_set:
                for other_index in compare.get(token, ()):
                    union(index, other_index)
            for token in token_set:
                all_tokens.setdefault(token, []).append(index)
                if doc["length"] < SHINGLE_SIZE:
                    short_tokens.setdefault(token, []).append(index)
    components: dict[int, list[int]] = {}
    for index in range(len(docs)):
        components.setdefault(find(index), []).append(index)
    labels = [""] * len(docs)
    for members in components.values():
        label = min(docs[index]["sha256"] for index in members)
        for index in members:
            labels[index] = label
    return labels, {
        "n_documents": len(docs), "n_distinct_hashes": len({d["sha256"] for d in docs}),
        "n_clusters": len(components), "shingle_size": SHINGLE_SIZE,
        "jaccard_threshold": NEAR_DUPLICATE_JACCARD,
        "containment_threshold": NEAR_DUPLICATE_CONTAINMENT,
        "short_rule": "shared token conservatively clusters documents when either is short",
        "independence": "heuristic grouping does not prove independent source documents",
    }


def _distinct_document_window_plan(corpus, *, count: int, width: int, seed: int,
                                   cluster_storage: str = "memory", cluster_temp_dir=None):
    """Freeze one fixed-length window per heuristically distinct source cluster."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("sampling_seed must be an unsigned 64-bit integer")
    if cluster_storage not in {"memory", "disk"}:
        raise ValueError("cluster_storage must be memory or disk")
    if cluster_storage == "disk" and width < SHINGLE_SIZE:
        raise ValueError("disk clustering requires window length at least SHINGLE_SIZE")
    if cluster_storage == "disk":
        from .sda_disk_cluster import cluster_long_documents_disk
        docs = [{"index": i, "tokens": seq, "length": len(seq), "sha256": document_sha256(seq)}
                for i, seq in enumerate(corpus) if len(seq) >= width]
        labels, clusters = cluster_long_documents_disk((doc["tokens"] for doc in docs), parent_dir=cluster_temp_dir)
    else:
        docs = [_document_record({"source": "reference", "index": i, "tokens": seq}, role="train")
                for i, seq in enumerate(corpus) if len(seq) >= width]
        labels, clusters = cluster_control_documents(docs)
    seed_bytes = seed.to_bytes(8, "little")
    order = sorted(range(len(docs)), key=lambda i: (
        hashlib.sha256(seed_bytes + bytes.fromhex(docs[i]["sha256"])).digest(), docs[i]["index"]))
    selected_clusters, selected_windows, plan = set(), set(), []
    for i in order:
        doc, label = docs[i], labels[i]
        if label in selected_clusters:
            continue
        start = int.from_bytes(hashlib.sha256(b"SDA-window" + seed_bytes + bytes.fromhex(doc["sha256"])).digest()[:8],
                               "little") % (doc["length"] - width + 1)
        content_hash = document_sha256(doc["tokens"][start:start + width])
        if content_hash in selected_windows:
            continue
        selected_clusters.add(label)
        selected_windows.add(content_hash)
        plan.append((doc["index"], start, width))
        if len(plan) == count:
            break
    if len(plan) != count:
        raise ValueError(f"insufficient distinct source clusters and windows: need {count}, found {len(plan)}")
    windows = [corpus[i][start:start + length] for i, start, length in plan]
    if cluster_storage == "disk":
        _, window_clusters = cluster_long_documents_disk(windows, parent_dir=cluster_temp_dir)
    else:
        window_docs = [_document_record({"source": "window", "index": i, "tokens": tokens}, role="train")
                       for i, tokens in enumerate(windows)]
        _, window_clusters = cluster_control_documents(window_docs)
    if window_clusters["n_clusters"] != count:
        raise ValueError("selected windows contain near duplicates; deduplicate sources before sampling")
    return plan, {"mode": "distinct_document_windows", "seed": seed, "window_length": width,
                  "selected_documents": len(plan), "unique_window_contents": len(selected_windows),
                  "eligible_documents": len(docs), "clustering": clusters,
                  "window_clustering": window_clusters,
                  "independence_status": "heuristic_deduplication_does_not_prove_source_independence"}


def _select_control_document(
    control_docs: Sequence[dict], *, min_tokens: int = 1
) -> dict:
    """Select a nonempty control after the cross-role isolation check."""
    for document in control_docs:
        if document["length"] >= min_tokens:
            return document
    raise ValueError("no nonempty held-out SDA control document")


def _role_summary(documents: Sequence[dict], *, cap: int = ROLE_LISTING_CAP) -> dict:
    """Counts plus a bounded hash listing and a role-level root digest."""
    hashes = [document["sha256"] for document in documents]
    root = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest() if hashes else None
    listed = list(documents[:cap])
    return {
        "count": len(documents),
        "root_sha256": root,
        "listed": len(listed),
        "truncated": len(documents) > len(listed),
        "documents": [
            {
                "role": document["role"],
                "source": document["source"],
                "index": int(document["index"]),
                "length": int(document["length"]),
                "sha256": document["sha256"],
            }
            for document in listed
        ],
    }


def _eligible_windows(length: int, *, min_len: int = 4, max_len: int = 32) -> int:
    """Number of (start, width) crops the pair generator can draw from a document."""
    return int(sum(max(0, int(length) - width + 1) for width in range(min_len, max_len + 1)))


def _frequency_order(values: Sequence[int]) -> list[int]:
    """Symbols by count descending, first appearance in ``values`` as tiebreak."""
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for position, value in enumerate(values):
        token = int(value)
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, position)
    return sorted(counts, key=lambda token: (-counts[token], first_seen[token]))


def _prior_baseline(
    control_cipher: Sequence[int],
    control_plain: Sequence[int],
    training_ids: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    prior_order: Sequence[int] | None = None,
) -> dict:
    """Trivial attacker baselines so the control can be read against chance.

    Neither baseline uses the held-out plaintext to build its prediction: the
    identity baseline is the raw cipher stream, and the frequency-prior baseline
    ranks the held-out *ciphertext* symbols and maps rank ``k`` to the ``k``-th
    most frequent plaintext symbol of the attacker-owned training corpus. The
    held-out plaintext is only used for scoring, exactly as for the model.
    """
    cipher = np.asarray(control_cipher, dtype=np.int64).reshape(-1)
    truth = np.asarray(control_plain, dtype=np.int64).reshape(-1)
    identity = _token_accuracy(cipher, truth)

    cipher_rank = {
        int(symbol): rank for rank, symbol in enumerate(_frequency_order(cipher.tolist()))
    }
    if prior_order is None:
        from itertools import chain
        prior_order = _frequency_order(chain.from_iterable(training_ids)) or [0]
    prior_order = list(prior_order)
    if len(cipher_rank) > len(prior_order):
        seen = set(prior_order)
        for token in range(vocab_size):
            if token not in seen:
                prior_order.append(token)
                if len(prior_order) >= len(cipher_rank):
                    break
    frequency_prior = [
        (prior_order[cipher_rank[int(symbol)]] if cipher_rank[int(symbol)] < len(prior_order) else -1)
        for symbol in cipher
    ]
    prior_scored = _token_accuracy(frequency_prior, truth)
    return {
        "identity": {
            **identity,
            "prediction": cipher.tolist(),
            "description": "assume cipher symbol == plaintext symbol",
        },
        "frequency_prior": {
            **prior_scored,
            "prediction": [int(value) for value in frequency_prior],
            "description": (
                "map the k-th most frequent held-out cipher symbol to the k-th "
                "most frequent training-plaintext symbol; unseen vocabulary IDs break ties in ID order; "
                "-1 denotes an unmapped symbol when the ciphertext has more types than the vocabulary"
            ),
        },
        "chance_uniform": 1.0 / max(int(vocab_size), 1),
    }



def _unconditional_bigram(training_ids: Sequence[Sequence[int]]):
    """Build a greedy public-corpus bigram LM without cipher or target access."""
    starts = Counter()
    transitions: dict[int, Counter] = {}
    for seq in training_ids:
        values = [int(t) for t in seq]
        if values:
            starts[values[0]] += 1
        for left, right in zip(values, values[1:]):
            transitions.setdefault(left, Counter())[right] += 1
    if not starts:
        raise ValueError("language prior needs non-empty public training documents")
    initial = starts.most_common(1)[0][0]
    next_token = {token: counts.most_common(1)[0][0] for token, counts in transitions.items()}

    def generate(length: int) -> list[int]:
        token = initial
        output = []
        for _ in range(length):
            output.append(token)
            token = next_token.get(token, initial)
        return output

    return generate


def _control_cell(row: dict, baseline: str, metric: str) -> tuple[int, int, int]:
    """Numerators are ordered so a positive difference favors the model."""
    reference = row["baselines"][baseline]
    if metric == "token_accuracy":
        accuracy = row["accuracy"]
        n = int(accuracy["denominator"])
        model, prior = int(accuracy["numerator"]), int(reference["numerator"])
        if n <= 0 or not (0 <= model <= n and 0 <= prior <= n):
            raise ValueError("invalid paired control counts")
        return model, prior, n
    if metric == "token_edit_error":
        model, prior = row["edit_error"], reference["edit_error"]
        n = int(model["reference_tokens"])
        model_errors, prior_errors = int(model["edits"]), int(prior["edits"])
        if n <= 0 or int(prior["reference_tokens"]) != n or min(model_errors, prior_errors) < 0:
            raise ValueError("invalid paired control edit counts")
        return prior_errors, model_errors, n
    raise ValueError("unknown control comparison metric")


def _paired_control_interval(rows: Sequence[dict], baseline: str, *, seed: int = 0,
                             metric: str = "token_accuracy") -> dict:
    """Paired document bootstrap, conditional on the one fixed control key."""
    clusters: dict[str, list[int]] = {}
    documents = {row["source_sha256"] for row in rows}
    clustered = all("source_cluster" in row for row in rows)
    for row in rows:
        label = row.get("source_cluster", row["source_sha256"])
        cluster = clusters.setdefault(label, [0, 0, 0])
        for index, amount in enumerate(_control_cell(row, baseline, metric)):
            cluster[index] += amount
    values = np.asarray(list(clusters.values()), dtype=np.float64).reshape(-1, 3)
    if len(values) < 2 or not values[:, 2].sum():
        return {"status": "not_evaluated", "n_documents": len(documents), "n_clusters": len(values),
                "reason": "at least two independent document clusters are required", "n_boot": 0}
    point = float((values[:, 0].sum() - values[:, 1].sum()) / values[:, 2].sum())
    rng = np.random.default_rng(seed)
    samples = np.empty(2000)
    for i in range(len(samples)):
        drawn = values[rng.integers(0, len(values), size=len(values))]
        samples[i] = (drawn[:, 0].sum() - drawn[:, 1].sum()) / drawn[:, 2].sum()
    return {"status": "descriptive", "difference": point,
            "lo": float(np.quantile(samples, .025)), "hi": float(np.quantile(samples, .975)),
            "n_documents": len(documents), "n_clusters": len(values), "n_boot": len(samples),
            "resampling_unit": "near-duplicate document cluster" if clustered else "distinct document hash",
            "conditional_on": "one fixed key",
            "document_independence": ("heuristic grouping does not prove independence" if clustered else
                                      "exact duplicates grouped; within-role near-duplicates unverified"),
            "metric": metric}


def _crossed_control_interval(rows: Sequence[dict], baseline: str, *, seed: int = 0,
                              metric: str = "token_accuracy") -> dict:
    """Paired crossed document/key bootstrap; duplicates are one document cluster."""
    cells = {}
    first_values = {}
    doc_clusters = {}
    clustered = all("source_cluster" in row for row in rows)
    for row in rows:
        doc, replica = row["source_sha256"], row["control_replica"]
        group = row.get("source_cluster", doc)
        if doc_clusters.setdefault(doc, group) != group:
            raise ValueError("same control document has inconsistent cluster identity")
        value = _control_cell(row, baseline, metric)
        label = (replica, doc)
        previous = first_values.setdefault(label, value)
        if previous != value:
            raise ValueError("duplicate control document has inconsistent paired scores")
        aggregate = cells.setdefault(label, [0, 0, 0])
        for index, amount in enumerate(value):
            aggregate[index] += amount
    keys = sorted({key for key, _ in cells})
    docs = sorted(doc_clusters)
    clusters = sorted(set(doc_clusters.values()))
    if len(keys) < 2 or len(clusters) < 2:
        return {"status": "not_evaluated", "n_keys": len(keys), "n_documents": len(docs),
                "n_clusters": len(clusters), "n_boot": 0,
                "reason": "at least two independent keys and document clusters are required"}
    if len(cells) != len(keys) * len(docs):
        raise ValueError("crossed control grid is incomplete")
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    key_index = {key: index for index, key in enumerate(keys)}
    values = np.zeros((len(keys), len(clusters), 3), dtype=np.float64)
    for (key, doc), counts in cells.items():
        values[key_index[key], cluster_index[doc_clusters[doc]]] += counts
    point = float((values[:, :, 0].sum() - values[:, :, 1].sum()) / values[:, :, 2].sum())
    rng = np.random.default_rng(seed)
    samples = np.empty(2000)
    for i in range(len(samples)):
        key_indices = rng.integers(0, len(keys), size=len(keys))
        doc_indices = rng.integers(0, len(clusters), size=len(clusters))
        sample = values[key_indices[:, None], doc_indices[None, :]]
        samples[i] = (sample[:, :, 0].sum() - sample[:, :, 1].sum()) / sample[:, :, 2].sum()
    return {"status": "descriptive", "difference": point,
            "lo": float(np.quantile(samples, 0.025)), "hi": float(np.quantile(samples, 0.975)),
            "n_keys": len(keys), "n_documents": len(docs), "n_clusters": len(clusters), "n_boot": len(samples),
            "resampling_unit": "crossed key and near-duplicate document cluster" if clustered else "crossed independent key and distinct document",
            "method": "paired crossed percentile bootstrap", "key_scope": "synthetic control only",
            "document_independence": ("heuristic grouping does not prove independence" if clustered else
                                      "exact duplicates grouped; within-role near-duplicates unverified"),
            "metric": metric}


def _control_significance(accuracy: dict, baseline: dict) -> dict:
    """A single document cannot support a clustered paired significance test."""
    return {
        "status": "not_evaluated",
        "reason": "independent documents and paired frequency and LM baseline predictions required",
        "beat_best_baseline": None,
    }


def _stable_symbol(payload: bytes) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _rounded_vector_bytes(vector: np.ndarray) -> bytes:
    # float64 avoids float32 overflow during decimal rounding; signed zeros are equal.
    rounded = np.round(np.asarray(vector, dtype=np.float64), 4)
    rounded[rounded == 0] = 0.0
    return rounded.astype("<f8", copy=False).tobytes()


def wire_symbol_ids(vectors: np.ndarray, mapping: dict[int, int] | None = None):
    """Quantise wire vectors into stable symbols and a compact cipher alphabet."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("wire vectors must be a finite two-dimensional array with nonzero width")
    mapping = {} if mapping is None else mapping
    symbols = []
    for position in range(vectors.shape[0]):
        symbol = _stable_symbol(_rounded_vector_bytes(vectors[position]))
        if symbol not in mapping:
            mapping[symbol] = len(mapping)
        symbols.append(int(mapping[symbol]))
    return symbols, mapping



class PublicVectorSymbols:
    """Online vector clustering within server-visible groups; no truth inputs."""

    def __init__(self, mode: str = "rounded", distance_threshold: float | None = None):
        if mode not in {"rounded", "nearest_l2"}:
            raise ValueError("symbol frontend must be rounded or nearest_l2")
        if mode == "nearest_l2" and (isinstance(distance_threshold, (bool, np.bool_)) or
                not isinstance(distance_threshold, (int, float, np.integer, np.floating)) or
                not np.isfinite(distance_threshold) or distance_threshold < 0):
            raise ValueError("nearest_l2 requires a finite non-negative threshold")
        self.mode = mode
        self.threshold = distance_threshold
        self.groups: dict[tuple, dict] = {}
        self.rounded: dict[tuple, dict[bytes, int]] = {}
        self.dimensions: dict[tuple, int] = {}
        self.next_id = 0

    def encode(self, vectors, *, group_id, request_id: int) -> list[int]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("wire vectors must be a finite two-dimensional array")
        group = ("group", group_id) if group_id is not None else ("request", request_id)
        exact = self.rounded.setdefault(group, {})
        if group in self.dimensions and vectors.shape[1] != self.dimensions[group]:
            raise ValueError("wire vector dimensions changed within a group")
        self.dimensions[group] = vectors.shape[1]
        centers = self.groups.get(group)
        if self.mode == "nearest_l2" and centers is None:
            centers = {"vectors": np.empty((1, vectors.shape[1]), dtype=np.float32),
                       "ids": np.empty(1, dtype=np.int64), "size": 0}
            self.groups[group] = centers
        symbols = []
        for vector in vectors:
            key = _rounded_vector_bytes(vector) if self.mode == "rounded" else None
            symbol = exact.get(key) if self.mode == "rounded" else None
            if self.mode == "nearest_l2" and centers["size"]:
                size = centers["size"]
                distances = np.linalg.norm(centers["vectors"][:size].astype(np.float64) - vector.astype(np.float64), axis=1)
                closest = int(np.argmin(distances))
                if float(distances[closest]) <= self.threshold:
                    symbol = int(centers["ids"][closest])
            if symbol is None:
                symbol = self.next_id
                self.next_id += 1
                if self.mode == "rounded":
                    exact[key] = symbol
                else:
                    size = centers["size"]
                    if size == len(centers["ids"]):
                        capacity = max(1, size * 2)
                        expanded = np.empty((capacity, vectors.shape[1]), dtype=np.float32)
                        expanded[:size] = centers["vectors"][:size]
                        ids = np.empty(capacity, dtype=np.int64)
                        ids[:size] = centers["ids"][:size]
                        centers["vectors"], centers["ids"] = expanded, ids
                    centers["vectors"][size] = vector
                    centers["ids"][size] = symbol
                    centers["size"] = size + 1
            symbols.append(symbol)
        return symbols

def _reference_arrays(reference_ids: Sequence[Sequence[int]]) -> list[np.ndarray]:
    result = []
    for seq in reference_ids:
        array = np.asarray(seq)
        if array.ndim != 1:
            raise ValueError("reference documents must be one-dimensional integer sequences")
        if not array.size:
            continue
        if (not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_) or
                np.any(array < 0) or (np.issubdtype(array.dtype, np.unsignedinteger)
                                      and int(array.max()) > np.iinfo(np.int64).max)):
            raise ValueError("reference documents must contain non-negative int64 IDs")
        result.append(array.astype(np.int64, copy=False))
    return result


def _sda_code_identity() -> dict:
    """Bind checkpoints to the training implementation and current protocol."""
    from ..attacks.sda import _neural, sda2021_seq2seq, sda2023_recurrence_clm
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__), Path(_neural.__file__), Path(enc.__file__),
             Path(sda2021_seq2seq.__file__), Path(sda2023_recurrence_clm.__file__),
             root / "src/obfuscation_attacks/utils/seeding.py",
             root / "src/obfuscation_attacks/evaluation/sda_pair_store.py",
             root / "src/obfuscation_attacks/evaluation/sda_disk_cluster.py",
             root / "docs" / "PAPER_EVALUATION_PROTOCOL.md"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def _verify_window_manifest(evidence: dict) -> None:
    """Verify external window provenance whenever a checkpoint references it."""
    path = evidence.get("manifest_path")
    expected = evidence.get("manifest_sha256")
    if path is None and expected is None:
        return
    if not path or not expected:
        raise ValueError("incomplete SDA window manifest reference")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError("SDA window manifest is missing or unreadable") from exc
    if digest.hexdigest() != expected:
        raise ValueError("SDA window manifest content hash mismatch")


def load_sda_checkpoint(path, *, device: str = "cpu") -> dict:
    """Load a local checkpoint for repeated decoding without retraining."""
    import torch
    from ..attacks.sda._neural import validate_sda_device
    device = validate_sda_device(device)
    # Only the selected model is needed on the inference device.
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("format") != "sda_training_v1" or record.get("code_identity") != _sda_code_identity():
        raise ValueError("SDA checkpoint implementation or protocol identity mismatch")
    identity_material = record.get("identity_material")
    if not isinstance(identity_material, dict) or identity_material.get("code_identity") != record["code_identity"] or hashlib.sha256(
            json.dumps(identity_material, sort_keys=True).encode()).hexdigest() != record.get("identity"):
        raise ValueError("SDA checkpoint training identity mismatch")
    _verify_window_manifest(record.get("window_evidence") or {})
    method = record["method"]
    if method == "sda2021":
        cfg = Seq2SeqConfig(**record["model_config"])
        model = Seq2SeqCipherModel(cfg)
        objective, encoding = "L_Seq2Seq = -log P(Y|X)", "frequency_rank"
    elif method == "sda2023":
        cfg = CausalLMConfig(**record["model_config"])
        model = CausalCipherLM(cfg)
        objective, encoding = "L_CLM = -log P(X) - log P(Y|X)", "recurrence"
    else:
        raise ValueError("unknown SDA checkpoint method")
    model.load_state_dict(record.get("best_model_state") or record["model_state"])
    model.to(device).eval()
    return {"method": method, "model": model, "cfg": cfg,
            "pairs": record["pairs"], "epochs": record["epoch"],
            "optimizer_steps": record["optimizer_steps"], "examples_seen": record["examples_seen"],
            "final_loss": record["final_loss"], "objective": objective, "encoding": encoding,
            "max_cipher_len": record["max_cipher_len"], "trained_max_code": record["trained_max_code"],
            "homophonic": record["homophonic"], "window_evidence": record["window_evidence"],
            "homophone_symbols": identity_material["params"].get("homophone_symbols"),
            "length_mode": record["length_mode"], "max_output_len": record["max_output_len"],
            "space_token_id": identity_material["params"].get("space_token_id"),
            "synthesis_mode": record["synthesis_mode"], "pair_storage": record["pair_storage"],
            "checkpoint_identity": record["identity"], "checkpoint_epoch": record["epoch"],
            "training_history": record["training_history"],
            "validation_document_sha256": identity_material["validation_documents"],
            "validation_seed": identity_material["validation_seed"],
            "training_seed": identity_material["seed"],
            "sampling_evidence": record["sampling_evidence"],
            "validation_sampling": record["validation_sampling"],
            "validation_checkpoint": record.get("validation_checkpoint", {"status": "unused"})}


def _checkpoint_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_lineage(upper_path, upper_sha, lower_path, lower_sha, identity):
    """Verify archived hash links and epoch histories, not the training computation."""
    import torch

    expected_code = _sda_code_identity()
    seen, links = set(), 0
    path, expected = Path(upper_path), upper_sha
    previous = None
    while True:
        if expected in seen:
            raise ValueError("SDA checkpoint lineage contains a cycle")
        seen.add(expected)
        if _checkpoint_sha256(path) != expected:
            raise ValueError("SDA checkpoint lineage hash mismatch")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if _checkpoint_sha256(path) != expected:
            raise ValueError("SDA checkpoint changed during lineage verification")
        material = record.get("identity_material")
        if (record.get("format") != "sda_training_v1" or record.get("identity") != identity
                or record.get("code_identity") != expected_code or not isinstance(material, dict)
                or material.get("code_identity") != expected_code
                or hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest() != identity):
            raise ValueError("SDA checkpoint lineage training identity mismatch")
        epoch, steps, examples = record.get("epoch"), record.get("optimizer_steps"), record.get("examples_seen")
        history = record.get("training_history")
        if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (epoch, steps, examples))
                or not isinstance(history, list) or len(history) != epoch
                or any(not isinstance(row, dict) or row.get("epoch") != i + 1
                       or any(isinstance(row.get(field), bool) or not isinstance(row.get(field), int)
                              or row[field] <= 0 for field in ("optimizer_steps", "examples"))
                       for i, row in enumerate(history))
                or sum(row["optimizer_steps"] for row in history) != steps
                or sum(row["examples"] for row in history) != examples):
            raise ValueError("SDA checkpoint lineage has invalid training history")
        provenance = (record.get("model_config"), record.get("window_evidence"), record.get("validation_sampling"))
        if previous is not None:
            child_epoch, child_steps, child_examples, child_history, child_provenance = previous
            if (epoch + 1 != child_epoch or steps >= child_steps or examples >= child_examples
                    or history != child_history[:-1] or provenance != child_provenance):
                raise ValueError("SDA checkpoint lineage has inconsistent epoch or training history")
        parent_hash, parent_file = record.get("parent_sha256"), record.get("parent_file")
        previous = (epoch, steps, examples, history, provenance)
        del record
        if expected == lower_sha:
            return {"status": "hash_links_verified", "links_checked": links,
                    "scope": "file hashes, training identity and recorded history only; no signature or replay"}
        if parent_hash is None and parent_file is None:
            return {"status": "not_linked", "links_checked": links,
                    "reason": "checkpoint chain ended before the earlier checkpoint"}
        if (not isinstance(parent_hash, str) or len(parent_hash) != 64
                or any(c not in "0123456789abcdef" for c in parent_hash)
                or not isinstance(parent_file, str) or Path(parent_file).name != parent_file
                or not parent_file.endswith(".parent-" + parent_hash)):
            raise ValueError("invalid SDA checkpoint parent reference")
        parent = path.parent / parent_file
        if not parent.is_file():
            if parent_hash == lower_sha:
                parent = Path(lower_path)
            else:
                return {"status": "incomplete", "links_checked": links,
                        "missing_parent_sha256": parent_hash}
        path, expected = parent, parent_hash
        links += 1


def fit_sda(
    reference_ids: Sequence[Sequence[int]],
    cipher_sequences: Sequence[Sequence[int]], *, vocab_size: int,
    method: str = "sda2021", params: dict | None = None, seed: int = 0,
    device: str = "cpu", checkpoint_path: str | None = None, resume: bool = False,
    validation_ids: Sequence[Sequence[int]] | None = None, validation_seed: int = 0,
) -> dict:
    """Fit with in-memory or temporary disk-backed training examples."""
    from ..attacks.sda._neural import validate_sda_device
    validate_sda_device(device)
    options = dict(params or {})
    if not isinstance(resume, bool) or ("resume" in options and not isinstance(options["resume"], bool)):
        raise ValueError("resume must be a boolean")
    if "homophonic" in options and not isinstance(options["homophonic"], bool):
        raise ValueError("homophonic must be a boolean")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or
            not 0 <= seed < 2**32):
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (isinstance(validation_seed, (bool, np.bool_)) or not isinstance(validation_seed, (int, np.integer)) or
            not 0 <= validation_seed < 2**64):
        raise ValueError("validation_seed must be an unsigned 64-bit integer")
    if "seed" in options and (isinstance(options["seed"], (bool, np.bool_)) or
                              not isinstance(options["seed"], (int, np.integer))):
        raise ValueError("training seed must be an integer")
    storage = str(options.get("pair_storage", "auto"))
    if storage not in {"auto", "memory", "disk"}:
        raise ValueError("pair_storage must be auto, memory or disk")
    actual = ("disk" if int(options.get("num_train", 256)) >= 50000 else "memory") if storage == "auto" else storage
    kwargs = dict(vocab_size=vocab_size, method=method, params=options, seed=seed,
                  device=device, checkpoint_path=checkpoint_path, resume=resume,
                  pair_storage=actual, validation_ids=validation_ids, validation_seed=validation_seed)
    if actual == "memory":
        if options.get("window_manifest_path") is not None or options.get("pair_storage_dir") is not None:
            raise ValueError("window_manifest_path and pair_storage_dir require disk pair storage")
        return _fit_sda_impl(reference_ids, cipher_sequences, **kwargs)
    manifest = options.get("window_manifest_path")
    effective_checkpoint = checkpoint_path or options.get("checkpoint_path")
    if manifest is None and effective_checkpoint is not None:
        manifest = str(effective_checkpoint) + ".windows.jsonl"
    if manifest is None:
        raise ValueError("disk pair storage requires window_manifest_path or checkpoint_path for provenance")
    manifest_path = Path(manifest).resolve()
    if effective_checkpoint is not None and manifest_path == Path(effective_checkpoint).resolve():
        raise ValueError("window manifest and checkpoint must be different files")
    effective_resume = options.get("resume", resume)
    if manifest_path.exists() and not effective_resume:
        raise FileExistsError(f"window manifest already exists: {manifest_path}")
    if effective_resume and not manifest_path.is_file():
        raise ValueError("resume requires the original SDA window manifest")
    from .sda_pair_store import SQLitePairStore
    with SQLitePairStore(parent=options.get("pair_storage_dir")) as pairs:
        return _fit_sda_impl(reference_ids, cipher_sequences, pair_store=pairs,
                             window_manifest_path=str(manifest), **kwargs)


def _fit_sda_impl(
    reference_ids: Sequence[Sequence[int]],
    cipher_sequences: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    method: str = "sda2021",
    params: dict | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_path: str | None = None,
    resume: bool = False,
    pair_store=None,
    pair_storage: str = "memory",
    window_manifest_path: str | None = None,
    validation_ids: Sequence[Sequence[int]] | None = None,
    validation_seed: int = 0,
) -> dict:
    """Fit a lookup-embedding SDA variant on attacker-owned auxiliary plaintext."""
    import torch

    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    params = dict(params or {})
    common = {"num_train", "min_len", "max_len", "max_cipher_len", "epochs", "batch_size",
              "lr", "d_model", "nhead", "num_layers", "dim_ff", "dropout",
              "seed", "device", "beam_width", "decode_mode", "decode_cache", "checkpoint_path", "resume", "synthesis_mode", "sampling_mode", "sampling_seed", "pair_storage", "pair_storage_dir", "window_manifest_path", "cluster_storage", "cluster_temp_dir"}
    specific = {"noise_rate", "noise_kinds", "length_mode", "max_output_len", "space_token_id"} if method == "sda2021" else {"homophonic", "homophone_symbols", "representation"}
    unknown = set(params) - common - specific
    if unknown:
        raise ValueError(f"unsupported SDA training parameters: {sorted(unknown)}")
    if params.get("decode_mode", "raw") not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    if "decode_cache" in params and not isinstance(params["decode_cache"], bool):
        raise ValueError("decode_cache must be a boolean")
    if method != "sda2023" and params.get("decode_cache", False):
        raise ValueError("KV cache is currently supported only for SDA2023")
    if "seed" in params and int(params["seed"]) != int(seed):
        raise ValueError("training seed conflicts with fit seed")
    if "device" in params and str(params["device"]) != str(device):
        raise ValueError("training device conflicts with fit device")
    if "checkpoint_path" in params:
        if checkpoint_path is not None and str(params["checkpoint_path"]) != str(checkpoint_path):
            raise ValueError("conflicting checkpoint paths")
        checkpoint_path = params["checkpoint_path"]
    if "resume" in params:
        if resume and not params["resume"]:
            raise ValueError("conflicting resume settings")
        resume = params["resume"]
    if isinstance(vocab_size, (bool, np.bool_)) or not isinstance(vocab_size, (int, np.integer)) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    for name in ("num_train", "epochs", "batch_size", "min_len", "max_len", "max_cipher_len",
                 "d_model", "nhead", "num_layers", "dim_ff", "homophone_symbols", "beam_width"):
        if name in params and (isinstance(params[name], (bool, np.bool_)) or
                               not isinstance(params[name], (int, np.integer)) or params[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    corpus = _reference_arrays(reference_ids)
    if not corpus:
        raise ValueError("reference corpus is empty")
    validation_corpus = _reference_arrays([] if validation_ids is None else validation_ids)
    if set(map(document_sha256, corpus)) & set(map(document_sha256, validation_corpus)):
        raise ValueError("validation documents overlap training documents")
    seed_everything(int(seed))
    rng = np.random.default_rng(int(seed))
    num_train = int(params.get("num_train", 256))
    min_len = int(params.get("min_len", 4))
    max_len = int(params.get("max_len", 32))
    epochs = int(params.get("epochs", 15))
    batch_size = int(params.get("batch_size", 32))
    lr = float(params.get("lr", 1e-3 if method == "sda2021" else 5e-4))
    if num_train <= 0 or epochs <= 0 or batch_size <= 0 or min_len <= 0 or max_len < min_len:
        raise ValueError("training counts, batch and lengths must be positive and ordered")
    sampling_mode = params.get("sampling_mode", "with_replacement")
    if sampling_mode not in {"with_replacement", "distinct_document_windows"}:
        raise ValueError("unsupported SDA sampling_mode")
    cluster_storage = params.get("cluster_storage", "memory")
    if cluster_storage not in {"memory", "disk"}:
        raise ValueError("cluster_storage must be memory or disk")
    if sampling_mode != "distinct_document_windows" and ("cluster_storage" in params or "cluster_temp_dir" in params):
        raise ValueError("cluster storage requires distinct document sampling")
    if cluster_storage != "disk" and params.get("cluster_temp_dir") is not None:
        raise ValueError("cluster_temp_dir requires disk clustering")
    window_plan = None
    sampling_evidence = {"mode": "with_replacement", "independence_status": "not_verified"}
    validation_sampling = None
    cross_role_isolation = None
    if validation_corpus:
        roles = {role: [{"source": role, "index": i, "tokens": doc}
                        for i, doc in enumerate(documents)]
                 for role, documents in (("train", corpus), ("validation", validation_corpus))}
        if cluster_storage == "disk":
            from .sda_disk_cluster import check_document_isolation_disk
            cross_role_isolation = check_document_isolation_disk(roles, parent_dir=params.get("cluster_temp_dir"))
        else:
            cross_role_isolation = check_document_isolation({
                role: [_document_record(doc, role=role) for doc in documents]
                for role, documents in roles.items()})
    if cross_role_isolation is not None and not cross_role_isolation["passed"]:
        raise ValueError("training and validation documents have near overlap or unverifiable short-document overlap")
    if sampling_mode == "distinct_document_windows":
        if min_len != max_len:
            raise ValueError("distinct document sampling requires a fixed window length")
        sampling_seed = params.get("sampling_seed", 0)
        window_plan, sampling_evidence = _distinct_document_window_plan(
            corpus, count=num_train, width=min_len, seed=sampling_seed, cluster_storage=cluster_storage,
            cluster_temp_dir=params.get("cluster_temp_dir"))
        if validation_corpus:
            validation_plan, validation_sampling = _distinct_document_window_plan(
                validation_corpus, count=len(validation_corpus), width=min_len, seed=validation_seed,
                cluster_storage=cluster_storage, cluster_temp_dir=params.get("cluster_temp_dir"))
            validation_sampling["windows"] = [
                {"document_index": i, "start": start, "end": start + length,
                 "content_sha256": document_sha256(validation_corpus[i][start:start + length])}
                for i, start, length in validation_plan]
    elif "sampling_seed" in params:
        raise ValueError("sampling_seed requires distinct document sampling")
    sampling_evidence["cross_role_isolation"] = cross_role_isolation
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    length_mode = str(params.get("length_mode", "fixed"))
    max_output_len = params.get("max_output_len")
    if method == "sda2021":
        space_token_id = params.get("space_token_id")
        if space_token_id is not None and (isinstance(space_token_id, bool) or not isinstance(space_token_id, int)
                                           or not 0 <= space_token_id < vocab_size):
            raise ValueError("space_token_id must be an ID in the public vocabulary")
        if space_token_id is not None and length_mode == "fixed":
            raise ValueError("space removal requires variable or fixed-output length")
        if length_mode not in {"fixed", "variable", "fixed_output"}:
            raise ValueError("length_mode must be fixed, variable, or fixed_output")
        if length_mode in {"variable", "fixed_output"}:
            minimum = 1 if length_mode == "fixed_output" else 0
            if isinstance(max_output_len, bool) or not isinstance(max_output_len, int) or max_output_len < minimum:
                raise ValueError("variable/fixed-output length requires an explicit valid max_output_len")
        elif max_output_len is not None:
            raise ValueError("max_output_len is only valid for variable or fixed-output length")
        noise_rate = float(params.get("noise_rate", 0.0))
        noise_kinds = tuple(params.get("noise_kinds", ("sub",)))
        if not np.isfinite(noise_rate) or not 0 <= noise_rate <= 1:
            raise ValueError("noise_rate must be finite and between 0 and 1")
        if noise_rate and (not noise_kinds or any(k not in {"sub", "ins", "del"} for k in noise_kinds)):
            raise ValueError("noise_kinds must be a non-empty subset of sub, ins and del")
        if noise_rate > 0 and any(kind in {"ins", "del"} for kind in noise_kinds) and length_mode == "fixed":
            raise ValueError("insertion/deletion noise needs a variable-length decoder or fixed-output decoder")
    requested_synthesis = str(params.get("synthesis_mode", "auto"))
    if requested_synthesis not in {"auto", "legacy", "invariant"}:
        raise ValueError("synthesis_mode must be auto, legacy or invariant")
    key_invariant = ((method == "sda2021" and float(params.get("noise_rate", 0.0)) == 0.0)
                     or (method == "sda2023" and not bool(params.get("homophonic", False))))
    synthesis_mode = ("invariant" if key_invariant else "legacy") if requested_synthesis == "auto" else requested_synthesis
    if synthesis_mode == "invariant" and not key_invariant:
        raise ValueError("invariant synthesis requires a noiseless one-to-one cipher")
    max_cipher_len = int(params.get("max_cipher_len", 256))
    if max_cipher_len <= 0:
        raise ValueError("max_cipher_len must be positive")
    if any(len(cipher) > max_cipher_len for cipher in cipher_sequences):
        raise ValueError("cipher exceeds configured max_cipher_len")
    device = torch.device(device)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint_path")
    if checkpoint is not None and checkpoint.exists() and not resume:
        raise FileExistsError("checkpoint already exists; use resume or a new path")
    identity_material = {
        "method": method, "vocab_size": int(vocab_size), "seed": int(seed),
        "params": {k: v for k, v in params.items() if k not in {"epochs", "beam_width", "decode_mode", "decode_cache", "seed", "device", "checkpoint_path", "resume"}},
        "documents": [document_sha256(doc) for doc in corpus],
        "validation_documents": [document_sha256(doc) for doc in validation_corpus],
        "validation_seed": int(validation_seed),
        "device": str(device), "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__), "python_version": platform.python_version(),
        "cuda_version": str(torch.version.cuda),
        "code_identity": _sda_code_identity(),
    }
    identity = hashlib.sha256(json.dumps(identity_material, sort_keys=True).encode()).hexdigest()
    optimizer_steps = 0
    examples_seen = 0
    training_history = []
    validation_checkpoint = {"status": "unused", "n_documents": 0}
    best_model_state = None
    windows: dict[tuple[int, int, int], dict] = {}
    persisted_window_evidence = None

    def record_window(doc_index: int, start: int, tokens: np.ndarray) -> None:
        key = (doc_index, start, start + len(tokens))
        digest = document_sha256(tokens)
        if pair_store is not None:
            pair_store.record_window(*key, digest)
            return
        if key not in windows:
            windows[key] = {"document_index": doc_index, "start": start,
                            "end": key[2], "content_sha256": digest, "draws": 0}
        windows[key]["draws"] += 1

    def window_evidence() -> dict:
        if pair_store is not None:
            if persisted_window_evidence is None:
                raise RuntimeError("disk window provenance has not been published")
            return persisted_window_evidence
        records = list(windows.values())
        return {"windows": records, "unique_windows": len(records),
                "unique_content_hashes": len({r["content_sha256"] for r in records}),
                "draws": sum(r["draws"] for r in records),
                "independence_status": "unverified_overlapping_windows_are_not_independent"}

    def restore(model, opt):
        nonlocal optimizer_steps, examples_seen, validation_checkpoint, best_model_state, training_history
        if not resume:
            return 0, None
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SDA checkpoint not found: {checkpoint}")
        # RNG states must stay on the CPU. Model and optimizer load_state_dict
        # move training tensors to the device of their existing parameters.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format") != "sda_training_v1" or state.get("identity") != identity:
            raise ValueError("SDA checkpoint data, configuration, implementation or environment mismatch")
        _verify_window_manifest(state.get("window_evidence") or {})
        if pair_store is not None and state["window_evidence"]["manifest_sha256"] != window_evidence()["manifest_sha256"]:
            raise ValueError("SDA checkpoint window manifest differs from regenerated data")
        if int(state["epoch"]) > epochs:
            raise ValueError("checkpoint has more epochs than requested")
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        optimizer_steps = int(state["optimizer_steps"])
        examples_seen = int(state["examples_seen"])
        training_history = state["training_history"]
        validation_checkpoint = state.get("validation_checkpoint", validation_checkpoint)
        best_model_state = state.get("best_model_state")
        return int(state["epoch"]), state["final_loss"]

    def save(model, opt, cfg, epoch, final_loss, trained_max_code, pairs_count):
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        parent = _checkpoint_sha256(checkpoint) if epoch > 1 else None
        parent_file = None
        if parent is not None:
            import shutil
            parent_file = checkpoint.name + ".parent-" + parent
            archive = checkpoint.parent / parent_file
            if archive.exists():
                if _checkpoint_sha256(archive) != parent:
                    raise ValueError("checkpoint parent archive hash mismatch")
            else:
                with checkpoint.open("rb") as source:
                    try:
                        target = archive.open("xb")
                    except FileExistsError:
                        if _checkpoint_sha256(archive) != parent:
                            raise ValueError("checkpoint parent archive hash mismatch")
                    else:
                        try:
                            with target:
                                shutil.copyfileobj(source, target)
                        except Exception:
                            archive.unlink(missing_ok=True)
                            raise
        state = {"format": "sda_training_v1", "identity": identity,
                 "identity_material": identity_material, "code_identity": identity_material["code_identity"],
                 "parent_sha256": parent, "parent_file": parent_file, "method": method, "model_config": vars(cfg),
                 "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                 "best_model_state": best_model_state, "validation_checkpoint": validation_checkpoint,
                 "training_history": training_history, "sampling_evidence": sampling_evidence,
                 "validation_sampling": validation_sampling,
                 "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "epoch": epoch, "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
                 "pairs": pairs_count, "final_loss": final_loss, "trained_max_code": trained_max_code,
                 "max_cipher_len": max_cipher_len, "homophonic": bool(params.get("homophonic", False)),
                 "length_mode": length_mode, "max_output_len": max_output_len,
                 "synthesis_mode": synthesis_mode, "pair_storage": pair_storage,
                 "window_evidence": window_evidence()}
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix=checkpoint.name + ".", suffix=".tmp", dir=checkpoint.parent)
        os.close(fd)
        try:
            torch.save(state, temporary)
            if parent is None:
                os.link(temporary, checkpoint)
            else:
                os.replace(temporary, checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    validation_pairs = []
    if validation_corpus:
        validation_rng = np.random.default_rng(int(validation_seed))
        validation_items = ([(i, validation_corpus[i], [(0, start, length)]) for i, start, length in validation_plan]
                            if sampling_mode == "distinct_document_windows" else
                            [(i, document, None) for i, document in enumerate(validation_corpus)])
        generated_windows = []
        for document_index, document, frozen_window in validation_items:
            def record_validation_window(_local_index, start, tokens, document_index=document_index):
                generated_windows.append({"document_index": document_index, "start": start,
                                          "end": start + len(tokens), "content_sha256": document_sha256(tokens)})
            if method == "sda2021":
                generated = synthesize_pairs_2021(
                    [document], int(vocab_size), 1, validation_rng,
                    min_len=min_len, max_len=max_len,
                    noise_rate=float(params.get("noise_rate", 0.0)),
                    noise_kinds=tuple(params.get("noise_kinds", ("sub",))),
                    synthesis_mode=synthesis_mode,
                    space_token_id=space_token_id,
                    window_plan=frozen_window,
                    record_window=record_validation_window,
                )
            else:
                generated = synthesize_pairs_2023(
                    [document], int(vocab_size), 1, validation_rng,
                    min_len=min_len, max_len=max_len,
                    homophonic=bool(params.get("homophonic", False)),
                    approx_num_symbols=params.get("homophone_symbols"),
                    synthesis_mode=synthesis_mode,
                    window_plan=frozen_window,
                    record_window=record_validation_window,
                )
            if not generated or not generated[0][0]:
                raise ValueError("validation synthesis produced an empty ciphertext")
            validation_pairs.extend(generated)
        if validation_sampling is None:
            validation_sampling = {"mode": "one_sampled_window_per_document", "seed": int(validation_seed),
                                   "selected_documents": len(generated_windows), "windows": generated_windows,
                                   "independence_status": "source_independence_not_verified"}
        elif validation_sampling["windows"] != generated_windows:
            raise ValueError("generated validation windows do not match their frozen plan")

    def select_checkpoint(model, cfg, epoch):
        nonlocal validation_checkpoint, best_model_state
        if not validation_pairs:
            return
        total, count = 0.0, 0
        model.eval()
        with torch.no_grad():
            batches_rng = np.random.default_rng(int(validation_seed))
            if method == "sda2021":
                for src, tgt_in, tgt_out, src_kpm in _make_batches_2021(validation_pairs, cfg, batch_size, batches_rng):
                    src, tgt_in, tgt_out, src_kpm = (x.to(device) for x in (src, tgt_in, tgt_out, src_kpm))
                    loss = seq2seq_target_nll(model(src, tgt_in, src_kpm=src_kpm), tgt_out,
                                               tgt_pad_id=cfg.tgt_pad, reduction="sum")
                    total += float(loss)
                    count += int((tgt_out != cfg.tgt_pad).sum())
            else:
                for inputs, labels in _make_batches_2023(validation_pairs, cfg, batch_size, batches_rng):
                    _, _, loss, _, n_target = clm_losses(model(inputs.to(device)), labels.to(device), cfg)
                    total += float(loss)
                    count += n_target
        model.train()
        value = total / count if count else float("nan")
        if not np.isfinite(value):
            raise FloatingPointError("invalid SDA validation loss")
        history = validation_checkpoint.get("history", []) + [{"epoch": epoch, "target_nll": value}]
        if best_model_state is None or value < validation_checkpoint["best_target_nll"]:
            best_model_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            best_epoch, best_loss = epoch, value
        else:
            best_epoch, best_loss = validation_checkpoint["best_epoch"], validation_checkpoint["best_target_nll"]
        validation_checkpoint = {"status": "selected", "metric": "target_nll",
                                 "n_documents": len(validation_corpus), "seed": int(validation_seed),
                                 "window_policy": "one synthetic window per validation document",
                                 "best_epoch": best_epoch, "best_target_nll": best_loss, "history": history}

    if method == "sda2021":
        pairs = synthesize_pairs_2021(
            corpus, int(vocab_size), num_train, rng,
            min_len=min_len, max_len=max_len, record_window=record_window,
            noise_rate=float(params.get("noise_rate", 0.0)),
            noise_kinds=tuple(params.get("noise_kinds", ("sub",))),
            synthesis_mode=synthesis_mode, pair_store=pair_store,
            space_token_id=space_token_id,
            window_plan=window_plan,
        )
        if pair_store is not None:
            pair_store.finish()
        if not pairs:
            raise ValueError("no training pairs synthesised")
        if any(not src for src, _ in pairs):
            raise ValueError("training noise produced an empty ciphertext; reduce noise or change seed")
        if pair_store is not None:
            persisted_window_evidence = pair_store.publish_windows(window_manifest_path, reuse_existing=resume)
        trained_max_code = max((max(src, default=0) for src, _ in pairs), default=0)
        max_rank = max(max_cipher_len, trained_max_code,
                       max((max(src, default=0) for src, _ in validation_pairs), default=0))
        cfg = Seq2SeqConfig(
            vocab_size=int(vocab_size),
            src_vocab_size=max_rank + 1,
            d_model=int(params.get("d_model", 32)),
            nhead=int(params.get("nhead", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dim_ff=int(params.get("dim_ff", 64)),
            dropout=float(params.get("dropout", 0.0)),
            max_len=max(256, max_cipher_len + 2, (max_output_len or 0) + 1,
                        max(max(len(src), len(tgt) + 1) for src, tgt in pairs),
                        max((max(len(src), len(tgt) + 1) for src, tgt in validation_pairs), default=0)),
        )
        model = Seq2SeqCipherModel(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        start_epoch, final_loss = restore(model, opt)
        for epoch in range(start_epoch, epochs):
            epoch_nll, epoch_tokens, epoch_examples, epoch_steps = 0.0, 0, 0, 0
            for src, tgt_in, tgt_out, src_kpm in _make_batches_2021(
                pairs, cfg, batch_size, rng
            ):
                src, tgt_in, tgt_out, src_kpm = (
                    src.to(device), tgt_in.to(device), tgt_out.to(device), src_kpm.to(device)
                )
                opt.zero_grad()
                logits = model(src, tgt_in, src_kpm=src_kpm)
                loss = seq2seq_target_nll(
                    logits, tgt_out, tgt_pad_id=cfg.tgt_pad, reduction="mean"
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite SDA training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("non-finite SDA gradient")
                opt.step()
                optimizer_steps += 1
                examples_seen += int(src.shape[0])
                final_loss = float(loss.detach())
                n_target = int((tgt_out != cfg.tgt_pad).sum())
                epoch_nll += final_loss * n_target
                epoch_tokens += n_target
                epoch_examples += int(src.shape[0])
                epoch_steps += 1
            training_history.append({"epoch": epoch + 1, "target_nll": epoch_nll / epoch_tokens,
                                     "target_tokens": epoch_tokens, "examples": epoch_examples,
                                     "optimizer_steps": epoch_steps})
            select_checkpoint(model, cfg, epoch + 1)
            save(model, opt, cfg, epoch + 1, final_loss, trained_max_code, len(pairs))
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return {
            "method": "sda2021",
            "model": model, "cfg": cfg, "pairs": len(pairs),
            "final_loss": final_loss, "epochs": epochs, "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            "objective": "L_Seq2Seq = -log P(Y|X)",
            "encoding": "frequency_rank", "trained_max_code": trained_max_code,
            "max_cipher_len": max_cipher_len, "window_evidence": window_evidence(),
            "checkpoint_identity": identity if checkpoint else None,
            "length_mode": length_mode, "max_output_len": max_output_len,
            "space_token_id": space_token_id,
            "synthesis_mode": synthesis_mode, "pair_storage": pair_storage,
            "training_history": training_history,
            "sampling_evidence": sampling_evidence, "validation_sampling": validation_sampling,
            "validation_checkpoint": validation_checkpoint,
        }

    pairs = synthesize_pairs_2023(
        corpus, int(vocab_size), num_train, rng,
        homophonic=bool(params.get("homophonic", False)),
        min_len=min_len, max_len=max_len, record_window=record_window,
        approx_num_symbols=params.get("homophone_symbols"), synthesis_mode=synthesis_mode,
        pair_store=pair_store,
        window_plan=window_plan,
    )
    if pair_store is not None:
        pair_store.finish()
    if not pairs:
        raise ValueError("no training pairs synthesised")
    if pair_store is not None:
        persisted_window_evidence = pair_store.publish_windows(window_manifest_path, reuse_existing=resume)
    trained_max_code = max((max(rec, default=0) for rec, _ in pairs), default=0)
    max_rec = max(max_cipher_len, trained_max_code,
                  max((max(src, default=0) for src, _ in validation_pairs), default=0))
    cfg = CausalLMConfig(
        vocab_size=int(vocab_size),
        max_rec=max_rec,
        d_model=int(params.get("d_model", 48)),
        nhead=int(params.get("nhead", 4)),
        num_layers=int(params.get("num_layers", 2)),
        dim_ff=int(params.get("dim_ff", 96)),
        dropout=float(params.get("dropout", 0.0)),
        max_len=max(512, max_cipher_len * 2 + 4, max(len(src) + len(tgt) + 2 for src, tgt in pairs),
                    max((len(src) + len(tgt) + 2 for src, tgt in validation_pairs), default=0)),
        representation=str(params.get("representation", "lookup")),
    )
    model = CausalCipherLM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    start_epoch, final = restore(model, opt)
    for epoch in range(start_epoch, epochs):
        epoch_src, epoch_tgt, epoch_n_src, epoch_n_tgt, epoch_examples, epoch_steps = 0.0, 0.0, 0, 0, 0, 0
        for inputs, labels in _make_batches_2023(pairs, cfg, batch_size, rng):
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(inputs)
            total, l_src, l_tgt, n_src, n_tgt = clm_losses(logits, labels, cfg)
            loss = total / max(n_src + n_tgt, 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SDA training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite SDA gradient")
            opt.step()
            optimizer_steps += 1
            examples_seen += int(inputs.shape[0])
            final = {
                "total": float(total.detach()),
                "src": float(l_src.detach()),
                "tgt": float(l_tgt.detach()),
            }
            epoch_src += final["src"]
            epoch_tgt += final["tgt"]
            epoch_n_src += n_src
            epoch_n_tgt += n_tgt
            epoch_examples += int(inputs.shape[0])
            epoch_steps += 1
        training_history.append({"epoch": epoch + 1,
                                 "source_nll": epoch_src / epoch_n_src if epoch_n_src else None,
                                 "target_nll": epoch_tgt / epoch_n_tgt if epoch_n_tgt else None,
                                 "source_tokens": epoch_n_src, "target_tokens": epoch_n_tgt,
                                 "examples": epoch_examples, "optimizer_steps": epoch_steps})
        select_checkpoint(model, cfg, epoch + 1)
        save(model, opt, cfg, epoch + 1, final, trained_max_code, len(pairs))
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return {
        "method": "sda2023",
        "model": model, "cfg": cfg, "pairs": len(pairs),
        "final_loss": final, "epochs": epochs, "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
        "objective": "L_CLM = -log P(X) - log P(Y|X)",
        "encoding": "recurrence", "trained_max_code": trained_max_code,
        "max_cipher_len": max_cipher_len, "window_evidence": window_evidence(),
        "checkpoint_identity": identity if checkpoint else None,
        "synthesis_mode": synthesis_mode, "pair_storage": pair_storage,
        "training_history": training_history,
        "sampling_evidence": sampling_evidence, "validation_sampling": validation_sampling,
        "validation_checkpoint": validation_checkpoint,
        "homophonic": bool(params.get("homophonic", False)),
        "homophone_symbols": params.get("homophone_symbols"),
    }


def decode_sda(bundle: dict, cipher: Sequence[int], *, beam_width: int = 10,
               use_cache: bool = False, mode: str = "raw") -> dict:
    """Decode one cipher sequence with a fitted lookup-embedding variant."""
    from ..attacks.sda.sda2021_seq2seq import SDA2021
    from ..attacks.sda.sda2023_recurrence_clm import SDA2023

    if not isinstance(use_cache, bool):
        raise ValueError("use_cache must be a boolean")
    if mode not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    if isinstance(beam_width, (bool, np.bool_)) or not isinstance(beam_width, (int, np.integer)) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    method = bundle["method"]
    model, cfg = bundle["model"], bundle["cfg"]
    raw_cipher = np.asarray(cipher)
    if (raw_cipher.ndim != 1 or not raw_cipher.size or
            not np.issubdtype(raw_cipher.dtype, np.integer) or np.issubdtype(raw_cipher.dtype, np.bool_) or
            np.any(raw_cipher < 0) or (np.issubdtype(raw_cipher.dtype, np.unsignedinteger)
                                        and int(raw_cipher.max()) > np.iinfo(np.int64).max)):
        raise ValueError("cipher must be a nonempty sequence of non-negative int64 IDs")
    cipher_arr = raw_cipher.astype(np.int64, copy=False)
    if len(cipher_arr) > bundle["max_cipher_len"]:
        raise ValueError("cipher exceeds configured max_cipher_len")
    if method == "sda2021":
        if use_cache:
            raise ValueError("KV cache is currently supported only for SDA2023")
        result = SDA2021()._decode(model, cfg, cipher_arr, beam_width=int(beam_width), mode=mode,
                                  length_mode=bundle["length_mode"],
                                  max_output_len=bundle["max_output_len"])
    elif method == "sda2023":
        result = SDA2023()._decode(
            model, cfg, cipher_arr, enc.recurrence_encode(cipher_arr),
            beam_width=int(beam_width), mode=mode, homophonic=bundle.get("homophonic", False),
            use_cache=use_cache,
        )
    else:
        raise ValueError(f"unknown SDA method {method!r}")
    codes = enc.frequency_rank_encode(cipher_arr) if method == "sda2021" else enc.recurrence_encode(cipher_arr)
    result["source_code_coverage"] = {"trained_max_code": bundle["trained_max_code"],
                                      "unseen_positions": sum(int(code > bundle["trained_max_code"]) for code in codes),
                                      "total_positions": len(codes)}
    return result


def _token_accuracy(recovered: Sequence[int], truth: Sequence[int]) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    recovered = np.asarray(recovered, dtype=np.int64)
    n = int(truth.size)
    shared = int(min(recovered.size, n))
    hits = int((truth[:shared] == recovered[:shared]).sum())
    return {
        "value": float(hits / n) if n else None,
        "numerator": hits,
        "denominator": n,
    }


def _token_edit_error(recovered: Sequence[int], reference: Sequence[int]) -> dict:
    """Levenshtein token edit distance, divided by full reference length.

    This excludes the block shifts in general TER. It matches the 2021 paper's
    stated character-edit operations only when IDs are individual characters.
    """
    predicted, truth = list(recovered), list(reference)
    previous = list(range(len(predicted) + 1))
    for i, expected in enumerate(truth, start=1):
        current = [i]
        for j, actual in enumerate(predicted, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + int(actual != expected)))
        previous = current
    edits = previous[-1]
    return {"value": edits / len(truth) if truth else None,
            "edits": edits, "reference_tokens": len(truth),
            "predicted_tokens": len(predicted), "metric": "token_levenshtein_error_not_TER"}


def _score_symbol_frontend(cipher_sequences, truth_sequences, group_ids) -> dict:
    """Scoring-only split/merge rates, aggregated over legal public groups."""
    if not (len(cipher_sequences) == len(truth_sequences) == len(group_ids)):
        raise ValueError("frontend scoring inputs must have equal request counts")
    groups: dict[tuple, dict[str, Counter]] = {}
    for index, (cipher, truth, group_id) in enumerate(zip(cipher_sequences, truth_sequences, group_ids)):
        if len(cipher) != len(truth):
            raise ValueError("frontend scoring requires the full truth for each observed token")
        group = ("group", group_id) if group_id is not None else ("request", index)
        counters = groups.setdefault(group, {"truth": Counter(), "symbol": Counter(), "joint": Counter()})
        for symbol, token in zip(cipher, truth):
            symbol, token = int(symbol), int(token)
            counters["truth"][token] += 1
            counters["symbol"][symbol] += 1
            counters["joint"][(symbol, token)] += 1
    pairs = lambda n: n * (n - 1) // 2
    same_truth = same_symbol = same_both = 0
    for counters in groups.values():
        same_truth += sum(pairs(n) for n in counters["truth"].values())
        same_symbol += sum(pairs(n) for n in counters["symbol"].values())
        same_both += sum(pairs(n) for n in counters["joint"].values())
    splits, merges = same_truth - same_both, same_symbol - same_both
    return {"scope": "scoring_only", "group_count": len(groups),
            "same_truth_pair_count": same_truth, "split_pairs": splits,
            "split_rate": splits / same_truth if same_truth else None,
            "same_symbol_pairs": same_symbol, "merge_pairs": merges,
            "merge_rate": merges / same_symbol if same_symbol else None}


def _select_validation_beam(bundle: dict, documents: Sequence[dict], *, vocab_size: int,
                            candidates: Sequence[int], split_seed: int,
                            noise_rate: float = 0.0,
                            noise_kinds: Sequence[str] = ("sub",),
                            use_cache: bool = False, decode_mode: str = "raw") -> dict:
    """Select beam on public held-out synthetic ciphers, before private scoring."""
    if not documents:
        raise ValueError("beam selection requires non-empty isolated validation documents")
    if not candidates or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in candidates):
        raise ValueError("validation beam candidates must be positive integers")
    beams = sorted(set(candidates))
    rng = np.random.default_rng(split_seed + 1991)
    homophonic = bool(bundle.get("homophonic", False))
    key = (None if homophonic else
           enc.make_substitution_key(range(vocab_size), rng=rng,
                                     homophonic=False, cipher_alphabet_size=vocab_size))
    if not np.isfinite(noise_rate) or not 0 <= noise_rate <= 1:
        raise ValueError("validation noise_rate must be between 0 and 1")
    if noise_rate and (not noise_kinds or any(k not in {"sub", "ins", "del"} for k in noise_kinds)):
        raise ValueError("validation noise kinds must be non-empty and valid")
    if noise_rate and any(k in {"ins", "del"} for k in noise_kinds) and bundle.get("length_mode", "fixed") == "fixed":
        raise ValueError("validation insertion/deletion requires variable or fixed-output length")
    prepared = []
    for doc in documents:
        plain = doc["tokens"]
        space_token_id = bundle.get("space_token_id")
        if space_token_id is not None:
            plain = [token for token in plain if token != space_token_id]
        if homophonic:
            counts = {int(t): int(c) for t, c in zip(*np.unique(plain, return_counts=True))}
            if not counts:
                raise ValueError("validation plaintext is empty")
            key = enc.make_substitution_key(counts, rng=rng, homophonic=True, plain_counts=counts,
                                            approx_num_symbols=bundle.get("homophone_symbols") or 2 * len(counts))
        cipher = np.asarray(enc.apply_substitution(plain, key, rng=rng if homophonic else None), dtype=np.int64)
        if noise_rate:
            cipher = enc.apply_transcription_noise(cipher, rate=noise_rate, rng=rng,
                                                   kinds=noise_kinds, alphabet=range(vocab_size))
        if not len(cipher):
            raise ValueError("validation noise produced an empty ciphertext")
        prepared.append((doc, cipher))
    clusters, cluster_report = cluster_control_documents(documents)
    results = []
    variable = bundle.get("length_mode", "fixed") != "fixed"
    for beam in beams:
        rows = []
        for (doc, cipher), cluster in zip(prepared, clusters):
            decoded = decode_sda(bundle, cipher, beam_width=beam, use_cache=use_cache, mode=decode_mode)
            rows.append({"source_sha256": doc["sha256"], "source": doc["source"],
                         "source_cluster": cluster,
                         "source_index": int(doc["index"]), "recovered": decoded["tokens"],
                         "cipher_length": len(cipher), "cipher_sha256": document_sha256(cipher),
                         "reference_length": len(doc["tokens"]),
                         "accuracy": _token_accuracy(decoded["tokens"], doc["tokens"]),
                         "edit_error": _token_edit_error(decoded["tokens"], doc["tokens"]) if variable else None})
        hits = sum(row["accuracy"]["numerator"] for row in rows)
        total = sum(row["accuracy"]["denominator"] for row in rows)
        if not total:
            raise ValueError("validation set contains no scoring tokens")
        edits = sum(row["edit_error"]["edits"] for row in rows) if variable else None
        results.append({"beam_width": beam, "hits": hits, "count": total,
                        "token_accuracy": hits / total,
                        "token_edit_error": edits / total if edits is not None else None,
                        "rows": rows})
    if variable:
        best = min(results, key=lambda row: (row["token_edit_error"], row["beam_width"]))
        rule = "lowest micro token Levenshtein error; ties select smallest beam"
    else:
        best = min(results, key=lambda row: (-row["token_accuracy"], row["beam_width"]))
        rule = "highest micro token accuracy; ties select smallest beam"
    comparisons = []
    metric = "token_edit_error" if variable else "token_accuracy"
    for lower, upper in zip(results, results[1:]):
        paired = []
        for old, new in zip(lower["rows"], upper["rows"]):
            paired.append({"source_sha256": new["source_sha256"],
                           "source_cluster": new["source_cluster"],
                           "accuracy": new["accuracy"], "edit_error": new["edit_error"],
                           "baselines": {"smaller_beam": {**old["accuracy"],
                                                          "edit_error": old["edit_error"]}}})
        comparison = _paired_control_interval(paired, "smaller_beam", seed=split_seed, metric=metric)
        comparisons.append({"smaller_beam": lower["beam_width"], "larger_beam": upper["beam_width"],
                            **comparison})
    return {"status": "selected", "surface": ("synthetic_homophonic_substitution" if homophonic
                                               else "synthetic_fixed_substitution"), "decode_mode": decode_mode,
            "rule": rule,
            "selected_beam": best["beam_width"], "candidates": results,
            "beam_comparisons": comparisons, "document_clustering": cluster_report,
            "n_documents": len({doc["sha256"] for doc in documents}),
            "saturation_status": "not_assessed", "key_policy": ("one auxiliary key per validation window"
                                                                       if homophonic else "one auxiliary fixed key"),
            "noise": {"rate": noise_rate, "kinds": list(noise_kinds)}}


def compare_sda_checkpoint_budgets(checkpoint_paths: Sequence[str], validation_ids: Sequence[Sequence[int]], *,
                                   beam_width: int, device: str = "cpu", noise_rate: float = 0.0,
                                   noise_kinds: Sequence[str] = ("sub",), use_cache: bool = False,
                                   decode_mode: str = "raw") -> dict:
    """Compare checkpoints on the same frozen public validation windows, not test data."""
    if len(checkpoint_paths) < 2:
        raise ValueError("checkpoint budget comparison needs at least two checkpoints")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if any(len(doc) == 0 for doc in validation_ids):
        raise ValueError("validation documents must not be empty")
    corpus = _reference_arrays(validation_ids)
    hashes = [document_sha256(doc) for doc in corpus]
    snapshots, common_identity, common_sampling = [], None, None
    paths_by_hash = {}

    for path in checkpoint_paths:
        before = _checkpoint_sha256(path)
        bundle = load_sda_checkpoint(path, device=device)
        if _checkpoint_sha256(path) != before:
            raise ValueError("checkpoint changed during loading")
        paths_by_hash[before] = path
        if common_identity is None:
            common_identity = bundle["checkpoint_identity"]
            common_sampling = bundle["validation_sampling"]
        if bundle["checkpoint_identity"] != common_identity:
            raise ValueError("checkpoint training identities differ")
        if bundle["validation_document_sha256"] != hashes or bundle["validation_sampling"] != common_sampling:
            raise ValueError("checkpoint validation documents or window plans differ")
        if not isinstance(common_sampling, dict) or not common_sampling.get("windows"):
            raise ValueError("checkpoints need frozen validation windows")
        if bundle["validation_checkpoint"].get("status") != "selected":
            raise ValueError("checkpoint has no validation-selected model")
        documents = []
        for window in common_sampling["windows"]:
            index, start, end = window["document_index"], window["start"], window["end"]
            if not (0 <= index < len(corpus) and 0 <= start < end <= len(corpus[index])):
                raise ValueError("invalid checkpoint validation window")
            tokens = corpus[index][start:end]
            if document_sha256(tokens) != window["content_sha256"]:
                raise ValueError("checkpoint validation window content mismatch")
            documents.append(_document_record({"source": "validation", "index": index, "tokens": tokens},
                                               role="validation"))
        result = _select_validation_beam(bundle, documents, vocab_size=int(bundle["cfg"].vocab_size),
                                         candidates=[beam_width], split_seed=int(bundle["validation_seed"]),
                                         noise_rate=noise_rate, noise_kinds=noise_kinds, use_cache=use_cache,
                                         decode_mode=decode_mode)
        snapshots.append({"epoch_budget": int(bundle["checkpoint_epoch"]), "checkpoint_sha256": before,
                          "optimizer_steps": int(bundle["optimizer_steps"]),
                          "selected_epoch": bundle["validation_checkpoint"].get("best_epoch"),
                          "validation_surface": result["surface"], "key_policy": result["key_policy"],
                          "result": result["candidates"][0]})
        validation_seed = int(bundle["validation_seed"])
        del bundle
    snapshots.sort(key=lambda row: row["epoch_budget"])
    if len({row["epoch_budget"] for row in snapshots}) != len(snapshots):
        raise ValueError("checkpoint epoch budgets must be distinct")
    metric = "token_edit_error" if snapshots[0]["result"]["token_edit_error"] is not None else "token_accuracy"
    comparisons, lineage_checks = [], []
    for lower, upper in zip(snapshots, snapshots[1:]):
        lineage = _checkpoint_lineage(paths_by_hash[upper["checkpoint_sha256"]], upper["checkpoint_sha256"],
                                      paths_by_hash[lower["checkpoint_sha256"]], lower["checkpoint_sha256"],
                                      common_identity)
        lineage_checks.append({"smaller_epoch_budget": lower["epoch_budget"],
                               "larger_epoch_budget": upper["epoch_budget"], **lineage})
        old_rows, new_rows = lower["result"]["rows"], upper["result"]["rows"]
        if (len(old_rows) != len(new_rows) or lower["validation_surface"] != upper["validation_surface"] or
                lower["key_policy"] != upper["key_policy"]):
            raise ValueError("checkpoint validation rows differ")
        paired = []
        for old, new in zip(old_rows, new_rows):
            if any(old[field] != new[field] for field in ("source_sha256", "source_index", "source_cluster", "reference_length", "cipher_length", "cipher_sha256")):
                raise ValueError("checkpoint validation examples do not match")
            paired.append({"source_sha256": new["source_sha256"], "source_cluster": new["source_cluster"],
                           "accuracy": new["accuracy"], "edit_error": new["edit_error"],
                           "baselines": {"smaller_budget": {**old["accuracy"], "edit_error": old["edit_error"]}}})
        interval = _paired_control_interval(paired, "smaller_budget", seed=validation_seed, metric=metric)
        comparisons.append({"smaller_epoch_budget": lower["epoch_budget"],
                            "larger_epoch_budget": upper["epoch_budget"], **interval})
    return {"status": "descriptive", "checkpoint_identity": common_identity,
            "beam_width": beam_width, "decode_mode": decode_mode, "checkpoints": snapshots, "comparisons": comparisons,
            "validation_seed": validation_seed, "noise_rate": noise_rate,
            "noise_kinds": list(noise_kinds), "key_policy": snapshots[0]["key_policy"],
            "validation_surface": snapshots[0]["validation_surface"],
            "checkpoint_lineage": ("hash_links_verified" if all(item["status"] == "hash_links_verified" for item in lineage_checks)
                                   else "incomplete_or_unlinked"),
            "checkpoint_lineage_checks": lineage_checks, "saturation_status": "not_assessed",
            "scope_note": "validation-selected checkpoint per budget; source independence and formal saturation remain unverified"}


def run_sda_stream(
    dep,
    texts: Sequence[str] | Mapping[int, str],
    reference_texts: Sequence[str],
    *,
    method: str = "sda2021",
    num_train: int = 256,
    epochs: int = 6,
    beam_width: int = 10,
    seed: int = 0,
    max_requests: int | None = None,
    positive_control: bool = True,
    control_keys: int = 1,
    verbose: bool = False,
    control_texts: Sequence[str] | None = None,
    validation_texts: Sequence[str] | None = None,
    control_fraction: float = 0.1,
    validation_fraction: float = 0.1,
    n_validation: int | None = None,
    require_isolated_control: bool = True,
    split_seed: int = 0,
    training_params: dict | None = None,
    device: str = "cpu",
    symbol_frontend: str = "rounded",
    distance_threshold: float | None = None,
    group_joint: bool = False,
    checkpoint_path: str | None = None,
    resume: bool = False,
    validation_beams: Sequence[int] | None = None,
    validation_noise_rate: float = 0.0,
    validation_noise_kinds: Sequence[str] = ("sub",),
    frozen_uploads: Sequence | None = None,
    candidate_table: np.ndarray | None = None,
    candidate_k: int = 1,
    text_tokenizer=None,
    text_decoder=None,
    text_entities=None,
    text_alignment_max_cells: int = 2_000_000,
    decode_cache: bool = False,
    scoring_clusters: Mapping[int, Mapping[str, str]] | None = None,
    budget_checkpoints: Sequence[str] | None = None,
    require_public_groups: bool = False,
) -> dict:
    """Fit once and decode the real A+B symbol stream; report control separately.

    The auxiliary corpus is split into disjoint roles by :data:`SPLIT_RULE`
    before training pairs are synthesised (``control_texts`` / ``validation_texts``
    override the respective role drawn from ``reference_texts``). ``fit_sda``
    only receives the training role; the known-substitution positive control is
    built from a held-out control document and reported with a prior baseline.
    The evaluated private ``texts`` form the ``private`` role and must be
    disjoint from every auxiliary role: an exact, near-duplicate or
    unverifiable short-document overlap refuses the run.
    Short cross-role documents with shared tokens remain blocked when their
    isolation cannot be verified; disjoint token sets are accepted.
    With ``require_isolated_control=False`` an auxiliary-role overlap is
    recorded but not raised, for diagnostics only -- the control role is still
    the only control source, so there is no fallback to training documents. A
    private-test overlap is always raised: fitting on the evaluated document
    invalidates the measurement itself.
    Frozen uploads require a scoring-side mapping from request ID to private
    text; the complete ID sets must match before any request limit is applied.
    """
    t0 = time.time()
    for name, value, upper in (("seed", seed, 2**32), ("split_seed", split_seed, 2**64)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < upper:
            raise ValueError(f"{name} must be an unsigned integer less than {upper}")
    if n_validation is not None and (isinstance(n_validation, (bool, np.bool_)) or
            not isinstance(n_validation, (int, np.integer)) or n_validation < 0):
        raise ValueError("n_validation must be a nonnegative integer")
    if n_validation is not None and validation_texts is not None:
        raise ValueError("n_validation cannot be combined with explicit validation_texts")
    if validation_beams is not None and (not validation_beams or any(
            isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in validation_beams)):
        raise ValueError("validation beam candidates must be positive integers")
    if not isinstance(decode_cache, bool):
        raise ValueError("decode_cache must be a boolean")
    if not isinstance(group_joint, bool):
        raise ValueError("group_joint must be a boolean")
    if not isinstance(require_public_groups, bool):
        raise ValueError("require_public_groups must be a boolean")
    decode_mode = (training_params or {}).get("decode_mode", "raw")
    if decode_mode not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    if decode_mode == "constrained" and (training_params or {}).get("length_mode", "fixed") != "fixed":
        raise ValueError("constrained decoding requires fixed length")
    if decode_cache and method != "sda2023":
        raise ValueError("KV cache is currently supported only for SDA2023")
    if scoring_clusters is not None and (training_params or {}).get("length_mode", "fixed") != "fixed":
        raise ValueError("clustered positional token scoring requires fixed-length decoding")
    if "decode_cache" in (training_params or {}) and training_params["decode_cache"] != decode_cache:
        raise ValueError("conflicting decode_cache settings")
    if budget_checkpoints is not None and (validation_beams is None or
            not (checkpoint_path or (training_params or {}).get("checkpoint_path"))):
        raise ValueError("checkpoint budget comparison requires validation beams and a checkpoint path")
    if max_requests is not None and (isinstance(max_requests, bool) or
                                     not isinstance(max_requests, int) or max_requests <= 0):
        raise ValueError("max_requests must be a positive integer")
    if frozen_uploads is None:
        if isinstance(texts, Mapping):
            raise ValueError("request-ID keyed texts require frozen uploads")
        selected = list(texts[:max_requests])
    else:
        if not isinstance(texts, Mapping):
            raise ValueError("frozen uploads require scoring texts keyed by request ID")
        if any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) for key in texts):
            raise ValueError("scoring request IDs must be integers")
        uploads = list(frozen_uploads)
        ids = [upload.request_id for upload in uploads]
        if any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) for key in ids):
            raise ValueError("upload request IDs must be integers")
        if len(set(ids)) != len(ids):
            raise ValueError("upload request IDs must be unique")
        if set(ids) != set(texts):
            raise ValueError("scoring request IDs must exactly match frozen upload request IDs")
        uploads = uploads[:max_requests]
        selected = [texts[upload.request_id] for upload in uploads]
        if any(not isinstance(text, str) for text in selected):
            raise ValueError("scoring texts must be strings")
    if not selected:
        raise ValueError("SDA requires at least one private request")
    cipher_sequences = []
    truth_plain = []
    group_ids = []
    request_ids = []
    frontend_candidates = []
    public_uploads_sha256 = None
    processed_observations_sha256 = None
    text_surface = False
    if frozen_uploads is not None:
        from .sda_public import prepare_sda_observations
        from ..schemes.adapters import UploadSurface
        text_surface = bool(uploads) and all(upload.surface == UploadSurface.TEXT for upload in uploads)
        if text_surface and scoring_clusters is not None:
            raise ValueError("token scoring clusters cannot be applied to cross-tokenizer text")
        if text_surface and (not callable(text_tokenizer) or not callable(text_decoder)):
            raise ValueError("text SDA scoring requires character-alignment decoding and an explicit public tokenizer")
        if text_surface and group_joint:
            raise ValueError("group_joint text scoring has no verified cross-request character alignment")
        prepared = prepare_sda_observations(
            uploads, vector_method=symbol_frontend, distance_threshold=distance_threshold,
            candidate_table=candidate_table, candidate_k=candidate_k, text_tokenizer=text_tokenizer,
        )
        public_uploads_sha256 = prepared["public_uploads_sha256"]
        processed_observations_sha256 = prepared["processed_observations_sha256"]
        frontend_details = prepared["frontend"]
        for text, row in zip(selected, prepared["rows"]):
            request_ids.append(row["request_id"])
            group_ids.append(row["group_id"])
            cipher_sequences.append(row["symbols"])
            frontend_candidates.append(row["candidates"])
            # Private plaintext enters only this scoring-side collection.
            truth_plain.append(np.asarray(dep.encode(text), dtype=np.int64))
        if not text_surface and any(len(cipher) != len(truth) for cipher, truth in zip(cipher_sequences, truth_plain)):
            raise ValueError("frozen uploads are not position-aligned with scoring tokens")
    else:
        frontend = PublicVectorSymbols(symbol_frontend, distance_threshold)
        frontend_details = {"method": symbol_frontend, "distance_threshold": distance_threshold,
                            "grouping": "public group_id; missing groups isolated per request",
                            "status": "unvalidated_engineering_variant"}
        for i, text in enumerate(selected):
            wire = dep.observe(text, arm="ab", request_id=i, key_index=i, forward=False)
            if wire.public.truncated:
                raise ValueError("SDA cannot score a truncated private request; provide full observations")
            request_ids.append(i)
            group_ids.append(wire.public.group_id)
            symbols = frontend.encode(wire.public.wire_vectors, group_id=wire.public.group_id, request_id=i)
            cipher_sequences.append(symbols)
            frontend_candidates.append(None)
            truth_plain.append(np.asarray(wire.truth.plaintext_ids, dtype=np.int64))
    if require_public_groups and any(group is None for group in group_ids):
        raise ValueError("SDA result cannot report key groups: public group identifiers are missing")
    cipher_alphabet_size = len({(("group", group) if group is not None else ("request", rid), int(symbol))
                                for cipher, group, rid in zip(cipher_sequences, group_ids, request_ids)
                                for symbol in cipher})

    reference_docs, n_empty = _materialise_documents(dep, reference_texts, "reference")
    # The evaluated private stream is a fourth role: it must be disjoint from
    # every auxiliary role, otherwise the model would be fitted on the very
    # document it is scored on (protocol zero-text-overlap requirement).
    private_docs, n_empty_private = _materialise_documents(dep, selected, "private")
    n_empty += n_empty_private

    explicit_control_docs: list[dict] | None = None
    if control_texts is not None:
        explicit_control_docs, n_empty_control = _materialise_documents(
            dep, control_texts, "control_texts"
        )
        n_empty += n_empty_control
        if not explicit_control_docs:
            raise ValueError(
                "control_texts was provided but yielded no usable (non-empty) "
                "document; an empty control role cannot isolate the positive control"
            )
    explicit_validation_docs: list[dict] | None = None
    if validation_texts is not None:
        explicit_validation_docs, n_empty_validation = _materialise_documents(
            dep, validation_texts, "validation_texts"
        )
        n_empty += n_empty_validation
        if not explicit_validation_docs:
            raise ValueError(
                "validation_texts was provided but yielded no usable (non-empty) document"
            )

    # When an explicit corpus fills a role, draw zero documents for it from the
    # reference pool; a control drawn from the pool is mandatory whenever a
    # positive control is requested.
    roles = assign_document_roles(
        [document["tokens"] for document in reference_docs],
        seed=split_seed,
        control_fraction=control_fraction,
        validation_fraction=validation_fraction,
        n_control=0 if explicit_control_docs is not None else None,
        n_validation=0 if explicit_validation_docs is not None else n_validation,
        require_control=explicit_control_docs is None and bool(positive_control),
    )
    if n_validation is not None and len(roles["validation"]) != n_validation:
        raise ValueError("not enough reference documents for the requested n_validation while retaining training and control roles")

    role_docs: dict[str, list[dict]] = {
        "train": [], "validation": [], "control": [], "private": [],
    }
    disk_isolation = (training_params or {}).get("cluster_storage", "memory") == "disk"
    def evidence_record(document, role):
        if disk_isolation and role in {"train", "private"}:
            tokens = np.asarray(document["tokens"], dtype=np.int64).reshape(-1)
            return {"role": role, "source": str(document["source"]), "index": int(document["index"]),
                    "tokens": tokens, "length": int(tokens.size), "sha256": document_sha256(tokens)}
        return _document_record(document, role=role)
    for index in roles["train"]:
        role_docs["train"].append(evidence_record(reference_docs[index], role="train"))
    for index in roles["validation"]:
        role_docs["validation"].append(
            _document_record(reference_docs[index], role="validation")
        )
    if explicit_control_docs is None:
        for index in roles["control"]:
            role_docs["control"].append(
                _document_record(reference_docs[index], role="control")
            )
    else:
        for document in explicit_control_docs:
            role_docs["control"].append(_document_record(document, role="control"))
    if explicit_validation_docs is not None:
        for document in explicit_validation_docs:
            role_docs["validation"].append(
                _document_record(document, role="validation")
            )
    for document in private_docs:
        role_docs["private"].append(evidence_record(document, role="private"))

    if disk_isolation:
        from .sda_disk_cluster import check_document_isolation_disk
        overlap = check_document_isolation_disk(role_docs, parent_dir=(training_params or {}).get("cluster_temp_dir"))
    else:
        overlap = check_document_isolation(role_docs)
    private_by_pair = {
        pair: stats
        for pair, stats in overlap["by_role_pair"].items()
        if "private" in pair
    }
    private_pairs = sum(
        stats["exact"] + stats["near_duplicate"] + stats["unverifiable_short"]
        for stats in private_by_pair.values()
    )
    if private_pairs:
        # Fail-closed and unconditional: fitting on a document that is also
        # being scored invalidates the evaluation itself, so this is not
        # downgraded to a diagnostic by require_isolated_control=False.
        private_short = sum(
            stats["unverifiable_short"] for stats in private_by_pair.values()
        )
        private_short_detail = (
            f" {private_short} of them involve a document shorter than "
            f"SHINGLE_SIZE={SHINGLE_SIZE} tokens and are too short to verify."
            if private_short
            else ""
        )
        raise ValueError(
            "the evaluated private stream overlaps the SDA auxiliary corpus: "
            f"{private_pairs} exact / near-duplicate / unverifiable-short "
            "pair(s) remain between a private test document and a "
            "train/validation/control document."
            + private_short_detail
            + " The model must never be fitted on a document it is scored on; "
            "private texts must be disjoint from reference_texts, control_texts "
            "and validation_texts."
        )
    if not overlap["passed"] and require_isolated_control:
        short_pairs = overlap["unverifiable_short_documents"]["pairs"]
        short_detail = (
            f" {short_pairs} cross-role pair(s) involve a document shorter "
            f"than SHINGLE_SIZE={SHINGLE_SIZE} tokens and are too short to "
            "verify: a verbatim training window of that length cannot be "
            "excluded."
            if short_pairs
            else ""
        )
        raise ValueError(
            "cross-role document overlap makes the SDA run non-isolated: "
            f"{overlap['exact']['collisions']} exact and "
            f"{overlap['near_duplicate']['collisions']} near-duplicate "
            "train/validation/control pair(s) remain (thresholds: exact sha256; "
            f"shingle Jaccard >= {NEAR_DUPLICATE_JACCARD}, containment >= "
            f"{NEAR_DUPLICATE_CONTAINMENT})."
            + short_detail
            + " The held-out control must be a document the model never "
            "trained on."
        )

    if validation_beams is not None:
        if not overlap["passed"]:
            raise ValueError("validation selection requires isolated role documents")
        if not role_docs["validation"]:
            raise ValueError("beam selection requires non-empty isolated validation documents")

    control_docs: list[dict] = []
    if isinstance(control_keys, bool) or not isinstance(control_keys, int) or control_keys <= 0:
        raise ValueError("control_keys must be a positive integer")
    if positive_control:
        _select_control_document(role_docs["control"])
        control_docs = list(role_docs["control"])
    # Sizing only checks configured length; control plaintext never enters fit_sda.
    sizing_ciphers = list(cipher_sequences) + [doc["tokens"] for doc in control_docs]

    if group_joint and (training_params or {}).get("length_mode", "fixed") != "fixed":
        raise ValueError("group_joint is unavailable for variable-length decoding")
    train_ids = [document["tokens"] for document in role_docs["train"]]
    bundle = fit_sda(
        train_ids, sizing_ciphers, vocab_size=int(dep.vocab_size),
        method=method,
        params={**dict(training_params or {}), "num_train": num_train, "epochs": epochs},
        seed=seed, device=device, checkpoint_path=checkpoint_path, resume=resume,
        validation_ids=([document["tokens"] for document in role_docs["validation"]]
                        if overlap["passed"] else None),
        validation_seed=split_seed,
    )
    validation = {"status": "unused", "reason": "no validation beam candidates supplied",
                  "saturation_status": "not_assessed"}
    requested_beam = beam_width
    if validation_beams is not None:
        validation_documents = role_docs["validation"]
        if bundle["validation_sampling"] is not None:
            validation_documents = []
            for window in bundle["validation_sampling"]["windows"]:
                source = role_docs["validation"][window["document_index"]]
                cropped = source["tokens"][window["start"]:window["end"]]
                if len(cropped) != window["end"] - window["start"] or document_sha256(cropped) != window["content_sha256"]:
                    raise ValueError("frozen validation window does not match its source")
                validation_documents.append(_document_record({"source": source["source"], "index": source["index"],
                                                               "tokens": cropped}, role="validation"))
        validation = _select_validation_beam(
            bundle, validation_documents, vocab_size=int(dep.vocab_size),
            candidates=validation_beams, split_seed=split_seed,
            noise_rate=validation_noise_rate, noise_kinds=validation_noise_kinds,
            use_cache=decode_cache, decode_mode=decode_mode,
        )
        beam_width = validation["selected_beam"]
    epoch_budget_curve = None
    if budget_checkpoints is not None:
        epoch_budget_curve = compare_sda_checkpoint_budgets(
            budget_checkpoints, [document["tokens"] for document in role_docs["validation"]],
            beam_width=beam_width, device=device, noise_rate=validation_noise_rate,
            noise_kinds=validation_noise_kinds, use_cache=decode_cache, decode_mode=decode_mode)
        if epoch_budget_curve["checkpoint_identity"] != bundle["checkpoint_identity"]:
            raise ValueError("budget checkpoints do not match the current training identity")
    rows = []
    text_predictions = {}
    for index, (cipher, truth) in enumerate(zip(cipher_sequences, truth_plain)):
        decoded = decode_sda(bundle, cipher, beam_width=beam_width, use_cache=decode_cache, mode=decode_mode)
        if text_surface:
            predicted_text = text_decoder(decoded["tokens"]) if decoded["tokens"] else ""
            if not isinstance(predicted_text, str):
                raise TypeError("text_decoder must return a string")
            text_predictions[request_ids[index]] = predicted_text
            accuracy = {"status": "not_evaluated", "value": None,
                        "reason": "uploaded text and reference token positions are not assumed aligned"}
            edit_error = None
        else:
            accuracy = _token_accuracy(decoded["tokens"], truth)
            edit_error = (_token_edit_error(decoded["tokens"], truth)
                          if bundle.get("length_mode", "fixed") != "fixed" else None)
        rows.append({
            "request_id": request_ids[index], "frontend_candidates": frontend_candidates[index],
            "recovered": decoded["tokens"], "decode": decoded,
            "accuracy": accuracy, "edit_error": edit_error,
        })
    text_scores = None
    if text_surface:
        from .sda_text_scoring import score_sda_text_rows
        text_scores = score_sda_text_rows(dict(zip(request_ids, selected)), text_predictions,
                                          entities=text_entities, max_cells=text_alignment_max_cells)
        text_scores["decoder_provenance"] = "caller_supplied_unverified"
        frontend_details["text_alignment"] = "scoring_side_unicode_levenshtein"
    joint_rows = []
    if group_joint:
        from .sda_group import decode_sda_group
        groups: dict[tuple, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            key = ("group", group_id) if group_id is not None else ("request", request_ids[index])
            groups.setdefault(key, []).append(index)
        for group, indices in groups.items():
            constrained = decode_sda_group(bundle, [cipher_sequences[i] for i in indices],
                                           beam_width=beam_width, use_cache=decode_cache)
            if constrained.get("status") != "ok":
                joint_rows.append({"group": list(group), "request_ids": [request_ids[i] for i in indices],
                                   "status": "infeasible",
                                   "requests": [{"request_id": request_ids[i], "recovered": [],
                                                 "accuracy": _token_accuracy([], truth_plain[i])}
                                                for i in indices]})
                continue
            offset = 0
            request_rows = []
            for index in indices:
                length = len(cipher_sequences[index])
                recovered = constrained["tokens"][offset:offset + length]
                request_rows.append({"request_id": request_ids[index], "recovered": recovered,
                                     "accuracy": _token_accuracy(recovered, truth_plain[index])})
                offset += length
            joint_rows.append({"group": list(group), "request_ids": [request_ids[i] for i in indices], "status": "ok",
                               "mapping": constrained["mapping"], "score": constrained["score"],
                               "requests": request_rows})
    hits = sum(row["accuracy"]["numerator"] for row in rows) if not text_surface else None
    total = sum(row["accuracy"]["denominator"] for row in rows) if not text_surface else None
    joint_hits = sum(row["accuracy"]["numerator"] for group in joint_rows for row in group["requests"])
    joint_total = sum(row["accuracy"]["denominator"] for group in joint_rows for row in group["requests"])
    clustered_token_scoring = None
    if scoring_clusters is not None:
        from .sda_token_scoring import score_sda_token_rows
        clustered_token_scoring = score_sda_token_rows(
            dict(zip(request_ids, truth_plain)),
            {row["request_id"]: row["recovered"] for row in rows},
            clusters=scoring_clusters, seed=split_seed,
        )
    n_train_docs = len(role_docs["train"])
    result = {
        "method": method,
        "encoding": bundle["encoding"],
        "objective": bundle["objective"],
        "n_training_pairs": int(bundle["pairs"]),
        "n_train_sequences": None,  # repeated sampled windows are not independent sequences
        "optimizer_steps": bundle["optimizer_steps"],
        "examples_seen": bundle["examples_seen"],
        "training_history": bundle["training_history"],
        "sampling_evidence": bundle["sampling_evidence"],
        "validation_sampling": bundle["validation_sampling"],
        "checkpoint_identity": bundle["checkpoint_identity"],
        "pair_storage": {"mode": bundle["pair_storage"],
                         "window_provenance": "manifest" if bundle["pair_storage"] == "disk" else "in_memory"},
        "model_config": vars(bundle["cfg"]),
        "device": str(next(bundle["model"].parameters()).device),
        "epochs": int(epochs),
        "beam_width": int(beam_width),
        "decode_mode": decode_mode,
        "requested_beam_width": int(requested_beam),
        "decode_cache": "kv_incremental" if decode_cache else "full_prefix",
        "validation": validation,
        "epoch_budget_curve": epoch_budget_curve,
        "validation_checkpoint": bundle["validation_checkpoint"],
        "seed": int(seed),
        "n_requests": len(rows),
        "public_group_count": len(set(group_ids)) if all(g is not None for g in group_ids) else None,
        "reference_tokens": sum(int(doc["length"]) for doc in role_docs["train"]),
        "token_accuracy": ({
            "status": "not_evaluated", "value": None,
            "reason": "cross-tokenizer reference positions lack token-level alignment",
        } if text_surface else {
            "value": hits / total if total else None,
            "numerator": int(hits),
            "denominator": int(total),
            "alignment": "positionwise_diagnostic" if bundle.get("length_mode", "fixed") != "fixed" else "positionwise",
        }),
        "text_scoring": text_scores,
        "clustered_token_scoring": clustered_token_scoring,
        "token_edit_error": ({
            "value": sum(row["edit_error"]["edits"] for row in rows) / total if total else None,
            "edits": sum(row["edit_error"]["edits"] for row in rows),
            "reference_tokens": int(total), "metric": "token_levenshtein_error_not_TER",
        } if not text_surface and bundle.get("length_mode", "fixed") != "fixed" else None),
        "length_mode": bundle.get("length_mode", "fixed"),
        "space_token_id": bundle.get("space_token_id"),
        "rows": rows,
        "group_joint": {"enabled": group_joint, "variant": "shared_mapping_per_request_context_beam_probe",
                        "groups": joint_rows,
                        "token_accuracy": ({"value": joint_hits / joint_total if joint_total else None,
                                            "numerator": joint_hits, "denominator": joint_total}
                                           if group_joint else None)},
        "cipher_alphabet_size": cipher_alphabet_size,
        "public_uploads_sha256": public_uploads_sha256,
        "processed_observations_sha256": processed_observations_sha256,
        "scoring_join": "request_id" if frozen_uploads is not None else "same_observation_truth",
        "observation_frontend": {
            **frontend_details,
            "limitations": "symbolization may split repeated symbols or merge distinct symbols; threshold selection and front-end accuracy require independent validation",
        },
        "frontend_scoring": ({"status": "not_evaluated", "reason": "cross-tokenizer symbols lack token-level alignment"}
                             if text_surface else _score_symbol_frontend(cipher_sequences, truth_plain, group_ids)),
        "model_fidelity": ("experimental_" + getattr(bundle["cfg"], "representation", "lookup")
                           + "_unverified_against_author_code"),
        "training_pairs_source": (
            "attacker-owned auxiliary plaintext (training role only); "
            + ("direct key-invariant encoding without materialising keys"
               if bundle["synthesis_mode"] == "invariant" else "fresh random keys")
        ),
        "synthesis": {"mode": bundle["synthesis_mode"],
                      "rng_policy": ("window sampling without key RNG draws; differs from legacy at the same seed"
                                     if bundle["synthesis_mode"] == "invariant" else "legacy shared window/key RNG sequence")},
        "corpus_roles": {
            "rule": SPLIT_RULE,
            "seed": int(split_seed),
            "control_fraction": float(control_fraction),
            "validation_fraction": float(validation_fraction),
            "requested_validation_documents": n_validation,
            "counts": {
                "train": n_train_docs,
                "validation": len(role_docs["validation"]),
                "control": len(role_docs["control"]),
                "private": len(role_docs["private"]),
                "ignored_empty": int(n_empty),
            },
            "hashes": {
                role: _role_summary(documents)
                for role, documents in role_docs.items()
            },
            "private_usage": {
                "status": "evaluated",
                "reason": (
                    "the evaluated private texts form the 'private' role and "
                    "are checked for overlap against every auxiliary role"
                ),
            },
            "explicit_corpora": {
                "control_texts": control_texts is not None,
                "validation_texts": validation_texts is not None,
            },
            "validation_usage": validation,
        },
        "overlap_check": {
            **overlap,
            "required": bool(require_isolated_control),
            "enforced": bool(require_isolated_control),
        },
        "training_provenance": {
            "generator": "synthesize_pairs_2021" if bundle["method"] == "sda2021" else "synthesize_pairs_2023",
            "n_requested_pairs": int(num_train),
            "documents": [{"document_index": i, "source": doc["source"],
                           "source_index": int(doc["index"]), "sha256": doc["sha256"],
                           "length": int(doc["length"])}
                          for i, doc in enumerate(role_docs["train"])],
            **bundle["window_evidence"],
        },
    }
    if positive_control:
        control_rows = []
        lm_generate = _unconditional_bigram(train_ids)
        from itertools import chain
        prior_order = _frequency_order(chain.from_iterable(train_ids)) or [0]
        control_clusters, control_cluster_report = cluster_control_documents(control_docs)
        control_rng = np.random.default_rng(split_seed + 991)
        seen_controls = set()
        for replica in range(control_keys):
            key = enc.make_substitution_key(
                range(int(dep.vocab_size)), rng=control_rng, homophonic=False,
                cipher_alphabet_size=int(dep.vocab_size),
            )
            key_bytes = np.fromiter((key.plain_to_cipher[i][0] for i in range(int(dep.vocab_size))),
                                    dtype="<i8", count=int(dep.vocab_size)).tobytes()
            key_digest = hashlib.sha256(key_bytes).digest()
            if key_digest in seen_controls:
                raise ValueError("independent control key draw repeated; use a different control seed")
            seen_controls.add(key_digest)
            for doc_index, document in enumerate(control_docs):
                source = document["tokens"]
                if bundle.get("space_token_id") is not None:
                    source = [token for token in source if token != bundle["space_token_id"]]
                if not len(source):
                    raise ValueError("space removal produced an empty control ciphertext")
                cipher = np.asarray(enc.apply_substitution(source, key), dtype=np.int64)
                truth = np.asarray(document["tokens"], dtype=np.int64)
                decoded = decode_sda(bundle, cipher, beam_width=beam_width, use_cache=decode_cache, mode=decode_mode)
                accuracy = _token_accuracy(decoded["tokens"], truth)
                baselines = _prior_baseline(cipher, truth, train_ids, vocab_size=int(dep.vocab_size), prior_order=prior_order)
                lm_prediction = lm_generate(len(cipher))
                baselines["unconditional_bigram_lm"] = {
                    **_token_accuracy(lm_prediction, truth),
                    "prediction": lm_prediction,
                    "description": "greedy bigram LM trained on public training documents; sees length only",
                }
                if bundle.get("length_mode", "fixed") != "fixed":
                    for baseline_name in ("frequency_prior", "unconditional_bigram_lm"):
                        baselines[baseline_name]["edit_error"] = _token_edit_error(
                            baselines[baseline_name]["prediction"], truth)
                control_rows.append({
                    "control_replica": replica, "source_role": "control", "source": document["source"],
                    "source_index": int(document["index"]), "source_sha256": document["sha256"],
                    "source_cluster": control_clusters[doc_index],
                    "recovered": decoded["tokens"], "accuracy": accuracy,
                    "edit_error": (_token_edit_error(decoded["tokens"], truth)
                                   if bundle.get("length_mode", "fixed") != "fixed" else None), "baselines": baselines,
                })
        control_hits = sum(row["accuracy"]["numerator"] for row in control_rows)
        control_total = sum(row["accuracy"]["denominator"] for row in control_rows)
        interval = _crossed_control_interval if control_keys > 1 else _paired_control_interval
        control_metric = "token_edit_error" if bundle.get("length_mode", "fixed") != "fixed" else "token_accuracy"
        comparisons = {name: interval(control_rows, name, seed=split_seed, metric=control_metric)
                       for name in ("frequency_prior", "unconditional_bigram_lm")}
        first = control_rows[0]
        result["positive_control"] = {
            "name": "fixed_substitution_heldout_auxiliary",
            "value": control_hits / control_total if control_total else None,
            "numerator": control_hits, "denominator": control_total,
            "passed": None, "status": "not_evaluated",
            "pass_rule": "formal gate requires frozen validation selection, supported independent clusters and audited comparisons",
            "source_role": "control", "source": first["source"],
            "source_index": first["source_index"], "source_sha256": first["source_sha256"],
            "isolated_from_training": bool(overlap["passed"]),
            "isolation_enforced": bool(require_isolated_control),
            "baseline": first["baselines"], "prior_baseline": first["baselines"],
            "control_significance": _control_significance(first["accuracy"], first["baselines"]),
            "comparisons": comparisons, "rows": control_rows,
            "control_key_count": control_keys, "comparison_metric": control_metric,
            "document_clustering": control_cluster_report,
            "control_surface": "synthetic_direct_symbols_clean",
            "scope_note": "crossed intervals are descriptive; actual-deployment front-end controls remain required",
            "value_semantics": "positionwise_diagnostic" if control_metric == "token_edit_error" else "token_accuracy",
            "key_policy": ("independent synthetic fixed substitutions crossed with control documents"
                           if control_keys > 1 else
                           "one fixed substitution shared across control documents; conditional inference only"),
        }
    result["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  SDA {method}: acc={result['token_accuracy']['value']} "
              f"control={result.get('positive_control', {}).get('value')} "
              f"({result['seconds']}s)", flush=True)
    return result
