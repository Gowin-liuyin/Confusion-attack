"""Disk-backed exact equivalent of SDA's long-document near-duplicate grouping."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain
import sqlite3
import tempfile
from pathlib import Path

from .sda_stream import (NEAR_DUPLICATE_CONTAINMENT, NEAR_DUPLICATE_JACCARD,
                         SHINGLE_SIZE, check_document_isolation, document_sha256, shingle_hashes)


def cluster_long_documents_disk(documents: Iterable[Sequence[int]], *, parent_dir=None) -> tuple[list[str], dict]:
    """Group a stream of >=5-token documents with the same thresholds as the in-memory path.

    Only hashes, counts and union-find parents remain in memory; the temporary
    SQLite inverted index holds the shingle postings. This is an exact candidate
    search for the configured shingle heuristic, not proof of independence.
    """
    hashes, sizes, parent = [], [], []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    with tempfile.TemporaryDirectory(prefix="sda-cluster-", dir=parent_dir) as directory:
        connection = sqlite3.connect(str(Path(directory) / "index.sqlite3"))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE postings (shingle BLOB NOT NULL, doc INTEGER NOT NULL, PRIMARY KEY(shingle, doc)) WITHOUT ROWID")
            connection.execute("CREATE TEMP TABLE current (shingle BLOB PRIMARY KEY) WITHOUT ROWID")
            exact = {}
            for index, tokens in enumerate(documents):
                if len(tokens) < SHINGLE_SIZE:
                    raise ValueError("disk clustering requires documents at least SHINGLE_SIZE tokens long")
                digest = document_sha256(tokens)
                shingles = shingle_hashes(tokens)
                values = [(value.to_bytes(8, "little"),) for value in shingles]
                parent.append(index)
                hashes.append(digest)
                sizes.append(len(shingles))
                if digest in exact:
                    union(index, exact[digest])
                else:
                    exact[digest] = index
                connection.execute("DELETE FROM current")
                connection.executemany("INSERT INTO current VALUES (?)", values)
                # CROSS JOIN fixes the loop order: probe postings for each current shingle.
                query = ("SELECT p.doc, COUNT(*) FROM current AS c CROSS JOIN postings AS p "
                         "WHERE p.shingle = c.shingle GROUP BY p.doc")
                for other, intersection in connection.execute(query):
                    union_size = sizes[index] + sizes[other] - intersection
                    smaller = min(sizes[index], sizes[other])
                    if ((intersection / union_size if union_size else 1.0) >= NEAR_DUPLICATE_JACCARD
                            or (intersection / smaller if smaller else 1.0) >= NEAR_DUPLICATE_CONTAINMENT):
                        union(index, other)
                connection.executemany("INSERT INTO postings VALUES (?, ?)", ((value[0], index) for value in values))
                if index % 128 == 127:
                    connection.commit()
            connection.commit()
        finally:
            connection.close()
    components = {}
    for index in range(len(parent)):
        root = find(index)
        components[root] = min(components.get(root, hashes[index]), hashes[index])
    labels = [components[find(index)] for index in range(len(parent))]
    return labels, {"n_documents": len(hashes), "n_distinct_hashes": len(set(hashes)),
                    "n_clusters": len(components), "shingle_size": SHINGLE_SIZE,
                    "jaccard_threshold": NEAR_DUPLICATE_JACCARD,
                    "containment_threshold": NEAR_DUPLICATE_CONTAINMENT,
                    "short_rule": "documents shorter than shingle_size are rejected",
                    "independence": "heuristic grouping does not prove independent source documents",
                    "storage": "temporary_sqlite_index"}


def check_document_isolation_disk(roles: dict, *, parent_dir=None, max_examples: int = 10) -> dict:
    """Check cross-role exact, short-document and shingle overlap using disk postings.

    Each role holds dictionaries with source, index and tokens. Documents are
    compared directly across roles; within-role connected components are not
    used as a proxy for overlap.
    """
    report = check_document_isolation({}, max_examples=max_examples)
    exact_examples, near_examples, short_examples = [], [], []
    max_jaccard = max_containment = 0.0
    total_exact = total_near = total_short = 0

    def prepare(role):
        docs = []
        for doc in roles.get(role, ()):
            tokens = doc["tokens"]
            docs.append({"role": role, "source": str(doc["source"]), "index": int(doc["index"]),
                         "tokens": tokens, "length": len(tokens), "sha256": document_sha256(tokens)})
        return docs

    def label(doc):
        return {key: doc[key] for key in ("role", "source", "index", "sha256")}

    documents = {role: prepare(role) for role in ("train", "validation", "control", "private")}
    for left_role, right_role in report["role_pairs_checked"]:
        left, right = documents[left_role], documents[right_role]
        exact, near, short = 0, 0, 0
        hashes = {}
        for doc in left:
            hashes.setdefault(doc["sha256"], []).append(doc)
        for doc in right:
            for other in hashes.get(doc["sha256"], ()):
                exact += 1
                if len(exact_examples) < max_examples:
                    exact_examples.append({"left": label(other), "right": label(doc), "kind": "exact"})
        short_left = [doc for doc in left if doc["length"] < SHINGLE_SIZE]
        short_right = [doc for doc in right if doc["length"] < SHINGLE_SIZE]
        long_left = [doc for doc in left if doc["length"] >= SHINGLE_SIZE]
        long_right = [doc for doc in right if doc["length"] >= SHINGLE_SIZE]
        for a, b in chain(((a, b) for a in short_left for b in right),
                          ((a, b) for a in long_left for b in short_right)):
            if a["sha256"] == b["sha256"] or set(map(int, a["tokens"])).isdisjoint(map(int, b["tokens"])):
                continue
            short += 1
            if len(short_examples) < max_examples:
                short_doc, other = (a, b) if a["length"] < SHINGLE_SIZE else (b, a)
                short_examples.append({"short_document": {**label(short_doc), "length": short_doc["length"]},
                    "compared_with": label(other),
                    "reason": f"document shorter than SHINGLE_SIZE={SHINGLE_SIZE} shares tokens; near-duplicate status cannot be established"})
        indexed, queries = (long_right, long_left) if len(long_right) > len(long_left) else (long_left, long_right)
        if indexed and queries:
            with tempfile.TemporaryDirectory(prefix="sda-isolation-", dir=parent_dir) as directory:
                connection = sqlite3.connect(str(Path(directory) / "index.sqlite3"))
                try:
                    connection.execute("PRAGMA temp_store=FILE")
                    connection.execute("CREATE TABLE postings (shingle BLOB NOT NULL, doc INTEGER NOT NULL, PRIMARY KEY(shingle, doc)) WITHOUT ROWID")
                    connection.execute("CREATE TEMP TABLE current (shingle BLOB PRIMARY KEY) WITHOUT ROWID")
                    sizes = []
                    for index, doc in enumerate(indexed):
                        shingles = shingle_hashes(doc["tokens"])
                        sizes.append(len(shingles))
                        connection.executemany("INSERT INTO postings VALUES (?, ?)",
                            ((value.to_bytes(8, "little"), index) for value in shingles))
                        if index % 128 == 127:
                            connection.commit()
                    connection.commit()
                    query = ("SELECT p.doc, COUNT(*) FROM current AS c CROSS JOIN postings AS p "
                             "WHERE p.shingle = c.shingle GROUP BY p.doc")
                    for doc in queries:
                        shingles = shingle_hashes(doc["tokens"])
                        connection.execute("DELETE FROM current")
                        connection.executemany("INSERT INTO current VALUES (?)",
                                               ((value.to_bytes(8, "little"),) for value in shingles))
                        for index, intersection in connection.execute(query):
                            other = indexed[index]
                            if doc["sha256"] == other["sha256"]:
                                continue
                            union_size = len(shingles) + sizes[index] - intersection
                            smaller = min(len(shingles), sizes[index])
                            jaccard = intersection / union_size if union_size else 1.0
                            containment = intersection / smaller if smaller else 1.0
                            max_jaccard = max(max_jaccard, jaccard)
                            max_containment = max(max_containment, containment)
                            if jaccard >= NEAR_DUPLICATE_JACCARD or containment >= NEAR_DUPLICATE_CONTAINMENT:
                                near += 1
                                if len(near_examples) < max_examples:
                                    near_examples.append({"left": label(other), "right": label(doc),
                                        "jaccard": round(jaccard, 6), "containment": round(containment, 6),
                                        "kind": "near_duplicate"})
                finally:
                    connection.close()
        report["by_role_pair"][f"{left_role}~{right_role}"] = {
            "checked_pairs": len(left) * len(right), "exact": exact, "near_duplicate": near,
            "unverifiable_short": short}
        total_exact += exact
        total_near += near
        total_short += short
    report["checked_pairs"] = sum(row["checked_pairs"] for row in report["by_role_pair"].values())
    report["exact"].update({"collisions": total_exact, "examples": exact_examples, "passed": total_exact == 0})
    report["near_duplicate"].update({"collisions": total_near, "examples": near_examples,
        "max_jaccard_observed": round(max_jaccard, 6), "max_containment_observed": round(max_containment, 6),
        "passed": total_near == 0})
    report["unverifiable_short_documents"].update({"pairs": total_short, "examples": short_examples, "passed": total_short == 0})
    report["passed"] = total_exact == total_near == total_short == 0
    report["storage"] = "temporary_sqlite_index"
    return report
