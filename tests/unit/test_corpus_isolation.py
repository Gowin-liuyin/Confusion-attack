import json

import numpy as np
import pytest

from obfuscation_attacks.data.corpora import CLUSTER_ALGORITHM, CLUSTERED_SPLIT, audit_corpus_isolation, cluster_near_duplicate_texts, load_alpaca


def test_clustered_split_keeps_near_duplicates_in_one_role(tmp_path):
    rng = np.random.default_rng(12)
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    texts = ["".join(rng.choice(alphabet, 90)) for _ in range(30)]
    texts += [texts[0] + "extra ending", texts[1].upper(), texts[2][:60] + "new"]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"instruction": text} for text in texts]), encoding="utf-8")
    split = load_alpaca(path, seed=17, n_knn=5, n_vma=4, n_robust=3,
                        split_strategy=CLUSTERED_SPLIT)
    role_of = {row["source_index"]: role for role, items in split.items() for row in items.rows}
    assert all(index in role_of for index in (0, 1, 2))
    assert all(index not in role_of for index in (30, 31, 32))
    assert [len(split[role]) for role in ("knn", "vma", "robust")] == [5, 4, 3]
    meta = split["knn"].manifest
    assert meta["isolation"]["near_duplicate_status"] == "clustered_by_recorded_heuristic"
    assert meta["isolation"]["discarded_near_duplicates"] == 3
    assert len(meta["split_sha256"]) == 64
    repeat = load_alpaca(path, seed=17, n_knn=5, n_vma=4, n_robust=3,
                         split_strategy=CLUSTERED_SPLIT)
    assert repeat["knn"].manifest["split_sha256"] == meta["split_sha256"]
    legacy = load_alpaca(path, seed=17, n_knn=5, n_vma=4, n_robust=3,
                         split_strategy=CLUSTER_ALGORITHM)
    legacy_roles = {row["source_index"]: role for role, items in legacy.items() for row in items.rows}
    assert legacy_roles[0] == legacy_roles[30]
    assert not audit_corpus_isolation(legacy)["passed"]
    audit = audit_corpus_isolation(split)
    assert audit["passed"] and audit["split_sha256"] == meta["split_sha256"]
    split["knn"].rows.append(split["knn"].rows[0])
    within = audit_corpus_isolation(split)
    assert not within["passed"] and within["within_role_duplicate_clusters"] == 1
    split["knn"].rows.pop()
    split["vma"].rows.append(split["knn"].rows[0])
    tampered = audit_corpus_isolation(split)
    assert not tampered["passed"] and tampered["split_sha256"] != meta["split_sha256"]


def test_cluster_heuristic_detects_containment_and_fails_closed_on_limit():
    base = "the quick brown fox jumps over the old stone wall at sunset"
    short = "brown fox jumps over the old stone wall"
    groups, report = cluster_near_duplicate_texts([base, short, "an unrelated passage about astronomy and optics"])
    assert any(set(group) == {0, 1} for group in groups)
    assert report["n_clusters"] == 2
    with pytest.raises(ValueError, match="posting limit"):
        cluster_near_duplicate_texts([base, short], max_posting_visits=0)
    with pytest.raises(ValueError, match="five normalized characters"):
        cluster_near_duplicate_texts(["a"])


def test_cluster_candidates_match_exhaustive_reference():
    rng = np.random.default_rng(101)
    alphabet = np.array(list("abcdefghijklm"))
    originals = ["".join(rng.choice(alphabet, 70)) for _ in range(12)]
    texts = originals + [text[5:60] + "zzz" for text in originals[:6]]
    signatures = [{text[i:i + 5] for i in range(len(text) - 4)} for text in texts]
    parents = list(range(len(texts)))

    def find(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for left in range(len(texts)):
        for right in range(left):
            if 5 * len(signatures[left] & signatures[right]) >= 4 * min(len(signatures[left]), len(signatures[right])):
                a, b = find(left), find(right)
                parents[max(a, b)] = min(a, b)
    expected = {frozenset(i for i in range(len(texts)) if find(i) == root)
                for root in {find(i) for i in range(len(texts))}}
    actual, _ = cluster_near_duplicate_texts(texts)
    assert {frozenset(group) for group in actual} == expected
