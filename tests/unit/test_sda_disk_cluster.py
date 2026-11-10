import random

import pytest

from obfuscation_attacks.evaluation.sda_disk_cluster import cluster_long_documents_disk, check_document_isolation_disk
from obfuscation_attacks.evaluation.sda_stream import _document_record, cluster_control_documents, check_document_isolation


def test_disk_cluster_matches_full_memory_heuristic_and_transitive_components(tmp_path):
    rng = random.Random(19)
    base = [rng.randrange(100) for _ in range(50)]
    first = base[:]
    first[-1] = 101
    second = first[:]
    second[0] = 102
    documents = [base, first, second, base[:], list(range(5, 65)), list(range(30, 90))]
    documents.extend([[rng.randrange(1000) for _ in range(rng.randrange(5, 70))] for _ in range(25)])
    records = [_document_record({"source": "test", "index": i, "tokens": doc}, role="test")
               for i, doc in enumerate(documents)]
    expected, metadata = cluster_control_documents(records)
    actual, report = cluster_long_documents_disk(iter(documents), parent_dir=tmp_path)
    assert actual == expected
    assert report["n_clusters"] == metadata["n_clusters"]
    assert report["n_distinct_hashes"] == metadata["n_distinct_hashes"]
    assert not list(tmp_path.iterdir())


def test_disk_cluster_refuses_short_docs_and_cleans_temporary_index(tmp_path):
    with pytest.raises(ValueError, match="SHINGLE_SIZE"):
        cluster_long_documents_disk([[1, 2, 3, 4]], parent_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_disk_cross_role_isolation_matches_memory_counts_and_maxima(tmp_path):
    docs = {"train": [list(range(20)), [1, 2, 3], list(range(30, 50)), list(range(20))],
            "validation": [list(range(20)), list(range(19)) + [99], [3, 4], list(range(31, 51))],
            "control": [[88, 89, 90], list(range(31, 51))],
            "private": [list(range(30, 50)), [1, 90]]}
    plain = {role: [{"source": role, "index": index, "tokens": tokens}
                    for index, tokens in enumerate(rows)] for role, rows in docs.items()}
    records = {role: [_document_record(doc, role=role) for doc in rows] for role, rows in plain.items()}
    expected = check_document_isolation(records)
    actual = check_document_isolation_disk(plain, parent_dir=tmp_path)
    assert actual["passed"] == expected["passed"]
    assert actual["checked_pairs"] == expected["checked_pairs"]
    assert actual["by_role_pair"] == expected["by_role_pair"]
    for name, field in (("exact", "collisions"), ("near_duplicate", "collisions"),
                        ("unverifiable_short_documents", "pairs")):
        assert actual[name][field] == expected[name][field]
        assert actual[name]["passed"] == expected[name]["passed"]
    for field in ("max_jaccard_observed", "max_containment_observed"):
        assert actual["near_duplicate"][field] == expected["near_duplicate"][field]
    assert not list(tmp_path.iterdir())
