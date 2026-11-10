"""Acceptance tests for F11: stream cache binding, truth separation, matched negatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obfuscation_attacks.evaluation import real_battery
from obfuscation_attacks.evaluation.real_battery import (
    _matched_linkage_pairs,
    build_stream,
    run_c12,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit


class FakeDeployment:
    artifact = Path("fixture-artifact")
    clean_model_path = Path("fixture-clean")
    max_len = 512
    vocab_size = 32

    def table(self, arm: str, normalize: bool = False) -> np.ndarray:
        # Non-zero rows: a zero-norm deployment row is an invalid candidate
        # universe and the KNN now refuses it.
        return np.eye(32, 4, dtype=np.float32) + 0.1

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None):
        ids = np.asarray(text, dtype=np.int64)
        index = 0 if key_index is None else int(key_index)
        z = np.repeat((index * 1000 + ids).astype(np.float32)[:, None], 4, axis=1)
        public = WirePublic(
            arm="ab", request_id=request_id, seq_len=len(ids), wire_vectors=z,
            group_id=f"g-{index}", epoch=epoch, session_id=session_id,
        )
        truth = WireTruth(plaintext_ids=ids.copy(), permuted_ids=ids.copy())
        return Wire(public, truth)


TEXTS = [
    np.array([1, 2, 3, 4, 5]),
    np.array([1, 2, 6, 7, 8]),
    np.array([1, 3, 6, 9, 10]),
    np.array([2, 3, 7, 9, 11]),
    np.array([1, 4, 6, 8, 11]),
    np.array([2, 5, 7, 10, 11]),
    np.array([1, 2, 3, 6, 7]),
    np.array([1, 3, 5, 8, 9]),
    np.array([2, 4, 6, 9, 10]),
    np.array([1, 5, 7, 9, 11]),
]


def test_public_stream_carries_no_truth_or_key_material():
    stream = build_stream(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=1)
    for record in stream["requests"]:
        assert "plaintext_ids" not in record
        assert "permuted_ids" not in record
        assert "key_seed" not in record and "key_index" not in record
        assert record["group_id"].startswith("g-")
        assert "epoch" in record
    assert len(stream["truth"]) == len(stream["requests"])
    assert "plaintext_ids" in stream["truth"][0]


def test_cache_is_bound_to_the_full_public_configuration():
    cache: dict = {}
    dep = FakeDeployment()
    first = build_stream(dep, TEXTS, n_keys=2, rotate_every=4, seed=1, cache=cache)
    again = build_stream(dep, TEXTS, n_keys=2, rotate_every=4, seed=1, cache=cache)
    assert again is first
    changed = build_stream(dep, TEXTS, n_keys=4, rotate_every=4, seed=1, cache=cache)
    assert changed is not first
    assert changed["stream_key"] != first["stream_key"]


def test_negatives_are_position_matched_not_the_first_token():
    stream = build_stream(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=2)
    pairs = _matched_linkage_pairs(stream["requests"], stream["truth"], seed=2)
    assert pairs["positives"] and pairs["negatives"]
    first_token_fraction = np.mean([pb == 0 for _, _, _, pb, _ in pairs["negatives"]])
    assert first_token_fraction < 0.9
    for a, pa, b, pb, _ in pairs["negatives"]:
        assert stream["truth"][a]["plaintext_ids"][pa] != stream["truth"][b]["plaintext_ids"][pb]


def test_cluster_ci_resamples_sessions_not_independent_pairs():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    out = _cluster_rate_ci(
        [True, False], [("S1", "S2"), ("S1", "S3")], n_boot=200, seed=0
    )
    assert out["estimand"] == "session bootstrap"
    assert out["n_clusters"] == 3


def test_session_bootstrap_interval_contains_its_point_estimate():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    outcomes = [True] * 20 + [False] * 9
    clusters = [("S0", "S0")] * 20 + [("S0", "S1")] * 9
    out = _cluster_rate_ci(outcomes, clusters, n_boot=1000, seed=0)
    assert out["lo"] <= out["value"] <= out["hi"]


def test_degenerate_cluster_intervals_report_no_bounds():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    single_session = _cluster_rate_ci(
        [True, False], [("S1", "S1"), ("S1", "S1")], n_boot=200, seed=0
    )
    assert single_session["value"] == 0.5
    assert single_session["n_clusters"] == 1
    assert single_session["n_boot"] == 0
    assert single_session["lo"] is None and single_session["hi"] is None

    single_pair_cluster = _cluster_rate_ci(
        [True, False], ["c", "c"], n_boot=200, seed=0
    )
    assert single_pair_cluster["value"] == 0.5
    assert single_pair_cluster["n_clusters"] == 1
    assert single_pair_cluster["n_boot"] == 0
    assert single_pair_cluster["lo"] is None and single_pair_cluster["hi"] is None

    empty = _cluster_rate_ci([], [], n_boot=200, seed=0)
    assert empty["value"] is None and empty["n_boot"] == 0
    assert empty["lo"] is None and empty["hi"] is None


def test_fewer_than_two_replicates_reports_no_bounds():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    zero = _cluster_rate_ci([True, False], ["a", "b"], n_boot=0)
    assert zero["value"] == 0.5 and zero["n_boot"] == 0
    assert zero["lo"] is None and zero["hi"] is None

    one = _cluster_rate_ci([True, False], ["a", "b"], n_boot=1, seed=0)
    assert one["value"] == 0.5 and one["n_boot"] == 0
    assert one["lo"] is None and one["hi"] is None

    two = _cluster_rate_ci([True, False], ["a", "b"], n_boot=2, seed=0)
    assert two["value"] == 0.5 and two["n_boot"] == 2
    assert two["lo"] is not None and two["hi"] is not None
    assert 0.0 <= two["lo"] <= two["hi"] <= 1.0


def test_linkage_selects_thresholds_on_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(
        FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=3, verbose=False,
        session_sets=[[f"session-{index}" for index in range(len(TEXTS))]],
    )
    assert out["approximate"]
    for entry in out["approximate"].values():
        assert "threshold_from_validation" in entry
        assert entry["tpr"]["denominator"] > 0
        assert entry["fpr"]["denominator"] > 0
        assert entry["tpr"]["estimand"] in {"session bootstrap", "degenerate outcomes; cluster interval unavailable"}
        assert "n_clusters" in entry["tpr"]
    assert out["control"]["value"] is not None
    written = list(tmp_path.glob("**/result.json"))
    assert written


def test_stream_uses_supplied_sessions_not_rotation_epochs():
    sessions = ["alice", "bob", "alice", "bob"]
    stream = build_stream(FakeDeployment(), TEXTS[:4], n_keys=2, rotate_every=2,
                          seed=1, session_ids=sessions)
    assert [row["session_id"] for row in stream["requests"]] == sessions
    assert [row["epoch"] for row in stream["requests"]] == [0, 0, 1, 1]
    unknown = build_stream(FakeDeployment(), TEXTS[:4], n_keys=2, rotate_every=2, seed=1)
    assert all(row["session_id"] is None for row in unknown["requests"])


def test_exact_wire_equality_is_not_rounded_equality():
    vectors = [np.array([[1.0, 2.0]], dtype=np.float32),
               np.array([[1.00001, 2.0]], dtype=np.float32)]
    requests = [{"vectors": value} for value in vectors]
    assert np.array_equal(np.round(vectors[0], 4), np.round(vectors[1], 4))
    assert not real_battery._wire_bits_equal(requests, 0, 0, 1, 0)
    assert real_battery._wire_bits_equal(requests, 0, 0, 0, 0)


def test_same_key_across_epochs_and_unknown_keys_are_distinct():
    from obfuscation_attacks.attacks.linkage.core import enumerate_pairs
    from obfuscation_attacks.data.streams import RequestRecord

    records = [RequestRecord(i, np.array([1]), epoch=epoch, key_id=key)
               for i, (epoch, key) in enumerate(((0, "a"), (1, "b"), (2, "a"), (2, None)))]
    pairs = enumerate_pairs(records, within_epoch=False, max_pairs=100)
    assert pairs["total_pairs"] == 6
    index = np.flatnonzero((pairs["left_req"] == 0) & (pairs["right_req"] == 2))[0]
    assert pairs["same_key"][index] and not pairs["same_epoch"][index]
    unknown = np.flatnonzero(pairs["right_req"] == 3)
    assert not pairs["key_known"][unknown].any()
    grouped = enumerate_pairs(records, within_epoch=True, max_pairs=100)
    assert any(a == 0 and b == 2 for a, b in zip(grouped["left_req"], grouped["right_req"]))


def test_exact_and_cosine_use_the_same_sampled_pairs():
    from obfuscation_attacks.attacks.linkage.core import ExactLinkage, ApproximateLinkage
    from obfuscation_attacks.data.contracts import Capability
    from obfuscation_attacks.data.streams import RequestRecord, RequestStream, build_stream_context

    class Scheme:
        def exposed_capabilities(self):
            return frozenset({Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS, Capability.REQUEST_METADATA})

    records = [RequestRecord(i, np.array([1, 2]), vectors=np.array([[1., 0.], [0., 1.]]),
                             key_id="a" if i != 1 else "b", epoch=i) for i in range(3)]
    stream = RequestStream(records, [np.array([1, 2])] * 3, vocab_size=4)
    ctx = build_stream_context(Scheme(), stream)
    exact = ExactLinkage(max_pairs=7, seed=4).run_stream(ctx).stream_metrics["pairs"]
    approx = ApproximateLinkage(max_pairs=7, seed=4, max_queries=1).run_stream(ctx).stream_metrics["pairs"]
    for key in ("left_req", "left_pos", "right_req", "right_pos"):
        np.testing.assert_array_equal(exact[key], approx[key])
    assert len(exact["left_req"]) == 7 and exact["inclusion_probability"] == pytest.approx(7 / 12)


def test_unknown_sessions_do_not_create_a_session_confidence_interval():
    requests = [{"stream_id": 0, "session_id": None}, {"stream_id": 0, "session_id": None}]
    rows = [(0, 0, 1, 0, 5), (0, 1, 1, 1, 6)]
    interval = real_battery._cluster_rate_ci([True, False], real_battery._pair_clusters(requests, rows))
    assert interval["lo"] is None and interval["hi"] is None and interval["n_boot"] == 0
    assert "session unavailable" in interval["estimand"]


def test_zero_events_never_create_zero_risk_interval():
    interval = real_battery._cluster_rate_ci([False] * 20, ["a"] * 10 + ["b"] * 10)
    assert interval["value"] == 0 and interval["hi"] is None
    assert interval["empirical_resolution"] == pytest.approx(1 / 20)
    assert interval["n_clusters"] == 2
    repeated = real_battery._cluster_rate_ci([False] * 200, ["a"] * 100 + ["b"] * 100)
    assert repeated["n_clusters"] == 2 and repeated["hi"] is None
    assert "not independent" in repeated["resolution_scope"]


def test_zero_events_with_unknown_sessions_do_not_claim_cluster_count():
    requests = [{"stream_id": 0, "session_id": None}, {"stream_id": 0, "session_id": None}]
    clusters = real_battery._pair_clusters(requests, [(0, 0, 1, 0, 5)])
    interval = real_battery._cluster_rate_ci([False], clusters)
    assert interval["n_clusters"] is None and interval["hi"] is None


def test_recovered_text_uses_the_stage2_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    mapping = {index: index for index in range(32)}
    out = run_c12(
        FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5, verbose=False,
        stage2_mapping=mapping,
    )
    assert "knn_aligned" not in out
    assert out["recovered_text"]["status"] == "implemented"
    assert out["recovered_text"]["tpr"]["denominator"] > 0


def test_linkage_supports_independent_streams(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(
        FakeDeployment(), n_keys=2, rotate_every=4, seed=7, verbose=False,
        text_sets=[TEXTS[:5], TEXTS[5:]],
    )
    assert out["n_streams"] == 2
    assert len(out["stream"]["per_stream"]) == 2


def test_tfma_runs_on_the_split_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    ref_counts = {token: 1000 - token for token in range(20)}
    out = real_battery.run_c10(
        FakeDeployment(), TEXTS, ref_counts, n_keys=2, rotate_every=4, seed=4, verbose=False,
    )
    assert "SH-P" in out and "SH-B" in out
    assert out["positive_control"]["top1"] is not None


def test_enumerate_pairs_bounds_memory_and_records_the_deterministic_sample():
    from obfuscation_attacks.attacks.linkage.core import enumerate_pairs
    from obfuscation_attacks.data.streams import RequestRecord

    records = [
        RequestRecord(request_id=index, symbols=np.arange(30) + index * 30, epoch=0)
        for index in range(100)
    ]
    total = sum(30 * 30 for i in range(100) for j in range(i + 1, 100))
    pairs = enumerate_pairs(records, within_epoch=False, max_pairs=50_000, seed=7)
    assert pairs["total_pairs"] == total
    assert pairs["sampled"] is True
    assert 1 < len(pairs["delta"]) <= 60_000
    assert np.all(pairs["left_req"] < pairs["right_req"])
    assert np.all((pairs["left_pos"] >= 0) & (pairs["left_pos"] < 30))
    assert np.all((pairs["right_pos"] >= 0) & (pairs["right_pos"] < 30))
    again = enumerate_pairs(records, within_epoch=False, max_pairs=50_000, seed=7)
    assert np.array_equal(pairs["left_req"], again["left_req"])
    assert np.array_equal(pairs["left_pos"], again["left_pos"])
    small = enumerate_pairs(records[:10], within_epoch=False, max_pairs=1_000_000)
    assert small["sampled"] is False
    assert len(small["delta"]) == (10 * 9 // 2) * 900


def test_indexed_pairs_match_brute_force_with_unknown_groups_and_empty_requests():
    from obfuscation_attacks.attacks.linkage.core import enumerate_pairs
    from obfuscation_attacks.data.streams import RequestRecord

    rng = np.random.default_rng(17)
    for _ in range(20):
        records = [RequestRecord(i, np.arange(int(rng.integers(0, 5))),
                                 key_id=[None, "a", "b"][int(rng.integers(0, 3))])
                   for i in range(8)]
        for restricted in (False, True):
            expected = [(i, x, j, y) for i, a in enumerate(records)
                        for j, b in enumerate(records) if j > i
                        if not restricted or a.key_id is None or b.key_id is None or a.key_id == b.key_id
                        for x in range(len(a.symbols)) for y in range(len(b.symbols))]
            pairs = enumerate_pairs(records, within_epoch=restricted, max_pairs=10000)
            actual = list(zip(pairs["left_req"], pairs["left_pos"], pairs["right_req"], pairs["right_pos"]))
            assert actual == expected
            assert pairs["total_pairs"] == len(expected)
            if len(expected) > 7:
                selected = np.sort(np.random.default_rng(19).choice(len(expected), size=7, replace=False, shuffle=False))
                sampled = enumerate_pairs(records, within_epoch=restricted, max_pairs=7, seed=19)
                got = list(zip(sampled["left_req"], sampled["left_pos"], sampled["right_req"], sampled["right_pos"]))
                assert got == [expected[index] for index in selected]


def test_indexed_pairs_sample_large_request_set_without_enumerating_all_pairs():
    from obfuscation_attacks.attacks.linkage.core import enumerate_pairs
    from obfuscation_attacks.data.streams import RequestRecord

    records = [RequestRecord(i, np.array([i]), key_id=str(i % 5) if i % 7 else None)
               for i in range(5000)]
    pairs = enumerate_pairs(records, within_epoch=True, max_pairs=200, seed=2)
    assert len(pairs["delta"]) == 200
    assert pairs["total_pairs"] > 2_000_000
    assert pairs["inclusion_probability"] == pytest.approx(200 / pairs["total_pairs"])
