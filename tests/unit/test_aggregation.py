"""Acceptance tests for W4 multi-observation aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obfuscation_attacks.evaluation.aggregation import (
    aggregate_public_observations,
    aggregate_public_stream,
    aggregate_representations,
    run_multi_observation,
    score_public_aggregation,
    vote_candidates,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D = 20, 8
E = np.eye(V, D, dtype=np.float32)


class FakeDeployment:
    artifact = Path("fixture-artifact")
    clean_model_path = Path("fixture-clean")
    max_len = 64
    vocab_size = V

    def public_view(self, arm: str, scenario: str = "S-public") -> PublicDeployment:
        return PublicDeployment(
            arm=arm, scenario=scenario, artifact="fixture", vocab_size=V,
            hidden_size=D, n_layers=2, layers=(1,), max_len=64, tokenizer_id="fixture",
            E_server=E, E_clean=E, allow_clean_weights=True,
        )

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None):
        ids = np.asarray(text, dtype=np.int64)
        index = 0 if key_index is None else int(key_index)
        noise = np.full((len(ids), D), 0.05 * ((index % 3) - 1), dtype=np.float32)
        vectors = E[ids] + noise
        public = WirePublic(
            arm="ab", request_id=request_id, seq_len=len(ids), wire_vectors=vectors,
            group_id=f"g-{index}", epoch=epoch, session_id=session_id,
        )
        return Wire(public, WireTruth(plaintext_ids=ids.copy(), permuted_ids=ids.copy()))


def test_candidate_voting_recovers_a_missed_token():
    observations = [
        np.array([[1, 2], [7, 8]]),
        np.array([[2, 3], [7, 9]]),
        np.array([[2, 4], [7, 6]]),
    ]
    aggregated = vote_candidates(observations, k=2)
    assert aggregated[0, 0] == 2
    assert aggregated[1, 0] == 7


def test_candidate_voting_keeps_union_and_marks_unfilled_slots():
    first = np.array([[1, 1], [5, -1]])
    second = np.array([[2], [-1]])
    assert vote_candidates([first, second], k=2).tolist() == [[1, 2], [5, -1]]


def test_normalized_representation_aggregation_uses_the_mean_direction():
    public = FakeDeployment().public_view("ab")
    observations = [E[[3]] * 10.0, E[[3]] * 0.1]
    aggregated = aggregate_representations(public, observations, k=3)
    assert aggregated[0, 0] == 3


def test_multi_observation_reports_every_m_without_truth_grouping():
    out = run_multi_observation(
        FakeDeployment(), [np.array([1, 2, 3, 4])], m_values=(1, 2, 4), k=5,
        verbose=False,
    )
    assert {"m=1", "m=2", "m=4"}.issubset(out)
    assert "controlled repeats" in out["grouping"]
    for key in ("m=1", "m=2", "m=4"):
        assert out[key]["single"]["denominator"] > 0
        assert out[key]["knn_aligned_then_aggregate"]["status"] == "not_run"


def test_single_baseline_keeps_positions_missing_from_later_observations():
    class ShorterRepeat(FakeDeployment):
        def observe(self, text, **kwargs):
            wire = super().observe(text, **kwargs)
            if kwargs["key_index"] == 1:
                wire.public.wire_vectors = wire.public.wire_vectors[:2]
                wire.public.seq_len = 2
            return wire

    out = run_multi_observation(ShorterRepeat(), [[1, 2, 3, 4]], m_values=(2,), k=3)
    assert out["m=2"]["single"] == {"value": 1.0, "numerator": 4, "denominator": 4}
    assert out["m=2"]["candidate_vote"]["denominator"] == 4


def test_zero_vectors_do_not_gain_arbitrary_candidates_or_groups():
    dep = FakeDeployment()
    public = dep.public_view("ab")
    wires = [dep.observe([1], arm="ab", request_id=i).public for i in range(2)]
    for wire in wires:
        wire.wire_vectors[:] = 0
    result = aggregate_public_observations(public, wires, similarity_threshold=0, k=3)
    assert result["pairs"] == [] and result["candidate_pairs"] == 0
    assert all(rows.tolist() == [[-1, -1, -1]] for rows in result["aggregated_candidates"])


def test_public_topk_breaks_boundary_ties_by_token_id():
    from types import SimpleNamespace
    from obfuscation_attacks.evaluation.aggregation import _topk_queries

    table = np.asarray([[0., 1.], [1., 0.], [1., 0.], [0., 0.]], dtype=np.float32)
    public = SimpleNamespace(arm="ab", table=lambda arm: table)
    assert _topk_queries(public, [[1., 0.]], k=1).tolist() == [[1]]
    assert _topk_queries(public, [[1., 0.]], k=4).tolist() == [[1, 2, 0, -1]]


def test_public_aggregation_groups_and_aligns_insertions_without_truth():
    dep = FakeDeployment()
    public = dep.public_view("ab")
    sequences = [[1, 2, 3], [1, 4, 2, 3], [5, 6, 7]]
    wires = [dep.observe(seq, arm="ab", request_id=i, key_index=1,
                         session_id="same-session").public for i, seq in enumerate(sequences)]
    result = aggregate_public_observations(public, wires, similarity_threshold=0.7, k=3)
    assert [(pair["left"], pair["right"]) for pair in result["pairs"]] == [(0, 1)]
    assert result["pairs"][0]["aligned_positions"] == [(0, 0), (1, 2), (2, 3)]
    score = score_public_aggregation(result, sequences, ["repeated", "repeated", "different"])
    assert score["pairing"] == {"predicted_pairs": 1, "true_pairs": 1, "correct_pairs": 1,
                               "false_pairs": 0, "missed_pairs": 0}
    assert score["alignment"]["matched_pairs"] == 3
    assert score["aggregated"]["denominator"] == sum(map(len, sequences))
    shuffled = aggregate_public_observations(public, [wires[2], wires[1], wires[0]],
                                            similarity_threshold=0.7, k=3)
    assert [(shuffled["request_ids"][p["left"]], shuffled["request_ids"][p["right"]])
            for p in shuffled["pairs"]] == [(1, 0)]
    with pytest.raises(TypeError, match="WirePublic"):
        aggregate_public_observations(public, [dep.observe([1], arm="ab")], similarity_threshold=0.7)


def test_public_aggregation_rejects_excess_work_and_records_missed_pairs():
    dep = FakeDeployment()
    public = dep.public_view("ab")
    wires = [dep.observe([1, 2], arm="ab", request_id=i, key_index=1).public for i in range(3)]
    with pytest.raises(ValueError, match="pair budget"):
        aggregate_public_observations(public, wires, similarity_threshold=0.8, max_pairs=2)
    with pytest.raises(ValueError, match="alignment budget"):
        aggregate_public_observations(public, wires, similarity_threshold=0.8, max_alignment_cells=8)
    with pytest.raises(ValueError, match="pair enumeration budget"):
        aggregate_public_observations(public, wires, similarity_threshold=0.8, max_pair_visits=3)
    separated = aggregate_public_observations(public, wires, similarity_threshold=1.0,
                                              same_session_only=True)
    score = score_public_aggregation(separated, [[1, 2]] * 3, [0, 0, 0])
    assert score["pairing"]["missed_pairs"] == 3
    assert score["single"]["denominator"] == 6


def test_public_stream_adapter_never_reads_scorer_truth():
    class PublicOnly(dict):
        def __getitem__(self, key):
            if key in {"truth", "plaintext_ids", "permuted_ids"}:
                raise AssertionError("private field read")
            return super().__getitem__(key)

    dep = FakeDeployment()
    public = dep.public_view("ab")
    requests = []
    for index, ids in enumerate(([1, 2], [1, 2])):
        wire = dep.observe(ids, arm="ab", request_id=index, key_index=1).public
        requests.append(PublicOnly(request_id=index, vectors=wire.wire_vectors,
                                   session_id="s", group_id=wire.group_id, epoch=0))
    result = aggregate_public_stream(public, requests, similarity_threshold=0.8, k=2)
    assert len(result["pairs"]) == 1
    assert score_public_aggregation(result, [[1, 2], [1, 2]], ["x", "x"])["aggregated"]["value"] == 1
    assert score_public_aggregation(result, [[4, 2], [4, 2]], ["x", "x"])["aggregated"]["value"] == 0.5
    assert result["single_candidates"][0][0, 0] == 1
    with pytest.raises(ValueError, match="unknown labels"):
        score_public_aggregation(result, [[1, 2], [1, 2]], [None, None])
    with pytest.raises(ValueError, match="scoring alignment budget"):
        score_public_aggregation(result, [[1, 2], [1, 2]], ["x", "x"], max_alignment_cells=8)
