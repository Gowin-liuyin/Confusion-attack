import numpy as np
import pytest

from obfuscation_attacks.attacks.nn_knn.norm import (
    compose_candidates, norm_topk, request_radius_statistics,
)


def test_radius_ranking_uses_public_table_and_breaks_ties_by_id():
    table = np.array([[1., 0.], [0., 2.], [-2., 0.], [0., 3.]])
    wire = np.array([[2., 0.], [0., 3.]])
    assert norm_topk(wire, table, k=2).tolist() == [[1, 2], [3, 1]]
    assert norm_topk(wire, table, k=2, direction_weight=1.).tolist()[0][0] == 0
    assert compose_candidates([[1, 2], [3, 1]], {1: 7, 3: 8}) == [[7], [8, 7]]


def test_radius_statistics_do_not_assume_request_alignment():
    table = np.array([[1., 0.], [0., 2.], [0., 3.]])
    result = request_radius_statistics([table[[1, 2]], table[[2]]], table)
    assert result["positions"] == 3
    assert result["top1_counts"] == {1: 1, 2: 2}
    assert [row["positions"] for row in result["per_request"]] == [2, 1]


def test_radius_rejects_bad_parameters():
    with pytest.raises(ValueError):
        norm_topk([[1., 0.]], [[1., 0.]], k=2)
    with pytest.raises(ValueError):
        norm_topk([[float("nan"), 0.]], [[1., 0.]], k=1)
