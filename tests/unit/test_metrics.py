"""Unit tests for the recovery metrics and their counting rules (Goal section 7)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.metrics import (
    asr_at_k,
    exact_sequence_match,
    mapping_asr_at_k,
    token_prf1,
    total_and_clean_asr,
    ttrsr,
)

pytestmark = pytest.mark.unit


def test_asr_at_k_counts_per_position():
    candidates = [[1, 2, 3], [4, 5, 6]]
    truth = [2, 9]
    assert asr_at_k(candidates, truth, 1) == 0.0
    assert asr_at_k(candidates, truth, 2) == 0.5  # only position 0 hits within top-2
    assert asr_at_k(candidates, truth, 3) == 0.5


def test_asr_at_k_excludes_special_positions():
    candidates = [[2], [9]]
    truth = [2, 9]
    special = [False, True]  # position 1 is a special token -> excluded
    assert asr_at_k(candidates, truth, 1, special_mask=special) == 1.0


def test_token_prf1_multiset_duplicates():
    # pred bag {1:2, 2:1, 3:1}, truth bag {1:1, 2:2, 3:1}
    prf = token_prf1([1, 1, 2, 3], [1, 2, 2, 3])
    assert prf.tp == 3 and prf.fp == 1 and prf.fn == 1
    assert prf.precision == pytest.approx(0.75)
    assert prf.recall == pytest.approx(0.75)
    assert prf.f1 == pytest.approx(0.75)


def test_token_prf1_excludes_special_ids():
    prf = token_prf1([0, 1, 2], [0, 1, 9], special_ids={0})
    # after dropping id 0: pred {1,2}, truth {1,9} -> tp=1
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 1


def test_exact_sequence_match_is_order_sensitive():
    assert exact_sequence_match([1, 2, 3], [1, 2, 3]) is True
    assert exact_sequence_match([1, 2, 3], [3, 2, 1]) is False
    assert exact_sequence_match([5, 1, 2, 3], [1, 2, 3], special_ids={5}) is True


def test_ttrsr_is_positional():
    assert ttrsr([1, 2, 9], [1, 2, 3]) == pytest.approx(2 / 3)
    # right tokens, wrong order -> low positional recovery
    assert ttrsr([3, 2, 1], [1, 2, 3]) == pytest.approx(1 / 3)


def test_total_and_clean_asr_split():
    candidates = [[1], [2], [3], [4]]
    truth = [1, 2, 3, 4]
    content = [True, False, True, False]  # positions 1,3 are stop words
    total, clean = total_and_clean_asr(candidates, truth, 1, content_mask=content)
    assert total == 1.0  # all four recovered
    assert clean == 1.0  # the two content positions recovered
    # a miss on a content position lowers Clean but Total stays high
    candidates2 = [[1], [2], [99], [4]]
    total2, clean2 = total_and_clean_asr(candidates2, truth, 1, content_mask=content)
    assert total2 == 0.75
    assert clean2 == 0.5  # one of two content tokens missed


def test_mapping_asr_at_k():
    cands = {0: [5, 6, 7], 1: [2, 3]}
    true = {0: 6, 1: 9}
    assert mapping_asr_at_k(cands, true, 1) == 0.0
    assert mapping_asr_at_k(cands, true, 2) == 0.5
