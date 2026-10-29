"""End-to-end correctness tests for the non-neural families via synthetic scenarios.

Each attack runs through the capability-gated context against a scenario that
genuinely contains the leakage it exploits, and must recover the secret.
"""

from __future__ import annotations

import pytest

from obfuscation_attacks.attacks.ia.attn import AttnIA
from obfuscation_attacks.attacks.ia.ednn import EDNN
from obfuscation_attacks.attacks.ia.gate import GateIA
from obfuscation_attacks.attacks.nn_knn.cosine_knn import CosineKNN
from obfuscation_attacks.attacks.nn_knn.euclidean_knn import EuclideanKNN
from obfuscation_attacks.attacks.tfma.frequency_matching import IndependentTFMA
from obfuscation_attacks.attacks.vma.autoregressive import AutoregressiveVMA
from obfuscation_attacks.attacks.vma.osnip_projection import OSNIPProjectionVMA
from obfuscation_attacks.attacks.vma.weight_rowsort import WeightRowSortVMA
from obfuscation_attacks.data.contracts import AttackStatus
from obfuscation_attacks.data.synthetic import build_scenario
from obfuscation_attacks.evaluation.scoring import compute_metrics
from obfuscation_attacks.schemes.base import build_context

pytestmark = pytest.mark.regression


def _run(attack, scenario):
    ctx = build_context(scenario.scheme, scenario.sample, attacker_model=scenario.attacker_model)
    res = attack.run(ctx)
    return res, compute_metrics(res, scenario.sample)


def test_nn_euclidean_recovers_under_small_noise():
    res, m = _run(EuclideanKNN(top_k=10), build_scenario("nn", noise=0.02, seed=0))
    assert res.status == AttackStatus.OK
    assert m["token_recovery"]["asr@1"] >= 0.9


def test_nn_cosine_recovers_under_small_noise():
    res, m = _run(CosineKNN(top_k=10), build_scenario("nn", noise=0.02, seed=1))
    assert m["token_recovery"]["asr@1"] >= 0.9


def test_vma_weight_recovers_permutations_exactly():
    res, m = _run(WeightRowSortVMA(), build_scenario("vma_weight", seed=0))
    assert res.status == AttackStatus.OK
    assert m["permutation_recovery"]["verified"] is True
    assert m["permutation_recovery"]["Z1_exact"] is True


def test_vma_autoregressive_recovers_tokens():
    res, m = _run(AutoregressiveVMA(), build_scenario("vma_autoregressive", seed=0))
    assert res.status == AttackStatus.OK
    assert m["token_recovery"]["ttrsr"] == 1.0  # exact causal recovery, plain L1


def test_vma_osnip_projection_recovers_and_reports_clean_asr():
    res, m = _run(OSNIPProjectionVMA(top_k=10), build_scenario("vma_osnip", noise=0.02, seed=0))
    assert m["token_recovery"]["asr@1"] >= 0.9
    # Total/Clean split is reported because the scenario marks stop words
    assert "total_asr@1" in m["token_recovery"]
    assert "clean_asr@1" in m["token_recovery"]


def test_ia_ednn_recovers_tokens_exactly():
    res, m = _run(EDNN(top_k=10), build_scenario("ia_ednn", seed=0))
    assert m["token_recovery"]["ttrsr"] == 1.0


def test_ia_gate_recovers_mapping_exactly():
    res, m = _run(GateIA(top_k=10), build_scenario("ia_gate", seed=0))
    assert res.status == AttackStatus.OK
    assert m["mapping_recovery"]["mapping_top1_accuracy"] == 1.0


def test_ia_attn_recovers_mapping_exactly():
    res, m = _run(AttnIA(block_size=2), build_scenario("ia_attn", seed=0))
    assert res.status == AttackStatus.OK
    assert m["mapping_recovery"]["mapping_top1_accuracy"] == 1.0


def test_tfma_independent_beats_chance():
    res, m = _run(IndependentTFMA(top_k=100), build_scenario("tfma", seed=0))
    assert res.status == AttackStatus.OK
    vocab = 30
    assert m["mapping_recovery"]["mapping_top1_accuracy"] > 2.0 / vocab
    assert m["mapping_recovery"]["mapping_asr@10"] >= 0.3
