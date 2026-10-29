"""Integration tests for the experiment runner and every scenario builder."""

from __future__ import annotations

import pytest

from obfuscation_attacks.data.synthetic import BUILDERS
from obfuscation_attacks.evaluation.runner import run_from_config, run_scenario
from obfuscation_attacks.utils import load_json, load_yaml, RunPaths

pytestmark = pytest.mark.integration

# scenario builder -> an attack that applies to it
NUMPY_PAIRS = {
    "nn": "nn_knn.euclidean",
    "vma_weight": "vma.weight_rowsort",
    "vma_autoregressive": "vma.autoregressive",
    "vma_osnip": "vma.osnip_projection",
    "ia_ednn": "ia.ednn",
    "ia_gate": "ia.gate",
    "ia_attn": "ia.attn",
    "tfma": "tfma.independent",
    "custom_nullspace": "nn_knn.euclidean",
}
NEURAL_PAIRS = {
    "isa": "isa.er",
    "ima": "ima.classify",
    "sda": "sda.sda2021",
}


@pytest.mark.parametrize("scenario,attack", sorted(NUMPY_PAIRS.items()))
def test_run_scenario_numpy_families_ok(scenario, attack):
    out = run_scenario(attack, scenario, seed=0)
    assert out["metrics"]["status"] == "ok"


@pytest.mark.neural
@pytest.mark.parametrize("scenario,attack", sorted(NEURAL_PAIRS.items()))
def test_run_scenario_neural_families_ok(scenario, attack):
    out = run_scenario(attack, scenario, seed=0)
    assert out["metrics"]["status"] == "ok"


def test_all_builders_have_coverage():
    covered = set(NUMPY_PAIRS) | set(NEURAL_PAIRS)
    assert set(BUILDERS) == covered  # every scenario builder is exercised


def test_run_from_config_smoke_writes_artifacts(tmp_path):
    cfg = {
        "name": "smoke_nn",
        "mode": "smoke",
        "seed": 0,
        "attack": {"name": "nn_knn.euclidean", "params": {"top_k": 10}},
        "scenario": {"name": "nn", "params": {"noise": 0.02}},
        "metrics": {"ks": [1, 5, 10]},
    }
    out = run_from_config(cfg, reports_root=str(tmp_path))
    assert out["status"] == "ok"
    paths = RunPaths(str(tmp_path), "smoke_nn")
    for p in (paths.resolved_config, paths.environment, paths.metrics, paths.predictions, paths.fidelity, paths.run_log):
        assert p.exists()
    metrics = load_json(paths.metrics)
    assert "token_recovery" in metrics
    fidelity = load_json(paths.fidelity)
    # smoke run must not silently claim any approximation is enabled
    assert all(v is False for v in fidelity["approximations"].values())


def test_run_from_config_full_is_resource_blocked_not_faked(tmp_path):
    cfg = {
        "name": "full_nn",
        "mode": "full",
        "seed": 0,
        "attack": {"name": "nn_knn.euclidean", "params": {}},
        "resource_requirement": "needs real 32B model weights + GPU memory not available here",
    }
    out = run_from_config(cfg, reports_root=str(tmp_path))
    assert out["status"] == "resource_blocked"
    paths = RunPaths(str(tmp_path), "full_nn")
    fidelity = load_json(paths.fidelity)
    assert fidelity["status"] == "resource_blocked"
    metrics = load_json(paths.metrics)
    # crucially, NO fabricated numeric metrics for a blocked full run
    assert metrics["status"] == "resource_blocked"
    assert "token_recovery" not in metrics
