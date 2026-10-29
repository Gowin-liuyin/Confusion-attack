"""Experiment runner.

``run_scenario`` runs one attack against one synthetic scenario and scores it.
``run_from_config`` executes a declarative config (smoke or full), writes the
Section-11 artifacts (``config.resolved.yaml``, ``environment.json``,
``metrics.json``, ``predictions.json``, ``run.log``, ``fidelity.json``), and -
crucially - refuses to fabricate results for *full* paper-scale configs whose
real models/data are not available on this machine. Those are recorded with a
``resource_blocked`` status (see docs/RESOURCE_REQUIREMENTS.md), never as fake
numbers.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from ..attacks import get_attack, load_all
from ..data.synthetic import build_scenario
from ..schemes.base import build_context
from ..utils import environment_snapshot, RunPaths, save_json, save_yaml, seed_everything
from .scoring import compute_metrics


def _jsonable_predictions(result) -> dict:
    def arr(x):
        return None if x is None else np.asarray(x).tolist()

    return {
        "status": result.status.value,
        "status_reason": result.status_reason,
        "recovered_token_ids": arr(result.recovered_token_ids),
        "candidates": result.candidates,
        "recovered_mapping": (
            {int(k): int(v) for k, v in result.recovered_mapping.items()}
            if result.recovered_mapping is not None
            else None
        ),
        "mapping_candidates": (
            {int(k): [int(x) for x in v] for k, v in result.mapping_candidates.items()}
            if result.mapping_candidates is not None
            else None
        ),
        "recovered_permutations": (
            {k: arr(v) for k, v in result.recovered_permutations.items()}
            if result.recovered_permutations is not None
            else None
        ),
        "recovered_text": result.recovered_text,
    }


def run_scenario(
    attack_name: str,
    scenario_name: str,
    *,
    attack_params: Optional[dict] = None,
    scenario_params: Optional[dict] = None,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> dict:
    """Build a scenario, run the attack through the gated context, and score."""
    load_all()
    seed_everything(seed)
    attack = get_attack(attack_name, **(attack_params or {}))
    scenario = build_scenario(scenario_name, **(scenario_params or {}))
    ctx = build_context(scenario.scheme, scenario.sample, attacker_model=scenario.attacker_model)
    result = attack.run(ctx)
    metrics = compute_metrics(result, scenario.sample, ks)
    return {
        "attack": attack.describe(),
        "scheme": scenario.scheme.describe(),
        "scenario": scenario.name,
        "metrics": metrics,
        "result": result,
    }


def _fidelity_record(config: dict, attack_desc: dict, scheme_desc: Optional[dict], status: str) -> dict:
    fid = {
        "mode": config.get("mode"),
        "status": status,
        "attack": attack_desc,
        "scheme": scheme_desc,
        "seed": config.get("seed"),
        # Declared fidelity flags: a faithful full run must assert these are all
        # false/empty. The audit script checks they are not silently enabled.
        "approximations": config.get("fidelity", {}).get("approximations", {
            "sampling": False,
            "truncation": False,
            "quantization": False,
            "low_precision": False,
            "pruning": False,
            "early_stopping": False,
            "candidate_truncation": False,
            "approx_nearest_neighbor": False,
        }),
        "formula_refs": config.get("formula_refs", []),
        "notes": config.get("notes", ""),
    }
    return fid


def run_from_config(config: dict, reports_root: str | None = None) -> dict:
    """Execute a declarative experiment config.

    Smoke configs (``mode: smoke``) run a synthetic scenario end-to-end. Full
    configs (``mode: full``) describe the real paper setup; if the real backend
    is not available the run is recorded as ``resource_blocked`` with the exact
    reason rather than producing fabricated metrics.
    """
    config = copy.deepcopy(config)
    name = config.get("name", "run")
    mode = config.get("mode", "smoke")
    seed = int(config.get("seed", 0))
    ks = tuple(config.get("metrics", {}).get("ks", [1, 5, 10]))
    attack_spec = config["attack"]
    attack_name = attack_spec["name"]
    attack_params = attack_spec.get("params", {})

    load_all()
    attack = get_attack(attack_name, **attack_params)

    paths = None
    if reports_root is not None:
        paths = RunPaths(reports_root, name)

    if mode == "full":
        # Full paper-scale run: requires real models/datasets. This harness does
        # not ship those weights/data, and this machine cannot host them (see
        # docs/RESOURCE_REQUIREMENTS.md). Report the blockage honestly.
        status = "resource_blocked"
        reason = config.get(
            "resource_requirement",
            "full-scale models/datasets and GPU memory are required and not available here",
        )
        fidelity = _fidelity_record(config, attack.describe(), None, status)
        fidelity["resource_requirement"] = reason
        result_payload = {
            "name": name,
            "mode": mode,
            "status": status,
            "status_reason": reason,
            "metrics": {"status": status, "status_reason": reason},
        }
        if paths is not None:
            save_yaml(paths.resolved_config, config)
            save_json(paths.environment, environment_snapshot())
            save_json(paths.metrics, result_payload["metrics"])
            save_json(paths.predictions, {"status": status, "status_reason": reason})
            save_json(paths.fidelity, fidelity)
            paths.run_log.write_text(
                f"[{name}] mode=full -> {status}: {reason}\n", encoding="utf-8"
            )
        return result_payload

    # --- smoke mode ---------------------------------------------------------
    seed_everything(seed)
    scenario_spec = config["scenario"]
    scenario = build_scenario(scenario_spec["name"], **scenario_spec.get("params", {}))
    ctx = build_context(scenario.scheme, scenario.sample, attacker_model=scenario.attacker_model)
    result = attack.run(ctx)
    metrics = compute_metrics(result, scenario.sample, ks)

    fidelity = _fidelity_record(config, attack.describe(), scenario.scheme.describe(), result.status.value)
    result_payload = {
        "name": name,
        "mode": mode,
        "status": result.status.value,
        "status_reason": result.status_reason,
        "metrics": metrics,
    }
    if paths is not None:
        save_yaml(paths.resolved_config, config)
        save_json(paths.environment, environment_snapshot())
        save_json(paths.metrics, metrics)
        save_json(paths.predictions, _jsonable_predictions(result))
        save_json(paths.fidelity, fidelity)
        paths.run_log.write_text(
            f"[{name}] mode=smoke attack={attack_name} scenario={scenario.name} "
            f"status={result.status.value}\n",
            encoding="utf-8",
        )
    return result_payload
