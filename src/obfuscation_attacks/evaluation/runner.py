"""Experiment runner.

``run_scenario`` runs one attack against one synthetic scenario and scores it.
``run_from_config`` executes a declarative config (smoke or full), writes the
Section-11 artifacts (``config.resolved.yaml``, ``environment.json``,
``metrics.json``, ``predictions.json``, ``result.json``, ``run.log``,
``fidelity.json``), and never fabricates results. A ``full`` config is dispatched
to ``evaluation.real_battery.run_full_config``: if its ``full_backend`` files are
on disk the real P0 battery runs against the packaged artifact, and if they are
missing the run is recorded as ``resource_blocked`` **with the specific missing
path** rather than a blanket hardware excuse (the A+B artifact and the Alpaca
snapshot are both available locally, so a blanket excuse would be false).

``result.json`` is the plan's App.-A payload: the attack model (surface,
knowledge level, key/pair flags, rotation awareness, candidate mode), the
positive control, the sample sizes, the metrics, the CIs and the two hashes. It
is produced by :func:`obfuscation_attacks.evaluation.attack_model.result_payload`,
so a run cannot emit metrics without declaring its attack model.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from ..attacks import get_attack, load_all
from ..data.synthetic import build_scenario
from ..data.streams import build_stream_context
from ..schemes.base import build_context
from ..utils import environment_snapshot, RunPaths, save_json, save_yaml, seed_everything
from .attack_model import result_payload, spec_for, validate_payload
from .scoring import compute_metrics, compute_stream_metrics


def _jsonable_predictions(result) -> dict:
    def arr(x):
        return None if x is None else np.asarray(x).tolist()

    stream_metrics = None
    if result.stream_metrics is not None:
        # The pair table can be hundreds of thousands of rows; keep the summary
        # and a hash of the evidence instead of dumping it into every artifact.
        import hashlib

        sm = dict(result.stream_metrics)
        pairs = sm.pop("pairs", None)
        same_symbol = sm.pop("same_symbol", None)
        cosine = sm.pop("cosine", None)
        if pairs is not None:
            sm["pairs_hash"] = hashlib.sha256(
                b"".join(
                    np.asarray(pairs[k]).tobytes() for k in sorted(pairs)
                    if isinstance(pairs[k], np.ndarray)
                )
            ).hexdigest()
        if same_symbol is not None:
            sm["same_symbol_positive"] = int(np.asarray(same_symbol, dtype=bool).sum())
        if cosine is not None:
            sm["cosine_summary"] = {
                "mean": float(np.mean(cosine)) if len(cosine) else None,
                "min": float(np.min(cosine)) if len(cosine) else None,
                "max": float(np.max(cosine)) if len(cosine) else None,
            }
        stream_metrics = sm

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
        "stream_metrics": stream_metrics,
    }


def _sizes(scenario) -> dict:
    """Sample sizes for the App.-A block, taken from the scenario itself."""
    if scenario.stream is not None:
        return {
            "n_prompts": scenario.stream.num_requests(),
            "n_tokens": scenario.stream.num_positions(),
            "n_keys": len({r.key_id for r in scenario.stream.requests if r.key_id}),
            "n_types": len(
                {int(s) for r in scenario.stream.requests for s in np.asarray(r.symbols).ravel()}
            ),
            "reference_tokens": 0,
        }
    sample = scenario.sample
    if sample is None:
        return {"n_prompts": 0, "n_tokens": 0, "n_keys": 1, "n_types": 0, "reference_tokens": 0}
    ids = sample.plaintext_token_ids
    cipher = sample.ciphertext_token_ids
    symbols = ids if ids is not None else cipher
    ref = sample.reference_corpus or []
    return {
        "n_prompts": 1,
        "n_tokens": int(sample.seq_len),
        "n_keys": 1,
        "n_types": int(len(set(int(t) for t in np.asarray(symbols).ravel()))) if symbols is not None else 0,
        "reference_tokens": int(sum(len(np.asarray(c).ravel()) for c in ref)),
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

    if scenario.stream is not None:
        sctx = build_stream_context(
            scenario.scheme,
            scenario.stream,
            attacker_model=scenario.attacker_model,
            reference_token_counts=scenario.sample.reference_token_counts if scenario.sample else None,
        )
        result = attack.run_stream(sctx)
        metrics = compute_stream_metrics(result, scenario.stream)
    else:
        ctx = build_context(
            scenario.scheme,
            scenario.sample,
            attacker_model=scenario.attacker_model,
            attacker_obfuscator=scenario.attacker_obfuscator,
        )
        result = attack.run(ctx)
        metrics = compute_metrics(result, scenario.sample, ks)
    return {
        "attack": attack.describe(),
        "scheme": scenario.scheme.describe(),
        "scenario": scenario.name,
        "metrics": metrics,
        "result": result,
        "model_scenario": scenario,
    }


def _attack_specs(config: dict) -> list[tuple[str, dict]]:
    """Resolve one or several attacks from a config.

    A config may carry a single ``attack: {name, params}`` or a list
    ``attack: {runs: [{name, params}, ...]}`` (e.g. the InvBERT / ISA full
    configs that bundle two inverters). Returns ``[(name, params), ...]``.
    """
    spec = config["attack"]
    if "runs" in spec:
        return [(r["name"], dict(r.get("params", {}))) for r in spec["runs"]]
    return [(spec["name"], dict(spec.get("params", {})))]


def _fidelity_record(config: dict, attack_descs: list[dict], scheme_desc: Optional[dict], status: str) -> dict:
    fid = {
        "mode": config.get("mode"),
        "status": status,
        "attacks": attack_descs,
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


def _payload_for(
    config: dict,
    attack_name: str,
    scheme_name: str,
    metrics: dict,
    diagnostics: dict,
    sizes: dict,
    source: dict,
    data: dict,
) -> dict:
    """Build + validate the plan App.-A result JSON for one attack run."""
    pc = diagnostics.get("positive_control") or {}
    overrides = {
        "candidate_mode": diagnostics.get("candidate_mode", "full"),
    }
    if diagnostics.get("attacker_knowledge"):
        overrides["attacker_knowledge"] = diagnostics["attacker_knowledge"]
    if "rotation_aware" in diagnostics:
        overrides["rotation_aware"] = diagnostics["rotation_aware"]
    if diagnostics.get("paper_parameter_undisclosed") is not None:
        overrides["paper_parameter_undisclosed"] = diagnostics["paper_parameter_undisclosed"]
    payload = result_payload(
        attack_name=attack_name,
        scheme_name=scheme_name,
        target=config.get("target") or scheme_name,
        metrics=metrics,
        diagnostics=diagnostics,
        positive_control=pc,
        ci=metrics.get("ci") or {},
        source=source,
        data=data,
        overrides=overrides,
        **sizes,
    )
    problems = validate_payload(payload)
    payload["schema_valid"] = not problems
    payload["schema_problems"] = problems
    return payload


def _refuse_unstarted_paper_v1(config: dict) -> dict | None:
    """Preflight paper-v1 before any full backend. Smoke and unprofiled full are unchanged."""
    from .paper_profile import (
        PreflightError,
        ProfileError,
        preflight_formal_start,
        require_profile,
    )

    try:
        profile = require_profile("paper-v1")
        preflight_formal_start(profile, config.get("manifest") or {}, config)
    except (PreflightError, ProfileError) as exc:
        shortages = list(getattr(exc, "shortages", None) or [str(exc)])
        return {
            "name": config.get("name", "run"),
            "mode": config.get("mode", "full"),
            "profile": "paper-v1",
            "status": "preflight_failed",
            "status_reason": str(exc),
            "shortages": shortages,
            "metrics": {"status": "preflight_failed", "shortages": shortages},
            "result": None,
        }
    # A passing preflight is not an execution path: the legacy full backend uses
    # probe defaults (for example max_len=64 and its own class runners) and must
    # never write results under the paper-v1 identity. Only the paper executor is
    # the formal entry.
    reason = (
        "profile=paper-v1 must run through the paper executor "
        "(python scripts/run_full_battery.py --profile paper-v1 --execute "
        "--manifest <manifest>); the legacy full backend uses probe defaults and "
        "is refused for the formal profile"
    )
    return {
        "name": config.get("name", "run"),
        "mode": config.get("mode", "full"),
        "profile": "paper-v1",
        "status": "refused",
        "status_reason": reason,
        "shortages": [reason],
        "metrics": {"status": "refused", "reason": reason},
        "result": None,
    }


def run_from_config(config: dict, reports_root: str | None = None) -> dict:
    """Execute a declarative experiment config.

    Smoke configs (``mode: smoke``) run a synthetic scenario end-to-end. Full
    configs (``mode: full``) describe the real paper setup; if the real backend
    is not available the run is recorded as ``resource_blocked`` with the exact
    reason rather than producing fabricated metrics. ``profile: paper-v1`` is
    preflighted before that backend and is never stamped onto an unprofiled run.
    """
    config = copy.deepcopy(config)
    if config.get("mode", "smoke") == "full" and config.get("profile") == "paper-v1":
        refused = _refuse_unstarted_paper_v1(config)
        if refused is not None:
            return refused
    name = config.get("name", "run")
    mode = config.get("mode", "smoke")
    seed = int(config.get("seed", 0))
    ks = tuple(config.get("metrics", {}).get("ks", [1, 5, 10]))
    specs = _attack_specs(config)

    load_all()
    attacks = [get_attack(n, **p) for n, p in specs]
    attack_descs = [a.describe() for a in attacks]

    paths = None
    if reports_root is not None:
        paths = RunPaths(reports_root, name)

    if mode == "full":
        # Full paper-scale run. Two honest outcomes only:
        #  * a ``full_backend`` is declared and its files are on disk -> run the real
        #    battery (plan P0 attacks against the packaged artifact + local corpora);
        #  * or the specific missing resource is named (never a blanket "GPU missing",
        #    which stopped being true once the A+B artifact was packaged).
        from .real_battery import run_full_config

        result_payload_doc = run_full_config(config)
        status = result_payload_doc["status"]
        reason = result_payload_doc.get("status_reason", "")
        fidelity = _fidelity_record(config, attack_descs, None, status)
        fidelity["full_backend"] = config.get("full_backend")
        if paths is not None:
            save_yaml(paths.resolved_config, config)
            save_json(paths.environment, environment_snapshot())
            save_json(paths.metrics, result_payload_doc["metrics"])
            save_json(paths.predictions, {"status": status, "status_reason": reason})
            save_json(paths.result, result_payload_doc)
            save_json(paths.fidelity, fidelity)
            paths.run_log.write_text(
                f"[{name}] mode=full -> {status}: {reason}\n", encoding="utf-8"
            )
        return result_payload_doc

    # --- smoke mode ---------------------------------------------------------
    seed_everything(seed)
    scenario_spec = config["scenario"]
    scenario = build_scenario(scenario_spec["name"], **scenario_spec.get("params", {}))
    sizes = _sizes(scenario)

    if scenario.stream is not None:
        sctx = build_stream_context(
            scenario.scheme,
            scenario.stream,
            attacker_model=scenario.attacker_model,
            reference_token_counts=(
                scenario.sample.reference_token_counts
                if scenario.sample is not None
                else config.get("reference_token_counts")
            ),
        )
        results = [a.run_stream(sctx) for a in attacks]
        metrics_list = [compute_stream_metrics(r, scenario.stream) for r in results]
    else:
        ctx = build_context(
            scenario.scheme,
            scenario.sample,
            attacker_model=scenario.attacker_model,
            attacker_obfuscator=scenario.attacker_obfuscator,
        )
        results = [a.run(ctx) for a in attacks]
        metrics_list = [compute_metrics(r, scenario.sample, ks) for r in results]

    result = results[0]
    metrics = metrics_list[0]
    diagnostics = dict(result.diagnostics or {})
    payload = _payload_for(
        config,
        specs[0][0],
        scenario.scheme.name,
        metrics,
        diagnostics,
        sizes,
        source={"config": config, "attack": attack_descs[0]},
        data={"scenario": scenario.name, "params": scenario_spec.get("params", {})},
    )

    fidelity = _fidelity_record(config, attack_descs, scenario.scheme.describe(), result.status.value)
    result_doc = {
        "name": name,
        "mode": mode,
        "status": result.status.value,
        "status_reason": result.status_reason,
        "metrics": metrics,
        "result": payload,
    }
    if len(attacks) > 1:
        result_doc["runs"] = [
            {"attack": d["name"], "status": r.status.value, "metrics": m}
            for d, r, m in zip(attack_descs, results, metrics_list)
        ]
    if paths is not None:
        save_yaml(paths.resolved_config, config)
        save_json(paths.environment, environment_snapshot())
        save_json(paths.metrics, metrics)
        save_json(paths.predictions, _jsonable_predictions(result))
        save_json(paths.result, payload)
        save_json(paths.fidelity, fidelity)
        paths.run_log.write_text(
            f"[{name}] mode=smoke attack={specs[0][0]} scenario={scenario.name} "
            f"status={result.status.value} class={spec_for(specs[0][0]).class_id} "
            f"knowledge={payload['attacker_knowledge']} schema_valid={payload['schema_valid']}\n",
            encoding="utf-8",
        )
    return result_doc
