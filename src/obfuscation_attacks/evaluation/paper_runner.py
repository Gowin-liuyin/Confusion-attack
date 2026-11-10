"""paper-v1 execution dispatcher (W7).

The plan is the contract; this module binds the implemented cells to their real
entry points, registers them as faithful, and dispatches only cells that the
preflight marks ``configured``. Model/data loading is deliberately injected: on
this machine the paper-v1 models and corpora are absent, so a run without an
executor returns ``resource_blocked`` with the exact missing context instead of
falling through to the legacy battery.

No cell is marked faithful unless its entry actually implements the paper
algorithm (the W1-W6 tests are the evidence). External arms stay unregistered.
"""

from __future__ import annotations

from typing import Callable, Mapping

from . import real_battery
from .paper_profile import (
    PAPER_PROFILE_NAME,
    PreflightError,
    plan_jobs,
    preflight_formal_start,
    register_faithful_entry,
)

#: attack id -> {arms, models, scenarios, runner attribute on real_battery}
#: Only cells the paper executor can actually run end-to-end are registered. The
#: algorithms below (VMA/ISA/Gate/Attn/linkage/TFMA/SDA/output) have implementations
#: but no verified paper-v1 executor entry, so they stay explicitly unconfigured.
IMPLEMENTED_ENTRIES: dict[str, dict] = {
    "knn_l2": {
        "arms": {"plaintext", "aloepri", "ab"},
        "models": {"llama-3.2-1b-base"},
        "scenarios": {"S-public"},
        "runner": "run_c01",
        "params": {"metric": "l2"},
    },
    "knn_cosine": {
        "arms": {"plaintext", "aloepri", "ab"},
        "models": {"llama-3.2-1b-base"},
        "scenarios": {"S-public"},
        "runner": "run_c01",
        "params": {"metric": "cosine"},
    },
    "knn_l2_key_robustness": {
        "arms": {"ab"},
        "models": {"llama-3.2-1b-base"},
        "scenarios": {"S-public"},
        "runner": "run_c01",
        "params": {"metric": "l2", "keys": 5, "texts": 200},
    },
}

#: declaration kept for the plan's audit trail: implementation exists, executor does not
UNWIRED_ENTRIES: dict[str, str] = {
    "vma_h_recovered_prefix": "real_vma implements the algorithm; no paper-v1 executor entry yet",
    "vma_h_observed_prefix": "real_vma implements the algorithm; no paper-v1 executor entry yet",
    "isa_er": "real_isa binds the original ER; the 50k x 5-key x 4-layer CPU budget is not schedulable here",
    "isa_tbs": "real_isa binds the original TBS; the 50k x 5-key x 4-layer CPU budget is not schedulable here",
    "vma_w": ("Table 9 public weight matching and composition exist; formal execution "
              "requires verified two-sided AloePri artifacts and a paper-v1 executor"),
    "gate_ia": "real_weight_attacks implements the public-candidate gate attack; no executor entry",
    "attn_ia": "real_weight_attacks implements the public-candidate attention attack; no executor entry",
    "linkage_exact": "run_c12 implements exact linkage; no paper-v1 stream executor entry",
    "tfma": "run_c10 implements TFMA; the 10M-token reference corpus is not prepared",
    "sda": ("public-upload and engineering decode paths exist; formal SDA remains unwired: "
            "independent training data, validation/saturation, control evidence, character alignment "
            "where needed, and end-to-end frozen-stream verification are not established"),
    "output_pii": "real_battery runs real generation; no paper-v1 executor entry or PII annotation set",
}


class DispatchError(RuntimeError):
    pass


def register_implemented_entries() -> None:
    """Register only the entries with a verified real implementation."""
    for attack_id, entry in IMPLEMENTED_ENTRIES.items():
        register_faithful_entry(
            attack_id,
            arms=set(entry["arms"]),
            models=set(entry["models"]),
            scenarios=set(entry.get("scenarios") or {"S-public", "S-deploy", "S-encoder", "S-pair"}),
        )


def configured_jobs(profile, manifest: Mapping | None, *,
                    requested: Mapping | None = None) -> list[dict]:
    jobs = plan_jobs(profile, manifest, root=None)["jobs"]
    selected = [
        job for job in jobs
        if job["status"] == "configured" and job.get("faithful_entry") and job["applicable"]
    ]
    if requested:
        for key in ("attacks", "arms", "scenarios"):
            if key in requested and not requested[key]:
                # An explicitly empty filter means "no cells", never "all cells".
                return []
        attacks = set(requested.get("attacks") or [])
        arms = set(requested.get("arms") or [])
        scenarios = set(requested.get("scenarios") or [])
        if attacks:
            selected = [job for job in selected if job["attack"] in attacks]
        if arms:
            selected = [job for job in selected if job["arm"] in arms]
        if scenarios:
            selected = [job for job in selected if job["scenario"] in scenarios]
    if selected:
        # The plan's per-job blockers are a subset of the formal preflight; the
        # dispatcher must never execute a cell the full preflight would refuse.
        by_attack: dict[str, dict] = {}
        for job in selected:
            bucket = by_attack.setdefault(
                job["attack"], {"arms": set(), "models": set(), "scenarios": set()}
            )
            bucket["arms"].add(job["arm"])
            bucket["models"].add(job["model"])
            bucket["scenarios"].add(job["scenario"])
        for attack, bucket in by_attack.items():
            try:
                preflight_formal_start(
                    profile,
                    manifest,
                    {
                        "attacks": [attack],
                        "arms": sorted(bucket["arms"]),
                        "scenarios": sorted(bucket["scenarios"]),
                    },
                    models=set(bucket["models"]),
                )
            except PreflightError:
                return []
    return selected


def dispatch_configured(
    profile,
    manifest: Mapping | None = None,
    *,
    requested: Mapping | None = None,
    executor: Callable[[dict], dict] | None = None,
) -> dict:
    """Dispatch configured cells; without an executor report resource_blocked."""
    if getattr(profile, "name", None) != PAPER_PROFILE_NAME:
        raise DispatchError("paper-v1 execution requires the paper-v1 profile")
    jobs = configured_jobs(profile, manifest, requested=requested)
    results = []
    for job in jobs:
        entry = IMPLEMENTED_ENTRIES.get(job["attack"])
        if entry is None:
            results.append({**job, "run_status": "not_run",
                            "reason": "no implemented entry bound"})
            continue
        runner = getattr(real_battery, str(entry["runner"]), None)
        if runner is None:
            results.append({**job, "run_status": "not_run",
                            "reason": f"runner {entry['runner']} is absent"})
            continue
        if executor is None:
            results.append({
                **job,
                "run_status": "resource_blocked",
                "reason": (
                    "no paper-v1 executor supplied; pass --execute with a verified "
                    "manifest to run this configured cell"
                ),
                "runner": entry["runner"],
                "params": dict(entry.get("params", {})),
            })
            continue
        try:
            record = executor(job)
        except Exception as exc:  # executor failures stay visible per cell
            results.append({**job, "run_status": "failed", "reason": str(exc)})
            continue
        if (
            not isinstance(record, dict)
            or record.get("outcome") not in {"executed", "control_failed", "failed"}
        ):
            results.append({
                **job,
                "run_status": "failed",
                "reason": "executor returned no explicit outcome/result_path",
            })
            continue
        results.append({**job, "run_status": str(record["outcome"]), "result": record})
    statuses = {row["run_status"] for row in results}
    if results and statuses == {"executed"}:
        overall = "executed"
    elif "resource_blocked" in statuses:
        overall = "resource_blocked"
    elif "failed" in statuses:
        overall = "failed"
    elif "control_failed" in statuses:
        overall = "control_failed"
    else:
        overall = "not_run"
    return {
        "profile": PAPER_PROFILE_NAME,
        "protocol_sha256": getattr(profile, "protocol_sha256", None),
        "configured": len(jobs),
        "results": results,
        "status": overall,
    }
