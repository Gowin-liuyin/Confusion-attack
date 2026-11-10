"""Positive-control + protocol gate for the privacy-plan attack families.

Runs the synthetic gate arms listed in ``ARMS`` (classes C01, C02, C04, C07,
C12, C13, C14) through the same code path, writes the plan's App.-A
``result.json`` per arm under ``reports/privacy_v1/phase1/<class>/<attack>/<arm>/``
(plus ``status.txt`` and ``.done``, as required by plan §8.9), validates the
payload schema, and exits non-zero if

* any payload is schema-invalid,
* any positive control misses its floor (plan §7.1: control failure == attack
  invalid, and an invalid attack may not report a number), or
* an attack that should be ``not_applicable`` (no generative LM for the GPT-4
  substitute, no stream for a single-request attack) instead produced a score.

This gate does **not** cover every registered class: C03/C05/C06/C08/C09/C10/C11
are exercised by their own runners and tests, and absence here is not evidence
that their controls passed. See ``docs/ATTACK_PROVENANCE_14.md`` for the mapping.

Usage::

    python scripts/privacy_family_gates.py            # run all gate arms
    python scripts/privacy_family_gates.py --only C12 # one class (merges into the summary)

The numbers produced here come from the synthetic scenarios in
``data/synthetic_privacy.py``; they are *protocol* evidence (the attack runs and
its control passes), not paper-scale claims. ``docs/ATTACK_STATUS_14.md`` records
which class is at which stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.attack_model import (  # noqa: E402
    result_payload,
    validate_payload,
)
from obfuscation_attacks.evaluation.runner import _sizes, run_scenario  # noqa: E402
from obfuscation_attacks.utils import save_json  # noqa: E402

REPORTS = ROOT / "reports" / "privacy_v1" / "phase1"


@dataclass
class Arm:
    """One gate arm: an attack run plus the control it must satisfy."""

    class_id: str
    attack: str
    scenario: str
    attack_params: dict = field(default_factory=dict)
    scenario_params: dict = field(default_factory=dict)
    target: str = ""
    control_name: str = ""
    control_check: str = ""  # dotted path into metrics, e.g. "token_recovery.asr@1"
    control_floor: float = 0.0
    expect_not_applicable: bool = False


def _dig(metrics: dict, path: str):
    cur = metrics
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


ARMS: list[Arm] = [
    # --- positive controls (plaintext / single-key / identity arms) ---------
    Arm(
        "C01", "nn_knn.euclidean", "nn",
        scenario_params={"noise": 0.0},
        target="plaintext",
        control_name="clean KNN top-1", control_check="token_recovery.asr@1",
        control_floor=0.99,
    ),
    Arm(
        "C07", "beamclean.gaussian", "beamclean",
        scenario_params={"noise": 0.0, "n": 6},
        target="plaintext",
        control_name="BeamClean on clean wire (naive NN)", control_check="token_recovery.asr@1",
        control_floor=0.99,
    ),
    Arm(
        "C12", "linkage.exact", "linkage",
        scenario_params={"key_size": 1, "n_requests": 12, "rotation_n": 64},
        target="plaintext",
        control_name="single-key exact link rate", control_check="linkage.link_rate",
        control_floor=0.99,
    ),
    Arm(
        "C13", "pii.extract", "pii",
        scenario_params={"identity": True},
        target="plaintext",
        control_name="identity output PIIRSR", control_check="pii.pii_rsr",
        control_floor=0.99,
    ),
    Arm(
        "C14", "lm_recon.ngram", "lm_recon",
        scenario_params={"identity": True},
        target="plaintext",
        control_name="identity output 1-gram overlap",
        control_check="output_reconstruction.ngram_overlap.1-gram",
        control_floor=0.99,
    ),
    Arm(
        "C02", "ima.cross_instance", "ima_cross_instance",
        attack_params={"epochs": 120},
        target="kI_self_instance",
        control_name="attacker in-instance accuracy",
        control_check="diagnostics.heldout_instance_accuracy",
        control_floor=0.95,
    ),
    Arm(
        "C04", "ima.cross_checkpoint", "cross_checkpoint",
        attack_params={"epochs": 120},
        target="kI_transfer",
        control_name="same-checkpoint in-instance accuracy",
        control_check="diagnostics.heldout_instance_accuracy",
        control_floor=0.95,
    ),
    # --- protected arms -----------------------------------------------------
    Arm(
        "C07", "beamclean.gaussian", "beamclean",
        scenario_params={"noise": 1.2, "n": 8},
        target="wire_gaussian",
    ),
    Arm(
        "C07", "beamclean.laplacian", "beamclean",
        scenario_params={"noise": 1.2, "n": 8, "noise_model": "laplacian"},
        target="wire_laplacian",
    ),
    Arm(
        "C12", "linkage.exact", "linkage",
        scenario_params={"key_size": 4, "rotation_n": 8, "n_requests": 24},
        target="rotation_SHP",
    ),
    Arm(
        "C12", "linkage.exact", "linkage",
        scenario_params={"key_size": 4, "rotation_n": 8, "n_requests": 24, "metadata_visible": False},
        target="rotation_SHB",
    ),
    Arm(
        "C12", "linkage.approximate", "linkage",
        attack_params={"max_queries": 60},
        scenario_params={"key_size": 4, "rotation_n": 8, "n_requests": 16},
        target="rotation_SHP_approx",
    ),
    Arm(
        "C13", "pii.extract", "pii",
        scenario_params={"n_entities": 3},
        target="output_permutation",
    ),
    Arm(
        "C14", "lm_recon.ngram", "lm_recon",
        target="output_permutation",
    ),
    Arm(
        "C14", "lm_recon.local_lm", "lm_recon",
        target="gpt4_substitute",
        expect_not_applicable=True,
    ),
    Arm(
        "C02", "ima.cross_instance", "ima_cross_instance",
        attack_params={"epochs": 120},
        target="ki_cross_instance_attack",
    ),
    Arm(
        "C04", "ima.cross_checkpoint", "cross_checkpoint",
        attack_params={"epochs": 120},
        target="ki_cross_checkpoint_attack",
    ),
]


def run_arm(arm: Arm) -> dict:
    out = run_scenario(
        arm.attack,
        arm.scenario,
        attack_params=arm.attack_params,
        scenario_params=arm.scenario_params,
    )
    result = out["result"]
    metrics = out["metrics"]
    scenario = out["model_scenario"]
    diagnostics = dict(result.diagnostics or {})
    identity = {
        "attack": arm.attack,
        "target": arm.target,
        "scenario": arm.scenario,
        "attack_params": arm.attack_params,
        "scenario_params": arm.scenario_params,
        "metrics": metrics,
    }
    run_id = "run-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    arm_dir = (
        REPORTS / arm.class_id / arm.attack.replace(".", "_")
        / f"{arm.target}__{run_id}"
    )
    arm_dir.mkdir(parents=True, exist_ok=True)

    sizes = _sizes(scenario)
    payload = result_payload(
        attack_name=arm.attack,
        scheme_name=scenario.scheme.name,
        target=arm.target or scenario.scheme.name,
        metrics=metrics,
        diagnostics=diagnostics,
        positive_control=(diagnostics.get("positive_control") or {}),
        source={"attack": arm.attack, "scenario": arm.scenario},
        data={"scenario": arm.scenario, "params": arm.scenario_params},
        overrides={
            "candidate_mode": diagnostics.get("candidate_mode", "full"),
            **(
                {"attacker_knowledge": diagnostics["attacker_knowledge"]}
                if diagnostics.get("attacker_knowledge")
                else {}
            ),
            **(
                {"rotation_aware": diagnostics["rotation_aware"]}
                if "rotation_aware" in diagnostics
                else {}
            ),
        },
        **sizes,
    )
    problems = validate_payload(payload)
    payload["schema_valid"] = not problems
    payload["schema_problems"] = problems
    payload["run_id"] = run_id
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["identity"] = identity
    save_json(arm_dir / "result.json", payload)
    (arm_dir / "status.txt").write_text(
        f"{result.status.value}\n{result.status_reason}\n", encoding="utf-8"
    )
    (arm_dir / ".done").write_text("done\n", encoding="utf-8")
    return {"arm": arm, "status": result.status.value, "metrics": metrics,
            "payload": payload, "problems": problems}


def _merged_summary(existing: dict, only: str | None, rows: list[str],
                    failures: list[str], n_arms: int) -> dict:
    """Build the gate summary without truncating other classes on a subset run."""
    if only is None:
        return {
            "n_arms": n_arms,
            "n_failures": len(failures),
            "failures": failures,
            "rows": rows,
        }
    prefix = f"{only:4s} "
    merged_rows = [
        row for row in existing.get("rows") or [] if not str(row).startswith(prefix)
    ] + rows
    merged_failures = [
        failure for failure in existing.get("failures") or []
        if not str(failure).startswith(str(only))
    ] + failures
    return {
        "n_arms": len(merged_rows),
        "n_failures": len(merged_failures),
        "failures": merged_failures,
        "rows": merged_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run one class id, e.g. C12")
    args = ap.parse_args()
    arms = [a for a in ARMS if args.only is None or a.class_id == args.only]
    failures: list[str] = []
    rows: list[str] = []

    for arm in arms:
        res = run_arm(arm)
        status = res["status"]
        payload = res["payload"]
        if res["problems"]:
            failures.append(f"{arm.class_id} {arm.attack} [{arm.target}]: schema {res['problems']}")
        if arm.expect_not_applicable:
            ok = status == "not_applicable"
            if not ok:
                failures.append(
                    f"{arm.class_id} {arm.attack} [{arm.target}]: expected not_applicable, got {status}"
                )
            rows.append(f"{arm.class_id:4s} {arm.attack:24s} {arm.target:28s} "
                        f"{status:15s} expected=not_applicable ok={ok}")
            continue
        if status != "ok":
            failures.append(f"{arm.class_id} {arm.attack} [{arm.target}]: status {status}")
            rows.append(f"{arm.class_id:4s} {arm.attack:24s} {arm.target:28s} {status}")
            continue

        value = _dig(res["metrics"], arm.control_check) if arm.control_check else None
        if arm.control_check:
            v = None if value is None else float(value)
            ok = v is not None and v >= arm.control_floor
            if not ok:
                failures.append(
                    f"{arm.class_id} {arm.attack} [{arm.target}]: control "
                    f"{arm.control_name}={v} < floor {arm.control_floor}"
                )
            rows.append(
                f"{arm.class_id:4s} {arm.attack:24s} {arm.target:28s} "
                f"control {arm.control_name}={v} floor={arm.control_floor} ok={ok}"
            )
        else:
            summary = _summary_line(res["metrics"])
            rows.append(f"{arm.class_id:4s} {arm.attack:24s} {arm.target:28s} ok  {summary}")
        assert payload["schema_valid"], payload["schema_problems"]

    print("\n".join(rows))
    summary_path = REPORTS / "gate_summary.json"
    if args.only is None:
        existing: dict = {}
    else:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    summary = _merged_summary(existing, args.only, rows, failures, len(arms))
    save_json(summary_path, summary)
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"\nGATE PASSED: {len(arms)} arms, all controls met, all payloads schema-valid.")
    return 0


def _summary_line(metrics: dict) -> str:
    tr = metrics.get("token_recovery") or {}
    lk = metrics.get("linkage") or {}
    pii = metrics.get("pii") or {}
    rec = metrics.get("output_reconstruction") or {}
    parts = []
    if "asr@1" in tr:
        parts.append(f"top1={tr['asr@1']:.3f}")
    if lk.get("link_rate") is not None:
        parts.append(f"link_rate={lk['link_rate']:.3f}")
    if lk.get("tpr_at_default") is not None:
        parts.append(f"tpr@0.5={lk['tpr_at_default']:.3f} fpr={lk['fpr_at_default']:.3f}")
    if pii.get("pii_rsr") is not None:
        parts.append(f"pii_rsr={pii['pii_rsr']:.3f}")
    if rec.get("ngram_overlap"):
        parts.append(f"1gram={rec['ngram_overlap']['1-gram']:.3f}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
