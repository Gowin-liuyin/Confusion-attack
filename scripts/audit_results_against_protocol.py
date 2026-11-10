"""Read-only protocol audit of existing results (protocol §4, §6, §8; F10).

The auditor never edits a measurement and never upgrades a cell because a
payload flag says so. It reads each ``result.json``, applies the paper-v1 floors
for the exact attack variant through
:mod:`obfuscation_attacks.evaluation.protocol_audit`, and writes a separate
audit document with its own provenance.

Historical probe results lack a ``paper-v1`` binding and are therefore recorded
as ``below_budget`` with the specific gap -- they are not silently presented as
paper-scale.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "privacy_v1" / "full"
AUDIT_DIR = ROOT / "reports" / "privacy_rebuild_20261107" / "audit"

from obfuscation_attacks.evaluation.paper_profile import (  # noqa: E402
    require_profile,
    protocol_sha256,
)
from obfuscation_attacks.evaluation.protocol_audit import audit_payload  # noqa: E402


def _git() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip().splitlines()
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "error": str(exc)}
    return {"commit": commit, "n_uncommitted": len(dirty), "uncommitted_sample": dirty[:12]}


def main() -> None:
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    payloads = sorted(RESULTS.glob("**/result.json"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "read_only_protocol_audit",
        "protocol": {
            "path": "docs/PAPER_EVALUATION_PROTOCOL.md",
            "sha256": digest,
        },
        "code": _git(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "classification_note": (
            "Read-only audit. Historical probe payloads without an explicit paper-v1 "
            "profile and protocol hash are below_budget; 'audited_paper_scale' requires "
            "every floor, the positive control, per-unit evidence and saturation where "
            "the attack is iterative."
        ),
        "cells": {},
    }
    for path in payloads:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            audit["cells"][str(path.relative_to(ROOT))] = {
                "verdict": "unreadable", "reason": str(exc),
                "auditor": "protocol_auditor", "protocol_sha256": digest,
            }
            continue
        entry = audit_payload(payload, profile=profile, protocol_sha256=digest)
        entry.update({
            "path": str(path.relative_to(ROOT)),
            "class_id": payload.get("class_id"),
            "attack": payload.get("attack"),
            "target": payload.get("target"),
            "observation_surface": payload.get("observation_surface"),
            "attacker_knowledge": payload.get("attacker_knowledge"),
            "payload_main_table_eligible": payload.get("main_table_eligible"),
            "n_prompts": payload.get("n_prompts"),
            "n_tokens": payload.get("n_tokens"),
            "n_types": payload.get("n_types"),
            "n_keys": payload.get("n_keys"),
            "reference_tokens": payload.get("reference_tokens"),
        })
        key = (
            f"{payload.get('class_id') or '?'}:{payload.get('attack')}"
            f"@{payload.get('target')}:{payload.get('run_id', 'no-run-id')}"
        )
        audit["cells"][key] = entry
    counts: dict[str, int] = {}
    for entry in audit["cells"].values():
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    audit["counts"] = counts
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / f"protocol_audit_{stamp}.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(audit['cells'])} cells, verdicts={counts}")


if __name__ == "__main__":
    main()
