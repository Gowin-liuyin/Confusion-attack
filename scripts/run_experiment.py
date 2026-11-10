"""Run one experiment config, or every smoke config.

Usage:
    python scripts/run_experiment.py --config configs/smoke/nn_knn.yaml
    python scripts/run_experiment.py --all-smoke
    python scripts/run_experiment.py --config configs/osnip/knn.yaml   # full -> resource_blocked

Smoke configs run synthetic scenarios end-to-end and write full artifacts under
``reports/runs/<name>/``. Full paper-scale configs are recorded as
``resource_blocked`` (real models/data/GPU not available here) - never faked.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.runner import run_from_config  # noqa: E402
from obfuscation_attacks.utils import load_yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run obfuscation-attack experiments")
    ap.add_argument("--config", help="path to a single experiment config (yaml)")
    ap.add_argument("--all-smoke", action="store_true", help="run every configs/smoke/*.yaml")
    ap.add_argument("--reports", default=str(ROOT / "reports" / "runs"), help="artifact root")
    args = ap.parse_args()

    if args.all_smoke:
        configs = sorted(glob.glob(str(ROOT / "configs" / "smoke" / "*.yaml")))
    elif args.config:
        configs = [args.config]
    else:
        ap.error("provide --config <file> or --all-smoke")
        return 2

    if not configs:
        print("no configs found", file=sys.stderr)
        return 1

    failures = 0
    for cfg_path in configs:
        cfg = load_yaml(cfg_path)
        out = run_from_config(cfg, reports_root=args.reports)
        status = out["status"]
        line = f"[{out['name']}] mode={out['mode']} -> status={status}"
        token = out.get("metrics", {}).get("token_recovery")
        mapping = out.get("metrics", {}).get("mapping_recovery")
        perm = out.get("metrics", {}).get("permutation_recovery")
        if token:
            line += f"  asr@1={token.get('asr@1')} ttrsr={token.get('ttrsr')}"
        if mapping:
            line += f"  map_top1={mapping.get('mapping_top1_accuracy')}"
        if perm:
            line += f"  Z1_exact={perm.get('Z1_exact')} verified={perm.get('verified')}"
        print(line)
        # A smoke run that failed (not merely not-applicable) is an error.
        if cfg.get("mode", "smoke") == "smoke" and status == "failed":
            failures += 1

    if failures:
        print(f"{failures} smoke run(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
