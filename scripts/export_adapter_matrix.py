#!/usr/bin/env python3
"""Export the eight-arm adapter x attack matrix.

Without ``--load-deployment`` the matrix is declaration-only (every arm blocked
with its exact shortages). With a verified manifest it builds a table-only
:class:`ABDeployment` so the local arms (and OSNIP-only's server side) are
adapter-ready, while the external arms stay blocked until an author backend is
supplied.

Usage::

    python scripts/export_adapter_matrix.py --manifest configs/paper/manifest.paper-v1.robust.json \
        --load-deployment --out reports/privacy_rebuild_20261107/round4/adapter_matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.generic_runner import adapter_attack_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--load-deployment", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = None
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    deployments: dict = {}
    if args.load_deployment:
        if manifest is None:
            raise SystemExit("--load-deployment needs --manifest")
        from obfuscation_attacks.schemes.ab_deployment import ABDeployment

        artifact = manifest["artifacts"]["ab"]["path"]
        model = manifest["models"]["llama-3.2-1b-base"]["path"]
        root_key_file = ROOT / ".secrets" / "paper-v1.root"
        root_key = root_key_file.read_bytes()[:32] if root_key_file.is_file() else None
        deployment = ABDeployment(
            artifact=Path(artifact), clean_model=Path(model),
            dtype="float32", layers=(1,), max_len=None,
            load_models=False, strict_length=True,
            expected_variant="base", root_key=root_key,
        )
        for arm in ("plaintext", "aloepri", "ab", "osnip"):
            deployments[arm] = deployment

    matrix = adapter_attack_matrix(manifest=manifest, deployments=deployments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    for arm, row in matrix["arms"].items():
        print(f"{arm:14s} ready={row['ready']!s:5s} surface={row['upload_surface']:8s} "
              f"shortages={row['shortages'][:2]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
