"""Record a sampled embedding-lineage probe for a standalone A export."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  # Load before torch on the installed Windows runtime.

from build_paper_manifest import column_alignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--sample-rows", type=int, default=1024)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite an existing lineage report")
    if args.sample_rows < 1:
        raise ValueError("sample-rows must be positive")
    result = column_alignment(args.artifact, args.base_model, n_rows=args.sample_rows)
    report = {
        "kind": "standalone_a_lineage_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact.resolve()), "base_model": str(args.base_model.resolve()),
        "artifact_manifest_sha256": _sha256(args.artifact / "manifest.json"),
        "base_weights_sha256": _sha256(args.base_model / "model.safetensors"),
        "lineage_implementation_sha256": _sha256(Path(__file__).with_name("build_paper_manifest.py")),
        "sample_rows_requested": args.sample_rows, "result": result,
        "scope": "sampled embedding-lineage probe; not formal privacy, utility, or A+B compatibility evidence",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Recorded lineage probe ({result.get('verdict')}); report: {args.report}")


if __name__ == "__main__":
    main()
