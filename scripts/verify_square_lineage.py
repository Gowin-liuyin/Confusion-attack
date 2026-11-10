"""Record exact decoder-lineage verification for a square A export."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  # Load before torch on this Windows environment.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obfuscation_attacks.evaluation import square_lineage  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite a previous report")
    result = square_lineage.verify_square_lineage(args.artifact, args.base)
    paths = [path for root in (args.artifact / "server", args.base) for path in root.iterdir()
             if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith(".json"))]
    identity = {str(path.resolve()): digest(path) for path in sorted(paths)}
    report = {"kind": "standalone_a_exact_decoder_lineage", "created_utc": datetime.now(timezone.utc).isoformat(),
              "artifact": str(args.artifact.resolve()), "base": str(args.base.resolve()),
              "file_sha256": identity, "verifier_sha256": digest(Path(square_lineage.__file__)),
              "result": result,
              "scope": "exact decoder comparison only; embedding and output-head noise, A+B compatibility, privacy and utility are separate"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Verified {result['checked_tensors']} decoder tensors; report: {args.report}")


if __name__ == "__main__":
    main()
