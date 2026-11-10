"""Record complete square-export decoder and noisy-endpoint lineage checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  # Load before torch on the installed Windows runtime.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obfuscation_attacks.evaluation import export_randomness, square_endpoints, square_lineage  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite a previous report")
    started = time.perf_counter()
    randomness = export_randomness.verify_export_randomness(args.artifact)
    decoder = square_lineage.verify_square_lineage(args.artifact, args.base)
    endpoints = square_endpoints.verify_square_endpoints(args.artifact, args.base)
    servers = [path for path in (args.artifact / "server", args.artifact / "model_server") if path.is_dir()]
    if len(servers) != 1:
        raise ValueError("artifact must contain exactly one server model directory")
    inputs = [path for root in (servers[0], args.base) for path in root.iterdir()
              if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith(".json"))]
    report = {
        "kind": "square_export_full_weight_lineage", "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact.resolve()), "base": str(args.base.resolve()),
        "file_sha256": {str(path.resolve()): sha256(path) for path in sorted(inputs)},
        "client_file_commitment_sha256": sha256(args.artifact / "client" / "client_secret.pt"),
        "checker_sha256": {Path(module.__file__).name: sha256(Path(module.__file__))
                           for module in (export_randomness, square_lineage, square_endpoints)},
        "driver_sha256": sha256(Path(__file__)), "command": list(sys.argv),
        "protocol_sha256": sha256(ROOT / "docs" / "PAPER_EVALUATION_PROTOCOL.md"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": square_lineage.torch.__version__, "platform": platform.platform(),
                        "torch_threads": square_lineage.torch.get_num_threads(),
                        "torch_interop_threads": square_lineage.torch.get_num_interop_threads()},
        "randomness": randomness, "decoder": decoder, "endpoints": endpoints,
        "seconds": time.perf_counter() - started,
        "scope": "square A weight lineage only; not A+B compatibility, privacy, utility, or formal experiment evidence",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Verified decoder and endpoints; report: {args.report}")


if __name__ == "__main__":
    main()
