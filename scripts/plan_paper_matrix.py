"""Write an expanded, non-executing paper-v1 matrix as a framed JSONL stream."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.paper_matrix import iter_expanded_specs  # noqa: E402
from obfuscation_attacks.evaluation.paper_profile import require_profile  # noqa: E402
from obfuscation_attacks.evaluation.paper_runner import register_implemented_entries  # noqa: E402
from obfuscation_attacks.evaluation.public_guard import assert_public_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    for option in ("arms", "models", "scenarios", "attacks"):
        parser.add_argument("--" + option[:-1], action="append", dest=option)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing matrix")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    profile = require_profile("paper-v1")
    register_implemented_entries()
    filters = {name: set(getattr(args, name)) for name in ("arms", "models", "scenarios", "attacks")
               if getattr(args, name) is not None}
    header = {"kind": "matrix_header", "profile": profile.name, "protocol_sha256": profile.protocol_sha256,
              "scope": "planning only; complete file requires the matrix_end trailer",
              "filters": {name: sorted(values) for name, values in filters.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        def write(value):
            assert_public_payload(value, label="matrix")
            line = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            return line

        digest.update(write(header).encode("utf-8"))
        for spec in iter_expanded_specs(profile, manifest, filters=filters):
            digest.update(write({"kind": "plan_spec", "spec": spec}).encode("utf-8"))
            count += 1
        write({"kind": "matrix_end", "spec_count": count, "preceding_lines_sha256": digest.hexdigest()})
    print(f"Wrote {count} planned units to {args.output}")


if __name__ == "__main__":
    main()
