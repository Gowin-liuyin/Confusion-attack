#!/usr/bin/env python3
"""RETIRED: never rewrite a stored measurement.

This tool used to re-render App.-A metadata *in place* on existing
``reports/privacy_v1/full/.../result.json`` files. Even when it only changed
booleans, it changed the run identity hashes and could flip
``uses_known_pairs``/``main_table_eligible`` without a new measurement -- which
the protocol (§4.4, §6) and CLAUDE.md §5 forbid. The historical payloads it
touched are preserved as-is and flagged in the read-only protocol audit.

Use ``scripts/audit_results_against_protocol.py`` for a verdict. If a run's
flags are wrong, re-run the measurement with a new run identity; do not edit the
old result.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "refresh_payload_flags.py is retired: stored measurements must not be "
        "rewritten. Use scripts/audit_results_against_protocol.py and a new run id."
    )


if __name__ == "__main__":
    main()
