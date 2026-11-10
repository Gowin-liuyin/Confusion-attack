#!/usr/bin/env python3
"""CLI for the real-artifact attack battery (implementation lives in the package).

Usage::

    python scripts/run_full_battery.py --classes C01,C06W,C08 --knn-texts 400
    python scripts/run_full_battery.py --classes C10,C12 --requests 1000
"""

from __future__ import annotations


import os

# BLAS threading must be set before numpy is imported: without it the
# 1000x2048x128256 GEMM runs at ~1 GFLOPS (>10 min on this box), with it ~350 GFLOPS (1.5 s).
# Measured with scripts/_bench_blas.py; ``setdefault`` keeps an explicit caller override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(24, os.cpu_count() or 8)))

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.data.corpora import load_alpaca, overlap_report, reference_counts  # noqa: E402
from obfuscation_attacks.evaluation.real_remaining import (  # noqa: E402
    run_c02,
    run_c03,
    run_c07,
    run_c09,
    run_c14,
)
from obfuscation_attacks.evaluation.real_battery import (  # noqa: E402
    run_c06h,
    run_c08_attn,
    OUT_ROOT,
    _json_default,
    _partition_texts,
    run_c01,
    run_c05,
    run_c06w,
    run_c08,
    run_c10,
    run_c12,
    run_c13_c14,
    run_composed,
    recover_mapping_gate_ia,
    _resolve_c11_probe_config,
    _run_c11_bridge,
)
from obfuscation_attacks.schemes.ab_deployment import (  # noqa: E402
    ABDeployment,
    DEFAULT_ARTIFACT,
    DEFAULT_CLEAN,
)
from obfuscation_attacks.evaluation.paper_profile import (  # noqa: E402
    gate_battery_args,
    protocol_sha256 as _paper_protocol_sha256,
)


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def _load_sda_probe_config(path: Path, *, seed: int, requests: int) -> dict:
    import yaml
    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            self.flatten_mapping(node)
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("SDA configuration keys must be strings")
                if key in result:
                    raise ValueError(f"duplicate SDA configuration field: {key}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result
    try:
        config = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError("invalid YAML or JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("SDA configuration must be a mapping")
    config = dict(config)
    supported = {"profile", "seed", "split_seed", "requests", "sda_val", "device", "symbol_frontend",
                 "distance_threshold", "group_joint", "validation_beams", "control_keys",
                 "validation_noise_rate", "validation_noise_kinds", "attack"}
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"unsupported SDA configuration fields: {sorted(unknown)}")
    if isinstance(config.get("attack"), dict) and set(config["attack"]) - {"name", "variant", "params"}:
        raise ValueError("unsupported SDA attack fields")
    config.setdefault("profile", "probe")
    config.setdefault("seed", seed)
    config.setdefault("requests", requests)
    _resolve_c11_probe_config(config)
    return config


# The key-based forbidden-material scanner is shared with every result writer
# (real_battery.write_payload); only the corpus-text scan lives in this script.
from obfuscation_attacks.evaluation.public_guard import (  # noqa: E402
    forbidden_material_gaps as _forbidden_material_gaps,
)


def _corpus_text_leaks(obj, corpus_texts, *, min_length: int = 20,
                       limit: int = 8, path: str = "summary") -> list[str]:
    """Corpus texts (>= ``min_length`` chars) appearing verbatim in a string value.

    Walks the parsed object's STRING values (not a raw JSON substring search), so
    JSON-escaped quotes/backslashes/newlines cannot hide a leak. The offending
    text itself is never echoed; only its path, index and length are reported.
    """
    texts: list[str] = []
    seen: set[str] = set()
    for text in corpus_texts:
        value = str(text)
        if len(value) < int(min_length) or value in seen:
            continue
        seen.add(value)
        texts.append(value)
    leaks: list[str] = []

    def _walk(value, where: str) -> None:
        if len(leaks) >= int(limit):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(child, f"{where}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, f"{where}[{index}]")
            return
        if isinstance(value, str):
            for index, text in enumerate(texts):
                if text in value:
                    leaks.append(
                        f"{where}: evaluated corpus text[{index}] "
                        f"({len(text)} chars) appears verbatim"
                    )
                    return

    _walk(obj, path)
    return leaks


def _guard_public_write(payload, *, label: str, corpus_texts=None) -> str:
    """Serialise a public artifact and fail closed on truth/secret material.

    Runs the forbidden-key scan and (when the evaluated corpus is known) the
    verbatim corpus-text scan. Raises ``SystemExit`` before the caller writes
    anything, so no summary/plan/report can be created from a leaking payload.
    """
    serialized = json.dumps(payload, indent=2, ensure_ascii=False,
                            default=_json_default)
    # Both scans walk the JSON view, not the pre-serialisation object: json
    # converts ndarray -> list, so material wrapped in an array cannot bypass
    # the key or corpus-text scan.
    scanned = json.loads(serialized)
    forbidden = _forbidden_material_gaps(scanned, path=str(label))
    if corpus_texts:
        forbidden += _corpus_text_leaks(
            scanned, corpus_texts, path=str(label)
        )
    if forbidden:
        raise SystemExit(
            f"refusing to write {label} with truth/secret material: "
            + ", ".join(forbidden[:8])
        )
    return serialized


def _requested_from_argv(args, argv: list[str]) -> dict:
    """Paper-v1 must not inherit CLI defaults. Only explicit flags are a request."""
    requested: dict = {}
    if _has_flag(argv, "--knn-texts"):
        requested["knn_texts"] = args.knn_texts
    if _has_flag(argv, "--isa-prompts"):
        requested["isa_prompts"] = args.isa_prompts
    if _has_flag(argv, "--vma-prompts"):
        requested["vma_prompts"] = args.vma_prompts
    if _has_flag(argv, "--isa-steps"):
        requested["isa_steps"] = [int(part) for part in str(args.isa_steps).split(",") if part.strip()]
    if _has_flag(argv, "--isa-layers"):
        requested["layers"] = [int(part) for part in str(args.isa_layers).split(",") if part.strip()]
    if _has_flag(argv, "--classes"):
        requested["classes"] = [part.strip().upper() for part in args.classes.split(",") if part.strip()]
    if _has_flag(argv, "--attacks"):
        requested["attacks"] = [part.strip() for part in args.attacks.split(",") if part.strip()]
    if _has_flag(argv, "--arms"):
        requested["arms"] = [part.strip() for part in args.arms.split(",") if part.strip()]
    if _has_flag(argv, "--scenarios"):
        requested["scenarios"] = [part.strip() for part in args.scenarios.split(",") if part.strip()]
    if _has_flag(argv, "--seed"):
        requested["training_seeds"] = [args.seed]
    return requested


def _load_root_key(path: Path) -> bytes:
    """Load (or create once) the secret 256-bit deployment root for a paper run.

    The key never enters the manifest, the result payload or stdout; the payload
    stores only a SHA-256 commitment.
    """
    if path.is_file():
        data = path.read_bytes()
        if len(data) < 32:
            raise SystemExit(f"root key file {path} is shorter than 32 bytes")
        return data[:32]
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    import stat
    path.write_bytes(key)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    print(f"[battery] created a new secret root key at {path} (not printed)", file=sys.stderr)
    return key


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="C01,C06W,C08")
    ap.add_argument("--attacks", default="", help="paper-v1 cell filter (comma-separated)")
    ap.add_argument("--arms", default="", help="paper-v1 arm filter (comma-separated)")
    ap.add_argument("--scenarios", default="", help="paper-v1 scenario filter (comma-separated)")
    ap.add_argument("--execute", action="store_true",
                    help="construct the real paper-v1 executor (long-running, real artifacts)")
    ap.add_argument("--root-key-file", type=Path, default=ROOT / ".secrets" / "paper-v1.root",
                    help="secret root key for derived per-request keys (never printed)")
    ap.add_argument("--result-root", type=Path, default=None,
                    help="immutable result root (default: reports/privacy_rebuild_20261107/runs)")
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--clean-model", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--alpaca", type=Path, default=None)
    ap.add_argument("--knn-texts", type=int, default=200)
    ap.add_argument("--weight-types", type=int, default=1000)
    ap.add_argument("--stage2-types", type=int, default=4000)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--sda-config", type=Path, default=None,
                    help="explicit C11 probe configuration (YAML or JSON)")
    ap.add_argument("--keys", type=int, default=16)
    ap.add_argument("--rotate-every", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--output-mapping", type=Path, default=None,
                    help="probe-only JSON object mapping recovered deployment IDs to plaintext IDs")
    ap.add_argument("--output-annotations", type=Path, default=None,
                    help="probe-only scorer-side JSON list with one annotation record per prompt")
    ap.add_argument("--isa-prompts", type=int, default=1)
    ap.add_argument("--isa-keys", type=int, default=1,
                    help="number of independent key observations per ISA prompt")
    ap.add_argument("--vma-prompts", type=int, default=4)
    ap.add_argument("--vma-propose", type=int, default=400)
    ap.add_argument("--vma-positions", type=int, default=6)
    ap.add_argument("--gen-prompts", type=int, default=3)
    ap.add_argument("--isa-steps", default="20,100")
    ap.add_argument("--isa-lr", type=float, default=0.1)
    ap.add_argument(
        "--isa-lambda-dm", type=float, default=None,
        help=("distribution-matching weight for ISA; omitted -> taken from the "
              "validation study when supplied, else 0.0"),
    )
    ap.add_argument("--isa-layers", default="1")
    ap.add_argument("--isa-method", choices=("er", "tbs", "aloepri_state"), default="er",
                    help="ISA optimizer family executed by C05")
    ap.add_argument("--isa-arm", choices=("ab", "aloepri", "osnip"), default="ab",
                    help="C05 probe arm; aloepri is the A-only engineering artifact")
    ap.add_argument("--isa-restarts", type=int, default=1,
                    help="ISA restarts per (layer, checkpoint, wire); protocol floor 3")
    ap.add_argument(
        "--isa-validation", type=Path, default=None,
        help=("JSON validation study for ISA hyper-parameter selection "
              "(n_prompts>=40, strata>=10 per band, lr, lambda_dm); "
              "without it the ISA payload records no selection and the audit "
              "reports it missing"),
    )
    ap.add_argument(
        "--saturation-study", type=Path, default=None,
        help=("JSON measured saturation study for C10: "
              '{"saturation": {...}, "saturation_curve": {...}, '
              '"saturation_variant": "SH-P"}; never fabricated when absent'),
    )
    ap.add_argument("--weights-only", action="store_true",
                    help="skip loading both 1B models (weight-only classes C06W/C08); saves ~8 GiB")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--summary", type=Path, default=OUT_ROOT / "summary.json")
    ap.add_argument("--profile", default=None, help="required: paper-v1 or probe")
    ap.add_argument("--plan-only", action="store_true", help="write the paper-v1 job plan and exit")
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument("--run-out", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None, help="identity manifest required for a paper-v1 run")

    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()
    classes = [c.strip().upper() for c in args.classes.split(",") if c.strip()]
    isa_layers = ()
    if "C05" in classes and args.profile == "probe":
        if args.isa_arm == "osnip":
            ap.error("C05 OSNIP-only requires an independently trained clean-backbone encoder")
        try:
            isa_layers = tuple(int(value) for value in args.isa_layers.split(","))
        except ValueError:
            ap.error("--isa-layers must be a comma-separated list of integers")
        if min(isa_layers) < 0 or len(set(isa_layers)) != len(isa_layers):
            ap.error("--isa-layers must contain distinct nonnegative layers")
    elif args.isa_arm != "ab":
        ap.error("--isa-arm requires C05 with --profile probe")

    if args.output_mapping is not None or args.output_annotations is not None:
        if args.profile != "probe" or "C13" not in {item.strip().upper() for item in args.classes.split(",")}:
            ap.error("output mapping and annotations are supported only by the probe C13 path")


    manifest = None
    if args.manifest is not None:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read identity manifest {args.manifest}: {exc}", file=sys.stderr)
            raise SystemExit(2)
    if args.profile == "paper-v1":
        from obfuscation_attacks.evaluation.paper_runner import register_implemented_entries

        register_implemented_entries()
    decision = gate_battery_args(
        args.profile,
        plan_only=args.plan_only,
        requested=_requested_from_argv(args, sys.argv),
        manifest=manifest,
        root=ROOT,
    )
    if decision["action"] == "refuse":
        print(decision.get("message", "refused"), file=sys.stderr)
        raise SystemExit(decision["code"])
    if decision["action"] == "plan":
        plan_out = args.plan_out or (
            ROOT / "reports" / "privacy_rebuild_20261107" / "plans" / "paper-v1.plan.json"
        )
        # Key scan only: the corpus must not be loaded before the profile gate.
        serialized = _guard_public_write(decision["plan"], label=plan_out.name)
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        plan_out.write_text(serialized + "\n", encoding="utf-8")
        print(f"[battery] wrote plan {plan_out}", flush=True)
        return
    if decision["action"] == "run_paper":
        from obfuscation_attacks.evaluation.paper_profile import require_profile
        from obfuscation_attacks.evaluation.paper_runner import dispatch_configured

        profile = require_profile("paper-v1")
        executor = None
        if args.execute:
            from obfuscation_attacks.evaluation.paper_executor import PaperExecutor

            executor = PaperExecutor(
                profile,
                manifest,
                root_key=_load_root_key(args.root_key_file),
                result_root=args.result_root,
            )
        report = dispatch_configured(
            profile,
            manifest,
            requested=_requested_from_argv(args, sys.argv),
            executor=executor,
        )
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = args.run_out or (
            ROOT / "reports" / "privacy_rebuild_20261107" / f"paper_run_status_{stamp}.json"
        )
        # The paper-v1 branch writes a result document too: guard it exactly like
        # the battery summary before anything touches disk.
        # Key scan only: no corpus load before/outside the battery loop.
        serialized = _guard_public_write(report, label=out.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized, encoding="utf-8")
        print(
            f"[battery] paper-v1 dispatch status={report['status']} "
            f"configured={report['configured']} -> {out}",
            file=sys.stderr,
        )
        for row in report.get("results", []):
            print(
                f"  {row.get('job_id')}: {row.get('run_status')} "
                f"{row.get('result_path') or row.get('reason') or ''}",
                file=sys.stderr,
            )
        raise SystemExit(0 if report["status"] == "executed" else 2)

    sda_config = None
    if "C11" in classes:
        if args.sda_config is None:
            ap.error("C11 requires an explicit --sda-config probe configuration")
        try:
            sda_config = _load_sda_probe_config(args.sda_config, seed=args.seed, requests=args.requests)
        except (OSError, ValueError) as exc:
            ap.error(f"invalid SDA configuration: {exc}")
    t_start = time.time()
    print(f"[battery] classes={classes} artifact={args.artifact}", flush=True)

    corpus = (load_alpaca(seed=args.seed, n_knn=max(args.knn_texts, 1)) if args.alpaca is None
              else load_alpaca(args.alpaca, seed=args.seed, n_knn=max(args.knn_texts, 1)))
    # Evaluated prompts (the sets the classes actually consume). These strings
    # must never appear verbatim in the public summary.
    evaluated_texts = [
        str(text)
        for name in ("knn", "vma", "private")
        for text in corpus[name].texts()
    ]
    print(f"[battery] alpaca: knn={len(corpus['knn'])} vma={len(corpus['vma'])} "
          f"reference={len(corpus['reference'])} private={len(corpus['private'])}", flush=True)

    weight_only_classes = {"C06W", "C08", "C08A"}
    weights_only = args.weights_only and set(classes).issubset(weight_only_classes)
    observation_layers = tuple(sorted({1, 5, 10, 15, *isa_layers}))
    dep = ABDeployment(artifact=args.artifact, clean_model=args.clean_model, layers=observation_layers,
                       max_len=64, load_models=not weights_only)
    print(f"[battery] deployment loaded: {dep.describe()['vocab_size']} vocab, "
          f"{dep.n_layers} layers, arms={dep.describe()['arms_available']}", flush=True)

    ref_counts, ref_tokens, _ = reference_counts(
        corpus["reference"].texts(), dep.encode, vocab_size=dep.vocab_size
    )
    print(f"[battery] reference corpus: {ref_tokens} tokens, {len(ref_counts)} types, "
          f"overlap with knn set = {overlap_report(corpus['knn'].texts(), corpus['reference'].texts())}",
          flush=True)

    knn_texts = corpus["knn"].texts()[: args.knn_texts]
    summary: dict = {
        "artifact": str(args.artifact),
        "clean_model": str(args.clean_model),
        "artifact_metadata": dep.secret_metadata,
        "corpora": {
            "knn": corpus["knn"].manifest,
            "private_docs": len(corpus["private"]),
            "reference_tokens": int(ref_tokens),
            "reference_types": int(len(ref_counts)),
        },
        "classes": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stream_cache: dict = {}
    if "C01" in classes:
        print("[battery] C01 KNN", flush=True)
        summary["classes"]["C01"] = run_c01(dep, knn_texts, ref_counts)
    if "C06W" in classes:
        print("[battery] C06-W VMA weight", flush=True)
        summary["classes"]["C06W"] = run_c06w(dep, n_types=args.weight_types, seed=args.seed)
    if "C08" in classes:
        print("[battery] C08 Gate-IA", flush=True)
        summary["classes"]["C08"] = run_c08(dep, n_types=args.weight_types, seed=args.seed)
    if "C06H" in classes:
        print("[battery] C06 VMA-H", flush=True)
        summary["classes"]["C06H"] = run_c06h(dep, corpus["vma"].texts()[: args.vma_prompts],
                                              n_propose=args.vma_propose,
                                              max_positions=args.vma_positions)
    if "C08A" in classes:
        print("[battery] C08 Attn-IA", flush=True)
        summary["classes"]["C08A"] = run_c08_attn(dep, n_types=args.weight_types, seed=args.seed)
    if "D3" in classes:
        print("[battery] D3 composed chain", flush=True)
        summary["classes"]["D3"] = run_composed(dep, knn_texts, ref_counts,
                                                n_types=args.stage2_types, seed=args.seed)
    if "C10" in classes:
        print("[battery] C10 TFMA-embed", flush=True)
        c10_texts = corpus["private"].texts()[: args.requests]
        # Protocol-scale runs get 3 disjoint near-equal streams; below the
        # request floor the run stays single-stream (recorded honestly).
        c10_text_sets = _partition_texts(c10_texts)
        saturation_study = None
        if args.saturation_study is not None:
            try:
                saturation_study = json.loads(
                    args.saturation_study.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"cannot read saturation study {args.saturation_study}: {exc}"
                )
            if not isinstance(saturation_study, dict):
                raise SystemExit(
                    "saturation study JSON must be an object with "
                    "'saturation'/'saturation_curve'"
                )
        # Bind the payload the same way the other cells are bound: probe runs
        # keep "probe", a paper-v1 run carries the audited protocol digest.
        c10_result = run_c10(
            dep, c10_texts, ref_counts,
            n_keys=args.keys, rotate_every=args.rotate_every, seed=args.seed,
            cache=stream_cache, profile=args.profile or "probe",
            protocol_sha256=(_paper_protocol_sha256()
                             if args.profile == "paper-v1" else ""),
            text_sets=c10_text_sets,
            saturation=(saturation_study or {}).get("saturation"),
            saturation_curve=(saturation_study or {}).get("saturation_curve"),
            saturation_variant=(saturation_study or {}).get("saturation_variant"),
        )
        c10_result["stream_partition"] = (
            "multi" if c10_text_sets is not None else "single"
        )
        c10_result["stream_sizes"] = (
            [len(row) for row in c10_text_sets] if c10_text_sets is not None
            else [len(c10_texts)]
        )
        c10_result["saturation_study"] = (
            "supplied" if saturation_study is not None else "missing"
        )
        summary["classes"]["C10"] = c10_result
    if "C12" in classes:
        print("[battery] C12 linkage", flush=True)
        mapping = recover_mapping_gate_ia(
            dep, n_types=args.stage2_types, top_k=1, seed=args.seed
        )["mapping"]
        stage2 = {
            int(row): (int(values[0]) if isinstance(values, (list, tuple)) and values else values)
            for row, values in mapping.items()
        }
        summary["classes"]["C12"] = run_c12(
            dep, corpus["private"].texts()[: args.requests],
            n_keys=args.keys, rotate_every=args.rotate_every,
            seed=args.seed, cache=stream_cache, stage2_mapping=stage2,
            profile=args.profile or "probe",
            protocol_sha256=(_paper_protocol_sha256()
                             if args.profile == "paper-v1" else ""),
        )
    if "C02" in classes:
        print("[battery] C02 learned inversion", flush=True)
        summary["classes"]["C02"] = run_c02(dep, knn_texts)
    if "C09" in classes:
        print("[battery] C09 IMA (ridge + MLP)", flush=True)
        summary["classes"]["C09"] = run_c09(dep, knn_texts)
    if "C07" in classes:
        print("[battery] C07 BeamClean", flush=True)
        summary["classes"]["C07"] = run_c07(dep, corpus["vma"].texts()[: args.gen_prompts])
    if "C03" in classes:
        print("[battery] C03 Vec2Text", flush=True)
        summary["classes"]["C03"] = run_c03(dep, corpus["vma"].texts()[: args.gen_prompts])
    if "C11" in classes:
        print("[battery] C11 SDA on the real stream", flush=True)
        summary["classes"]["C11"] = _run_c11_bridge(dep, corpus, ref_counts, sda_config)
    if "C14" in classes:
        print("[battery] C14 LM reconstruction", flush=True)
        texts = corpus["vma"].texts()[: args.gen_prompts]
        wires = [
            dep.observe(text, arm="ab", request_id=i, key_index=i)
            for i, text in enumerate(texts)
        ]
        mapping = recover_mapping_gate_ia(
            dep, n_types=args.stage2_types, top_k=1, seed=args.seed
        )["mapping"]
        stage2 = {
            int(d): (int(values[0]) if isinstance(values, (list, tuple)) and values else values)
            for d, values in mapping.items()
        }
        summary["classes"]["C14"] = run_c14(
            dep.public_view("ab", "S-public"),
            [wire.public for wire in wires],
            stage2_mapping=stage2,
            truth=texts,
        )
    if "C13" in classes:
        print("[battery] C13/C14 output side", flush=True)
        output_mapping = None
        output_annotations = None
        if args.output_mapping is not None:
            output_mapping = json.loads(args.output_mapping.read_text(encoding="utf-8"))
            if not isinstance(output_mapping, dict):
                raise ValueError("output mapping must be a JSON object")
        if args.output_annotations is not None:
            output_annotations = json.loads(args.output_annotations.read_text(encoding="utf-8"))
            if not isinstance(output_annotations, list):
                raise ValueError("output annotations must be a JSON list")
        summary["classes"]["C13/C14"] = run_c13_c14(
            dep,
            prompts=corpus["vma"].texts()[: args.gen_prompts],
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            output_mapping=output_mapping,
            annotations=output_annotations,
        )
    if "C05" in classes:
        print("[battery] C05 ISA budget curve", flush=True)
        isa_validation = None
        if args.isa_validation is not None:
            try:
                isa_validation = json.loads(
                    args.isa_validation.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"cannot read ISA validation study {args.isa_validation}: {exc}"
                )
        c05_result = run_c05(
            dep, corpus["vma"].texts(), ref_counts,
            prompts=args.isa_prompts,
            steps=tuple(int(x) for x in args.isa_steps.split(",")),
            lr=(args.isa_lr if _has_flag(sys.argv, "--isa-lr") else None),
            lambda_dm=args.isa_lambda_dm,
            seed=args.seed,
            isa_layers=isa_layers,
            arm=args.isa_arm,
            method=str(args.isa_method),
            restarts=int(args.isa_restarts),
            keys_per_prompt=int(args.isa_keys),
            selected_on_validation=isa_validation,
            profile=args.profile or "probe",
            protocol_sha256=(_paper_protocol_sha256()
                             if args.profile == "paper-v1" else ""),
        )
        if isa_validation is None:
            # Do not fabricate a selection: record the concrete blocker so the
            # summary states why the audit will report it missing.
            c05_result["selected_on_validation"] = {
                "status": "missing",
                "reason": (
                    "no ISA validation study supplied (--isa-validation); a real "
                    "40-prompt / 10-per-length-band selection is required"
                ),
            }
        summary["classes"]["C05"] = c05_result

    summary["elapsed_seconds"] = round(time.time() - t_start, 1)
    # Fail closed on truth/secret keys or verbatim evaluated corpus text before
    # writing any public summary (never echoes the offending text).
    serialized = _guard_public_write(
        summary, label=args.summary.name, corpus_texts=evaluated_texts
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(serialized, encoding="utf-8")
    print(f"[battery] wrote {args.summary} in {summary['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
