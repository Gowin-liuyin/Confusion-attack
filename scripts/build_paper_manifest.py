#!/usr/bin/env python3
"""Build a verifiable paper-v1 identity manifest from real files.

Every field is either computed from file bytes or explicitly marked unknown; the
script never fills a hash from a directory name. The generated manifest is the
only input the paper executor trusts, and the executor re-hashes the same files
before running, so a stale manifest is refused rather than silently used.

Usage::

    python scripts/build_paper_manifest.py \
        --artifact F:/OSNIP_package_20261101/ab_runnable \
        --model-id llama-3.2-1b-base --model-path F:/Models/Llama-3.2-1B \
        --alpaca F:/OSNIP_package_20261101/references/online/stanford_alpaca-alpaca_data.json \
        --out configs/paper/manifest.paper-v1.json
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

import numpy as np  # noqa: E402

from obfuscation_attacks.data.corpora import CLUSTERED_SPLIT, audit_corpus_isolation, load_alpaca, overlap_report  # noqa: E402
from obfuscation_attacks.evaluation.paper_executor import (  # noqa: E402
    _combined_weight_sha,
    sha256_file,
    texts_sha256,
)
from obfuscation_attacks.evaluation.paper_profile import (  # noqa: E402
    PAPER_ARMS,
    SOURCE_ANCHORS,
    require_profile,
)
from obfuscation_attacks.schemes.ab_deployment import local_model_revision, verify_model_identity  # noqa: E402


def _load_table(root: Path, name: str = "model.embed_tokens.weight") -> np.ndarray:
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.is_file():
        shard = root / json.loads(index.read_text())["weight_map"][name]
    else:
        shard = root / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).float().numpy()


def column_alignment(artifact: Path, model_path: Path, *, n_rows: int = 8192) -> dict:
    """Monomial-colour test: undo the vocab permutation, then match columns.

    The AloePri ``square_monomial`` transform is a coordinate permutation times a
    per-coordinate sign and a scale (``square_transform.py``). If the artifact's
    backbone is this model, every deployment column has exactly one model column
    with |corr| near 1 (noise aside); a different backbone yields no bijection.
    """
    import torch

    servers = [path for path in (artifact / "model_server", artifact / "server") if path.is_dir()]
    if len(servers) != 1:
        raise ValueError("artifact must contain exactly one server model directory")
    server = servers[0]
    E_dep = _load_table(server)
    secret = torch.load(artifact / "client" / "client_secret.pt", map_location="cpu",
                        weights_only=True)
    perm = secret["perm_vocab"].long().numpy()
    E_clean = _load_table(model_path)
    if E_dep.shape != E_clean.shape:
        return {"verdict": "shape_mismatch", "deployment": list(E_dep.shape),
                "model": list(E_clean.shape)}
    A = E_dep[perm]
    rng = np.random.default_rng(20261107)
    rows = rng.choice(len(A), min(n_rows, len(A)), replace=False)
    As, Cs = A[rows], E_clean[rows]

    def zscore(x):
        return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-9)

    corr = (zscore(As).T @ zscore(Cs)) / As.shape[0]
    best = np.abs(corr).max(axis=1)
    arg = np.abs(corr).argmax(axis=1)
    bijection = len(set(arg.tolist())) / len(arg)
    row_ratio = np.linalg.norm(A, axis=1) / np.maximum(
        np.linalg.norm(E_clean, axis=1), 1e-9
    )
    median_best = float(np.median(best))
    # Decisive test: the square_monomial transform adds alpha_e * sigma Gaussian
    # noise and nothing else. Subtract the best-fit monomial image of the candidate
    # backbone; a true backbone leaves exactly alpha_e*sigma, a fine-tuned relative
    # (same-family) leaves more.
    signs = np.sign(corr[np.arange(len(arg)), arg])
    Ez = Cs - Cs.mean(0, keepdims=True)
    Az = As - As.mean(0, keepdims=True)
    denom = float((Ez[:, arg] ** 2).sum())
    scale = float((Az * (Ez[:, arg] * signs)).sum() / denom) if denom else 1.0
    residual = As - scale * signs * Cs[:, arg]
    sigma = float(E_clean[: min(len(E_clean), 128000)].std())
    residual_ratio = float(residual.std() / sigma) if sigma else None
    alpha_e = float(secret.get("metadata", {}).get("alpha_e", 0.0) or 0.0)
    if bijection >= 0.98 and residual_ratio is not None and alpha_e and residual_ratio <= alpha_e + 0.02:
        verdict = "matches"
    elif bijection >= 0.98 and residual_ratio is not None and alpha_e:
        verdict = "same_family_excess_residual"
    else:
        verdict = (
            "matches" if bijection >= 0.98 and median_best >= 0.90
            else "mismatch" if bijection < 0.50 or median_best < 0.50
            else "unverified"
        )
    return {
        "method": "undo vocab permutation, match embedding columns (monomial invariant), "
                  "then compare the residual with the documented alpha_e noise",
        "n_rows_sampled": int(len(rows)),
        "bijection_fraction": float(bijection),
        "median_best_abs_corr": median_best,
        "p10_best_abs_corr": float(np.percentile(best, 10)),
        "row_norm_ratio_median": float(np.median(row_ratio)),
        "alpha_e": alpha_e,
        "residual_over_sigma": residual_ratio,
        "expected_noise_ratio": alpha_e,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path,
                        default=ROOT.parent / "OSNIP_package_20261101" / "ab_runnable")
    parser.add_argument("--model-id", default="llama-3.2-1b-base")
    parser.add_argument("--model-path", type=Path, default=ROOT.parent / "Models" / "Llama-3.2-1B")
    parser.add_argument("--model-source", default="unsloth/Llama-3.2-1B")
    parser.add_argument("--variant", default="base", choices=["base", "instruct"])
    parser.add_argument("--revision", default=None,
                        help="unverified claimed remote revision; local bytes remain authoritative")
    parser.add_argument("--license", default=None)
    parser.add_argument("--alpaca", type=Path,
                        default=ROOT.parent / "OSNIP_package_20261101" / "references"
                        / "online" / "stanford_alpaca-alpaca_data.json")
    parser.add_argument("--split-seed", type=int, default=20261107)
    parser.add_argument("--knn-texts", type=int, default=1000)
    parser.add_argument("--robust-texts", type=int, default=200)
    parser.add_argument("--robust-keys", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lineage-model", type=Path, action="append", default=None,
                        help="extra model directory to fingerprint against the artifact")
    args = parser.parse_args()

    profile = require_profile("paper-v1")
    manifest: dict = {
        "format": "paper_v1_identity_manifest",
        "profile": "paper-v1",
        "protocol_sha256": profile.protocol_sha256,
        "source_anchors": SOURCE_ANCHORS,
        "created_by": "scripts/build_paper_manifest.py",
        "models": {},
        "data": {},
        "arms": {},
        "artifacts": {},
    }

    # -- model identity ------------------------------------------------------
    if not args.model_path.is_dir():
        raise SystemExit(f"model path missing: {args.model_path}")
    model = verify_model_identity(args.model_path, expected_variant=args.variant)
    revision = local_model_revision(model)
    manifest["models"][args.model_id] = {
        "name": model["name"],
        "path": str(args.model_path),
        "source": args.model_source,
        "revision": revision,
        "revision_evidence": "local content fingerprint; remote revision not verified",
        "claimed_remote_revision": args.revision or model.get("revision"),
        "variant": args.variant,
        "license": args.license,
        "config_sha256": model["config_sha256"],
        "tokenizer_sha256": model["tokenizer_sha256"],
        "weights_sha256": model["weights_sha256"],
        "files_sha256": model["files"],
        "variant_evidence": model["variant_evidence"],
    }

    # -- artifact identity ---------------------------------------------------
    artifact_manifest_path = args.artifact / "manifest.json"
    artifact_doc = (
        json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        if artifact_manifest_path.is_file() else {}
    )
    files: dict[str, str] = {}
    for path in sorted(args.artifact.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(args.artifact).as_posix()
        files[relative] = sha256_file(path)
    lineage = {}
    for candidate in [args.model_path] + list(args.lineage_model or []):
        try:
            lineage[str(candidate)] = column_alignment(args.artifact, candidate)
        except Exception as exc:
            lineage[str(candidate)] = {"verdict": "error", "error": str(exc)}
    manifest["artifacts"]["ab"] = {
        "path": str(args.artifact),
        "format": artifact_doc.get("format"),
        "declared_backbone": artifact_doc.get("components", {}).get(
            "A_server_model", {}
        ).get("backbone"),
        "declared_source": artifact_doc.get("components", {}).get(
            "A_server_model", {}
        ).get("source"),
        "declared_export": artifact_doc.get("components", {}).get(
            "A_server_model", {}
        ).get("export"),
        "files_sha256": files,
        "artifact_manifest_sha256": files.get("manifest.json"),
        "weights_sha256": _combined_weight_sha(files),
        "lineage": lineage,
    }

    # -- data identity -------------------------------------------------------
    if not args.alpaca.is_file():
        raise SystemExit(f"alpaca snapshot missing: {args.alpaca}")
    corpus = load_alpaca(args.alpaca, seed=args.split_seed, n_knn=args.knn_texts,
                        n_robust=args.robust_texts, split_strategy=CLUSTERED_SPLIT)
    knn_texts = corpus["knn"].texts()
    robust_texts = corpus["robust"].texts()
    others = (corpus["vma"].texts() + corpus["robust"].texts()
              + corpus["reference"].texts() + corpus["private"].texts())
    overlap = overlap_report(knn_texts, others)
    isolation = audit_corpus_isolation(corpus)
    if not isolation["passed"] or isolation["split_sha256"] != corpus["knn"].manifest["split_sha256"]:
        raise SystemExit("refusing to publish a corpus manifest with near-duplicate documents")
    lengths = [len(text) for text in knn_texts]
    robust_lengths = [len(text) for text in robust_texts]
    manifest["data"] = {
        "path": str(args.alpaca),
        "source": "tatsu-lab/stanford_alpaca alpaca_data.json",
        "source_sha256": sha256_file(args.alpaca),
        "split_seed": args.split_seed,
        "knn_texts": len(knn_texts),
        "knn_sha256": texts_sha256(knn_texts),
        "char_min": min(lengths),
        "char_max": max(lengths),
        "robust_texts": len(robust_texts),
        "robust_keys": args.robust_keys,
        "robust_sha256": texts_sha256(robust_texts),
        "robust_char_min": min(robust_lengths) if robust_lengths else None,
        "robust_char_max": max(robust_lengths) if robust_lengths else None,
        "text_rule": "instruction + '\n' + input",
        "hashes_disjoint": overlap["intersection"] == 0,
        "split_strategy": CLUSTERED_SPLIT,
        "split_sha256": isolation["split_sha256"],
        "isolation": isolation,
        "reference_source_scope": "single Alpaca source; three independent TFMA prior sources are not established",
        "overlap_with_other_roles": overlap,
        "roles": {
            "knn": corpus["knn"].manifest,
            "vma": corpus["vma"].manifest,
            "robust": corpus["robust"].manifest,
            "reference": corpus["reference"].manifest,
            "private": corpus["private"].manifest,
        },
    }

    # -- arm declarations ----------------------------------------------------
    implemented = {
        "plaintext": {
            "adapter": "src/obfuscation_attacks/schemes/ab_deployment.py",
            "status": "implemented",
            "source_verified": True,
        },
        "aloepri": {
            "adapter": "src/obfuscation_attacks/schemes/ab_deployment.py",
            "status": "implemented_engineering_variant",
            "source_verified": True,
        },
        "ab": {
            "adapter": "src/obfuscation_attacks/schemes/ab_deployment.py",
            "status": "implemented",
            "source_verified": True,
        },
    }
    external_gaps = {
        "osnip": "needs an encoder trained on the clean backbone; the packaged encoder is A-conditioned",
        "sentinellms": "no public checkpoint (HF link 401); classification-only; decoder adaptation unvalidated",
        "obfuslm_plus": "no official repository/license; must be reimplemented from the paper",
        "inferdpt": "no license; two-module client/server path and extraction module not wired",
        "dp_forward": "no license; classification-only BERT; noisy fine-tuning required",
    }
    for arm in PAPER_ARMS:
        if arm in implemented:
            manifest["arms"][arm] = implemented[arm]
        else:
            manifest["arms"][arm] = {
                "adapter": None,
                "status": "unverified",
                "source_verified": False,
                "reason": external_gaps.get(arm, "not integrated"),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"model {args.model_id}: variant={args.variant} revision={revision}")
    print(f"data knn={len(knn_texts)} chars=[{min(lengths)},{max(lengths)}] "
          f"robust={len(robust_texts)} overlap={overlap['intersection']}")
    for path, evidence in lineage.items():
        print(f"lineage vs {path}: {evidence.get('verdict')} "
              f"bijection={evidence.get('bijection_fraction')} "
              f"median|corr|={evidence.get('median_best_abs_corr')}")


if __name__ == "__main__":
    main()
