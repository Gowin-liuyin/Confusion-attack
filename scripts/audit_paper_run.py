#!/usr/bin/env python3
"""Independent audit of paper-v1 C01 runs (read-only, re-runs the evidence).

This script does not import the executor. It loads a finished ``result.json``,
recomputes every Top-k numerator from the stored per-position ranks, re-hashes
the model/artifact/corpus content, and re-runs the exact KNN from the stored
query matrix against the public candidate table. A tampered prediction, budget
count, positive control or hash is refused; the executor's own flags are not
trusted.

Usage::

    python scripts/audit_paper_run.py --result reports/privacy_rebuild_20261107/runs/C01/.../result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from obfuscation_attacks.evaluation.attack_model import validate_payload  # noqa: E402
from obfuscation_attacks.evaluation.export_randomness import verify_export_randomness  # noqa: E402
from obfuscation_attacks.evaluation.square_lineage import verify_square_lineage  # noqa: E402
from obfuscation_attacks.evaluation.square_endpoints import verify_square_endpoints  # noqa: E402
from obfuscation_attacks.evaluation.encoder_lineage import verify_encoder_binding  # noqa: E402
from obfuscation_attacks.evaluation.paper_profile import (  # noqa: E402
    protocol_sha256,
    require_profile,
)
from obfuscation_attacks.evaluation.protocol_audit import audit_payload  # noqa: E402
from obfuscation_attacks.evaluation.paper_executor import (  # noqa: E402
    sha256_file,
    texts_sha256,
)
from obfuscation_attacks.evaluation.run_store import content_sha256  # noqa: E402
from obfuscation_attacks.schemes.ab_deployment import knn_topk  # noqa: E402

AUDITOR = "independent_paper_run_auditor"

def _cached_sha(path: Path) -> str:
    return sha256_file(path)


def _cached_model_identity(path: Path, variant: str | None) -> dict:
    from obfuscation_attacks.schemes.ab_deployment import verify_model_identity

    return verify_model_identity(path, expected_variant=variant)


@lru_cache(maxsize=2)
def _corpus_by_content(path: str, data_sha: str, implementation_sha: str, *,
                       seed: int, n_knn: int, n_robust: int, split_strategy: str):
    from obfuscation_attacks.data.corpora import load_alpaca

    return load_alpaca(Path(path), seed=seed, n_knn=n_knn, n_robust=n_robust,
                       split_strategy=split_strategy)


def _rebuild_corpus(path: Path, *, seed: int, n_knn: int, n_robust: int, split_strategy: str):
    # Rehash on every access; timestamps alone cannot detect in-place changes.
    implementation = ROOT / "src" / "obfuscation_attacks" / "data" / "corpora.py"
    return _corpus_by_content(str(path.resolve()), sha256_file(path), sha256_file(implementation),
                              seed=seed, n_knn=n_knn, n_robust=n_robust,
                              split_strategy=split_strategy)


def _load_table(root: Path, name: str = "model.embed_tokens.weight") -> np.ndarray:
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.is_file():
        shard = root / json.loads(index.read_text())["weight_map"][name]
    else:
        shard = root / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).float().numpy()


def _resolve(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _source_paths(artifact: Path) -> list[Path]:
    candidates = [
        artifact / "manifest.json",
        artifact / "client" / "client_secret.pt",
        artifact / "osnip" / "encryptor.pt",
        artifact / "osnip" / "config.json",
        artifact / "model_server" / "config.json",
    ]
    candidates.extend(sorted((ROOT / "src" / "obfuscation_attacks").rglob("*.py")))
    candidates.extend([ROOT / "scripts" / "run_full_battery.py", ROOT / "scripts" / "audit_paper_run.py"])
    return [path for path in candidates if path.is_file()]


def _gaps_from_ranks(payload: dict) -> tuple[list[str], dict]:
    gaps: list[str] = []
    diag = payload.get("diagnostics") or {}
    rows = diag.get("per_unit_predictions") or []
    metrics = (payload.get("metrics") or {}).get("stage1_permuted_id") or {}
    hits_total = 0
    counts_total = 0
    keys = int(payload.get("n_keys") or 1)
    n_prompts = int(payload.get("n_prompts") or 0)
    top = {k: 0 for k in (1, 5, 10, 100)}
    for index, row in enumerate(rows):
        ranks = row.get("rank")
        count = row.get("count")
        if not isinstance(ranks, list) or len(ranks) != count:
            gaps.append(f"unit {index}: rank list missing or length != count")
            continue
        text_index = int(row.get("text", index // max(keys, 1)))
        key_index = int(row.get("key", index % max(keys, 1)))
        if text_index != index // max(keys, 1) or key_index != index % max(keys, 1):
            gaps.append(
                f"unit {index}: text/key grouping {text_index}/{key_index} != "
                f"{index // max(keys, 1)}/{index % max(keys, 1)}"
            )
        if not 0 <= text_index < max(n_prompts, 1):
            gaps.append(f"unit {index}: text index {text_index} outside n_prompts")
        hits_total += sum(1 for value in ranks if value <= 1)
        counts_total += int(count)
        for k in top:
            top[k] += sum(1 for value in ranks if value <= k)
        if int(row.get("hits", -1)) != sum(1 for value in ranks if value <= 1):
            gaps.append(f"unit {index}: stored hits != ranks<=1")
        predicted, truth = row.get("predicted"), row.get("truth")
        if isinstance(predicted, list) and isinstance(truth, list):
            if len(predicted) != count or len(truth) != count:
                gaps.append(f"unit {index}: predicted/truth length != count")
            else:
                for position, (pred, true, rank) in enumerate(zip(predicted, truth, ranks)):
                    if (rank == 1) != (pred == true):
                        gaps.append(f"unit {index}: position {position} rank/prediction mismatch")
                        break
    declared = metrics.get("top1") or {}
    if declared.get("numerator") != hits_total or declared.get("denominator") != counts_total:
        gaps.append(
            f"metrics top1 {declared.get('numerator')}/{declared.get('denominator')} != "
            f"recomputed {hits_total}/{counts_total}"
        )
    for key in (5, 10, 100):
        block = metrics.get(f"top{key}") or {}
        if block.get("numerator") != top[key] or block.get("denominator") != counts_total:
            gaps.append(f"metrics top{key} does not match recomputed ranks")
    composed = (payload.get("metrics") or {}).get("composed_with_oracle_permutation")
    if isinstance(composed, dict):
        for k in (1, 5, 10, 100):
            oracle = composed.get(f"top{k}") or {}
            stage = metrics.get(f"top{k}") or {}
            if (
                oracle.get("numerator") != stage.get("numerator")
                or oracle.get("denominator") != stage.get("denominator")
            ):
                gaps.append(
                    f"metrics composed top{k} does not match stage1 "
                    f"(a secret permutation preserves the hit count)"
                )
    if payload.get("n_tokens") != counts_total:
        gaps.append(f"n_tokens {payload.get('n_tokens')} != per-unit counts {counts_total}")
    if len(rows) != n_prompts * keys:
        gaps.append(
            f"n_prompts*n_keys {n_prompts}*{keys} != per-unit rows {len(rows)}"
        )
    return gaps, {"hits": hits_total, "counts": counts_total, "top": top}


def _gaps_from_ci(payload: dict) -> tuple[list[str], dict]:
    """Recompute micro and macro intervals from the per-unit rows."""
    from obfuscation_attacks.evaluation.real_battery import bootstrap_ci, micro_bootstrap

    gaps: list[str] = []
    rows = (payload.get("diagnostics") or {}).get("per_unit_predictions") or []
    ci = payload.get("ci") or {}
    micro = ci.get("top1_micro") if isinstance(ci.get("top1_micro"), dict) else {}
    macro = ci.get("top1_per_text_macro") if isinstance(ci.get("top1_per_text_macro"), dict) else {}
    if not micro or not macro:
        return ["ci: micro/macro interval missing"], {}
    keys = int(payload.get("n_keys") or 1)
    n_prompts = int(payload.get("n_prompts") or 0)
    hits = [int(row.get("hits", 0)) for row in rows]
    counts = [int(row.get("count", 0)) for row in rows]
    clusters = [int(row.get("text", index // max(keys, 1)))
                for index, row in enumerate(rows)]
    raw_n_boot = micro.get("n_boot")
    if raw_n_boot is None:
        gaps.append("ci: micro.n_boot missing")
        n_boot = 2000
    else:
        n_boot = int(raw_n_boot)
        if n_boot < 2000:
            gaps.append(f"ci: n_boot {n_boot} < 2000 (protocol floor)")
    raw_macro_boot = macro.get("n_boot")
    if raw_macro_boot is None:
        gaps.append("ci: macro.n_boot missing")
    elif int(raw_macro_boot) < 2000:
        gaps.append(f"ci: macro n_boot {raw_macro_boot} < 2000 (protocol floor)")
    seed = int(ci.get("bootstrap_seed") or 0)
    recomputed_micro = micro_bootstrap(hits, counts, clusters=clusters,
                                       n_boot=n_boot, seed=seed)
    text_hits = [0] * n_prompts
    text_counts = [0] * n_prompts
    for hit, count, cluster in zip(hits, counts, clusters):
        if 0 <= cluster < n_prompts:
            text_hits[cluster] += hit
            text_counts[cluster] += count
    per_text = [
        float(hit / count) if count else 0.0
        for hit, count in zip(text_hits, text_counts)
    ]
    recomputed_macro = bootstrap_ci(per_text, n_boot=n_boot, seed=seed)

    def close(stored, recomputed) -> bool:
        if stored is None or recomputed is None:
            return stored is None and recomputed is None
        try:
            return abs(float(stored) - float(recomputed)) <= 1e-9
        except (TypeError, ValueError):
            return False

    for field in ("micro", "lo", "hi", "denominator"):
        if not close(micro.get(field), recomputed_micro.get(field)):
            gaps.append(
                f"ci: micro.{field} {micro.get(field)!r} != recomputed "
                f"{recomputed_micro.get(field)!r}"
            )
    for field in ("mean", "lo", "hi"):
        if not close(macro.get(field), recomputed_macro.get(field)):
            gaps.append(
                f"ci: macro.{field} {macro.get(field)!r} != recomputed "
                f"{recomputed_macro.get(field)!r}"
            )
    for field in ("n_clusters", "n_units", "n_boot"):
        if not close(micro.get(field), recomputed_micro.get(field)):
            gaps.append(
                f"ci: micro.{field} {micro.get(field)!r} != recomputed "
                f"{recomputed_micro.get(field)!r}"
            )
        if field == "n_clusters" and macro.get("n_clusters") is None:
            # Legacy payloads recorded only n_units for the macro block; the
            # resampled unit was the cluster in that bootstrap.
            if not close(macro.get("n_units"), recomputed_macro.get("n_clusters")):
                gaps.append(
                    f"ci: macro.n_clusters missing and n_units "
                    f"{macro.get('n_units')!r} != recomputed clusters "
                    f"{recomputed_macro.get('n_clusters')!r}"
                )
            continue
        if not close(macro.get(field), recomputed_macro.get(field)):
            gaps.append(
                f"ci: macro.{field} {macro.get(field)!r} != recomputed "
                f"{recomputed_macro.get(field)!r}"
            )
    if micro.get("estimand") != "micro":
        gaps.append(f"ci: micro.estimand {micro.get('estimand')!r} != 'micro'")
    if macro.get("estimand") != "macro":
        gaps.append(f"ci: macro.estimand {macro.get('estimand')!r} != 'macro'")
    return gaps, {"micro": recomputed_micro, "macro": recomputed_macro}


def _gaps_from_evidence(payload: dict, identity: dict) -> tuple[list[str], dict]:
    gaps: list[str] = []
    diag = payload.get("diagnostics") or {}
    evidence = diag.get("evidence")
    if not isinstance(evidence, dict) or not evidence.get("path"):
        return ["evidence: query matrix reference missing"], {}
    path = _resolve(evidence["path"])
    if path is None or not path.is_file():
        return [f"evidence: query matrix missing on disk: {evidence.get('path')}"], {}
    try:
        data = np.load(path, allow_pickle=False)
        queries = np.asarray(data["queries"], dtype=np.float32)
        lengths = [int(x) for x in data["lengths"]]
        evidence_arm = str(data["arm"])
        metric = str(data["metric"])
    except Exception as exc:
        return gaps + [f"evidence: query matrix unreadable: {exc}"], {}
    if evidence.get("arrays_sha256"):
        from obfuscation_attacks.evaluation.real_battery import _arrays_digest

        recomputed = _arrays_digest(
            queries, np.asarray(lengths, dtype=np.int64), evidence_arm, metric
        )
        if recomputed != evidence["arrays_sha256"]:
            gaps.append("evidence: query array digest != declared arrays_sha256")
        if evidence.get("file_sha256") and _cached_sha(path) != evidence["file_sha256"]:
            gaps.append("evidence: query matrix file sha256 drift")
    else:
        digest = _cached_sha(path)
        if digest != evidence.get("sha256"):
            gaps.append(
                f"evidence: query matrix sha256 drift: stored {evidence.get('sha256')} != file {digest}"
            )
    arm = str(
        payload.get("arm")
        or (payload.get("scheme_name") or "").split(":")[-1]
        or (payload.get("target") or "").split("/")[-1]
    )
    if evidence_arm != arm or arm != ((payload.get("target") or "").split("/")[-1] or arm):
        gaps.append("evidence: arm in query matrix != payload arm")
    if metric != (payload.get("diagnostics") or {}).get("metric"):
        gaps.append("evidence: metric in query matrix != payload metric")
    if list(queries.shape) != list(evidence.get("shape") or []):
        gaps.append("evidence: query matrix shape != declared shape")
    rows = diag.get("per_unit_predictions") or []
    positions = sum(int(row.get("count", 0)) for row in rows)
    if queries.shape[0] != positions:
        gaps.append(f"evidence: query rows {queries.shape[0]} != per-unit positions {positions}")
        return gaps, {}
    if lengths != [int(row.get("count", 0)) for row in rows]:
        gaps.append("evidence: per-text lengths != per-unit counts")
        return gaps, {}

    model_path = _resolve(identity.get("model", {}).get("path"))
    artifact_path = _resolve(identity.get("artifact", {}).get("path"))
    if arm == "plaintext":
        if model_path is None or not model_path.is_dir():
            return gaps + ["evidence: clean model path missing"], {}
        table = _load_table(model_path)
        truth = np.concatenate(
            [np.asarray(row.get("truth"), dtype=np.int64) for row in rows]
        )
        recomputed_q = table[truth]
        if not np.array_equal(recomputed_q, queries):
            gaps.append("evidence: plaintext queries != clean_table[truth ids]")
        candidates = knn_topk(queries, table, k=100, metric=metric)
        predicted = np.concatenate(
            [np.asarray(row.get("predicted"), dtype=np.int64) for row in rows]
        )
        if not np.array_equal(candidates[:, 0], predicted):
            gaps.append("evidence: re-run KNN top-1 != stored predictions")
        ranks = np.concatenate([np.asarray(row.get("rank"), dtype=np.int64) for row in rows])
        from obfuscation_attacks.evaluation.real_battery import truth_rank

        if not np.array_equal(truth_rank(candidates, truth), ranks):
            gaps.append("evidence: re-run KNN ranks != stored ranks")
        details = {"arm": arm, "recomputed_top1": float(np.mean(ranks <= 1))}
        return gaps, details
    if arm in {"ab", "aloepri"}:
        if artifact_path is None:
            return gaps + ["evidence: artifact path missing"], {}
        table = _load_table(artifact_path / "model_server")
        candidates = knn_topk(queries, table, k=100, metric=metric)
        predicted = np.concatenate(
            [np.asarray(row.get("predicted"), dtype=np.int64) for row in rows]
        )
        truth = np.concatenate([np.asarray(row.get("truth"), dtype=np.int64) for row in rows])
        ranks = np.concatenate([np.asarray(row.get("rank"), dtype=np.int64) for row in rows])
        from obfuscation_attacks.evaluation.real_battery import truth_rank

        if not np.array_equal(candidates[:, 0], predicted):
            gaps.append("evidence: re-run KNN top-1 != stored predictions")
        if not np.array_equal(truth_rank(candidates, truth), ranks):
            gaps.append("evidence: re-run KNN ranks != stored ranks")
        if arm == "aloepri":
            identity_value = (payload.get("diagnostics") or {}).get("wire_id_identity")
            if identity_value is None:
                gaps.append("evidence: AloePri-only row must declare wire_id_identity")
            elif abs(float(identity_value) - 1.0) > 1e-12:
                gaps.append("evidence: wire_id_identity must be 1.0")
        return gaps, {"arm": arm, "recomputed_top1": float(np.mean(ranks <= 1))}
    gaps.append(f"evidence: unknown arm {arm!r}")
    return gaps, {}


def _gaps_from_hashes(payload: dict, identity: dict) -> tuple[list[str], dict]:
    gaps: list[str] = []
    data_info = identity.get("data") or {}
    data_path = _resolve(data_info.get("path"))
    if data_path is None or not data_path.is_file():
        gaps.append("hash: data file missing")
    else:
        digest = _cached_sha(data_path)
        if digest != data_info.get("source_sha256"):
            gaps.append("hash: data file sha256 != identity")
        if digest != payload.get("data_sha256"):
            gaps.append("hash: data file sha256 != payload.data_sha256")
        from obfuscation_attacks.data.corpora import CLUSTERED_SPLIT, audit_corpus_isolation

        try:
            role = str(data_info.get("role") or "knn")
            main_n = int(data_info.get("main_knn_texts") or (
                data_info.get("n_texts") if role == "knn" else 0
            ) or 0)
            robust_n = int(data_info.get("robust_texts") or (
                data_info.get("n_texts") if role == "robust" else 0
            ) or 0)
            corpus = _rebuild_corpus(
                data_path, seed=int(data_info.get("split_seed")),
                n_knn=main_n or 1000, n_robust=robust_n,
                split_strategy=str(data_info.get("split_strategy") or "exact_v1"),
            )
            if data_info.get("split_strategy") != CLUSTERED_SPLIT:
                gaps.append("corpus: near-duplicate clustered split not established")
            else:
                isolation = audit_corpus_isolation(corpus)
                if not isolation["passed"] or isolation["split_sha256"] != data_info.get("split_sha256"):
                    gaps.append("corpus: clustered split or role content hash mismatch")
            role_texts = corpus["robust"].texts() if role == "robust" else corpus["knn"].texts()
            if texts_sha256(role_texts) != data_info.get("knn_sha256"):
                gaps.append("hash: frozen KNN text digest != identity")
        except Exception as exc:
            gaps.append(f"corpus: cannot rebuild or verify frozen split: {exc}")
    artifact_path = _resolve((identity.get("artifact") or {}).get("path"))
    if artifact_path is None or not artifact_path.is_dir():
        gaps.append("hash: artifact path missing")
    else:
        if (payload.get("arm") or (payload.get("target") or "").split("/")[-1]) == "ab":
            try:
                verify_encoder_binding(artifact_path)
            except Exception as exc:
                gaps.append(f"artifact: encoder binding verification failed: {type(exc).__name__}: {exc}")
        try:
            verify_export_randomness(artifact_path)
        except Exception as exc:
            gaps.append(f"artifact: export randomness verification failed: {type(exc).__name__}: {exc}")
        else:
            try:
                base = _resolve((identity.get("model") or {}).get("path"))
                if base is None:
                    raise ValueError("base model path missing")
                verify_square_lineage(artifact_path, base)
            except Exception as exc:
                gaps.append(f"artifact: decoder lineage verification failed: {type(exc).__name__}: {exc}")
            else:
                try:
                    verify_square_endpoints(artifact_path, base)
                except Exception as exc:
                    gaps.append(f"artifact: endpoint lineage verification failed: {type(exc).__name__}: {exc}")
        paths = _source_paths(artifact_path)
        if content_sha256(paths) != payload.get("source_sha256"):
            gaps.append("hash: source file set sha256 != payload.source_sha256")
        files = (identity.get("artifact") or {}).get("files_sha256") or {}
        for relative, expected in files.items():
            path = artifact_path / relative
            if not path.is_file():
                gaps.append(f"hash: artifact file missing: {relative}")
                break
            if _cached_sha(path) != expected:
                gaps.append(f"hash: artifact file drift: {relative}")
                break
    model_info = identity.get("model") or {}
    model_path = _resolve(model_info.get("path"))
    if model_path is None or not model_path.is_dir():
        gaps.append("hash: model path missing")
    else:
        actual = _cached_model_identity(model_path, model_info.get("variant"))
        from obfuscation_attacks.schemes.ab_deployment import local_model_revision
        if model_info.get("revision") != local_model_revision(actual):
            gaps.append("hash: model revision is not the verified local content identity")
        for field in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
            if actual.get(field) != model_info.get(field):
                gaps.append(f"hash: model {field} != identity")
    return gaps, {"data_sha256": payload.get("data_sha256"),
                  "source_sha256": payload.get("source_sha256")}


def _gaps_from_corpus(payload: dict, identity: dict) -> tuple[list[str], dict, list]:
    """Tie every per-unit denominator to the frozen corpus re-tokenised here.

    Without this check a payload can keep 1000 rows but move most of the mass into
    zero-count placeholders and reset ``n_tokens``; the denominator would then be
    the attacker's claim rather than the frozen 1000-document split.
    """
    gaps: list[str] = []
    data = identity.get("data") or {}
    path = _resolve(data.get("path"))
    if path is None or not path.is_file():
        return ["corpus: frozen data file missing"], {}, []
    artifact = _resolve((identity.get("artifact") or {}).get("path"))
    if artifact is None or not (artifact / "model_server").is_dir():
        return ["corpus: artifact tokenizer missing"], {}, []
    try:
        from transformers import AutoTokenizer

        role = str(data.get("role") or "knn")
        main_n = int(data.get("main_knn_texts") or (
            data.get("n_texts") if role == "knn" else 0
        ) or 0)
        robust_n = int(data.get("robust_texts") or (
            data.get("n_texts") if role == "robust" else 0
        ) or 0)
        corpus = _rebuild_corpus(
            path, seed=int(data.get("split_seed")), n_knn=main_n or 1000,
            n_robust=robust_n,
            split_strategy=str(data.get("split_strategy") or "exact_v1"),
        )
        role_texts = corpus["robust"].texts() if role == "robust" else corpus["knn"].texts()
        tokenizer = AutoTokenizer.from_pretrained(
            str(artifact / "model_server"), local_files_only=True
        )
        encoded = [
            tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in role_texts
        ]
    except Exception as exc:
        return [f"corpus: cannot rebuild the frozen split: {exc}"], {}, []
    counts = [len(ids) for ids in encoded]
    keys = int(payload.get("n_keys") or 1)
    rows = (payload.get("diagnostics") or {}).get("per_unit_predictions") or []
    if len(rows) != len(encoded) * keys:
        gaps.append(
            f"corpus: per-unit rows {len(rows)} != frozen texts {len(encoded)} x {keys} keys"
        )
    for index, row in enumerate(rows):
        text_index = int(row.get("text", index // max(keys, 1)))
        if not 0 <= text_index < len(counts):
            gaps.append(f"corpus: per-unit row {index} has an invalid text index")
            break
        if int(row.get("count", -1)) != counts[text_index]:
            gaps.append(
                f"corpus: per-unit counts != frozen tokenisation "
                f"(first difference at unit {index})"
            )
            break
    total = int(sum(counts)) * keys
    if payload.get("n_tokens") != total:
        gaps.append(
            f"corpus: n_tokens {payload.get('n_tokens')} != frozen corpus tokens {total}"
        )
    if payload.get("n_prompts") != len(encoded):
        gaps.append(
            f"corpus: n_prompts {payload.get('n_prompts')} != frozen texts {len(encoded)}"
        )
    arm = payload.get("arm") or (payload.get("target") or "").split("/")[-1]
    if arm == "plaintext" and rows:
        truth = np.concatenate(
            [np.asarray(row.get("truth"), dtype=np.int64) for row in rows]
        )
        expected = np.concatenate([
            np.tile(np.asarray(ids, dtype=np.int64), keys) for ids in encoded
        ]) if encoded else np.asarray([], dtype=np.int64)
        if not np.array_equal(truth, expected):
            gaps.append("corpus: plaintext truth ids != frozen tokenisation")
    if arm in {"ab", "aloepri"} and rows:
        # The stage-1 truth lives in the permuted vocabulary; bind it to the frozen
        # corpus through the artifact's own client secret instead of trusting it.
        try:
            import torch

            secret = torch.load(
                artifact / "client" / "client_secret.pt",
                map_location="cpu",
                weights_only=False,
            )
            perm = secret["perm_vocab"].long().numpy()
        except Exception as exc:
            gaps.append(f"corpus: cannot load the artifact client secret: {exc}")
            perm = None
        if perm is not None:
            blocks = [
                np.tile(perm[np.asarray(ids, dtype=np.int64)], keys)
                for ids in encoded
            ]
            expected = (
                np.concatenate(blocks) if blocks else np.asarray([], dtype=np.int64)
            )
            truth = np.concatenate(
                [np.asarray(row.get("truth"), dtype=np.int64) for row in rows]
            )
            if truth.shape != expected.shape or not np.array_equal(truth, expected):
                gaps.append(
                    "corpus: permuted truth ids != perm[tokenised frozen text]"
                )
    detail = {
        "n_texts": len(encoded),
        "n_tokens": total,
        "counts_sha256": hashlib.sha256(
            json.dumps(counts, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return gaps, detail, encoded


_WIRE_DEPLOYMENTS: dict[tuple, object] = {}


def _build_wire_deployment(identity: dict, arm: str, root_key: bytes | None):
    from obfuscation_attacks.schemes.ab_deployment import ABDeployment

    artifact = _resolve((identity.get("artifact") or {}).get("path"))
    model = _resolve((identity.get("model") or {}).get("path"))
    return ABDeployment(
        artifact=artifact,
        clean_model=model,
        dtype="float32",
        layers=(1,),
        max_len=None,
        load_models="deployment" if arm == "aloepri" else False,
        strict_length=True,
        expected_variant=(identity.get("model") or {}).get("variant"),
        root_key=root_key,
    )


def _gaps_from_wire(
    payload: dict, identity: dict, root_key: bytes | None
) -> tuple[list[str], dict]:
    """Recompute the uploaded observation from the frozen text + secret and compare.

    The KNN audit trusts the stored query matrix; without this check a crafted matrix
    of rows from the deployment table would audit clean. AB needs the root key to
    rebuild ``z``; AloePri needs the deployment model's layer-1 hidden state.
    """
    gaps: list[str] = []
    arm = payload.get("arm") or (payload.get("target") or "").split("/")[-1]
    if arm not in {"ab", "aloepri"}:
        return gaps, {}
    if arm == "ab" and root_key is None:
        return ["wire: root key not supplied; cannot recompute the uploaded observation"], {}
    commitment = (identity.get("key_schedule") or {}).get("root_commitment")
    if root_key is not None and commitment:
        if hashlib.sha256(root_key).hexdigest() != commitment:
            gaps.append("wire: key root commitment != sha256(root key)")
    key_digest = (
        hashlib.sha256(root_key).hexdigest() if root_key is not None else None
    )
    key = (
        str((identity.get("artifact") or {}).get("path")),
        arm,
        key_digest,
    )
    deployment = _WIRE_DEPLOYMENTS.get(key)
    if deployment is None:
        try:
            deployment = _build_wire_deployment(identity, arm, root_key)
        except Exception as exc:
            return [f"wire: cannot build the verification deployment: {exc}"], {}
        _WIRE_DEPLOYMENTS[key] = deployment
    data = identity.get("data") or {}
    try:
        role = str(data.get("role") or "knn")
        main_n = int(data.get("main_knn_texts") or (
            data.get("n_texts") if role == "knn" else 0
        ) or 0)
        robust_n = int(data.get("robust_texts") or (
            data.get("n_texts") if role == "robust" else 0
        ) or 0)
        corpus = _rebuild_corpus(
            _resolve(data.get("path")),
            seed=int(data.get("split_seed")),
            n_knn=main_n or 1000,
            n_robust=robust_n,
            split_strategy=str(data.get("split_strategy") or "exact_v1"),
        )
        texts = corpus["robust"].texts() if role == "robust" else corpus["knn"].texts()
        keys = int(payload.get("n_keys") or 1)
        vectors = []
        for text_index, text in enumerate(texts):
            for key_offset in range(keys):
                index = text_index * keys + key_offset
                wire = deployment.observe(
                    text, arm=arm, request_id=index, key_index=index,
                    forward=arm == "aloepri",
                )
                if arm == "ab":
                    q = wire.public.wire_vectors
                else:
                    q = wire.public.hidden_states[0]
                vectors.append(np.asarray(q, dtype=np.float32))
        recomputed = np.concatenate(vectors, axis=0)
    except Exception as exc:
        return [f"wire: recomputation failed: {exc}"], {}
    evidence = (payload.get("diagnostics") or {}).get("evidence") or {}
    path = _resolve(evidence.get("path"))
    if path is None or not path.is_file():
        return gaps + ["wire: query matrix missing for recomputation"], {}
    try:
        stored = np.asarray(
            np.load(path, allow_pickle=False)["queries"], dtype=np.float32
        )
    except Exception as exc:
        return gaps + [f"wire: query matrix unreadable: {exc}"], {}
    if stored.shape != recomputed.shape:
        return gaps + [
            f"wire: stored queries shape {stored.shape} != recomputed {recomputed.shape}"
        ], {}
    max_diff = float(np.abs(stored - recomputed).max()) if stored.size else 0.0
    if max_diff > 1e-4:
        gaps.append(
            f"wire: stored queries != recomputed upload (max abs diff {max_diff:.3e})"
        )
    return gaps, {"arm": arm, "max_abs_diff": max_diff,
                  "n_rows": int(recomputed.shape[0])}


def _audit_result(path: Path, profile, digest: str, *, root_key: bytes | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "run_id": None, "attack": None, "attack_id": None, "arm": None,
            "target": None, "scenario": None, "n_prompts": None, "n_tokens": None,
            "verdict": "evidence_mismatch",
            "reason": "result payload is not a JSON object",
            "gaps": ["payload: not a JSON object"],
            "base_audit": {}, "rank_recompute": {}, "corpus_recompute": {},
            "ci_recompute": {}, "hash_recompute": {}, "knn_recompute": {},
            "wire_recompute": {}, "control": {}, "model_variant": None,
            "auditor": "protocol_auditor", "protocol_sha256": digest,
        }

    def _safe_two(label, fn):
        try:
            return fn()
        except Exception as exc:  # malformed payload must become a verdict, not a crash
            return [f"{label}: malformed payload ({exc})"], {}

    def _safe_three(label, fn):
        try:
            return fn()
        except Exception as exc:
            return [f"{label}: malformed payload ({exc})"], {}, []

    try:
        base = audit_payload(payload, profile=profile, protocol_sha256=digest)
    except Exception as exc:
        base = {
            "verdict": "below_budget",
            "status": "below_budget",
            "reason": f"malformed payload: {exc}",
            "gaps": [f"audit_payload: malformed payload ({exc})"],
            "control": {"passed": None},
            "saturation": None,
            "auditor": "protocol_auditor",
            "protocol_sha256": digest,
        }
    identity = payload.get("identity") or {}
    gaps: list[str] = []
    try:
        schema = validate_payload(payload)
    except Exception as exc:
        schema = [f"validate_payload raised: {exc}"]
    if schema:
        gaps.extend(f"schema: {item}" for item in schema[:8])
    rank_gaps, rank_detail = _safe_two("rank", lambda: _gaps_from_ranks(payload))
    gaps.extend(rank_gaps)
    corpus_gaps, corpus_detail, encoded_texts = _safe_three(
        "corpus", lambda: _gaps_from_corpus(payload, identity)
    )
    gaps.extend(corpus_gaps)
    ci_gaps, ci_detail = _safe_two("ci", lambda: _gaps_from_ci(payload))
    gaps.extend(ci_gaps)
    hash_gaps, hash_detail = _safe_two(
        "hashes", lambda: _gaps_from_hashes(payload, identity)
    )
    gaps.extend(hash_gaps)
    evidence_gaps, evidence_detail = _safe_two(
        "evidence", lambda: _gaps_from_evidence(payload, identity)
    )
    gaps.extend(evidence_gaps)
    wire_gaps, wire_detail = _safe_two(
        "wire", lambda: _gaps_from_wire(payload, identity, root_key)
    )
    gaps.extend(wire_gaps)

    raw_control = payload.get("positive_control")
    if raw_control is None:
        control = {}
    elif not isinstance(raw_control, dict):
        gaps.append("control: positive_control is not a mapping")
        control = {}
    else:
        control = raw_control
    control_detail = None
    arm_value = payload.get("arm") or (payload.get("target") or "").split("/")[-1]
    control_passed = False
    if arm_value == "plaintext":
        recomputed_top1 = None
        if rank_detail.get("counts"):
            recomputed_top1 = float(rank_detail["top"][1] / rank_detail["counts"])
        control_passed = recomputed_top1 is not None and recomputed_top1 >= 0.99
        control_detail = {"self": True, "recomputed_top1": recomputed_top1}
        if control.get("top1") is not None and recomputed_top1 is not None:
            try:
                if abs(float(control["top1"]) - recomputed_top1) > 1e-9:
                    gaps.append("control: self-reported top1 != recomputed per-unit top1")
            except (TypeError, ValueError):
                gaps.append("control: self-reported top1 is not numeric")
    else:
        # Resolve the control from the immutable layout, not from a path the audited
        # payload supplies; a fake control file inside the result tree is then only
        # usable if it also passes its own full audit.
        control_path = None
        try:
            result_root = path.parents[4]
        except IndexError:
            result_root = None
        if result_root is not None and control.get("run_id"):
            candidate = (
                result_root / "C01" / str(payload.get("attack"))
                / "plaintext" / str(control["run_id"]) / "result.json"
            )
            if candidate.is_file():
                control_path = candidate
        if control_path is None and result_root is not None:
            reported = _resolve(control.get("result_path"))
            if reported is not None and reported.is_file():
                try:
                    reported.relative_to(result_root)
                    control_path = reported
                except ValueError:
                    control_path = None
        if control_path is None or not control_path.is_file():
            gaps.append("control: positive-control result not found in the immutable layout")
        else:
            try:
                control_payload = json.loads(control_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                gaps.append(f"control: control result unreadable ({exc})")
                control_payload = {}
            if not isinstance(control_payload, dict):
                gaps.append("control: control result is not a JSON object")
                control_payload = {}
            control_audit = audit_payload(control_payload, profile=profile, protocol_sha256=digest)
            control_rows = (control_payload.get("diagnostics") or {}).get(
                "per_unit_predictions"
            ) or []
            recomputed = sum(
                1 for row in control_rows for value in (row.get("rank") or []) if value <= 1
            )
            total = sum(int(row.get("count", 0)) for row in control_rows)
            recomputed_top1 = (recomputed / total) if total else None
            control_detail = {
                "path": str(control_path),
                "verdict": control_audit.get("verdict"),
                "recomputed_top1": recomputed_top1,
                "data_sha256": control_payload.get("data_sha256"),
            }
            control_passed = recomputed_top1 is not None and recomputed_top1 >= 0.99
            control_identity = control_payload.get("identity") or {}
            control_rank_gaps, _ = _gaps_from_ranks(control_payload)
            control_corpus_gaps, _, _ = _gaps_from_corpus(control_payload, control_identity)
            control_evidence_gaps, _ = _gaps_from_evidence(control_payload, control_identity)
            control_problems = control_rank_gaps + control_corpus_gaps + control_evidence_gaps
            if control_problems:
                gaps.append(
                    "control: control evidence does not recompute: "
                    + "; ".join(control_problems[:3])
                )
            control_model = control_identity.get("model") or {}
            arm_model = identity.get("model") or {}
            if control_model.get("weights_sha256") != arm_model.get("weights_sha256"):
                gaps.append("control: control run used a different model")
            control_role = (control_identity.get("data") or {}).get("role", "knn")
            arm_role = (identity.get("data") or {}).get("role", "knn")
            if control_role != arm_role:
                gaps.append(
                    f"control: control data role {control_role!r} != {arm_role!r}"
                )
            if control_payload.get("profile") != "paper-v1":
                gaps.append("control: control run is not bound to paper-v1")
            # The control may legitimately sit below the main-cell floor (the robust
            # control uses 200 texts); its integrity is enforced by the recomputation
            # above, not by reusing the paper-scale verdict.
            if control_payload.get("attack") != payload.get("attack"):
                gaps.append("control: control run used a different attack/metric")
            if (control_payload.get("arm") or "") != "plaintext":
                gaps.append("control: control run is not the plaintext arm")
            if control_payload.get("data_sha256") != payload.get("data_sha256"):
                gaps.append("control: positive control used a different corpus")
            if control_payload.get("protocol_sha256") != digest:
                gaps.append("control: positive control is bound to another protocol")
            if control.get("top1") is not None and recomputed_top1 is not None:
                try:
                    if abs(float(control["top1"]) - recomputed_top1) > 1e-9:
                        gaps.append("control: self-reported top1 != recomputed control top1")
                except (TypeError, ValueError):
                    gaps.append("control: self-reported control top1 is not numeric")
    model_variant = (identity.get("model") or {}).get("variant")
    if model_variant != "base":
        gaps.append(f"model: variant {model_variant!r} != paper-v1 primary 'base'")

    if gaps:
        verdict = "evidence_mismatch"
        reason = "; ".join(gaps[:8])
    elif base.get("verdict") != "audited_paper_scale":
        verdict = base.get("verdict", "below_budget")
        reason = base.get("reason", "")
    elif not control_passed:
        verdict = "control_failed"
        reason = "positive control top1 < 0.99 or missing"
    else:
        verdict = "audited_paper_scale"
        reason = "all floors, per-unit evidence, hashes and re-run KNN agree"
    return {
        "path": str(path),
        "run_id": payload.get("run_id"),
        "attack": payload.get("attack"),
        "attack_id": payload.get("attack_id"),
        "arm": payload.get("arm") or (payload.get("scheme_name") or "").split(":")[-1],
        "target": payload.get("target"),
        "scenario": payload.get("scenario"),
        "n_prompts": payload.get("n_prompts"),
        "n_tokens": payload.get("n_tokens"),
        "verdict": verdict,
        "reason": reason,
        "gaps": gaps,
        "base_audit": base,
        "rank_recompute": rank_detail,
        "corpus_recompute": corpus_detail,
        "ci_recompute": ci_detail,
        "hash_recompute": hash_detail,
        "knn_recompute": evidence_detail,
        "wire_recompute": wire_detail,
        "control": {"passed": control_passed, "detail": control_detail},
        "model_variant": model_variant,
        "auditor": AUDITOR,
        "protocol_sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--root-key-file", type=Path,
                        default=ROOT / ".secrets" / "paper-v1.root")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    root_key = (
        args.root_key_file.read_bytes()[:32]
        if args.root_key_file.is_file() else None
    )
    entries = []
    for path in args.result:
        try:
            entry = _audit_result(path, profile, digest, root_key=root_key)
        except Exception as exc:
            entry = {
                "path": str(path), "run_id": None, "attack": None,
                "attack_id": None, "arm": None, "target": None, "scenario": None,
                "n_prompts": None, "n_tokens": None,
                "verdict": "evidence_mismatch",
                "reason": f"audit crashed on malformed result: {exc}",
                "gaps": [f"audit: malformed result ({exc})"],
                "base_audit": {}, "rank_recompute": {}, "corpus_recompute": {},
                "ci_recompute": {}, "hash_recompute": {}, "knn_recompute": {},
                "wire_recompute": {}, "control": {}, "model_variant": None,
                "auditor": "protocol_auditor", "protocol_sha256": digest,
            }
        entries.append(entry)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (
        ROOT / "reports" / "privacy_rebuild_20261107" / "audit"
        / f"paper_run_audit_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "independent_paper_run_audit",
        "protocol_sha256": digest,
        "counts": counts,
        "results": entries,
    }
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}: {len(entries)} results, verdicts={counts}")
    for entry in entries:
        print(f"  {entry['attack']}/{entry['arm']} {entry['target']}: {entry['verdict']} "
              f"({entry['reason'][:120]})")
    if any(entry["verdict"] != "audited_paper_scale" for entry in entries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
