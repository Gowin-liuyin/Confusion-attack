"""Real-artifact attack battery (plan classes C01..C14) over the packaged A+B build.

Moved here from ``scripts/run_full_battery.py`` so the experiment runner's ``mode:
full`` path can execute it: the runner may not import from ``scripts/``, and a
``resource_blocked`` answer that no longer corresponds to reality (the artifact and
the Alpaca snapshot are both on disk) would be a fabricated blockage.

Every attack function here takes the *public* half of a :class:`Wire` plus the arm's
candidate table; ground truth is used only for scoring (``wire.truth``), which is the
plan's §1.7-9 isolation rule. Results are written through
:func:`obfuscation_attacks.evaluation.attack_model.result_payload`, so each cell
carries its surface / knowledge / scenario / key flags / candidate mode / sample
sizes / positive control / CI.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..data.corpora import load_alpaca, reference_counts
from ..metrics.intervals import macro_bootstrap, micro_cluster_bootstrap
from ..metrics.ratio_ci import paired_gain_ci, ratio_ci
from ..metrics.recovery import DELTA_BUCKETS, delta_bucket
from ..schemes.ab_deployment import ABDeployment, DEFAULT_ARTIFACT, DEFAULT_CLEAN, knn_topk
from .attack_model import result_payload, validate_payload
from .public_guard import assert_public_payload
from .run_store import content_sha256, make_run_id, write_immutable
from .tfma_ranking import rank_candidates

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "reports" / "privacy_v1" / "full"
KS = (1, 5, 10, 100)
#: Adapter scenarios in which the clean reference weights are public. Every
#: other scenario forces ``clean_model_public=False`` (or a clear refusal) so a
#: runner can never silently run a clean-access job as S-public.
CLEAN_ACCESS_SCENARIOS = frozenset({"S-public", "S-encoder", "S-pair"})


def _stable_symbol(payload: bytes) -> int:
    """Deterministic 64-bit symbol id (never Python's salted ``hash``)."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _prompt_identity(index: int, text) -> dict:
    """Non-reversible prompt identity for serialized per-prompt rows.

    The raw prompt is ground truth; rows written into result payloads and
    summaries carry only the prompt index and a SHA-256 commitment, never the
    text itself.
    """
    import hashlib

    return {
        "prompt_index": int(index),
        "prompt_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# scoring helpers (numerator / denominator explicit, plan §5.4)
# ---------------------------------------------------------------------------
def per_position_asr(candidates: np.ndarray, truth: np.ndarray, ks: Sequence[int] = KS) -> dict:
    """Occurrence-level ASR@k with explicit numerator/denominator (plan §5.4(1))."""
    candidates = np.asarray(candidates)
    truth = np.asarray(truth).ravel()
    n = truth.size
    out = {}
    for k in ks:
        kk = min(k, candidates.shape[1])
        hit = (candidates[:, :kk] == truth[:, None]).any(axis=1)
        out[f"top{k}"] = {
            "value": float(hit.mean()) if n else None,
            "numerator": int(hit.sum()),
            "denominator": int(n),
        }
    return out


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Macro (mean-of-units) interval. Micro ratios use :func:`micro_cluster_bootstrap`."""
    units = np.arange(len(values))
    result = macro_bootstrap(values, units, n_boot=n_boot, seed=seed)
    return {
        "mean": result["macro"],
        "lo": result["lo"],
        "hi": result["hi"],
        "n_units": result["n_units"],
        "n_clusters": int(result["n_clusters"]),
        "n_boot": result["n_boot"],
        "estimand": "macro",
    }


def micro_bootstrap(hits: Sequence[float], counts: Sequence[float], *, n_boot: int = 2000,
                    seed: int = 0, clusters: Optional[Sequence[int]] = None) -> dict:
    """Micro ratio ``sum(hits)/sum(counts)`` with a cluster-bootstrap interval."""
    if clusters is None:
        clusters = np.arange(len(hits))
    result = micro_cluster_bootstrap(hits, counts, clusters, n_boot=n_boot, seed=seed)
    result["estimand"] = "micro"
    return result


def frequency_buckets(ref_counts: dict[int, int]) -> dict[str, set[int]]:
    """High / mid / low token buckets from the reference corpus (plan §5.2-4)."""
    ordered = sorted(ref_counts.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ordered)
    return {
        "high": {t for t, _ in ordered[: max(1, n // 10)]},
        "mid": {t for t, _ in ordered[max(1, n // 10): max(2, n // 2)]},
        "low": {t for t, _ in ordered[max(2, n // 2):]},
    }


def bucket_asr(candidates: np.ndarray, truth: np.ndarray, buckets: dict[str, set[int]], k: int = 1) -> dict:
    truth = np.asarray(truth).ravel()
    hit = (candidates[:, :k] == truth[:, None]).any(axis=1)
    all_known = buckets["high"] | buckets["mid"] | buckets["low"]
    out = {}
    for name, members in list(buckets.items()) + [("unseen", None)]:
        if members is None:
            mask = np.array([int(t) not in all_known for t in truth])
        else:
            mask = np.array([int(t) in members for t in truth])
        out[name] = {"value": float(hit[mask].mean()) if mask.any() else None, "n": int(mask.sum())}
    return out


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _code_digest(code: Optional[dict]) -> str | None:
    import hashlib

    code = code if isinstance(code, dict) else {}
    files = code.get("files_sha256") if isinstance(code.get("files_sha256"), dict) else {}
    if not files:
        return None
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(str(name).encode("utf-8"))
        digest.update(str(files[name]).encode("ascii"))
    return digest.hexdigest()


def _artifact_training_seeds(dep: ABDeployment) -> list[int]:
    """Training seeds declared by the released artifact recipe (if any)."""
    import re

    manifest = Path(str(dep.artifact)) / "manifest.json"
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return []
    return sorted({int(match) for match in re.findall(r"seed\s*[=:]?\s*(\d+)", text)})


def identity_binding(identity: Optional[dict], protocol_sha256: str | None) -> dict:
    """Hash-bearing identity fields that must separate two run identities.

    The run id covers the model weights/tokenizer/config, the full artifact file
    map (deployment weights, client secret and OSNIP encoder) and the secret-key
    commitment, so a re-export, a re-keyed client or a protocol change can never
    collide with an older completed run.
    """
    import hashlib

    identity = identity if isinstance(identity, dict) else {}
    model = identity.get("model") if isinstance(identity.get("model"), dict) else {}
    artifact = identity.get("artifact") if isinstance(identity.get("artifact"), dict) else {}
    data = identity.get("data") if isinstance(identity.get("data"), dict) else {}
    code = identity.get("code") if isinstance(identity.get("code"), dict) else {}
    files = artifact.get("files_sha256") if isinstance(artifact.get("files_sha256"), dict) else {}
    files_digest = None
    if files:
        combined = hashlib.sha256()
        for name in sorted(files):
            combined.update(str(name).encode("utf-8"))
            combined.update(str(files[name]).encode("ascii"))
        files_digest = combined.hexdigest()
    data_digest = (
        hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        if data else None
    )
    return {
        "protocol_sha256": protocol_sha256,
        "protocol_declared": identity.get("protocol_sha256"),
        "model_weights_sha256": model.get("weights_sha256"),
        "model_tokenizer_sha256": model.get("tokenizer_sha256"),
        "model_config_sha256": model.get("config_sha256"),
        "artifact_weights_sha256": artifact.get("weights_sha256"),
        "artifact_files_digest": files_digest,
        "client_secret_sha256": files.get("client/client_secret.pt"),
        "encoder_sha256": files.get("osnip/encryptor.pt"),
        "key_root_commitment": (identity.get("key_schedule") or {}).get("root_commitment"),
        # Split/role/n_texts are part of the frozen data identity; without them two
        # runs over different splits of the same corpus file collide.
        "data_identity_sha256": data_digest,
        "code_digest": _code_digest(code),
        "code_commit": code.get("commit"),
        "code_n_uncommitted": code.get("n_uncommitted"),
    }


def write_payload(name: str, attack: str, arm: str, payload: dict, extra: Optional[dict] = None,
                  *, result_root: Optional[Path] = None) -> dict:
    payload = dict(payload)
    payload["schema_valid"] = True
    extra = dict(extra or {})
    payload["run_extra"] = extra
    if "executed_steps" in extra:
        payload["executed_steps"] = extra["executed_steps"]
    if "profile" in extra:
        payload["profile"] = extra["profile"]
    if "protocol_sha256" in extra:
        payload["protocol_sha256"] = extra["protocol_sha256"]
    if "per_unit_predictions" in extra:
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics["per_unit_predictions"] = list(extra["per_unit_predictions"])
        payload["diagnostics"] = diagnostics
    if "per_unit_predictions_by_layer" in extra:
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics["per_unit_predictions_by_layer"] = extra[
            "per_unit_predictions_by_layer"
        ]
        payload["diagnostics"] = diagnostics
    for key in ("identity", "timing", "errors", "run_outcome", "scenario"):
        if key in extra:
            payload[key] = extra[key]
    for role in ("data", "source"):
        paths = extra.get(f"{role}_paths")
        if paths:
            if not isinstance(paths, (list, tuple)) or not all(Path(path).is_file() for path in paths):
                raise ValueError(f"{role}_paths must contain existing files")
            payload[f"{role}_sha256"] = content_sha256(paths)
    # Schema + fail-closed public guard run on the FULLY MERGED payload: an
    # ``extra`` block must never smuggle truth/secret material past the scan
    # (the only sanctioned ground-truth fields are the strict per-unit evidence
    # rows validated inside the guard).
    problems = validate_payload(payload)
    if problems:
        raise ValueError(
            f"refusing invalid payload for {name}/{attack}/{arm}: {problems[:5]}"
        )
    # Scan the serialised JSON view, not the pre-serialisation object: json
    # converts ndarray -> list (and numpy scalars -> Python scalars), so material
    # wrapped in an array cannot slip past a scan that only walks list/tuple.
    scanned = json.loads(json.dumps(payload, default=_json_default))
    assert_public_payload(scanned, label=f"{name}/{attack}/{arm}")
    run_id = make_run_id(
        profile=str(payload.get("profile") or "probe"),
        model=str(payload.get("target") or "?"),
        arm=arm,
        scenario=str(payload.get("scenario") or "?"),
        attack=attack,
        workpoint=str(extra.get("workpoint") or payload.get("candidate_mode") or "?"),
        seed=int(extra.get("seed", 0)),
        data_sha256=str(payload.get("data_sha256") or payload.get("data_label_sha256")),
        binding=identity_binding(extra.get("identity"), payload.get("protocol_sha256")),
    )
    root = Path(result_root) if result_root is not None else OUT_ROOT
    steps = payload.get("executed_steps")
    status = ("completed" if extra.get("run_outcome") == "completed"
              and isinstance(steps, (int, float)) and not isinstance(steps, bool)
              and steps > 0 else "not_verified")
    path = write_immutable(root / name / attack / arm, run_id, payload, status=status)
    payload["run_id"] = run_id
    try:
        payload["result_path"] = str(path.relative_to(ROOT))
    except ValueError:
        payload["result_path"] = str(path)
    return payload


# ---------------------------------------------------------------------------
# C01 - KNN on the real arms (plan P0-1 / §6.1)
# ---------------------------------------------------------------------------
def truth_rank(candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1-based rank of the true id in each candidate row; ``k+1`` means not found."""
    candidates = np.asarray(candidates)
    truth = np.asarray(truth).ravel()
    match = candidates == truth[:, None]
    found = match.any(axis=1)
    first = match.argmax(axis=1)
    return np.where(found, first + 1, candidates.shape[1] + 1).astype(np.int32)


def asr_from_ranks(ranks: np.ndarray, ks: Sequence[int] = KS) -> dict:
    """Top-k ASR from per-position ranks; every position stays in the denominator."""
    ranks = np.asarray(ranks).ravel()
    out = {}
    for k in ks:
        hit = ranks <= int(k)
        out[f"top{k}"] = {
            "value": float(hit.mean()) if ranks.size else None,
            "numerator": int(hit.sum()),
            "denominator": int(ranks.size),
        }
    return out


def knn_cell(dep: ABDeployment, texts: Sequence[str], *, arm: str,
             metric: str = "cosine", forward: bool = True,
             oracle_diagnostics: bool = True, ks: Sequence[int] = KS,
             keys_per_text: int = 1, key_base: int = 0,
             recovered_mapping: Mapping[int, int] | None = None) -> dict:
    """One arm's C01 measurement: real wire -> exact full-vocabulary KNN -> per-position ranks.

    The attack side reads only ``wire.public`` and the arm's public table. Ground
    truth is used to score. ``oracle_diagnostics`` adds the secret-permutation
    composition, clearly labelled as an oracle upper bound.

    ``keys_per_text`` repeats each text under fresh derived keys (the protocol's
    key-robustness sub-experiment: 200 texts x 5 keys). The bootstrap resamples
    **texts**, not (text, key) observations, so the five observations of one text
    stay in the same cluster.
    """
    if int(keys_per_text) < 1:
        raise ValueError("keys_per_text must be >= 1")
    queries, truth_plain, truth_perm, lengths, clusters = [], [], [], [], []
    for text_index, text in enumerate(texts):
        for key_offset in range(int(keys_per_text)):
            request_index = int(key_base) + text_index * int(keys_per_text) + key_offset
            wire = dep.observe(
                text, arm=arm, request_id=request_index, key_seed=request_index,
                forward=forward,
            )
            ids = wire.truth.plaintext_ids
            if arm == "plaintext":
                q = dep.E_clean[wire.public.wire_token_ids]
            elif arm == "ab":
                q = wire.public.wire_vectors
            else:
                q = wire.public.hidden_states[0]
            queries.append(np.asarray(q, dtype=np.float32))
            truth_plain.append(ids)
            truth_perm.append(
                wire.truth.permuted_ids if wire.truth.permuted_ids is not None else dep.perm[ids]
            )
            lengths.append(len(ids))
            clusters.append(text_index)
    Q = np.concatenate(queries, axis=0)
    T_plain = np.concatenate(truth_plain)
    T_perm = np.concatenate(truth_perm)
    table = dep.table(arm)

    if metric in {"norm", "norm_direction"}:
        from ..attacks.nn_knn.norm import norm_topk
        cand = norm_topk(Q, table, k=max(ks), direction_weight=1.0 if metric == "norm_direction" else 0.0)
        wire_id_identity = 1.0 if arm == "aloepri" else None
    elif arm == "aloepri":
        # The wire itself is the permuted id, so a KNN on the *received id* is trivial.
        # The measured attack uses the server-visible layer-1 hidden state against the
        # public deployment table; the trivial id identity is reported separately.
        cand = knn_topk(Q, table, k=max(ks), metric=metric)
        wire_id_identity = 1.0
    else:
        cand = knn_topk(Q, table, k=max(ks), metric=metric)
        wire_id_identity = None
    stage1_truth = T_plain if arm == "plaintext" else T_perm
    ranks = truth_rank(cand, stage1_truth)
    stage1 = asr_from_ranks(ranks, ks)
    if oracle_diagnostics and arm != "plaintext":
        composed = per_position_asr(dep.inv_perm[cand], T_plain, ks)
    else:
        composed = stage1 if arm == "plaintext" else None
    end_to_end = None
    if recovered_mapping is not None:
        from ..attacks.nn_knn.norm import compose_candidates
        translated = compose_candidates(cand, recovered_mapping)
        end_to_end = {}
        for k in ks:
            hits = sum(int(int(target) in row[:int(k)]) for row, target in zip(translated, T_plain))
            end_to_end[f"top{k}"] = {"value": hits / len(T_plain) if len(T_plain) else None,
                                     "numerator": hits, "denominator": len(T_plain)}

    per_hits, per_unit, start = [], [], 0
    text_hits = [0] * len(texts)
    text_counts = [0] * len(texts)
    for index, n in enumerate(lengths):
        block = ranks[start:start + n]
        block_cand = cand[start:start + n]
        hits = int((block <= 1).sum())
        per_hits.append(hits)
        text_hits[clusters[index]] += hits
        text_counts[clusters[index]] += n
        per_unit.append({
            "unit": index,
            "text": int(clusters[index]),
            "key": int(index % int(keys_per_text)),
            "hits": hits,
            "count": int(n),
            "rank": block.tolist(),
            "predicted": block_cand[:, 0].tolist(),
            "truth": stage1_truth[start:start + n].tolist(),
        })
        start += n
    per_text = [
        float(hits / count) if count else 0.0
        for hits, count in zip(text_hits, text_counts)
    ]
    return {
        "arm": arm,
        "metric": metric,
        "queries": Q,
        "lengths": lengths,
        "clusters": clusters,
        "keys_per_text": int(keys_per_text),
        "n_texts": len(texts),
        "candidates": cand,
        "ranks": ranks,
        "stage1_truth": stage1_truth,
        "truth_plain": T_plain,
        "truth_perm": T_perm,
        "stage1": stage1,
        "composed": composed,
        "end_to_end_recovered_mapping": end_to_end,
        "end_to_end_candidate_depth": int(cand.shape[1]) if end_to_end is not None else None,
        "end_to_end_candidate_scope": (
            "full_vocabulary" if cand.shape[1] == table.shape[0] else "retrieval_top_k_only"
        ) if end_to_end is not None else None,
        "wire_id_identity": wire_id_identity,
        "per_unit": per_unit,
        "per_hits": per_hits,
        "per_text": per_text,
        "micro_ci": micro_bootstrap(per_hits, lengths, clusters=clusters),
        "macro_ci": bootstrap_ci(per_text),
        "n_queries": int(Q.shape[0]),
    }


def _arrays_digest(queries: np.ndarray, lengths: np.ndarray, arm: str, metric: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(queries, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(lengths, dtype=np.int64).tobytes())
    digest.update(str(arm).encode("utf-8"))
    digest.update(str(metric).encode("utf-8"))
    return digest.hexdigest()


def save_query_evidence(directory: Path, cell: dict) -> dict:
    """Write the query matrix an independent auditor re-runs the KNN on.

    The file name is a digest of the array contents, so re-executing a cell never
    overwrites another run's evidence (``np.savez`` embeds zip timestamps, so file
    bytes are not a stable identity).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    queries = np.asarray(cell["queries"], dtype=np.float32)
    lengths = np.asarray(cell["lengths"], dtype=np.int64)
    digest = _arrays_digest(queries, lengths, cell["arm"], cell["metric"])
    path = directory / f"queries_{digest[:16]}.npz"
    if not path.exists():
        temporary = directory / f".tmp_{os.getpid()}_{digest[:8]}.npz"
        np.savez(
            temporary,
            queries=queries,
            lengths=lengths,
            arm=np.asarray(cell["arm"]),
            metric=np.asarray(cell["metric"]),
        )
        os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": digest,
        "arrays_sha256": digest,
        "file_sha256": content_sha256(path),
        "shape": [int(x) for x in queries.shape],
        "dtype": "float32",
    }


def run_c01(dep: ABDeployment, texts: Sequence[str], ref_counts: dict[int, int], *,
            verbose: bool = True, profile: str = "probe",
            protocol_sha256: str = "", metric: str = "cosine",
            arms: Sequence[str] = ("plaintext", "ab", "aloepri"),
            forward: Optional[bool | dict] = None,
            model_id: str = "llama-3.2-1b", scenario: str = "S-public",
            seed: int = 0, data_paths: Optional[list] = None,
            source_paths: Optional[list] = None,
            control: Optional[dict] = None,
            identity: Optional[dict] = None,
            oracle_diagnostics: bool = True,
            emit_not_applicable: bool = True,
            result_root: Optional[Path] = None,
            evidence_dir: Optional[Path] = None,
            write_queries: bool = False,
            workpoint: Optional[str] = None,
            keys_per_text: int = 1,
            key_base: int = 0,
            attack_id: Optional[str] = None) -> dict:
    """KNN embedding inversion over the real wire, per arm.

    * ``plaintext``: positive control -- clean embeddings against the clean table;
      the plan requires Top-1 >= 99%.
    * ``ab``: the server's continuous ``z`` against the *deployment* table gives
      stage-1 (permuted id); mapping through the secret permutation is reported
      separately as an **oracle-permutation upper bound**, because the real stage-2
      is C06-W/C08 and this attack does not recover it (§1.5 discipline).
    * ``aloepri``: the wire *is* the permuted id, so stage-1 is not an attack
      outcome for this arm; the cell records that instead of a fake 100%.
    """
    results: dict[str, dict] = {}
    for arm in arms:
        t0 = time.time()
        if isinstance(forward, dict):
            arm_forward = bool(forward.get(arm, True))
        else:
            arm_forward = True if forward is None else bool(forward)
        cell = knn_cell(dep, texts, arm=arm, metric=metric, forward=arm_forward,
                        oracle_diagnostics=oracle_diagnostics,
                        keys_per_text=keys_per_text, key_base=key_base)
        stage1 = cell["stage1"]
        composed = cell["composed"]
        buckets = frequency_buckets(ref_counts)
        if arm == "plaintext":
            control_record = {
                "name": "clean_embedding_retrieval",
                "top1": stage1["top1"]["value"],
                "required_top1": 0.99,
                "n_texts": len(cell["lengths"]),
                "top_k": int(max(KS)),
                "metric": metric,
            }
            pc_top1 = stage1["top1"]["value"]
        else:
            control_record = dict(control or {
                "name": "not_run",
                "top1": None,
                "required_top1": 0.99,
            })
            pc_top1 = control_record.get("top1")
        evidence = None
        if write_queries and evidence_dir is not None:
            evidence = save_query_evidence(Path(evidence_dir), cell)
        attack_name = "nn_knn.euclidean" if metric == "l2" else "nn_knn.cosine"
        payload = result_payload(
            attack_name=attack_name,
            scheme_name=f"ab_deployment:{arm}",
            target=f"{model_id}/{arm}",
            positive_control=control_record,
            metrics={
                "stage1_permuted_id": stage1,
                "composed_with_oracle_permutation": composed,
                "random_baseline": {f"top{k}": k / dep.vocab_size for k in KS},
                "frequency_buckets_top1": bucket_asr(
                    cell["candidates"] if arm == "plaintext"
                    else dep.inv_perm[cell["candidates"]],
                    cell["truth_plain"],
                    buckets,
                ),
                "frequency_buckets_scope": (
                    "plaintext token ids (clean frequency buckets)"
                    if arm == "plaintext"
                    else "clean ids via the oracle permutation (diagnostic only)"
                ),
            },
            diagnostics={
                "candidate_mode": "full",
                "n_candidates": int(dep.vocab_size),
                "positive_control": control_record,
                "positive_control_passed": (
                    None if pc_top1 is None else bool(float(pc_top1) >= 0.99)
                ),
                "metric": metric,
                "table": "clean" if arm == "plaintext" else "deployment",
                "stage1_definition": "recover the id in the server's own (permuted) vocabulary",
                "composed_definition": "stage-1 hit mapped through the secret permutation "
                                       "(oracle tau upper bound; real stage-2 is C06-W/C08)",
                "n_queries": cell["n_queries"],
                "n_texts": int(cell["n_texts"]),
                "n_observations": len(cell["lengths"]),
                "keys_per_text": int(keys_per_text),
                "arm": arm,
                "variant": "full_vocabulary_exact",
                "forward_computed": bool(arm_forward),
                "wire_id_identity": cell.get("wire_id_identity"),
                "paper_parameter_undisclosed": True,
                "evidence": evidence,
                "note": (
                    "AloePri-only wire is the permuted token id itself; the trivial id "
                    "identity is reported separately and the measured attack is the KNN on "
                    "the server-visible layer-1 hidden state against the public deployment table."
                    if arm == "aloepri" else ""
                ),
            },
            ci={"top1_micro": cell["micro_ci"], "top1_per_text_macro": cell["macro_ci"],
                "bootstrap_seed": 0},
            n_prompts=int(cell["n_texts"]),
            n_tokens=int(cell["truth_plain"].size),
            n_keys=int(keys_per_text),
            n_types=int(len(set(cell["truth_plain"].tolist()))),
            reference_tokens=int(sum(ref_counts.values())),
            source={"artifact": str(dep.artifact), "arm": arm},
            data={"texts": len(texts), "filter": "alpaca instruction+input, 20-512 chars"},
        )
        payload["top_k"] = int(max(KS))
        payload["attack_id"] = attack_id or ("knn_l2" if metric == "l2" else "knn_cosine")
        payload["arm"] = arm
        payload["keys_per_prompt"] = int(keys_per_text)
        payload["n_observations"] = int(len(cell["lengths"]))
        if arm != "plaintext":
            payload["defense_training_seeds"] = _artifact_training_seeds(dep)
            payload["defense_training_seed_source"] = "artifact manifest recipe"
        if identity:
            payload["identity"] = identity
        written = write_payload("C01", attack_name, arm, payload, {
            "seconds": round(time.time() - t0, 1),
            "profile": profile,
            "protocol_sha256": protocol_sha256,
            "per_unit_predictions": cell["per_unit"],
            "executed_steps": int(cell["n_queries"]),
            "run_outcome": "completed",
            "seed": int(seed),
            "scenario": scenario,
            "identity": identity or {},
            "timing": {"seconds": round(time.time() - t0, 1)},
            "errors": [],
            "data_paths": list(data_paths or []),
            "source_paths": list(source_paths or []),
            "workpoint": workpoint,
        }, result_root=result_root)
        results[arm] = {
            "stage1_top1": stage1["top1"]["value"],
            "stage1_top5": stage1["top5"]["value"],
            "stage1_top10": stage1["top10"]["value"],
            "stage1_top100": stage1["top100"]["value"],
            "composed_top1_oracle": composed["top1"]["value"] if composed is not None else None,
            "ci_top1_micro": cell["micro_ci"],
            "ci_top1_macro": cell["macro_ci"],
            "n_texts": int(cell["n_texts"]),
            "n_tokens": int(cell["truth_plain"].size),
            "n_queries": cell["n_queries"],
            "seconds": round(time.time() - t0, 1),
            "run_id": written.get("run_id"),
            "result_path": written.get("result_path"),
            "positive_control": control_record,
            "positive_control_passed": (
                None if pc_top1 is None else bool(float(pc_top1) >= 0.99)
            ),
            "evidence": evidence,
        }
        if verbose:
            print(f"  C01 {arm:9s} stage1@1={results[arm]['stage1_top1']:.4f} "
                  f"@10={results[arm]['stage1_top10']:.4f} "
                  f"composed@1={results[arm]['composed_top1_oracle']} "
                  f"({results[arm]['seconds']}s, n={len(cell['lengths'])})", flush=True)
    if emit_not_applicable:
        results["osnip"] = emit_not_applicable_cells(dep)["C01"]
    return results


def emit_not_applicable_cells(dep: ABDeployment) -> dict:
    """Write the cells that are N/A for this artifact (plan §1.6: N/A, never 0%).

    Kept separate so it can be regenerated without re-running any measured arm, and so
    ``main_table_eligible`` is explicitly false for a cell that carries no measurement.
    """
    na = dep.arm_status("osnip")
    if na.get("status") != "not_applicable":
        # A missing artifact/encoder is a resource block, not a conceptual
        # inapplicability; writing a not_applicable payload would launder it.
        return {"C01": {"status": na.get("status", "resource_blocked"),
                        "reason": na.get("reason", "")}}
    payload = result_payload(
        attack_name="nn_knn.cosine",
        scheme_name="ab_deployment:osnip",
        target="llama-3.2-1b/osnip",
        metrics={"status": "not_applicable", "reason": na["reason"]},
        diagnostics={"candidate_mode": "full", "positive_control": {},
                     "not_applicable_reason": na["reason"],
                     "not_applicable_premise": "real_premise_absent",
                     "not_applicable_evidence": na["reason"]},
        n_prompts=0, n_tokens=0, n_keys=0, n_types=0, reference_tokens=0,
        source={"artifact": str(dep.artifact), "arm": "osnip"}, data={"arm": "osnip"},
        overrides={"main_table_eligible": False},
    )
    write_payload("C01", "nn_knn.cosine", "osnip", payload, {"not_applicable": True})
    return {"C01": {"status": "not_applicable", "reason": na["reason"]}}


# ---------------------------------------------------------------------------
# C06-W / C08 - weight-only attacks on the real deployment weights (S3/KW)
# ---------------------------------------------------------------------------
def _mapping_recovery_rate(predictions, truth) -> float:
    """Exact-match rate over ALL truth rows; an invalid prediction is a miss.

    Predictions outside the truth range (e.g. ``-1`` for an unassigned row) are
    never dropped from the denominator: a partial recovery must report
    ``hits / len(truth)``, not ``hits / #valid``.
    """
    predicted = np.asarray(predictions).ravel()
    expected = np.asarray(truth).ravel()
    n = int(expected.size)
    if n == 0:
        return 0.0
    if predicted.size != n:
        # Missing predictions are misses; surplus predictions are ignored.
        adjusted = np.full(n, -1, dtype=np.int64)
        keep = min(n, int(predicted.size))
        adjusted[:keep] = predicted[:keep]
        predicted = adjusted
    valid = (predicted >= 0) & (predicted < n)
    hits = int(np.sum(valid & (predicted == expected)))
    return float(hits / n)


def _monomial_precondition(E_clean: np.ndarray, E_dep: np.ndarray, perm: np.ndarray,
                           *, n_types: int = 1000, seed: int = 0) -> dict:
    """Test the premise the plan's weight attacks assume before quoting a number.

    VMA-W (Thomas et al. RowSort) and the IA invariants assume the obfuscation is a
    monomial re-basing: a permutation of rows/columns times a per-row/column scale,
    so that sorted-row magnitudes and column statistics survive. This function
    measures that premise on the real artifact, with a random-pairing control.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(perm), size=min(n_types, len(perm)), replace=False)
    X = E_clean[idx]
    Y = E_dep[perm[idx]]
    # A monomial re-basing is ``E_dep[perm[i], :] = (E_clean[i, :] * z)[:, pi]``:
    # the coordinate permutation is undone by *column* normalisation, and the
    # per-coordinate scale ``z`` is what makes a naive sorted-row test fail. So the
    # precondition is measured on column-standardised rows, plus a direct per-column
    # correlation test (each observed column must have one high-correlation partner
    # in the reference, and the argmax must be a near-bijection).
    def _col_z(mat: np.ndarray) -> np.ndarray:
        return (mat - mat.mean(0, keepdims=True)) / (mat.std(0, keepdims=True) + 1e-12)

    Xz, Yz = _col_z(X), _col_z(Y)
    sx, sy = np.sort(Xz, axis=1), np.sort(Yz, axis=1)
    d_aligned = float(np.mean(np.abs(sx - sy).sum(1) / (np.abs(sx).sum(1) + 1e-12)))
    ry = np.sort(_col_z(E_dep[perm[rng.permutation(idx)]]), axis=1)
    d_random = float(np.mean(np.abs(ry - sx).sum(1) / (np.abs(sx).sum(1) + 1e-12)))

    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc, axis=0, keepdims=True) + 1e-12)
    corr = np.abs(Xn.T @ Yn)
    best = corr.max(axis=1)
    second = np.partition(corr, -2, axis=1)[:, -2]
    argmax = corr.argmax(axis=1)
    bijection = len(set(argmax.tolist())) / corr.shape[0]
    return {
        "n_types": int(len(idx)),
        "column_normalised_sorted_row_l1_aligned": d_aligned,
        "column_normalised_sorted_row_l1_random_control": d_random,
        "separation_ratio": float(d_random / (d_aligned + 1e-12)),
        "column_corr_best_mean": float(best.mean()),
        "column_corr_second_mean": float(second.mean()),
        "column_corr_margin": float(best.mean() - second.mean()),
        "column_argmax_is_bijection_rate": float(bijection),
        "monomial_supported": bool(d_random > 1.5 * d_aligned and best.mean() > 0.8 and bijection > 0.95),
    }


def _dense_mixing_control(X: np.ndarray, e_clean: np.ndarray, *, seed: int,
                          extra_rows: int = 32, heldout_rows: int = 256,
                          forward_rows: int = 64,
                          tolerance: float = 1e-6,
                          required_top1: float = 0.99) -> dict:
    """Positive control for the dense-mixing solver on a *known* transform.

    S02: the old control solved ``M`` from an ``X`` with far fewer rows than
    columns (256 rows vs d=2048) and demanded ``allclose(Mhat, M0)``. That system
    is underdetermined: least squares returns the minimum-norm solution, which is
    generally not ``M0``, so the control failed mathematically and forced the
    dense-mixing control top-1 to 0.

    The verdict therefore comes only from an **identifiable** construction:

    * probe ``Xc`` is column-full-rank with at least ``d`` rows (d = number of
      columns; d + ``extra_rows`` Gaussian rows), ``Yc = Xc @ M0``;
    * the same solver the attack uses (:func:`solve_mixing_matrix`, float64)
      recovers ``Mhat``; the relative residual on the probe must be < ``tolerance``;
    * independent held-out rows (not used in the solve) must satisfy
      ``||Y_ho - X_ho @ Mhat|| / ||Y_ho|| < tolerance``;
    * ``forward_match`` with the *recovered* ``Mhat`` (not the oracle ``M0``)
      must recover the probed row identities at top-1 >= ``required_top1``.

    The cell's own diagnostic sample ``X`` is still solved and recorded as an
    explicit ``underdetermined_probe`` negative result; it never enters the
    verdict (``used_for_verdict: False``).
    """
    from ..attacks.vma.dense_mixing import forward_match, solve_mixing_matrix

    x = np.asarray(X, dtype=np.float64)
    n_rows, d = int(x.shape[0]), int(x.shape[1])
    rng = np.random.default_rng(int(seed))
    m0 = rng.standard_normal((d, d)) / np.sqrt(d)

    # (a) identifiable probe: at least d rows, column-full-rank.
    probe_n = d + max(0, int(extra_rows))
    xc = rng.standard_normal((probe_n, d))
    yc = xc @ m0
    solved = solve_mixing_matrix(xc, yc)
    mhat = np.asarray(solved["M"], dtype=np.float64)
    relative_residual = float(solved["relative_residual"])
    probe_rank = int(np.linalg.matrix_rank(xc))

    # (b) held-out rows, never used in the solve.
    x_ho = rng.standard_normal((max(1, int(heldout_rows)), d))
    y_ho = x_ho @ m0
    heldout_residual = float(
        np.linalg.norm(y_ho - x_ho @ mhat) / (np.linalg.norm(y_ho) + 1e-12)
    )

    # (c) row-identity recovery through the *recovered* transform. Deployment rows
    # carry the true transform M0; the candidate table carries the recovered Mhat.
    e = np.asarray(e_clean, dtype=np.float32)
    rows = np.sort(rng.choice(len(e), size=min(int(forward_rows), len(e)), replace=False))
    m0_32 = m0.astype(np.float32)
    mhat_32 = mhat.astype(np.float32)
    recovered = forward_match(e[rows] @ m0_32, e @ mhat_32)
    forward_top1 = float(np.mean(recovered == rows)) if len(rows) else 0.0

    passed = bool(
        probe_rank == d
        and relative_residual < float(tolerance)
        and heldout_residual < float(tolerance)
        and forward_top1 >= float(required_top1)
    )

    # Explicit negative probe: the old underdetermined construction. It is kept as
    # evidence that the solver cannot identify M0 from n_rows < d, and it must not
    # decide the control (see ``used_for_verdict``).
    under = solve_mixing_matrix(x, x @ m0)
    if n_rows < d:
        under_status = "not_identifiable"
        under_note = (
            f"{n_rows} rows < {d} columns; rank deficiency means lstsq cannot "
            "identify M0"
        )
    else:
        under_status = "identifiable_but_unused"
        under_note = (
            f"{n_rows} rows >= {d} columns; the cell's own diagnostic sample is "
            "solvable but still never enters the control verdict"
        )
    underdetermined = {
        "status": under_status,
        "n_rows": n_rows,
        "n_columns": d,
        "note": under_note,
        "relative_residual": float(under["relative_residual"]),
        "min_norm_solution_matches_m0": bool(
            np.allclose(np.asarray(under["M"]), m0, atol=1e-6)
        ),
        "used_for_verdict": False,
    }
    return {
        "name": "known_dense_transform_lstsq_recovery",
        "top1": forward_top1,
        "required_top1": float(required_top1),
        "passed": passed,
        "uses_recovered_M": True,
        "dtype": "float64 solve; float32 forward-match tables",
        "probe": {
            "n_rows": int(probe_n),
            "n_columns": d,
            "rank": probe_rank,
            "column_full_rank": bool(probe_rank == d),
            "relative_residual": relative_residual,
            "heldout_rows": int(x_ho.shape[0]),
            "heldout_relative_residual": heldout_residual,
            "tolerance": float(tolerance),
        },
        "forward_match": {"n_rows": int(len(rows)), "top1": forward_top1},
        "underdetermined_probe": underdetermined,
    }


def _mixing_layer_coverage(mix: Mapping, n_layers: int) -> tuple[list[int], list[dict]]:
    used = sorted({int(str(key).split(".")[0].removeprefix("layer"))
                   for key in (mix.get("per_matrix") or {}) if str(key).startswith("layer")})
    skipped = {}
    for entry in mix.get("skipped_groups") or []:
        try:
            skipped.setdefault(int(entry.get("layer")), entry)
        except (TypeError, ValueError, OverflowError):
            continue
    missing = []
    for layer in range(int(n_layers)):
        if layer in used:
            continue
        record = skipped.get(layer) or {}
        missing.append({"layer": layer, "code": str(record.get("code") or "weight_missing"),
                        "reason": str(record.get("reason") or "no q/k/v mixing matrix recovered for this layer")})
    return used, missing


def run_c06w(dep: ABDeployment, *, n_types: int = 1000, diag_types: int = 256, seed: int = 0,
             verbose: bool = True, clean_model_public: bool = True,
             scenario: str = "S-public") -> dict:
    """Measure a public dense-mixing attack and separately labelled oracle diagnostics."""
    from ..attacks.vma.dense_mixing import (
        forward_match_stats,
        recover_mixing,
    )
    from ..attacks.vma.weight_rowsort import recover_permutations

    t0 = time.time()
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        # No public clean weights in this scenario: the cell is not run
        # (scenario/resource block), never a conceptual not_applicable payload.
        clean_model_public = False
    if not clean_model_public:
        return {
            "status": "not_run",
            "scenario": scenario,
            "reason": (
                "clean reference weights are not public in scenario "
                f"{scenario!r}"
            ),
        }
    # AloePri's threat model grants the attacker the *plain* model as well as the obfuscated one:
    # §5.1 says the sub-obfuscations exist "to prevent the attacker from recovering tau by
    # comparing plaintext and obfuscated weights", and App. D.1 Table 9 is built from "Known X"
    # (plain) versus "Observed Y" (obfuscated). What the attacker does NOT hold is the key
    # matrices, tau, or the noise seed. The parameter that decides the attack's strength is the
    # noise level (§7.3: at alpha_e=0.5 VMA already recovers >30% of tokens and the paper
    # recommends alpha_e=1.0/alpha_h=0.2 for exactly that reason).
    paper_working_point = {
        "paper_alpha_e": 1.0,
        "paper_alpha_h": 0.2,
        "artifact_alpha_e": dep.secret_metadata.get("alpha_e"),
        "artifact_alpha_h": dep.secret_metadata.get("alpha_h"),
        "artifact_transform_family": dep.secret_metadata.get("transform_family"),
        "artifact_paper_algorithm1_reproduced": dep.secret_metadata.get("paper_algorithm1_reproduced"),
        "why_it_matters": ("Artifact metadata and the paper's noise point are recorded separately; "
                           "different implementations and working points require separate evaluation."),
    }
    diag_types = min(int(diag_types or n_types), n_types, 256)
    t_stage = time.time()
    pre = _monomial_precondition(dep.E_clean, dep.E_dep, dep.perm, n_types=diag_types, seed=seed)
    if verbose:
        print(f"    premise diagnostic ({diag_types} types): {time.time()-t_stage:.1f}s", flush=True)
    rng = np.random.default_rng(seed)
    ref_types = np.sort(rng.choice(len(dep.E_clean), size=min(n_types, len(dep.E_clean)), replace=False))
    diag_rows = np.sort(rng.choice(len(dep.E_clean), size=min(diag_types, len(dep.E_clean)), replace=False))
    X = dep.E_clean[diag_rows]
    Y = dep.E_dep[dep.perm[diag_rows]]
    truth = np.arange(len(diag_rows))

    def _score(p1: np.ndarray) -> float:
        # All truth rows stay in the denominator; -1/out-of-range predictions
        # are misses (never dropped as if they had not been attempted).
        return _mapping_recovery_rate(p1, truth)

    # --- variant 1: paper-literal RowSort (assumes a pure permutation) ------
    literal: dict = {"status": "not_run", "mapping_recovery": None}
    if pre["monomial_supported"]:
        rec = recover_permutations(X, Y)
        literal = {
            "status": rec.get("status"),
            "mapping_recovery": (_score(np.asarray(rec["Z1"]))
                                 if rec.get("status") in ("ok", "partial") else None),
            "min_margin": rec.get("min_margin"),
            "reconstruction_residual": rec.get("reconstruction_residual"),
        }

    # --- variant 2: scale-normalised RowSort (monomial with per-column scale) ---
    def _col_z(mat: np.ndarray) -> np.ndarray:
        return (mat - mat.mean(0, keepdims=True)) / (mat.std(0, keepdims=True) + 1e-12)

    t_stage = time.time()
    Xs, Ys = np.sort(_col_z(X), axis=1), np.sort(_col_z(Y), axis=1)
    D = np.zeros((len(truth), len(truth)), dtype=float)
    features = int(Xs.shape[1])
    budget_chunk = max(
        1, int(128 * 1024 ** 2 / max(1, len(truth) * features * 8))
    )
    for start in range(0, len(truth), budget_chunk):
        blk = Ys[start:start + budget_chunk]
        D[start:start + blk.shape[0]] = np.abs(
            blk[:, None, :] - Xs[None, :, :]
        ).sum(axis=2)
    scaled = {"status": "ok", "mapping_recovery": _score(D.argmin(axis=1))}
    if verbose:
        print(f"    scale-normalised rowwise diagnostic: {time.time()-t_stage:.1f}s", flush=True)

    # --- variant 3: dense-mixing inversion (what this artifact actually permits) ---
    t_stage = time.time()
    mix = recover_mixing(dep, getter=dep.weight_tensor)
    if verbose:
        print(f"    mixing solve (4 x q_proj lstsq): {time.time()-t_stage:.1f}s", flush=True)
    # Evaluate on a random sample of *deployment* rows: the attacker picks rows of the table it
    # holds and asks which clean token each one is. Un-mixing only those rows keeps the run to
    # seconds (the full-table GEMM is 0.54 TFLOP and dominated wall time).
    t_stage = time.time()
    eval_rows = np.sort(rng.choice(len(dep.E_dep), size=min(n_types, len(dep.E_dep)), replace=False))
    transformed_clean = (dep.E_clean @ np.asarray(mix["M"], dtype=np.float32))
    if verbose:
        print(f"    forward-transform clean table: {time.time()-t_stage:.1f}s", flush=True)
    t_stage = time.time()
    recovered_clean, top1_cos, match_margin = forward_match_stats(
        dep.E_dep[eval_rows], transformed_clean
    )
    truth_clean = dep.inv_perm[eval_rows]
    dense = {
        "accuracy": float(np.mean(recovered_clean == truth_clean)),
        "n_types": int(len(eval_rows)),
        "match_mode": "forward (E' vs E M; no matrix inverse)",
        "mixing_cond": mix["conditional_number"],
        "top1_cosine_mean": float(np.mean(top1_cos)),
        "margin_mean": float(np.mean(match_margin)),
    }
    if verbose:
        print(f"    forward match over clean table: {time.time()-t_stage:.1f}s "
              f"(cond {mix['conditional_number']:.3g}, "
              f"cos {dense['top1_cosine_mean']:.3f})", flush=True)
    dense_fit_residual = mix["residual_mean"]
    heldout_residual = mix.get("heldout_residual_mean")
    # Alignment summaries are descriptive; no validation-calibrated threshold
    # establishes or excludes the dense model for this artifact.
    dense_premise = None

    # --- real positive control: the same recovery code on known transforms ------
    # The old record cited a synthetic-scenario gate run elsewhere; that is not a
    # control for this cell. Here both code paths are run on known transforms built
    # from this cell's real clean weights and embedding table.
    ctl_rng = np.random.default_rng(int(seed) + 20261107)
    z1_ctl = ctl_rng.permutation(X.shape[0])
    z2_ctl = ctl_rng.permutation(X.shape[1])
    rec_ctl = recover_permutations(X, X[np.ix_(z1_ctl, z2_ctl)])
    rowsort_ctl = (
        float(np.mean(np.asarray(rec_ctl["Z1"]) == z1_ctl))
        if rec_ctl.get("status") in ("ok", "partial") else 0.0
    )
    rowsort_control = {
        "name": "known_row_column_permutation",
        "top1": float(rowsort_ctl),
        "required_top1": 0.99,
        "passed": bool(rowsort_ctl >= 0.99),
        "n_rows": int(X.shape[0]),
    }

    def _scale_normalised_top1(clean: np.ndarray, observed: np.ndarray,
                               truth_ids: np.ndarray) -> float:
        """Rowwise matching on column-zscored sorted rows (the `scaled` path)."""
        Xs = np.sort(_col_z(clean), axis=1)
        Ys = np.sort(_col_z(observed), axis=1)
        D = np.zeros((Ys.shape[0], Xs.shape[0]), dtype=float)
        features = int(Xs.shape[1])
        budget_chunk = max(
            1, int(128 * 1024 ** 2 / max(1, Ys.shape[0] * features * 8))
        )
        for start in range(0, Ys.shape[0], budget_chunk):
            block = Ys[start:start + budget_chunk]
            D[start:start + block.shape[0]] = np.abs(
                block[:, None, :] - Xs[None, :, :]
            ).sum(axis=2)
        predicted = D.argmin(axis=1)
        truth = np.asarray(truth_ids)
        return float(np.mean(predicted == truth)) if truth.size else 0.0

    # Positive control for the scale-normalised variant: a known row permutation
    # times a known per-column scale, scored by that variant's own matcher.
    scaled_ctl_rng = np.random.default_rng(int(seed) + 20261110)
    scaled_rows_ctl = scaled_ctl_rng.permutation(X.shape[0])
    scaled_scale_ctl = scaled_ctl_rng.uniform(0.5, 2.0, size=X.shape[1])
    scaled_observed_ctl = X[scaled_rows_ctl] * scaled_scale_ctl
    scaled_ctl_top1 = _scale_normalised_top1(
        X, scaled_observed_ctl, scaled_rows_ctl
    )
    scaled_control = {
        "name": "known_row_permutation_and_per_column_scale",
        "top1": float(scaled_ctl_top1),
        "required_top1": 0.99,
        "passed": bool(scaled_ctl_top1 >= 0.99),
        "n_rows": int(X.shape[0]),
        "premise": "monomial (row permutation x per-column scale)",
    }
    # S02: the dense-mixing control is decided only by the identifiable probe
    # (>= d rows), never by the underdetermined diagnostic sample recorded inside.
    dense_control = _dense_mixing_control(
        X, dep.E_clean, seed=int(seed) + 20261107
    )

    alpha_e = dep.secret_metadata.get("alpha_e")
    # Chance must be computed against the candidate universe each variant
    # actually matches: the RowSort diagnostics see only the <=256 sampled
    # diagnostic rows, while the dense forward match sees the full clean table.
    chance_diagnostic_rows = 1.0 / max(1, len(diag_rows))
    chance_full_vocab = 1.0 / max(1, int(dep.vocab_size))
    chance_by_variant = {
        "paper_literal_rowsort": chance_diagnostic_rows,
        "scale_normalised_rowsort": chance_diagnostic_rows,
        "dense_mixing_inversion": chance_full_vocab,
    }
    variant_scores = {
        "paper_literal_rowsort": literal["mapping_recovery"],
        "scale_normalised_rowsort": scaled["mapping_recovery"],
        "dense_mixing_inversion": dense["accuracy"],
    }
    # PERMISSION: the RowSort variants score a secret-aligned pair
    # (Y = E_dep[dep.perm[diag_rows]]), i.e. they are oracle-aligned diagnostics,
    # not attacks. The public dense-mixing attempt remains reportable even when
    # its alignment is weak; low confidence is not a proof of inapplicability.
    applicable_variant = "dense_mixing_inversion"
    variant_top1 = {
        "dense_mixing_inversion": dense["accuracy"],
    }
    chance = chance_by_variant.get(applicable_variant)
    # The headline is the applicable variant's own measured number; an
    # oracle-aligned diagnostic can never become the headline.
    headline_top1 = variant_top1.get(applicable_variant)
    diagnostic_variants = ["paper_literal_rowsort", "scale_normalised_rowsort"]
    # Bind the cell's control to the variant it reports. A cell without an
    # applicable attack fails closed instead of borrowing another variant's
    # control verdict.
    control_by_variant = {
        "dense_mixing_inversion": dense_control,
    }
    selected_control = control_by_variant.get(applicable_variant)
    if selected_control is None:
        control_top1 = None
        control_passed = False
        control_reason = (
            f"applicable variant {applicable_variant!r} has no legitimate "
            "positive control; fail-closed (the dense premise did not hold and "
            "the RowSort variants are oracle-aligned diagnostics only)"
        )
    else:
        control_top1 = float(selected_control["top1"])
        control_passed = bool(selected_control["passed"])
        control_reason = (
            f"{applicable_variant} positive control (the variant this cell reports)"
        )
    # Layers recover_mixing actually solved (per_matrix is keyed layer<N>.<group>);
    # the default call iterates every layer 0..n_layers-1 and skips missing tensors.
    used_layers, missing_layers = _mixing_layer_coverage(mix, dep.n_layers)
    payload = result_payload(
        attack_name="vma.weight_rowsort",
        scheme_name="ab_deployment:aloepri_weights",
        target="llama-3.2-1b/aloepri_weights",
        metrics={
            "paper_literal_rowsort": {
                "mapping_recovery_rate": {
                    "value": literal["mapping_recovery"],
                    **({"numerator": int(round(literal["mapping_recovery"] * len(truth))),
                        "denominator": int(len(truth))} if literal["mapping_recovery"] is not None else {}),
                    "status": literal["status"],
                    "premise": "obfuscation is a pure row x column permutation",
                    "premise_holds": bool(pre["monomial_supported"]),
                    # The pair is aligned with the secret permutation: this row
                    # is an oracle diagnostic, never an attack estimate.
                    "oracle_aligned_diagnostic": True,
                    "uses_secret_permutation": True,
                },
                "min_margin": literal.get("min_margin"),
                "reconstruction_residual": literal.get("reconstruction_residual"),
            },
            "scale_normalised_rowsort": {
                "mapping_recovery_rate": {
                    "value": scaled["mapping_recovery"],
                    "numerator": int(round(scaled["mapping_recovery"] * len(truth))),
                    "denominator": int(len(truth)),
                    # Y is E_dep[dep.perm[rows]]: the pair is aligned by the
                    # secret permutation, so this is an oracle diagnostic.
                    "status": "oracle_aligned_diagnostic",
                    "premise": "monomial (permutation x per-column scale)",
                    "premise_holds": pre["monomial_supported"],
                    "oracle_aligned_diagnostic": True,
                    "uses_secret_permutation": True,
                },
                "scaled_control": scaled_control,
            },
            "dense_mixing_inversion": {
                "mapping_recovery_rate": {
                    "value": dense["accuracy"],
                    "numerator": int(round(dense["accuracy"] * len(ref_types))),
                    "denominator": int(len(ref_types)),
                    "status": "ok",
                },
                "mixing_fit_residual": dense_fit_residual,
                "conditional_number": mix["conditional_number"],
                "n_proposals": mix["n_proposals"],
                "match_mode": dense["match_mode"],
                "mixing_cond": dense["mixing_cond"],
                "top1_cosine_mean": dense["top1_cosine_mean"],
                "margin_mean": dense["margin_mean"],
                "premise": "hidden axis is a dense linear re-basing W' = W M",
                "premise_holds": dense_premise,
                "premise_status": "unverified; alignment statistics are descriptive",
                "heldout_residual": heldout_residual,
            },
            "chance_baseline": {
                "diagnostic_rows": chance_diagnostic_rows,
                "full_vocab": chance_full_vocab,
                "n_diagnostic_rows": int(len(diag_rows)),
                "vocab_size": int(dep.vocab_size),
            },
            "applicable_variant": applicable_variant,
            # Under the KW baseline (deployment weights only) the token permutation is not merely
            # hard: it is unobservable. Every relabelling of the vocabulary is an equivalent model
            # for an attacker holding nothing but the obfuscated checkpoint, so no statistic of
            # those weights can name a token. Writing 0% here would be as wrong as writing 99.5%.
            "kw_only": {
                "status": "not_applicable",
                "reason": (
                    "if the attacker had the obfuscated weights *without* the plain model, the token "
                    "permutation would be unobservable (any relabelling is an equivalent model). "
                    "AloePri's threat model does not do that -- App. D.1 Table 9 hands the attacker "
                    "both -- so this row documents the counterfactual, not the paper's setting."),
            },
            "paper_working_point": paper_working_point,
            "threat_model": {
                "clean_model_public": bool(clean_model_public),
                "paper_literal_rowsort": (
                    "uses the clean reference table AND the secret permutation to build the "
                    "aligned pair -> diagnostic, not an attack"),
                "scale_normalised_rowsort": "same construction -> diagnostic",
                "dense_mixing_inversion": (
                    "requires the (plain, obfuscated) weight pair, which AloePri's App. D.1 Table 9 "
                    "also assumes (Known X / Observed Y) -- in-model, not a KP key pair"),
                "not_assumed": "key matrices, tau, noise seed (all secret)",
            },
        },
        diagnostics={
            "candidate_mode": "full",
            "positive_control": {
                "name": "known_row_column_permutation_and_known_dense_transform",
                "variant": applicable_variant,
                "top1": control_top1,
                "required_top1": 0.99,
                "passed": control_passed,
                "protocol_floor": 0.99,
                "verdict_source": control_reason,
                "rowsort_control": rowsort_control,
                "dense_control": dense_control,
                "scaled_control": scaled_control,
            },
            "rowsort_control": rowsort_control,
            "dense_control": dense_control,
            "scaled_control": scaled_control,
            "positive_control_passed": control_passed,
            "attack_top1": headline_top1,
            "attack_top1_definition": (
                "measured top-1 of the applicable variant only; None when no "
                "variant's premise holds"
            ),
            "chance": chance,
            "chance_baseline": {
                "diagnostic_rows": chance_diagnostic_rows,
                "full_vocab": chance_full_vocab,
                "n_diagnostic_rows": int(len(diag_rows)),
                "vocab_size": int(dep.vocab_size),
            },
            "chance_by_variant": dict(chance_by_variant),
            "ratio_over_chance": ((headline_top1 / chance)
                                  if (headline_top1 is not None and chance) else None),
            "ratio_over_chance_by_variant": {
                name: ((score / chance_by_variant[name])
                       if (score is not None and chance_by_variant.get(name))
                       else None)
                for name, score in variant_scores.items()
            },
            "variant_top1": dict(variant_top1),
            "variant_scores": dict(variant_scores),
            "stage": "stage-2",
            "attack_variant": "stage-2 (token permutation recovery, known weight pair)",
            "uses_known_pairs_for_attack": False,
            "uses_plain_and_obfuscated_weights": True,
            "paper_working_point": paper_working_point,
            "attack_uses_secret_permutation": False,
            "diagnostic_variants_use_secret_permutation": list(diagnostic_variants),
            "row_sort_uses_secret_permutation": True,
            "paper_working_point": paper_working_point,
            "variants": {
                "paper_literal_rowsort": (
                    "raw sorted rows on a secret-permutation-aligned pair; "
                    "oracle-aligned diagnostic, not an attack"
                ),
                "scale_normalised_rowsort": (
                    "per-column standardisation then sorted-row match on a "
                    "secret-permutation-aligned pair; oracle-aligned diagnostic, "
                    "not an attack"
                ),
                "dense_mixing_inversion": "solve W' = W M from q_proj across layers, un-mix E', KNN",
            },
            "applicable_variant": applicable_variant,
            "alpha_e": alpha_e,
            "note": ("The two RowSort rows use secret-aligned diagnostic inputs. "
                     "The public dense-mixing attempt is an engineering variant, not a full "
                     "AloePri Table 9 implementation; alignment statistics do not gate its result."),
            "paper_parameter_undisclosed": True,
        },
        ci={},
        n_prompts=0, n_tokens=0, n_keys=0, n_types=int(len(ref_types)), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "row_sort + dense_mixing",
                "scenario": scenario},
        data={"n_types": int(len(ref_types)), "seed": seed, "scenario": scenario},
        overrides={
            # Protocol evidence, not a main-table row: the read-only audit decides
            # eligibility, and this payload does not claim it. In-model for AloePri
            # (plain + obfuscated weights), so uses_known_pairs stays False; the
            # row-sort rows stay diagnostics because their *construction* uses tau.
            "uses_known_pairs": False,
        },
    )
    payload["attack"] = "vma.dense_mixing_weight_probe"
    payload["attack_id"] = "least_squares"
    payload["paper"] = "single-sided dense-mixing engineering variant; not AloePri Table 9"
    # The layers the mixing solve actually used (derived from recover_mixing's
    # per-matrix records), plus any layer missing a required tensor.
    payload["layers"] = used_layers
    payload["missing_weight_layers"] = missing_layers
    payload["diagnostics"]["mixing_layers"] = used_layers
    payload["diagnostics"]["mixing_group_residuals"] = dict(mix.get("per_matrix") or {})
    payload["diagnostics"]["mixing_skipped_groups"] = list(
        mix.get("skipped_groups") or []
    )
    payload["diagnostics"]["mixing_groups"] = (
        "self_attn.{q,k,v}_proj.weight for every available layer "
        "(dense_mixing.HIDDEN_INPUT_GROUPS)"
    )
    payload["adapter_scenario"] = scenario
    # Permission record: the headline (dense) attack never reads the secret
    # permutation; the RowSort rows do, and are marked oracle diagnostics.
    payload["attack_uses_secret_permutation"] = False
    payload["diagnostic_variants_use_secret_permutation"] = list(diagnostic_variants)
    write_payload("C06", "vma.dense_mixing_weight_probe", "aloepri_weights", payload,
                  {"seconds": round(time.time() - t0, 1)})
    if verbose:
        chance_text = "n/a" if chance is None else f"{chance:.4f}"
        print(f"  C06W literal={literal['mapping_recovery']} "
              f"scale_norm={scaled['mapping_recovery']:.3f} "
              f"dense_mixing={dense['accuracy']:.4f} (M residual {dense_fit_residual:.4f}, "
              f"chance {chance_text}, {round(time.time()-t0,1)}s)", flush=True)
    return {"precondition": pre, "paper_literal": literal, "scale_normalised": scaled,
            "dense_mixing": dense}


def _token_permutation_subset(dep: ABDeployment, unmixed: np.ndarray,
                              ref_types: np.ndarray) -> dict:
    """NN map deployment row -> clean id for ``ref_types``, scored against the secret perm."""
    ref = np.asarray(ref_types, dtype=np.int64)
    queries = np.asarray(dep.E_clean, dtype=np.float32)[ref]
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    table = unmixed / (np.linalg.norm(unmixed, axis=1, keepdims=True) + 1e-12)
    hits = 0
    for start in range(0, len(ref), 256):
        block = qn[start:start + 256]
        nn = (block @ table.T).argmax(axis=1)
        hits += int(np.sum(nn == dep.perm[ref[start:start + 256]]))
    return {"accuracy": hits / max(1, len(ref)), "n_types": int(len(ref))}


def run_c08(dep: ABDeployment, *, n_types: int = 1000, seed: int = 0, verbose: bool = True,
            clean_model_public: bool = True, profile: str = "probe",
            protocol_sha256: str = "", scenario: str = "S-public") -> dict:
    """Gate-IA over ALL public deployment rows (F01); truth only in the scorer."""
    from .real_weight_attacks import gate_public_candidates

    t0 = time.time()
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        clean_model_public = False
    if not clean_model_public:
        return {
            "status": "not_run",
            "scenario": scenario,
            "reason": (
                "clean reference weights are not public in scenario "
                f"{scenario!r}"
            ),
        }
    rng = np.random.default_rng(seed)
    vocab = int(len(dep.perm))
    types = np.sort(rng.choice(vocab, size=min(int(n_types), vocab), replace=False))
    public = dep.public_view("aloepri", scenario)
    out = gate_public_candidates(public, types, top_k=max(KS))
    truth = dep.perm[types]  # scoring side only
    hits = {}
    for k in KS:
        matches = np.array(
            [int(truth[i]) in out["candidates"][int(token)][:k]
             for i, token in enumerate(types)]
        )
        hits[k] = float(matches.mean()) if matches.size else None
    per_unit = [
        {
            "unit": int(token),
            "hits": int(int(truth[index]) in out["candidates"][int(token)][:1]),
            "count": 1,
        }
        for index, token in enumerate(types)
    ]
    alpha_e = dep.secret_metadata.get("alpha_e")
    # In-cell positive control: the same gate-invariant pipeline on a known
    # construction (clean weights + a known row permutation). A number without
    # a passed in-cell control can never be a main-table row (S04).
    control = _gate_known_construction_control(public, types, seed=seed)
    control_passed = bool(control.get("passed"))
    paper_working_point = {
        "paper_alpha_e": 1.0,
        "paper_alpha_h": 0.2,
        "artifact_alpha_e": dep.secret_metadata.get("alpha_e"),
        "artifact_alpha_h": dep.secret_metadata.get("alpha_h"),
        "invariant_exact_at": "alpha_e = 0 (App. D.1); broken by sufficient embedding noise",
    }
    payload = result_payload(
        attack_name="ia.gate",
        scheme_name="ab_deployment:aloepri_weights",
        target="llama-3.2-1b/aloepri_weights",
        metrics={
            "mapping_asr": {
                f"top{k}": {
                    "value": value,
                    "numerator": int(round(value * len(types))) if value is not None else 0,
                    "denominator": int(len(types)),
                }
                for k, value in hits.items()
            },
            "precondition": {"alpha_e": alpha_e,
                             "invariant": "Avg(e W_gate) preserved only for alpha_e = 0"},
            "candidate_source": "all public deployment rows",
            "n_candidates": out["n_candidates"],
        },
        diagnostics={
            "candidate_mode": "full",
            "positive_control": control,
            "positive_control_passed": control_passed,
            "layers": [int(layer) for layer in out["layers"]],
            "missing_weight_layers": list(out.get("missing_weight_layers") or []),
            "paper_working_point": paper_working_point,
            "clean_model_public": bool(clean_model_public),
            "uses_known_pairs_for_attack": False,
            "uses_plain_and_obfuscated_weights": True,
            "note": ("Gate-IA's equality holds for the noiseless covariant construction. "
                     f"This measurement uses artifact alpha_e={alpha_e}."),
            "mapping_direction": "clean_to_deployment",
        },
        ci={},
        n_prompts=0, n_tokens=0, n_keys=0, n_types=int(len(types)), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "gate_invariant_public",
                "scenario": scenario},
        data={"n_types": int(len(types)),
              "layers": [int(layer) for layer in out["layers"]],
              "missing_weight_layers": list(out.get("missing_weight_layers") or []),
              "scenario": scenario},
        overrides={"uses_known_pairs": False},
    )
    payload["attack_id"] = "gate_ia"
    payload["adapter_scenario"] = scenario
    write_payload("C08", "ia.gate", "aloepri_weights", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "profile": profile, "protocol_sha256": protocol_sha256,
                   "per_unit_predictions": per_unit})
    if verbose:
        print(f"  C08 Gate-IA top1={hits[1]:.4f} top10={hits[10]:.4f} "
              f"({round(time.time()-t0,1)}s, alpha_e={alpha_e})", flush=True)
    return {"gate_ia": hits,
            "layers": [int(layer) for layer in out["layers"]],
            "missing_weight_layers": list(out.get("missing_weight_layers") or []),
            "positive_control": control}


# ---------------------------------------------------------------------------
# D3 composed chain: stage-1 (KNN on z) o stage-2 (weight-recovered permutation)
# ---------------------------------------------------------------------------
def recover_mapping_gate_ia(
    dep: ABDeployment, *, n_types: int = 4000, top_k: int = 3, seed: int = 0,
    scenario: str = "S-public",
) -> dict:
    """Stage-2 mapping ``deployment id -> ranked plaintext ids`` from deployment weights.

    Implements plan P0-5 (Gate-IA) at the size the plan asks for (>=1,000 types; the
    caller may raise it toward the full vocabulary). Returns the ranked mapping plus
    the type coverage this mapping gives over the evaluated token stream.
    """
    from .real_weight_attacks import recover_mapping_gate_ia as public_gate_mapping

    if scenario not in CLEAN_ACCESS_SCENARIOS:
        raise ValueError(
            "Gate-IA stage-2 mapping needs the clean reference weights; scenario "
            f"{scenario!r} does not expose them (allowed: "
            f"{sorted(CLEAN_ACCESS_SCENARIOS)})"
        )
    public = dep.public_view("aloepri", scenario)
    out = public_gate_mapping(public, n_types=n_types, top_k=top_k, seed=seed)
    mapping: dict[int, list[int]] = {
        int(deployment_row): [int(clean_id)]
        for deployment_row, clean_id in out["mapping"].items()
    }
    # Scoring only: compare each recovered mapping entry with the true inverse permutation.
    correct = sum(
        1 for deployment_row, candidates in mapping.items()
        if candidates and candidates[0] == int(dep.inv_perm[deployment_row])
    )
    return {
        "mapping": mapping,
        "n_types": int(out["n_types"]),
        "top1_correct": int(correct),
        "top1_rate": float(correct / max(1, len(mapping))),
        "candidate_source": "all public deployment rows",
        "mapping_candidates": out["mapping_candidates"],
        "collision_rows": out["collision_rows"],
        "selection_rule": out["selection_rule"],
        "layers": [int(layer) for layer in out.get("layers") or []],
        "missing_weight_layers": list(out.get("missing_weight_layers") or []),
    }


def _compose_dense_candidates(stage1_candidates: np.ndarray, deployment_table: np.ndarray,
                              transformed_clean: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map public stage-one candidates through the recovered dense transform."""
    from ..attacks.vma.dense_mixing import forward_match_stats

    candidates = np.asarray(stage1_candidates, dtype=np.int64)
    if candidates.ndim != 2 or not candidates.size:
        raise ValueError("stage-one candidates must be a non-empty matrix")
    if np.any((candidates < 0) | (candidates >= len(deployment_table))):
        raise ValueError("stage-one candidate outside the public deployment table")
    needed, inverse = np.unique(candidates, return_inverse=True)
    predicted, cosine, margin = forward_match_stats(np.asarray(deployment_table)[needed], transformed_clean)
    return predicted[inverse].reshape(candidates.shape), {
        "n_rows": int(len(needed)), "cosine_mean": float(np.mean(cosine)),
        "margin_mean": float(np.mean(margin)),
        "mapping": {int(row): int(token) for row, token in zip(needed, predicted)},
    }


def _composed_dense_control(*, seed: int) -> dict:
    """Exercise the same solver, KNN, mapping and scorer on a known construction."""
    from types import SimpleNamespace
    from ..attacks.vma.dense_mixing import recover_mixing

    rng = np.random.default_rng(int(seed))
    vocab, width = 128, 12
    clean = rng.normal(size=(vocab, width)).astype(np.float32)
    transform, _ = np.linalg.qr(rng.normal(size=(width, width)))
    perm = rng.permutation(vocab)
    deployment = np.empty_like(clean)
    deployment[perm] = clean @ transform
    weights = {}
    for group in ("q_proj", "k_proj", "v_proj"):
        matrix = rng.normal(size=(width, width))
        name = f"model.layers.0.self_attn.{group}.weight"
        weights[name] = {"clean": matrix, "deployment": matrix @ transform}
    mix = recover_mixing(SimpleNamespace(n_layers=1), getter=lambda name, side: weights[name][side])
    transformed = clean @ np.asarray(mix["M"], dtype=np.float32)
    targets = np.arange(vocab, dtype=np.int64)
    wire = deployment[perm[targets]]
    stage1 = knn_topk(wire, deployment, k=min(max(KS), vocab), metric="cosine")
    composed, _ = _compose_dense_candidates(stage1, deployment, transformed)
    stage1_score = per_position_asr(stage1, perm[targets])["top1"]
    combined_score = per_position_asr(composed, targets)["top1"]
    return {"name": "synthetic_dense_full_composition", "n_positions": vocab,
            "stage1_top1": stage1_score["value"], "top1": combined_score["value"],
            "numerator": combined_score["numerator"], "denominator": combined_score["denominator"],
            "required_top1": 0.99, "passed": bool(combined_score["value"] >= 0.99),
            "pipeline": "weight solve -> wire KNN -> public forward mapping -> scoring"}


def run_composed(dep: ABDeployment, texts: Sequence[str], ref_counts: dict[int, int], *,
                 n_types: int = 4000, seed: int = 0, verbose: bool = True,
                 scenario: str = "S-public") -> dict:
    """Plan D3: stage-1, stage-2 and the composed end-to-end chain, with full coverage.

    stage-1 = KNN over the real wire against the deployment table (recover the permuted id);
    stage-2 = the token permutation recovered from the weights by the **applicable** variant
    for this artifact (``dense_mixing_inversion``: solve ``W' = W M`` from q_proj and match
    ``E'`` against the forward image ``E M``; this avoids forming an ill-conditioned inverse),
    which covers every deployment id instead of a 20k-type sample;
    composed = the two chained.

    The oracle-tau row is kept as the upper bound so the composed number can be read against
    the stage-1 ceiling: if composed ~= stage-1, stage-2 is not the bottleneck; if it is far
    below, stage-2 is.
    """
    from ..attacks.vma.dense_mixing import (
        forward_match,
        forward_match_stats,
        recover_mixing,
    )

    if scenario not in CLEAN_ACCESS_SCENARIOS:
        raise ValueError(f"composed dense attack requires public clean weights; scenario {scenario!r} does not provide them")
    if len(texts) == 0 or int(n_types) < 1:
        raise ValueError("composed attack needs non-empty texts and a positive type budget")
    t0 = time.time()
    t_stage = time.time()
    mix = recover_mixing(dep, getter=dep.weight_tensor)
    if verbose:
        print(f"    mixing solve (4 x q_proj lstsq): {time.time()-t_stage:.1f}s", flush=True)
    transformed_clean = (dep.E_clean @ np.asarray(mix["M"], dtype=np.float32))
    premise_rng = np.random.default_rng(int(seed) + 20261107)
    premise_rows = np.sort(
        premise_rng.choice(len(dep.E_dep), size=min(256, len(dep.E_dep)), replace=False)
    )
    _, premise_cos, premise_margin = forward_match_stats(
        dep.E_dep[premise_rows], transformed_clean
    )
    premise_cos_mean = float(np.mean(premise_cos))
    premise_margin_mean = float(np.mean(premise_margin))
    # --- stage-1 on the wire ------------------------------------------------
    stage1_cands, oracle_cands, truth_plain, truth_perm, lengths = [], [], [], [], []
    for i, text in enumerate(texts):
        wire = dep.observe(text, arm="ab", request_id=i, key_seed=i)
        cand = knn_topk(np.asarray(wire.public.wire_vectors, dtype=np.float32), dep.table("ab"),
                        k=min(max(KS), len(dep.E_dep)), metric="cosine")
        stage1_cands.append(cand)
        oracle_cands.append(dep.inv_perm[cand])
        truth_plain.append(wire.truth.plaintext_ids)
        truth_perm.append(wire.truth.permuted_ids)
        lengths.append(len(wire.truth.plaintext_ids))
    C1 = np.concatenate(stage1_cands)
    Co = np.concatenate(oracle_cands)
    T = np.concatenate(truth_plain)
    Tp = np.concatenate(truth_perm)

    # --- stage-2 mapping for every id stage-1 can emit ----------------------
    composed_cands, mapping_evidence = _compose_dense_candidates(C1, dep.E_dep, transformed_clean)
    mapping = mapping_evidence["mapping"]
    # Sample observed IDs independently; consult truth after matching.
    rng = np.random.default_rng(seed)
    probe = np.sort(rng.choice(len(dep.E_dep), size=min(n_types, len(dep.E_dep)), replace=False))
    nn_clean = forward_match(dep.E_dep[probe], transformed_clean)
    stage2_top1 = float(np.mean(nn_clean == dep.inv_perm[probe]))

    stage1 = per_position_asr(C1, Tp)
    oracle = per_position_asr(Co, T)
    composed = per_position_asr(np.where(composed_cands < 0, 10 ** 9, composed_cands), T)
    covered = np.array([int(d) in mapping for d in C1[:, 0]])
    coverage = float(covered.mean()) if covered.size else None
    per_text_hits: list[int] = []
    per_text_counts: list[int] = []
    hit1 = composed_cands[:, 0] == T
    start = 0
    for n in lengths:
        block = hit1[start:start + n]
        per_text_hits.append(int(block.sum()))
        per_text_counts.append(int(n))
        start += n
    # S09: the reported point is micro (sum hits / sum positions). The interval
    # must resample texts and recompute that ratio; a macro interval around the
    # same point would answer the wrong estimand.
    composed_ci = ratio_ci(
        per_text_hits, per_text_counts,
        clusters=np.arange(len(per_text_hits)), seed=seed, unit="positions",
    )

    # --- real positive control: the same retrieval + scoring on the plaintext wire ---
    plain_cands = []
    for i, text in enumerate(texts):
        wire = dep.observe(text, arm="plaintext", request_id=i, key_seed=i)
        plain_cands.append(
            knn_topk(
                np.asarray(dep.E_clean[wire.public.wire_token_ids], dtype=np.float32),
                dep.E_clean, k=min(max(KS), len(dep.E_clean)), metric="cosine",
            )
        )
    pc_metrics = per_position_asr(np.concatenate(plain_cands), T)
    stage1_control_top1 = pc_metrics["top1"]["value"]
    control = _composed_dense_control(seed=seed)
    stage1_control_passed = bool(stage1_control_top1 is not None and stage1_control_top1 >= 0.99)
    control_passed = bool(control["passed"] and stage1_control_passed)
    import hashlib

    texts_sha256 = hashlib.sha256(json.dumps(list(texts), ensure_ascii=False,
                                              separators=(",", ":"), default=_json_default).encode("utf-8")).hexdigest()
    reference_sha256 = hashlib.sha256(json.dumps(sorted((int(key), int(value)) for key, value in ref_counts.items()),
                                                  separators=(",", ":")).encode("utf-8")).hexdigest()

    payload = result_payload(
        attack_name="vma.weight_rowsort",
        scheme_name="ab_deployment:ab_composed",
        target="llama-3.2-1b/ab",
        metrics={
            "stage1_permuted_id": stage1,
            "stage2_mapping_top1_rate": {
                "value": stage2_top1,
                "numerator": int(round(stage2_top1 * len(probe))),
                "denominator": int(len(probe)),
            },
            "stage2_source": "dense_mixing_inversion (W' = W M solved from q_proj, E' un-mixed)",
            "stage2_mixing_residual": mix["residual_mean"],
            "stage2_match_mode": "forward (E' vs E M)",
            "composed_end_to_end": composed,
            "candidate_coverage": {"value": coverage,
                                   "numerator": int(covered.sum()), "denominator": int(covered.size)},
            "oracle_tau_upper_bound": oracle,
            "random_baseline": {f"top{k}": min(k, dep.vocab_size) / dep.vocab_size for k in KS},
            "frequency_buckets_top1": bucket_asr(
                np.where(composed_cands < 0, 10 ** 9, composed_cands), T, frequency_buckets(ref_counts)),
        },
        diagnostics={
            "candidate_mode": "full",
            "clean_model_public": True,
            "positive_control": control,
            "stage1_plaintext_control": {"top1": stage1_control_top1, "n_prompts": len(texts),
                                         "required_top1": 0.99, "passed": stage1_control_passed},
            "positive_control_passed": control_passed,
            "stage": "composed(stage-1 KNN o stage-2 dense-mixing)",
            "stage2_premise": {
                "top1_cosine_mean": premise_cos_mean,
                "margin_mean": premise_margin_mean,
                "mixing_fit_residual": mix["residual_mean"],
                "mapping_cosine_mean": mapping_evidence["cosine_mean"],
                "mapping_margin_mean": mapping_evidence["margin_mean"],
                "decision": "diagnostic only; no unvalidated hard threshold",
            },
            "attack_variant": "dense_mixing_composed_engineering_variant",
            "stage2_types": int(len(mapping)),
            "stage2_seconds": None,
            "note": ("Dense hidden-axis matching is an engineering variant; low alignment "
                     "does not suppress its measured predictions or establish the paper algorithm."),
        },
        ci={"composed_top1_micro": composed_ci},
        n_prompts=len(lengths), n_tokens=int(T.size), n_keys=len(lengths),
        n_types=int(len(probe)), reference_tokens=int(sum(ref_counts.values())),
        source={"artifact": str(dep.artifact), "stage2": "dense_mixing", "scenario": scenario},
        data={"texts": len(texts), "texts_sha256": texts_sha256,
              "reference_counts_sha256": reference_sha256, "seed": int(seed),
              "stage2_probe_types": int(len(probe)), "scenario": scenario},
        overrides={
            # Matching public weights is not a target-request plaintext pair.
            "uses_known_pairs": False,
        },
    )
    payload["attack"] = "vma.dense_mixing_composed"
    payload["attack_id"] = "least_squares"
    payload["paper"] = "single-sided dense-mixing engineering variant; not AloePri Table 9"
    payload["scenario"] = scenario
    payload["layers"], payload["missing_weight_layers"] = _mixing_layer_coverage(mix, dep.n_layers)
    payload["diagnostics"]["mixing_group_residuals"] = dict(mix.get("per_matrix") or {})
    payload["diagnostics"]["mixing_skipped_groups"] = list(mix.get("skipped_groups") or [])
    write_payload("C06", "vma.dense_mixing_composed", "ab_composed", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": int(seed)})
    out = {
        "stage1_top1": stage1["top1"]["value"],
        "stage2_mapping_top1": stage2_top1,
        "composed_top1": composed["top1"]["value"],
        "composed_top10": composed["top10"]["value"],
        "oracle_top1": oracle["top1"]["value"],
        "coverage": coverage,
        "composed_ci": composed_ci,
        "n_texts": len(lengths),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  D3 stage1@1={out['stage1_top1']:.4f} stage2@1={out['stage2_mapping_top1']:.4f} "
              f"composed@1={out['composed_top1']:.4f} (coverage {coverage:.3f}) "
              f"({out['seconds']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# C10 / C12 - multi-request stream (plan S-C, SH-P vs SH-B)
# ---------------------------------------------------------------------------
def _localisation_accuracy(reqs, *, sample: int = 200, seed: int = 0) -> Optional[float]:
    """Nearest-neighbour localisation of a known shared token (plan §6.7).

    For each sampled position that shares its token with at least one other request,
    take the highest-cosine position *in a different request* and check that it is one of
    those same-token partners. That is the plan's "定位正确率"; reporting the exact-link
    rate here instead would just duplicate the symbol-equality result.
    """
    rng = np.random.default_rng(seed)
    flat = [(ri, pi) for ri, r in enumerate(reqs) for pi in range(len(r["plaintext_ids"]))]
    if len(flat) < 4:
        return None
    token_of = {(ri, pi): int(reqs[ri]["plaintext_ids"][pi]) for ri, pi in flat}
    # index positions by request for the cross-request search
    vecs = [np.asarray(r["vectors"], dtype=np.float32) for r in reqs]
    norms = [np.linalg.norm(v, axis=1) + 1e-12 for v in vecs]
    pick = rng.choice(len(flat), size=min(sample, len(flat)), replace=False)
    hits = trials = 0
    for idx in pick:
        ri, pi = flat[int(idx)]
        t = token_of[(ri, pi)]
        partners = {(rj, pj) for (rj, pj), tok in token_of.items() if tok == t and rj != ri}
        if not partners:
            continue
        q = vecs[ri][pi]
        best_cos, best_key = -2.0, None
        for rj in range(len(reqs)):
            if rj == ri:
                continue
            sims = (vecs[rj] @ q) / (norms[rj] * (np.linalg.norm(q) + 1e-12))
            j = int(np.argmax(sims))
            if float(sims[j]) > best_cos:
                best_cos, best_key = float(sims[j]), (rj, j)
        trials += 1
        if best_key in partners:
            hits += 1
    return float(hits / trials) if trials else None


def _texts_digest(texts: Sequence) -> str:
    """SHA-256 over the texts with explicit length framing.

    A NUL separator alone is ambiguous (``["a\\x00b"]`` and ``["a", "b"]`` would
    collide); each UTF-8 payload is prefixed with its 8-byte little-endian length.
    """
    import hashlib
    import struct

    hasher = hashlib.sha256()
    for text in texts:
        payload = str(text).encode("utf-8")
        hasher.update(struct.pack("<Q", len(payload)))
        hasher.update(payload)
    return hasher.hexdigest()


def _stream_sets_digest(sets: Sequence[Sequence]) -> str:
    """SHA-256 over a stream partition: stream count, text counts, framed texts.

    Flattening alone would let distinct partitions of the same tokens collide
    (``[["a"], ["b", "c"]]`` vs ``[["a", "b"], ["c"]]``); the structure is part
    of the content identity.
    """
    import hashlib
    import struct

    hasher = hashlib.sha256()
    hasher.update(struct.pack("<Q", len(sets)))
    for stream in sets:
        texts = list(stream)
        hasher.update(struct.pack("<Q", len(texts)))
        for text in texts:
            payload = str(text).encode("utf-8")
            hasher.update(struct.pack("<Q", len(payload)))
            hasher.update(payload)
    return hasher.hexdigest()


def _client_secret_commitment(dep) -> Optional[str]:
    """SHA-256 commitment to the client secret (never the secret itself).

    Prefers the artifact's ``client/client_secret.pt`` bytes; falls back to the
    in-memory key root + permutation bytes when the file is absent, so the run
    identity separates deployments rebuilt with a different client secret. A hash
    of high-entropy secret material is safe in the public identity; only the
    digest is emitted, never raw bytes or anything reversible.
    """
    import hashlib

    artifact = getattr(dep, "artifact", None)
    if artifact is not None:
        path = Path(artifact) / "client" / "client_secret.pt"
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1 << 20)
                    if not block:
                        break
                    hasher.update(block)
            return hasher.hexdigest()
    hasher = hashlib.sha256()
    found = False
    schedule = getattr(dep, "key_schedule", None)
    root = getattr(schedule, "_root", None) if schedule is not None else None
    if root is not None:
        hasher.update(b"root\x00")
        hasher.update(bytes(root))
        found = True
    perm = getattr(dep, "perm", None)
    if perm is not None:
        hasher.update(b"perm\x00")
        hasher.update(np.asarray(perm, dtype=np.int64).tobytes())
        found = True
    return hasher.hexdigest() if found else None


def _deployment_secret_digest(dep) -> bytes:
    """Opaque digest binding a stream cache to one deployment's key schedule.

    The key root and the secret permutation are hashed, never recorded or
    returned in clear; two deployments that share artifact/text/seed but derive
    different keys must not reuse each other's stream.
    """
    import hashlib

    hasher = hashlib.sha256()
    schedule = getattr(dep, "key_schedule", None)
    root = getattr(schedule, "_root", None) if schedule is not None else None
    if root is not None:
        hasher.update(b"key-root\x00")
        hasher.update(bytes(root))
    perm = getattr(dep, "perm", None)
    if perm is not None:
        hasher.update(b"permutation\x00")
        hasher.update(np.asarray(perm, dtype=np.int64).tobytes())
    return hasher.digest()


def _table_state_digest(dep) -> bytes:
    """SHA-256 of the loaded embedding tables, hashed fresh on every call.

    No memoization: an in-place mutation of ``E_dep``/``E_clean`` after a build
    must invalidate the cached stream. The cost is one full-table pass per
    cache-key computation (once per cell), which is acceptable.
    """
    import hashlib

    hasher = hashlib.sha256()
    for name in ("E_dep", "E_clean"):
        table = getattr(dep, name, None)
        if table is None:
            continue
        array = np.asarray(table, dtype=np.float32)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        hasher.update(name.encode("ascii"))
        hasher.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
        hasher.update(b"\x00")
        # Chunked to avoid a second full-table byte copy on the 1B artifact.
        for start in range(0, len(array), 8192):
            hasher.update(array[start:start + 8192].tobytes())
    return hasher.digest()


def _encoder_state_digest(dep) -> bytes:
    """SHA-256 of the in-memory encoder parameters, computed fresh each call.

    The encoder weights can be mutated after construction; hashing them on every
    cache-key computation keeps the stream cache sensitive to that mutation. The
    parameters are small next to the embedding tables.
    """
    import hashlib

    hasher = hashlib.sha256()
    encoder = getattr(dep, "_encoder", None)
    encoder_state = getattr(dep, "encoder_state", None)
    if encoder_state is None and encoder is not None:
        state_fn = getattr(encoder, "state_dict", None)
        if callable(state_fn):
            try:
                encoder_state = state_fn()
            except (RuntimeError, TypeError, AttributeError):
                encoder_state = None
    if isinstance(encoder_state, Mapping) and encoder_state:
        hasher.update(b"encoder-params\x00")
        for name in sorted(encoder_state):
            tensor = encoder_state[name]
            if hasattr(tensor, "detach"):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.asarray(tensor)
            array = np.ascontiguousarray(array)
            hasher.update(str(name).encode("utf-8"))
            hasher.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
            hasher.update(b"\x00")
            flat = array.ravel(order="C")
            for start in range(0, flat.size, 1 << 16):
                hasher.update(flat[start:start + (1 << 16)].tobytes())
    return hasher.digest()


def _artifact_state_digest(dep) -> bytes:
    """SHA-256 of the artifact files that shape the wire.

    Memoized together with a cheap stat fingerprint (relative label, size,
    mtime_ns) of every hashed file; any change or removal recomputes the digest.
    Known identity files are always hashed regardless of size (the Llama
    tokenizer.json is ~17 MB and would be skipped by the <=10 MB scan); every
    other small regular file under osnip/model_server is hashed too. Weight
    shards and large binaries (``*.safetensors`` or >10 MB) are skipped.
    """
    import hashlib

    artifact = getattr(dep, "artifact", None)
    if artifact is None:
        return hashlib.sha256(b"no-artifact").digest()
    artifact_root = Path(artifact)
    model_server = artifact_root / "model_server"
    osnip_dir = artifact_root / "osnip"
    known_paths: list[tuple[Path, str]] = []
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "config.json"):
        known_paths.append((model_server / name, f"model_server/{name}"))
    if model_server.is_dir():
        for model_path in sorted(model_server.glob("*.model")):
            known_paths.append((model_path, f"model_server/{model_path.name}"))
    known_paths.append((osnip_dir / "config.json", "osnip/config.json"))
    if osnip_dir.is_dir():
        for json_path in sorted(osnip_dir.glob("*.json")):
            known_paths.append((json_path, f"osnip/{json_path.name}"))
    announced: set[str] = set()
    known_entries: list[tuple[Path, str]] = []
    for path, label in known_paths:
        if label in announced:
            continue
        announced.add(label)
        known_entries.append((path, label))
    known_resolved = {str(path.resolve()) for path, _label in known_entries}
    small_entries: list[tuple[Path, str]] = []
    for directory in (osnip_dir, model_server):
        if not directory.is_dir():
            continue
        paths = sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(directory)),
        )
        for path in paths:
            if str(path.resolve()) in known_resolved:
                continue
            if path.name.endswith(".safetensors"):
                continue
            try:
                size = int(path.stat().st_size)
            except OSError:
                continue
            if size > 10 * 1024 * 1024:
                continue
            small_entries.append(
                (path, f"{directory.name}/{path.relative_to(directory).as_posix()}")
            )
    fingerprint: list[tuple[str, int, int]] = []
    for path, label in known_entries + small_entries:
        try:
            stat = path.stat()
            fingerprint.append((label, int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError:
            fingerprint.append((label, -1, 0))
    fingerprint_tuple = tuple(fingerprint)
    cached = getattr(dep, "_stream_artifact_digest", None)
    if (cached is not None
            and getattr(dep, "_stream_artifact_fingerprint", None) == fingerprint_tuple):
        return cached
    hasher = hashlib.sha256()

    def _hash_artifact_file(path: Path, label: str) -> None:
        hasher.update(b"artifact-file\x00")
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\x00")
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                hasher.update(block)

    hashed_known: set[str] = set()
    for path, label in known_entries:
        present = path.is_file()
        hasher.update(b"known-file\x00")
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\x01" if present else b"\x00")
        if present:
            _hash_artifact_file(path, label)
            hashed_known.add(str(path.resolve()))
    file_count = 0
    for path, label in small_entries:
        if str(path.resolve()) in hashed_known:
            continue
        _hash_artifact_file(path, label)
        file_count += 1
    hasher.update(b"artifact-file-count\x00")
    hasher.update(str(file_count).encode("ascii"))
    value = hasher.digest()
    try:
        setattr(dep, "_stream_artifact_digest", value)
        setattr(dep, "_stream_artifact_fingerprint", fingerprint_tuple)
    except (AttributeError, TypeError):
        pass
    return value


def _deployment_state_digest(dep) -> bytes:
    """SHA-256 over tables, artifact files and in-memory encoder parameters.

    The table and artifact parts are memoized (the artifact part with a stat
    fingerprint), while the encoder parameters are hashed fresh on every call, so
    post-construction mutation of the encoder or of encryptor/tokenizer files
    invalidates the stream cache instead of reusing a stale stream.
    """
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(b"tables\x00")
    hasher.update(_table_state_digest(dep))
    hasher.update(b"artifact\x00")
    hasher.update(_artifact_state_digest(dep))
    hasher.update(b"encoder\x00")
    hasher.update(_encoder_state_digest(dep))
    value = hasher.digest()
    try:
        setattr(dep, "_stream_state_digest", value)
    except (AttributeError, TypeError):
        pass
    return value


def _stream_cache_key(dep: ABDeployment, texts: Sequence[str], *, n_keys: int,
                      rotate_every: int, seed: int, profile: str,
                      protocol_sha256: str = "", session_ids: Optional[Sequence[str]] = None) -> str:
    import hashlib
    import struct

    digest = hashlib.sha256()
    digest.update(b"texts\x00")
    digest.update(bytes.fromhex(_texts_digest(texts)))
    digest.update(b"protocol\x00")
    protocol_bytes = str(protocol_sha256).encode("utf-8")
    digest.update(struct.pack("<Q", len(protocol_bytes)))
    digest.update(protocol_bytes)
    digest.update(b"deployment\x00")
    digest.update(_deployment_secret_digest(dep))
    digest.update(b"state\x00")
    digest.update(_deployment_state_digest(dep))
    digest.update(json.dumps({
        "artifact": str(dep.artifact),
        "clean_model": str(dep.clean_model_path),
        "n_keys": int(n_keys),
        "rotate_every": int(rotate_every),
        "seed": int(seed),
        "profile": str(profile),
        "max_len": dep.max_len,
        "session_ids": list(session_ids) if session_ids is not None else None,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def build_stream(dep: ABDeployment, texts: Sequence[str], *, n_keys: int = 16,
                 rotate_every: int = 64, seed: int = 0,
                 cache: Optional[dict] = None, profile: str = "probe",
                 protocol_sha256: str = "", session_ids: Optional[Sequence[str]] = None) -> dict:
    """Log the server-side public stream and the scorer-side truth separately (F11).

    * the public request record carries only what the server logs: request id,
      rotation epoch, opaque group id, session id, clustered symbols and vectors;
    * plaintext/permuted ids live in ``truth`` and are used only by the scorer;
    * the cache is keyed by the full public configuration, the protocol binding,
      the length-framed text content, an opaque digest of the deployment's key
      schedule (root/permutation hashed, never stored), and a digest of the loaded
      embedding tables, so a changed profile/protocol, key, weights or corpus
      rebuilds the stream;
    * key identity (opaque ``group_id``) and rotation ``epoch`` are separate fields.
    """
    if session_ids is not None and (len(session_ids) != len(texts) or any(not isinstance(item, str) or not item for item in session_ids)):
        raise ValueError("session_ids must contain one nonempty session ID per request")
    cache_key = _stream_cache_key(
        dep, texts, n_keys=n_keys, rotate_every=rotate_every, seed=seed,
        profile=profile, protocol_sha256=protocol_sha256, session_ids=session_ids,
    )
    if cache is not None and cache.get("stream_key") == cache_key and "stream" in cache:
        return cache["stream"]
    rng = np.random.default_rng(seed)
    key_indices = rng.integers(0, 2 ** 31 - 1, size=max(1, n_keys))
    requests = []
    truth = []
    for i, text in enumerate(texts):
        epoch = i // max(1, rotate_every)
        key_index = int(key_indices[epoch % len(key_indices)])
        wire = dep.observe(
            text, arm="ab", request_id=i, key_index=key_index, epoch=epoch,
            session_id=session_ids[i] if session_ids is not None else None,
        )
        if session_ids is not None and wire.public.session_id != session_ids[i]:
            raise ValueError("deployment did not preserve the supplied session identity")
        z = np.asarray(wire.public.wire_vectors, dtype=np.float32)
        # Plan §4.6/P0-8: for a continuous wire the attacker must *cluster* the vectors
        # into symbols first; quantising to 1e-4 is the symbol step.
        symbols = [_stable_symbol(np.round(z[p], 4).tobytes()) for p in range(z.shape[0])]
        requests.append({
            "request_id": i,
            "epoch": epoch,
            "group_id": wire.public.group_id,
            "session_id": wire.public.session_id,
            "session_source": "provided" if session_ids is not None else "unavailable",
            "symbols": symbols,
            "vectors": z,
        })
        truth.append({
            "plaintext_ids": wire.truth.plaintext_ids,
            "permuted_ids": wire.truth.permuted_ids,
        })
    n_pos = sum(len(row["plaintext_ids"]) for row in truth)
    out = {
        "requests": requests,
        "truth": truth,
        "stream_key": cache_key,
        "n_requests": len(requests),
        "n_positions": n_pos,
        "n_keys": int(len({r["group_id"] for r in requests})),
        "n_epochs": int(len(set(r["epoch"] for r in requests))),
        "n_symbols": int(len({s for r in requests for s in r["symbols"]})),
        "profile": profile,
    }
    if cache is not None:
        cache["stream_key"] = cache_key
        cache["stream"] = out
    return out


def _rank_window_candidates(
    group_symbols: dict[int, int],          # symbol -> count
    ref_counts: dict[int, int],
    *,
    window: int = 3,
    k: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> dict[int, list[int]]:
    """Exact nearest normalized frequency, with token-ID tie breaks."""
    top_k = int(k if k is not None else max(100, 2 * window + 1))
    return rank_candidates(group_symbols, ref_counts, top_k, vocab_size=vocab_size)


def _knn_deployment_ids(requests: Sequence[Mapping], table: np.ndarray, *,
                        chunk: int = 256) -> list[np.ndarray]:
    """Stage-1 KNN: nearest public deployment-table row for every wire vector.

    Uses only the public wire and the deployment table (no truth, key or secret).
    The table is not re-normalised into a second full copy; cosine similarity is
    computed chunk-wise as ``qn @ table.T / ||table_row||``. A zero or
    non-finite query vector has no direction: it is reported as ``-1``
    (unassigned), never silently mapped to table row 0 by ``argmax``.
    """
    table = np.asarray(table, dtype=np.float32)
    # The deployment table is the candidate universe: an invalid row cannot be
    # silently "repaired" or argmax would hand it a bogus id for every query.
    if not np.all(np.isfinite(table)):
        raise ValueError(
            "KNN deployment table contains non-finite values (NaN/inf); "
            "refusing to assign deployment ids from an undefined table"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        table_norms_raw = np.linalg.norm(table, axis=1)
    if not np.all(np.isfinite(table_norms_raw)):
        raise ValueError(
            "KNN deployment table contains a non-finite row norm; refusing to "
            "assign deployment ids"
        )
    if np.any(table_norms_raw <= 0):
        raise ValueError(
            "KNN deployment table contains an all-zero row (zero norm); cosine "
            "assignment is undefined"
        )
    table_norms = table_norms_raw + 1e-12
    out: list[np.ndarray] = []
    for request in requests:
        vectors = np.asarray(request["vectors"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(norms) & (norms > 0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            query = vectors / np.maximum(norms, 1e-12)[:, None]
        query = np.where(valid[:, None], query, 0.0).astype(np.float32)
        ids = np.full(len(query), -1, dtype=np.int64)
        for start in range(0, len(query), max(1, int(chunk))):
            stop = min(start + max(1, int(chunk)), len(query))
            block_valid = valid[start:stop]
            if not bool(block_valid.any()):
                continue
            sims = query[start:stop] @ table.T
            block_ids = (sims / table_norms[None, :]).argmax(axis=1)
            block_ids[~block_valid] = -1
            ids[start:stop] = block_ids
        out.append(ids)
    return out


def _tfma_group_counts(requests: Sequence[Mapping], mode: str) -> dict:
    """Symbol counts per attack group.

    ``SH-P`` groups by the **public key identity** ``group_id``: the same key
    reused across rotation epochs accumulates in one bucket (``epoch`` is
    rotation metadata only). ``SH-B`` pools every request into one bucket; that
    raw-symbol pooling is a diagnostic and is only meaningful if symbol ids are
    key-invariant (see :func:`_symbol_invariance`).
    """
    groups: dict = {}
    for request in requests:
        gid = request.get("group_id") if mode == "SH-P" else -1
        bucket = groups.setdefault(gid, {})
        for symbol in request["symbols"]:
            bucket[symbol] = bucket.get(symbol, 0) + 1
    return groups


def _symbol_invariance(stream: Mapping) -> dict:
    """Premise check for cross-key raw-symbol pooling (scoring-side diagnostic).

    Truth is used only to pair occurrences of the same plaintext token across
    keys; the attack's candidate tables never read it. ``premise_holds`` is
    measured from the artifact's observed symbols, never assumed.
    """
    token_symbols: dict[int, dict] = {}
    for request, truth in zip(stream["requests"], stream["truth"]):
        group = request.get("group_id")
        for token, symbol in zip(truth["plaintext_ids"], request["symbols"]):
            token_symbols.setdefault(int(token), {}).setdefault(group, set()).add(int(symbol))
    agreements = comparisons = 0
    for by_group in token_symbols.values():
        groups = sorted(by_group, key=str)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                comparisons += 1
                agreements += int(bool(by_group[groups[i]] & by_group[groups[j]]))
    n_groups = len({request.get("group_id") for request in stream["requests"]})
    if comparisons == 0:
        # Nothing was compared: the premise is not measured, so it must not be
        # reported as holding (the audit requires measured=True).
        rate = None
        holds = False
        measured = False
        status = "vacuous_single_key" if n_groups <= 1 else "no_cross_key_token"
        reason = (
            "no cross-key token occurrence exists, so symbol invariance was not "
            "measured"
        )
    else:
        rate = agreements / comparisons
        holds = bool(rate >= 0.99)
        measured = True
        status = "measured"
        reason = None
    return {
        "status": status,
        "premise": (
            "raw symbol ids are key-specific; pooling symbols across keys is valid "
            "only if every token keeps its symbol under every key"
        ),
        "measured": measured,
        "reason": reason,
        "n_cross_key_token_pairs": int(comparisons),
        "symbol_agreement": rate,
        "premise_holds": holds,
    }


def _candidate_table_digest(table) -> str:
    return content_sha256({"candidate_tables": table})


def _partition_texts(texts: Sequence, *, streams: int = 3,
                     min_requests: int = 15000) -> Optional[list[list]]:
    """Split a protocol-scale corpus into disjoint near-equal streams.

    Round-robin distribution keeps the streams balanced (<=1 request apart).
    Below the request floor the run stays single-stream (``n_streams=1`` is
    recorded honestly, never padded to three).
    """
    values = list(texts)
    if len(values) < int(min_requests):
        return None
    buckets: list[list] = [[] for _ in range(max(1, int(streams)))]
    for index, value in enumerate(values):
        buckets[index % len(buckets)].append(value)
    return buckets


def _saturation_study_sha256(saturation: Mapping, saturation_curve,
                             *, run_data_sha256: str, seed: int,
                             corpus_sha256: str, deployment_state_sha256: str,
                             reference_counts_sha256: str) -> str:
    """Identity hash binding a saturation study to one run.

    Includes the run seed and data identity so the same curve/final value from a
    different seed (or corpus/deployment/reference table) cannot bind.
    """
    import hashlib

    sanitized = {
        key: value for key, value in saturation.items()
        if key not in {"data_sha256", "study_sha256"}
    }
    canonical = {
        "saturation": sanitized,
        "saturation_curve": saturation_curve,
        "corpus_sha256": corpus_sha256,
        "deployment_state_sha256": deployment_state_sha256,
        "reference_counts_sha256": reference_counts_sha256,
        "run_data_sha256": str(run_data_sha256),
        "seed": int(seed),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def run_c10(dep: ABDeployment, texts: Sequence[str], ref_counts: dict[int, int], *,
            n_keys: int = 16, rotate_every: int = 64, seed: int = 0, verbose: bool = True,
            cache: Optional[dict] = None, profile: str = "probe",
            protocol_sha256: str = "",
            saturation: Optional[Mapping] = None,
            saturation_curve: Optional[Mapping] = None,
            saturation_variant: Optional[str] = None,
            text_sets: Optional[Sequence[Sequence[str]]] = None) -> dict:
    """TFMA-embed (plan P0-8): embedding-level frequency attack.

    Three paths are written from the same public stream: ``SH-P`` groups by the
    public key identity (``group_id``; rotation epochs of the same key merge),
    ``SH-B`` is the raw-symbol pooling diagnostic (valid only under measured
    symbol invariance; it is metadata-free, so it records ``attacker_knowledge=SH-B``),
    and ``SH-ALIGN`` runs a legitimate public stage-1 KNN to a deployment id and
    accumulates frequency-ranked candidates per deployment id across keys,
    scoring stage-1 errors end to end.

    A formal C10 run must pass a measured ``saturation`` block (>=2 budget/top1
    points); without it the payload carries no saturation and the audit reports
    ``not_converged``. The block is never fabricated here.

    ``text_sets`` builds one independent stream per text set (protocol §8.2:
    >=3 streams) with the same seed policy as C12 (``seed + stream_id``) and
    attacks the combined public observations; ``None`` keeps the single-stream
    probe behaviour and records ``n_streams=1``.

    Order matters here: the **synthetic known-permutation control is computed first**, and
    its verdict decides ``main_table_eligible`` for the cells. Plan §7.2-1 says a number
    whose positive control failed may not enter the main table, and at this corpus size the
    control does fail (the reference corpus is 8.6x below the §6.6 floor), so the cells are
    written as protocol evidence rather than main-table rows.
    """
    t0 = time.time()
    # Multi-stream support (protocol: >=3 independent streams): one stream per
    # text set, same seed/n_keys/rotation policy as C12 (seed + stream_id).
    # ``text_sets=None`` keeps the single-stream probe behaviour and is recorded
    # as n_streams=1, never padded to look like three.
    sets = [list(texts)] if text_sets is None else [list(row) for row in text_sets]
    if not sets or all(not row for row in sets):
        raise ValueError("C10 needs at least one non-empty text set")
    requests: list[dict] = []
    truth: list[dict] = []
    stream_summaries: list[dict] = []
    for stream_id, stream_texts in enumerate(sets):
        stream = build_stream(
            dep, stream_texts, n_keys=n_keys, rotate_every=rotate_every,
            seed=seed + stream_id, profile=profile,
            protocol_sha256=protocol_sha256,
            cache=cache if (cache is not None and len(sets) == 1) else None,
        )
        for record, truth_row in zip(stream["requests"], stream["truth"]):
            public = dict(record)
            public["stream_id"] = stream_id
            requests.append(public)
            truth.append(truth_row)
        stream_summaries.append({
            "stream_id": stream_id,
            "n_requests": stream["n_requests"],
            "n_positions": stream["n_positions"],
            "n_keys": stream["n_keys"],
            "n_epochs": stream["n_epochs"],
            # Per-stream type/symbol counts: the protocol floor is >=1000 types
            # *per stream*, which an aggregate count would mask.
            "n_types": int(len({
                int(token)
                for row in stream["truth"]
                for token in row["plaintext_ids"]
            })),
            "n_symbols": int(stream["n_symbols"]),
            "stream_key": stream["stream_key"],
            "profile": profile,
        })
    n_streams = len(sets)
    stream = {
        "requests": requests,
        "truth": truth,
        "n_requests": len(requests),
        "n_positions": int(sum(len(row["plaintext_ids"]) for row in truth)),
        "n_keys": int(len({record.get("group_id") for record in requests})),
        "n_epochs": int(len({record.get("epoch") for record in requests})),
        "n_symbols": int(len({sym for record in requests for sym in record["symbols"]})),
    }
    corpus_texts = [text for row in sets for text in row]
    # Content identity: two corpora/partitions with equal request or key counts
    # must not collide in data_sha256/run_id.
    corpus_sha256 = _stream_sets_digest(sets)
    deployment_state_sha256 = _deployment_state_digest(dep).hex()
    client_secret_sha256 = _client_secret_commitment(dep)
    # The reference frequency table determines the candidate rankings; two
    # corpora with the same token total but different frequencies must not share
    # a data_sha256/run_id.
    import hashlib
    import struct

    reference_hasher = hashlib.sha256()
    for token in sorted(ref_counts):
        entry = f"{int(token)}:{int(ref_counts[token])}".encode("ascii")
        reference_hasher.update(struct.pack("<Q", len(entry)))
        reference_hasher.update(entry)
    reference_counts_sha256 = reference_hasher.hexdigest()
    # The payload data identity is fixed before any candidate scoring; a
    # saturation study must commit to exactly this identity (same seed, corpus,
    # deployment and reference table) to be bindable.
    payload_data = {
        "requests": stream["n_requests"], "keys": stream["n_keys"],
        "rotate_every": rotate_every,
        "n_streams": int(n_streams),
        "stream_partition": ("multi" if text_sets is not None else "single"),
        "stream_sizes": [int(len(row)) for row in sets],
        "n_texts": int(len(corpus_texts)),
        "corpus_sha256": corpus_sha256,
        "reference_counts_sha256": reference_counts_sha256,
        "artifact": str(dep.artifact),
        "deployment_state_sha256": deployment_state_sha256,
        "client_secret_sha256": client_secret_sha256,
        "n_keys_requested": int(n_keys),
        "seed": int(seed),
    }
    run_data_sha256 = hashlib.sha256(
        json.dumps(payload_data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    # --- synthetic known-permutation positive control (plan §6.6 / §7.1) -----
    rng = np.random.default_rng(seed)
    fake_perm = rng.permutation(dep.vocab_size)
    sym_counts: dict[int, int] = {}
    control_rows: list[list[int]] = []
    for r, t in zip(stream["requests"], stream["truth"]):
        syms = [_stable_symbol(b"synthetic:" + int(fake_perm[token]).to_bytes(8, "little"))
                for token in t["plaintext_ids"]]
        for s in syms:
            sym_counts[s] = sym_counts.get(s, 0) + 1
        control_rows.append(syms)
    control_table = _rank_window_candidates(sym_counts, ref_counts, vocab_size=dep.vocab_size)
    control_cands = [[control_table.get(s, []) for s in syms] for syms in control_rows]
    control_truth = np.concatenate([np.asarray(t["plaintext_ids"]) for t in stream["truth"]])
    control_arr = np.asarray([(c + [-1] * max(KS))[:max(KS)] for row in control_cands for c in row],
                             dtype=np.int64)
    control = per_position_asr(control_arr, control_truth)
    control_pass = bool(control["top1"]["value"] is not None and control["top1"]["value"] >= 0.90)

    results: dict[str, dict] = {"positive_control": {"top1": control["top1"]["value"],
                                                     "top10": control["top10"]["value"],
                                                     "passed": control_pass,
                                                     "required_top1": 0.90}}
    # S06: SH-P groups by the public key identity (group_id), not by rotation
    # epoch. A key reused in several epochs accumulates in one bucket.
    invariance = _symbol_invariance(stream)
    cand_by_group: dict[str, dict] = {}
    for mode in ("SH-P", "SH-B"):
        groups = _tfma_group_counts(stream["requests"], mode)
        cand_by_group[mode] = {
            gid: _rank_window_candidates(counts, ref_counts, vocab_size=dep.vocab_size)
            for gid, counts in groups.items()
        }

    # S06: SH-ALIGN -- legitimate public stage-1 KNN maps every observed vector
    # to a deployment id (no truth/key/secret), then frequency-rank candidates
    # per deployment id across keys. Stage-1 errors stay in the denominator.
    deployment_ids = _knn_deployment_ids(stream["requests"], dep.table("ab"))
    aligned_counts: dict[int, int] = {}
    unassigned_positions = 0
    for ids in deployment_ids:
        for dep_id in np.asarray(ids).reshape(-1).tolist():
            if int(dep_id) < 0:
                # -1 is the "no direction" sentinel, never a table row: it must
                # not index the last deployment id (Python negative indexing).
                unassigned_positions += 1
                continue
            aligned_counts[int(dep_id)] = aligned_counts.get(int(dep_id), 0) + 1
    aligned_table = _rank_window_candidates(aligned_counts, ref_counts, vocab_size=dep.vocab_size)
    aligned_rows: list[list[list[int]]] = []
    secret_perm = getattr(dep, "perm", None)
    stage1_hits = stage1_total = 0
    for request, truth, ids in zip(stream["requests"], stream["truth"], deployment_ids):
        id_list = [int(value) for value in np.asarray(ids).reshape(-1).tolist()]
        aligned_rows.append([
            aligned_table.get(dep_id, []) if dep_id >= 0 else []
            for dep_id in id_list
        ])
        truth_ids = np.asarray(truth["plaintext_ids"], dtype=np.int64)
        stage1_total += int(truth_ids.size)
        if secret_perm is not None:      # scoring side only
            stage1_hits += int(sum(
                1 for dep_id, token in zip(id_list, truth_ids.tolist())
                if dep_id >= 0 and dep_id == int(secret_perm[int(token)])
            ))

    path_specs = {
        "SH-P": {
            "rows": None,
            "scheme": "ab_SH-P",
            "grouping": ("grouped by public group_id (key identity); the same key "
                         "across rotation epochs is one bucket, epoch is metadata only"),
            "n_groups": len(cand_by_group["SH-P"]),
            "extra_metrics": {},
            "extra_diagnostics": {
                "epoch_is_rotation_metadata_only": True,
                "symbol_invariance": invariance,
            },
        },
        "SH-B": {
            "rows": None,
            "scheme": "ab_SH-B",
            "grouping": "raw-symbol pooling across keys (diagnostic only)",
            "n_groups": 1,
            "extra_metrics": {},
            "extra_diagnostics": {
                "diagnostic_only": True,
                "symbol_invariance": invariance,
                "premise": invariance,
                "premise_note": (
                    "raw symbol ids are key-specific; pooling them across keys is "
                    "valid only under symbol invariance, which is measured from the "
                    "artifact above and not assumed"
                ),
            },
        },
        "SH-ALIGN": {
            "rows": aligned_rows,
            "scheme": "ab_SH-ALIGN",
            "grouping": ("public stage-1 KNN deployment ids; symbols/vectors are "
                         "accumulated per deployment id across keys"),
            "n_groups": len(aligned_table),
            "exclude_unassigned_coverage": True,
            "unassigned_positions": int(unassigned_positions),
            "extra_metrics": {
                "stage1_deployment_id_top1": {
                    "value": ((stage1_hits / stage1_total)
                              if (stage1_total and secret_perm is not None) else None),
                    "numerator": int(stage1_hits),
                    "denominator": int(stage1_total),
                    "note": ("stage-1 KNN hits over all positions; errors stay in the "
                             "end-to-end denominator"
                             + ("" if secret_perm is not None else
                                "; secret permutation unavailable on this fixture, "
                                "stage-1 scoring disabled")),
                },
            },
            "extra_diagnostics": {
                "alignment": {
                    "stage1": "KNN over the public wire vectors against the public deployment table",
                    "aggregation": "frequency-ranked candidates per deployment id, accumulated across keys",
                    "uses_secret": False,
                    "n_deployment_ids": len(aligned_table),
                    "unassigned_positions": int(unassigned_positions),
                    "stage1_error_rate": ((1.0 - stage1_hits / stage1_total)
                                          if (stage1_total and secret_perm is not None)
                                          else None),
                },
                "symbol_invariance": invariance,
            },
        },
    }

    def _mode_candidate_rows(mode: str):
        spec = path_specs[mode]
        cand_rows, truth_rows = [], []
        for index, (request, truth) in enumerate(zip(stream["requests"], stream["truth"])):
            if spec["rows"] is None:
                gid = request.get("group_id") if mode == "SH-P" else -1
                table = cand_by_group[mode][gid]
                req_cands = [table.get(symbol, []) for symbol in request["symbols"]]
            else:
                req_cands = spec["rows"][index]
            cand_rows.append(req_cands)
            truth_rows.append(truth["plaintext_ids"])
        return cand_rows, truth_rows

    modes = ("SH-P", "SH-B", "SH-ALIGN")
    mode_rows = {mode: _mode_candidate_rows(mode) for mode in modes}
    mode_top1: dict[str, Optional[float]] = {}
    for mode, (cand_rows, truth_rows) in mode_rows.items():
        flat_cands = [c for row in cand_rows for c in row]
        flat_truth = np.concatenate([np.asarray(t) for t in truth_rows])
        arr = np.asarray(
            [(c + [-1] * max(KS))[:max(KS)] for c in flat_cands], dtype=np.int64
        )
        mode_top1[mode] = per_position_asr(arr, flat_truth)["top1"]["value"]
    # A supplied saturation study must describe the occurrence-level curve of a
    # variant this run actually measured. Validate before writing any payload so
    # a mismatched study can never leave a partially written cell. The study is
    # attached ONLY to the variant it is bound to: an ambiguous tie without an
    # explicit ``saturation_variant`` binds none, so one cell's earlier points
    # can never be validated against another variant's curve.
    saturation_modes: set[str] = set()
    saturation_binding: dict = {"status": "not_supplied"}
    saturation_identity: dict = {}
    if saturation is not None:
        points = (saturation.get("points")
                  if isinstance(saturation, Mapping) else None)
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError(
                "saturation must be a mapping with at least one measured point"
            )
        try:
            float(points[-1]["budget"])
            final_saturation_top1 = float(points[-1]["top1"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(
                "saturation final point must carry numeric budget and top1"
            ) from None
        # The study must commit to THIS run's data identity (seed + data hash):
        # final-top1 equality alone would let a different-seed study bind.
        expected_study_sha256 = _saturation_study_sha256(
            saturation, saturation_curve,
            run_data_sha256=run_data_sha256, seed=int(seed),
            corpus_sha256=corpus_sha256,
            deployment_state_sha256=deployment_state_sha256,
            reference_counts_sha256=reference_counts_sha256,
        )
        supplied_study_sha256 = saturation.get("study_sha256")
        supplied_data_sha256 = saturation.get("data_sha256")
        if supplied_study_sha256 is None and supplied_data_sha256 is None:
            raise ValueError(
                "saturation study must carry study_sha256 or data_sha256 tied to "
                "this run's data identity; refusing to write"
            )
        if (supplied_study_sha256 is not None
                and str(supplied_study_sha256) != expected_study_sha256):
            raise ValueError(
                "saturation study identity (study_sha256) does not match this "
                "run's corpus/deployment/reference identity; refusing to write"
            )
        if (supplied_data_sha256 is not None
                and str(supplied_data_sha256) != run_data_sha256):
            raise ValueError(
                "saturation study data_sha256 does not match this run's data "
                "identity; refusing to write"
            )
        study_identity_record = {
            "study_sha256": expected_study_sha256,
            "data_sha256": run_data_sha256,
        }
        saturation_identity = dict(study_identity_record)
        matching = {
            mode for mode, top1 in mode_top1.items()
            if top1 is not None
            and abs(final_saturation_top1 - float(top1)) <= 1e-9
        }
        if saturation_variant is not None:
            if saturation_variant not in modes:
                raise ValueError(
                    f"unknown saturation_variant {saturation_variant!r}; "
                    f"expected one of {list(modes)}"
                )
            if saturation_variant not in matching:
                raise ValueError(
                    f"saturation final point top1 {final_saturation_top1} does "
                    f"not match the computed occurrence-level top1 of "
                    f"{saturation_variant!r} ({mode_top1.get(saturation_variant)}); "
                    "refusing to write"
                )
            saturation_modes = {saturation_variant}
            saturation_binding = {
                "status": "bound", "variant": saturation_variant,
                "method": "explicit saturation_variant",
            }
        elif len(matching) == 1:
            saturation_modes = set(matching)
            saturation_binding = {
                "status": "bound", "variant": next(iter(matching)),
                "method": "unique final-top1 match",
            }
        elif len(matching) > 1:
            # Ambiguous tie: bind none rather than certifying every variant
            # against one study (the audit could otherwise cross-validate curves).
            saturation_modes = set()
            saturation_binding = {
                "status": "ambiguous", "candidates": sorted(matching),
                "reason": (
                    "final top1 matches several variants; pass saturation_variant "
                    "to choose the certified one"
                ),
            }
        else:
            raise ValueError(
                "saturation final point top1 "
                f"{final_saturation_top1} does not match any computed "
                f"occurrence-level top1 {mode_top1}; refusing to write"
            )

    for mode in modes:
        spec = path_specs[mode]
        cand_rows, truth_rows = mode_rows[mode]
        per_request_hits: list[int] = []
        per_request_counts: list[int] = []
        for req_cands, truth in zip(cand_rows, truth_rows):
            per_request_hits.append(int(sum(
                1 for candidates, token in zip(req_cands, truth)
                if candidates and int(candidates[0]) == int(token)
            )))
            per_request_counts.append(int(len(truth)))
        per_unit = []
        for unit_index, (req_cands, truth) in enumerate(zip(cand_rows, truth_rows)):
            truth_ids = [int(token) for token in np.asarray(truth).reshape(-1).tolist()]
            ranks: list[int] = []
            predicted: list[int] = []
            for candidates, token in zip(req_cands, truth_ids):
                cand_list = [int(value) for value in (candidates or [])]
                predicted.append(int(cand_list[0]) if cand_list else -1)
                if token in cand_list:
                    ranks.append(int(cand_list.index(token)) + 1)
                else:
                    # Absent from the truncated Top-100 candidate set: rank > 100
                    # so the audit's top-k reconciliation matches per_position_asr.
                    ranks.append(max(len(cand_list) + 1, max(KS) + 1))
            per_unit.append({
                "unit": int(unit_index),
                "request": int(unit_index),
                "hits": int(sum(1 for value in ranks if value <= 1)),
                "count": int(len(truth_ids)),
                "rank": ranks,
                "predicted": predicted,
                "truth": truth_ids,
            })
        flat_cands = [c for row in cand_rows for c in row]
        flat_truth = np.concatenate([np.asarray(t) for t in truth_rows])
        arr = np.asarray([(c + [-1] * max(KS))[:max(KS)] for c in flat_cands], dtype=np.int64)
        metrics = per_position_asr(arr, flat_truth)
        # Truth coverage: the fraction of positions whose TRUE token appears in
        # the reported candidate list. A non-empty list of wrong candidates is
        # not coverage; candidate-list presence is reported separately.
        # Unassigned positions remain in both denominators as misses.
        if spec.get("exclude_unassigned_coverage"):
            assigned_mask = [
                dep_id >= 0
                for ids in deployment_ids
                for dep_id in np.asarray(ids).reshape(-1).tolist()
            ]
        else:
            assigned_mask = [True] * len(flat_cands)
        coverage_denominator = len(flat_cands)
        truth_hits = sum(
            1 for cand, token, flag in zip(
                flat_cands, flat_truth.tolist(), assigned_mask)
            if flag and int(token) in [int(value) for value in cand]
        )
        presence_hits = sum(
            1 for cand, flag in zip(flat_cands, assigned_mask) if flag and cand
        )
        coverage = (truth_hits / coverage_denominator) if coverage_denominator else None
        presence_coverage = (
            (presence_hits / coverage_denominator) if coverage_denominator else None
        )
        # S09: interval resamples the independent unit and recomputes
        # sum(hits)/sum(positions); never a mean of per-request rates. SH-P keeps
        # the public key identity as the cluster (requests sharing a key are not
        # independent); the pooled/alignment paths cluster by request.
        if mode == "SH-P":
            clusters = [request.get("group_id") for request in stream["requests"]]
        else:
            clusters = list(range(len(per_request_hits)))
        ci = ratio_ci(
            per_request_hits, per_request_counts,
            clusters=clusters, seed=seed, unit="positions",
        )
        prior_id = min(ref_counts, key=lambda token: (-ref_counts[token], int(token)))
        prior_hits = [int(np.count_nonzero(np.asarray(row) == int(prior_id))) for row in truth_rows]
        prior_total = sum(prior_hits)
        prior_denominator = sum(per_request_counts)
        prior_ci = ratio_ci(prior_hits, per_request_counts, clusters=clusters, seed=seed)
        gain_ci = paired_gain_ci(per_request_hits, prior_hits, per_request_counts,
                                 clusters=clusters, seed=seed)
        prior_rows = [{"unit": i, "count": int(count), "hits": int(hits),
                       "predicted": int(prior_id)}
                      for i, (count, hits) in enumerate(zip(per_request_counts, prior_hits))]
        results[mode] = {
            "top1": metrics["top1"]["value"],
            "top5": metrics["top5"]["value"],
            "top10": metrics["top10"]["value"],
            "top100": metrics["top100"]["value"],
            "coverage": coverage,
            "candidate_presence_coverage": presence_coverage,
            "n_groups": spec["n_groups"],
            "n_positions": int(flat_truth.size),
            "ci_top1": ci,
            "group_candidates_sha256": _candidate_table_digest(
                aligned_table if mode == "SH-ALIGN" else cand_by_group[mode]
            ),
        }
        if mode == "SH-ALIGN":
            scorable = bool(stage1_total and secret_perm is not None)
            results[mode]["stage1_top1"] = (
                stage1_hits / stage1_total if scorable else None
            )
            results[mode]["stage1_error_rate"] = (
                1.0 - stage1_hits / stage1_total if scorable else None
            )
        payload_metrics = {
            "occurrence_level": metrics,
            "candidate_coverage": {
                "value": coverage,
                "numerator": int(truth_hits),
                "denominator": int(coverage_denominator),
                "definition": ("fraction of all positions whose true token appears "
                               "in the reported candidate list"),
            },
            "candidate_presence_coverage": {
                "value": presence_coverage,
                "numerator": int(presence_hits),
                "denominator": int(coverage_denominator),
                "definition": ("fraction of all positions with a non-empty "
                               "candidate list"),
            },
            "random_baseline": {f"top{k}": k / dep.vocab_size for k in KS},
            "topk_construction": (
                "full vocabulary ranked by absolute normalized frequency difference; "
                "ties use ascending token ID, including zero-frequency tokens"
            ),
            "prior_baseline": {
                "method": "most frequent public reference token; ties by token ID",
                "predicted": int(prior_id), "numerator": prior_total,
                "denominator": prior_denominator,
                "value": prior_total / prior_denominator if prior_denominator else None,
                "ci": prior_ci, "paired_gain": gain_ci, "per_unit": prior_rows,
            },
            **spec["extra_metrics"],
        }
        # Measured candidate length over ASSIGNED positions only: a -1 sentinel
        # row has an empty list by construction and must not drag the minimum
        # (and therefore top100_distinct) down to 0.
        assigned_lengths = [
            len(candidates)
            for candidates, flag in zip(flat_cands, assigned_mask) if flag
        ]
        measured_top100 = int(min(assigned_lengths, default=0))
        # Distinctness is verified, not assumed: a truncated list length says
        # nothing if the list repeats ids. Top-100 claims require unique ids.
        duplicate_lists = sum(
            1 for cand in flat_cands
            if len({int(value) for value in cand}) != len(cand)
        )
        candidate_lists_unique = bool(duplicate_lists == 0)
        diagnostics = {
            "candidate_mode": "full",
            "positive_control": {"name": "synthetic_known_permutation",
                                 "top1": control["top1"]["value"], "required_top1": 0.90},
            "positive_control_passed": control_pass,
            "attack_variant": "embedding_level",
            "rotation_grouping": mode,
            "grouping": spec["grouping"],
            "n_symbols": stream["n_symbols"],
            "n_groups": spec["n_groups"],
            "n_streams": int(n_streams),
            "per_stream": stream_summaries,
            "keys_per_prompt": 1,
            "distinct_rotation_keys": int(stream["n_keys"]),
            "top100_distinct": measured_top100,
            "candidate_lengths_min": measured_top100,
            "candidate_lists_unique": candidate_lists_unique,
            "candidate_lists_with_duplicates": int(duplicate_lists),
            "per_unit_scope": "one row per request; truth used for scoring only",
            "below_strength_floor": True,
            "floor": "reference >=4.45M tokens, >=20k occurrences, three priors (§6.6)",
            "note": ("Occurrence-level TFMA on the real wire with the full ranked reference "
                     "vocabulary as the candidate set. The synthetic control fails at this corpus "
                     "size, so these cells are protocol evidence, not main-table rows."),
            "paper_parameter_undisclosed": True,
            **spec["extra_diagnostics"],
        }
        if saturation is not None:
            diagnostics["saturation_binding"] = {
                **saturation_binding, **saturation_identity,
            }
        if mode in saturation_modes:
            # Never fabricated: the measured study is attached ONLY to the
            # variant it is bound to (never copied across variants).
            diagnostics["saturation"] = dict(saturation)
            diagnostics["saturation_variant"] = mode
            if saturation_curve is not None:
                diagnostics["saturation_curve"] = dict(saturation_curve)
        knowledge, rotation_aware = {
            "SH-P": ("SH-P", True),
            "SH-B": ("SH-B", False),
            "SH-ALIGN": ("SH", False),
        }[mode]
        # The auditor binds the saturation ladder to a measured reference-token
        # record; the top-level payload value alone is not a measurement record.
        diagnostics["reference_tokens"] = int(sum(ref_counts.values()))
        diagnostics["reference_types"] = int(len(ref_counts))
        payload = result_payload(
            attack_name="tfma.independent",
            scheme_name=f"ab_deployment:{spec['scheme']}",
            target=f"llama-3.2-1b/{spec['scheme']}",
            metrics=payload_metrics,
            diagnostics=diagnostics,
            ci={"top1_micro": ci},
            n_prompts=stream["n_requests"], n_tokens=int(flat_truth.size),
            # One request = one prompt observed under exactly one key; the number
            # of *distinct rotation keys* is recorded in diagnostics below.
            n_keys=1, n_types=int(len(set(flat_truth.tolist()))),
            reference_tokens=int(sum(ref_counts.values())),
            source={"artifact": str(dep.artifact), "stream": "ab"},
            data=payload_data,
            overrides={
                "main_table_eligible": bool(control_pass),
                # SH-B pools raw symbols across keys and is not protocol-aware;
                # SH-ALIGN needs no rotation metadata either.
                "attacker_knowledge": knowledge,
                "rotation_aware": rotation_aware,
            },
        )
        # S06/TFMA: the payload must carry the measured minimum candidate-list
        # length, never a hardcoded 100. If the reference vocabulary is smaller,
        # the audit must see the small measured value and flag it.
        payload["top100_distinct"] = measured_top100
        # The AB deployment is a vector surface: the canonical audit id is
        # tfma_after_inversion (paper_profile rejects "tfma" for vector arms).
        payload["attack_id"] = "tfma_after_inversion"
        payload["attack_variant"] = mode
        payload["n_streams"] = int(n_streams)
        # result_payload hardcodes main_table_eligible=False and ignores the
        # override. Record the control verdict explicitly instead of leaving the
        # flag implicit. The payload schema forbids a run from self-certifying
        # eligibility (validate_payload refuses True), so the persisted flag is
        # False and eligibility stays with the read-only audit; the audit hint
        # below carries the control verdict.
        payload["main_table_eligible"] = False
        payload["main_table_eligible_audit_hint"] = bool(control_pass)
        write_payload("C10", "tfma.independent", mode, payload,
                      {"seconds": round(time.time() - t0, 1),
                       "per_unit_predictions": per_unit,
                       "profile": profile,
                       "protocol_sha256": protocol_sha256})
        if verbose:
            print(f"  C10 TFMA {mode} top1={results[mode]['top1']:.4f} "
                  f"top10={results[mode]['top10']:.4f} cov={coverage:.3f} "
                  f"groups={spec['n_groups']} ({stream['n_requests']} req, "
                  f"{stream['n_keys']} keys)", flush=True)

    results["stream"] = {
        k: stream[k]
        for k in ("n_requests", "n_positions", "n_keys", "n_epochs", "n_symbols")
    }
    results["stream"]["n_streams"] = int(n_streams)
    results["symbol_invariance"] = invariance
    results["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  C10 positive control (known random permutation) top1={control['top1']['value']:.4f} "
              f"passed={control_pass}", flush=True)
    return results


def run_c06h(dep: ABDeployment, texts: Sequence[str], *, n_propose: int = 400,
             max_positions: int = 6, layers: Sequence[int] = (5,), seed: int = 0,
             full_vocab: bool = False, variant: str = "recovered",
             pose: str = "empty", bos_id: Optional[int] = None,
             keys_per_prompt: int = 1,
             profile: str = "probe", protocol_sha256: str = "",
             scenario: str = "S-public",
             verbose: bool = True) -> dict:
    """VMA-H on the public wire (F04).

    ``variant="recovered"`` is the paper's autoregressive recovered-prefix attack
    (attack id ``vma_h_recovered_prefix``); ``variant="observed"`` is the stronger
    public-observation variant (``vma_h_observed_prefix``). ``full_vocab=True`` is
    the paper candidate set; the probe default remains a labelled proposal.
    ``keys_per_prompt`` repeats each text under fresh derived keys; the paper floor
    is 200 prompts x 5 keys.
    """
    from .real_neural import stopword_definition, stopword_mask
    from .real_vma import score_vma, vma_h_observed, vma_h_recovered

    if int(keys_per_prompt) < 1:
        raise ValueError("keys_per_prompt must be >= 1")
    t0 = time.time()
    if variant not in {"recovered", "observed"}:
        raise ValueError(f"variant must be 'recovered' or 'observed'; got {variant!r}")
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        # The plaintext positive control needs the clean model and weights; the
        # cell cannot be run honestly in a deployment-only scenario.
        raise ValueError(
            "run_c06h needs the public clean model for its plaintext control; "
            f"scenario {scenario!r} does not expose it (allowed: "
            f"{sorted(CLEAN_ACCESS_SCENARIOS)})"
        )
    wires = []
    wire_prompt: list[int] = []
    wire_key: list[int] = []
    for prompt_index, text in enumerate(texts):
        for key_offset in range(int(keys_per_prompt)):
            wire_index = prompt_index * int(keys_per_prompt) + key_offset
            wires.append(
                dep.observe(text, arm="ab", request_id=wire_index, key_index=wire_index)
            )
            wire_prompt.append(prompt_index)
            wire_key.append(key_offset)
    public = dep.public_view("ab", scenario)
    attack = vma_h_recovered if variant == "recovered" else vma_h_observed
    if variant == "recovered":
        out = attack(
            public, [wire.public for wire in wires], layers=layers,
            full_vocab=full_vocab, n_propose=n_propose,
            max_positions=max_positions, top_k=max(KS), verbose=verbose,
            pose=pose, bos_id=bos_id,
        )
    elif pose != "empty" or bos_id is not None:
        raise ValueError("pose/bos_id apply only to the recovered-prefix variant")
    else:
        out = attack(
            public, [wire.public for wire in wires], layers=layers,
            full_vocab=full_vocab, n_propose=n_propose,
            max_positions=max_positions, top_k=max(KS), verbose=verbose,
        )
    ab_truth = [np.asarray(wire.truth.permuted_ids, dtype=np.int64)[:max_positions]
                for wire in wires]
    plain_truth = [np.asarray(wire.truth.plaintext_ids, dtype=np.int64)[:max_positions]
                   for wire in wires]
    masks = []
    for wire in wires:
        ids = wire.truth.plaintext_ids[:max_positions]
        masks.append(stopword_mask(dep, ids))
    scored = score_vma(out, ab_truth, layers=layers, clean_mask=masks)
    layer_key = f"layer{layers[0]}"
    per_unit = []
    token_total = 0
    for wire_index, (candidates, truth_row) in enumerate(
        zip(out["layers"][layer_key]["candidates"], ab_truth)
    ):
        count = int(len(truth_row))
        hits = sum(
            int(position < len(candidates) and candidates[position][0] == truth_row[position])
            for position in range(count)
        )
        per_unit.append({
            "unit": wire_index,
            "prompt": int(wire_prompt[wire_index]),
            "key": int(wire_key[wire_index]),
            "hits": int(hits),
            "count": int(count),
        })
        token_total += count

    plain_wires = [
        dep.observe(text, arm="plaintext", request_id=wire_index, key_index=wire_index)
        for prompt_index, text in enumerate(texts)
        for wire_index in range(
            prompt_index * int(keys_per_prompt),
            (prompt_index + 1) * int(keys_per_prompt),
        )
    ]
    plain_public = dep.public_view("plaintext", scenario)
    if variant == "recovered":
        control = attack(
            plain_public, [wire.public for wire in plain_wires], layers=layers,
            full_vocab=full_vocab, n_propose=n_propose, max_positions=max_positions,
            top_k=max(KS), verbose=False, pose=pose, bos_id=bos_id,
        )
    else:
        control = attack(
            plain_public, [wire.public for wire in plain_wires], layers=layers,
            full_vocab=full_vocab, n_propose=n_propose, max_positions=max_positions,
            top_k=max(KS), verbose=False,
        )
    control_scored = score_vma(control, plain_truth, layers=layers, clean_mask=masks)

    layer_key = f"layer{layers[0]}"
    layer_metrics = scored[layer_key]
    pc_top1 = control_scored[layer_key]["total_top1"]["value"]
    payload = result_payload(
        attack_name="vma.autoregressive",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={
            "occurrence_level": layer_metrics,
            "plaintext_control": control_scored[layer_key],
            "candidate_mode": "full" if full_vocab else "proposal",
            "n_propose": None if full_vocab else n_propose,
            "max_positions": max_positions,
            "prefix_mode": out["prefix_mode"],
            "pose": out.get("pose"),
            "bos_id": out.get("bos_id"),
            "clean_definition": stopword_definition(),
            "layers": list(layers),
        },
        diagnostics={
            "candidate_mode": "full" if full_vocab else "proposal",
            "positive_control": {
                "name": "plaintext_hidden_states",
                "top1": pc_top1,
                "required_top1": 0.99,
            },
            "positive_control_passed": bool(pc_top1 is not None and pc_top1 >= 0.99),
            "attack_variant": out["attack_id"],
            "uses_known_pairs": False,
            "paper_working_point": {
                "paper_alpha_e": 1.0, "paper_alpha_h": 0.2,
                "artifact_alpha_e": dep.secret_metadata.get("alpha_e"),
                "artifact_alpha_h": dep.secret_metadata.get("alpha_h"),
            },
        },
        ci={},
        n_prompts=len(texts), n_tokens=int(token_total),
        n_keys=int(keys_per_prompt),
        n_types=int(n_propose) if not full_vocab else int(dep.vocab_size),
        reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": out["attack_id"],
                "scenario": scenario},
        data={"texts": len(texts), "n_propose": n_propose, "max_positions": max_positions,
              "keys_per_prompt": int(keys_per_prompt),
              "observations": int(len(wires)),
              "scenario": scenario},
    )
    payload["attack_id"] = out["attack_id"]
    payload["adapter_scenario"] = scenario
    payload["keys_per_prompt"] = int(keys_per_prompt)
    payload["n_observations"] = int(len(wires))
    payload["layers"] = [int(layer) for layer in layers]
    payload["prefix"] = out["prefix_mode"]
    write_payload("C06", "vma.autoregressive", "ab", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "profile": profile, "protocol_sha256": protocol_sha256,
                   "per_unit_predictions": per_unit})
    if verbose:
        print(f"  C06-H ab {layer_key} ({out['attack_id']}): "
              f"top1={layer_metrics['total_top1']['value']:.3f} "
              f"({round(time.time()-t0,1)}s)", flush=True)
    return {"ab": out, "plaintext_control": control, "scored": scored}


def _known_row_permutation_view(public, types_array: np.ndarray,
                                rng: np.random.Generator, *,
                                corrupt: bool = False):
    """Known-construction view for the Gate-IA / Attn-IA in-cell controls.

    The deployment side is the clean side with a **known** token-row
    permutation of the embedding table and identical weights on both sides.
    Returns ``(view, truth_rows)`` where ``truth_rows[i]`` is the deployment
    row holding clean token ``types_array[i]`` under the construction's
    permutation. ``corrupt=True`` keeps the construction but scores against a
    different permutation, which the same pipeline cannot recover.
    """
    vocab = int(getattr(public, "vocab_size"))
    row_perm = rng.permutation(vocab)               # deployment row j <- clean row row_perm[j]
    scored_perm = rng.permutation(vocab) if corrupt else row_perm
    clean_table = np.asarray(public.table("plaintext"), dtype=np.float32)
    known_table = clean_table[row_perm]
    truth_rows = np.argsort(scored_perm)[types_array]

    class _KnownView:
        def __init__(self, base):
            object.__setattr__(self, "_base", base)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_base"), name)

        def table(self, arm, *args, **kwargs):
            base = object.__getattribute__(self, "_base")
            if str(arm) == str(getattr(base, "arm", "")):
                return known_table
            return base.table(arm, *args, **kwargs)

        def weight_tensor(self, name, side):
            base = object.__getattribute__(self, "_base")
            # Identity transform: both sides use the same clean weights.
            return base.weight_tensor(name, "clean" if str(side) == "deployment" else side)

    return _KnownView(public), truth_rows


def _attn_known_construction_control(public, types: Sequence[int], *,
                                     layers: Optional[Sequence[int]] = None,
                                     seed: int = 0,
                                     corrupt: bool = False) -> dict:
    """In-cell positive control for Attn-IA on a known construction.

    Builds a public view whose *deployment* side is the clean side with a known
    token-row permutation of the embedding table (identical clean weights on
    both sides), then runs the exact :func:`attn_public_candidates` invariant
    pipeline on it. The attack must recover the known row permutation at
    top-1 >= 0.99. ``corrupt=True`` scores against a different permutation,
    which the same pipeline cannot recover (used to prove the control can fail).
    """
    from .real_weight_attacks import attn_public_candidates

    types_array = np.asarray(types, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 20261111)
    view, truth_rows = _known_row_permutation_view(public, types_array, rng,
                                                   corrupt=corrupt)
    try:
        out = attn_public_candidates(view, types_array,
                                     layers=layers, top_k=1)
    except Exception as exc:  # noqa: BLE001 - a control failure must be recorded
        return {
            "name": "known_identity_weights_with_row_permutation",
            "top1": None,
            "required_top1": 0.99,
            "passed": False,
            "reason": f"control construction could not run: {type(exc).__name__}: {exc}",
        }
    hits = sum(
        1 for index, token in enumerate(types_array.tolist())
        if int(truth_rows[index]) in out["candidates"].get(int(token), [])[:1]
    )
    top1 = float(hits / len(types_array)) if len(types_array) else 0.0
    return {
        "name": "known_identity_weights_with_row_permutation",
        "top1": top1,
        "required_top1": 0.99,
        "passed": bool(top1 >= 0.99),
        "n_types": int(len(types_array)),
        "construction": (
            "deployment embedding table = clean table[known row permutation]; "
            "identical clean weights on both sides; same public invariant pipeline"
        ),
        "matches_recovered_construction": not bool(corrupt),
    }


def _gate_known_construction_control(public, types: Sequence[int], *,
                                     layers: Optional[Sequence[int]] = None,
                                     seed: int = 0,
                                     corrupt: bool = False) -> dict:
    """In-cell positive control for Gate-IA on a known construction (S04).

    Same known construction as :func:`_attn_known_construction_control` -- the
    deployment table is the clean table under a known row permutation and both
    sides share identical weights -- run through the exact
    :func:`gate_public_candidates` gate-invariant pipeline. The attack must
    recover the known row permutation at top-1 >= 0.99; ``corrupt=True`` scores
    against a different permutation the same pipeline cannot recover, proving
    the control can fail.
    """
    from .real_weight_attacks import gate_public_candidates

    types_array = np.asarray(types, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 20261112)
    view, truth_rows = _known_row_permutation_view(public, types_array, rng,
                                                   corrupt=corrupt)
    try:
        out = gate_public_candidates(view, types_array, layers=layers, top_k=1)
    except Exception as exc:  # noqa: BLE001 - a control failure must be recorded
        return {
            "name": "known_identity_weights_with_row_permutation",
            "top1": None,
            "required_top1": 0.99,
            "passed": False,
            "reason": f"control construction could not run: {type(exc).__name__}: {exc}",
        }
    hits = sum(
        1 for index, token in enumerate(types_array.tolist())
        if int(truth_rows[index]) in out["candidates"].get(int(token), [])[:1]
    )
    top1 = float(hits / len(types_array)) if len(types_array) else 0.0
    return {
        "name": "known_identity_weights_with_row_permutation",
        "top1": top1,
        "required_top1": 0.99,
        "passed": bool(top1 >= 0.99),
        "n_types": int(len(types_array)),
        "construction": (
            "deployment embedding table = clean table[known row permutation]; "
            "identical clean weights on both sides; same public invariant pipeline"
        ),
        "matches_recovered_construction": not bool(corrupt),
    }


def run_c08_attn(dep: ABDeployment, *, n_types: int = 1000, seed: int = 0,
                 layers: Optional[Sequence[int]] = None, verbose: bool = True,
                 clean_model_public: bool = True,
                 profile: str = "probe", protocol_sha256: str = "",
                 scenario: str = "S-public") -> dict:
    """Attn-IA on the real deployment weights (plan P0-6, AloePri App. D.1).

    ``layers=None`` lets the attack enumerate **all** applicable weight layers
    (0..n_layers-1); an explicit ``layers=`` sequence overrides it. The payload
    records the layers that actually participated, never the requested default.
    ``scenario`` is passed to the public view; a scenario without public clean
    weights forces ``clean_model_public=False`` and the cell is not run.

    The whitening invariant ``q_j G_j^-1 q_j^T`` over a RoPE block survives AloePri's attention
    obfuscation because the block transform cancels between ``q`` and the Gram ``G``. It is
    computed on the *plain* and the *obfuscated* side -- the same premise as App. D.1 (the paper
    shows the invariant between plaintext and obfuscated data) -- and matched per token.
    """
    from .real_weight_attacks import attn_public_candidates

    t0 = time.time()
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        clean_model_public = False
    if not clean_model_public:
        return {
            "status": "not_run",
            "scenario": scenario,
            "reason": (
                "clean reference weights are not public in scenario "
                f"{scenario!r}"
            ),
        }
    rng = np.random.default_rng(seed)
    vocab = int(len(dep.perm))
    types = np.sort(rng.choice(vocab, size=min(int(n_types), vocab), replace=False))
    public = dep.public_view("aloepri", scenario)
    out = attn_public_candidates(public, types, layers=layers, top_k=max(KS))
    # Skipped Gram blocks are evidence: a layer may still contribute invariant
    # columns while some of its blocks were singular/ill-conditioned. Record the
    # entries and annotate them per still-used layer instead of dropping them.
    skipped_blocks = [
        dict(entry) for entry in (out.get("skipped_gram_blocks") or [])
        if isinstance(entry, Mapping)
    ]
    skipped_by_layer: dict[str, list[dict]] = {}
    for entry in skipped_blocks:
        try:
            layer_number = int(entry.get("layer"))
        except (TypeError, ValueError, OverflowError):
            continue
        skipped_by_layer.setdefault(f"layer{layer_number}", []).append({
            "group": entry.get("group"),
            "head": entry.get("head"),
            "block": entry.get("block"),
            "condition": entry.get("condition"),
            "code": entry.get("code"),
            "reason": entry.get("reason"),
        })
    truth = dep.perm[types]  # scoring side only
    inv = {}
    for k in KS:
        matches = np.array(
            [int(truth[i]) in out["candidates"][int(token)][:k]
             for i, token in enumerate(types)]
        )
        inv[f"top{k}"] = {
            "value": float(matches.mean()) if matches.size else None,
            "numerator": int(matches.sum()),
            "denominator": int(matches.size),
        }
    per_unit = [
        {
            "unit": int(token),
            "hits": int(int(truth[index]) in out["candidates"][int(token)][:1]),
            "count": 1,
        }
        for index, token in enumerate(types)
    ]
    # In-cell positive control: the same invariant pipeline on a known
    # construction (clean weights + a known row permutation). A number without a
    # passed in-cell control can never be a main-table row.
    control = _attn_known_construction_control(
        public, types, layers=layers, seed=seed
    )
    control_passed = bool(control.get("passed"))
    working_point = {
        "paper_alpha_e": 1.0,
        "paper_alpha_h": 0.2,
        "artifact_alpha_e": dep.secret_metadata.get("alpha_e"),
        "artifact_alpha_h": dep.secret_metadata.get("alpha_h"),
        "invariant_exact_at": "alpha_e = 0 and no head/block permutation (App. D.1)",
    }
    payload = result_payload(
        attack_name="ia.attn",
        scheme_name="ab_deployment:aloepri_weights",
        target="llama-3.2-1b/aloepri_weights",
        metrics={
            "mapping_asr": inv,
            "n_invariant_columns": out.get("invariant_dims"),
            "invariant": "q_j G_j^-1 q_j^T over RoPE blocks (app. D.1), stacked over layers x heads x blocks",
            "pairing": out["pairing"],
            "layers": [int(layer) for layer in out["layers"]],
            "missing_weight_layers": list(out.get("missing_weight_layers") or []),
            "skipped_gram_blocks": skipped_blocks,
            "heads": int(public.n_heads),
            "paper_working_point": working_point,
        },
        diagnostics={
            "candidate_mode": "full",
            "positive_control": control,
            "positive_control_passed": control_passed,
            "stage": "stage-2",
            "attack_variant": "stage-2 (token permutation recovery, attention invariant)",
            "mapping_direction": "clean_to_deployment",
            "uses_known_pairs": False,
            "uses_plain_and_obfuscated_weights": True,
            "clean_model_public": bool(clean_model_public),
            "layers": [int(layer) for layer in out["layers"]],
            "missing_weight_layers": list(out.get("missing_weight_layers") or []),
            "skipped_gram_blocks": skipped_blocks,
            "skipped_gram_blocks_by_layer": skipped_by_layer,
            "paper_working_point": working_point,
            "note": ("The artifact's dense hidden-axis re-basing does not permute heads/RoPE blocks the "
                     "way Algorithm 2 does, so the invariant is expected to survive here; the paper's "
                     "defence against Attn-IA is the head/block permutation plus noise."),
        },
        ci={},
        n_prompts=0, n_tokens=0, n_keys=0, n_types=int(len(types)), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "whitening_invariant",
                "scenario": scenario},
        data={"n_types": int(len(types)),
              "layers": [int(layer) for layer in out["layers"]],
              "missing_weight_layers": list(out.get("missing_weight_layers") or []),
              "scenario": scenario},
    )
    payload["attack_id"] = "attn_ia"
    payload["adapter_scenario"] = scenario
    write_payload("C08", "ia.attn", "aloepri_weights", payload,
                  {"seconds": round(time.time() - t0, 1),
                   "profile": profile, "protocol_sha256": protocol_sha256,
                   "per_unit_predictions": per_unit})
    if verbose:
        print(f"  C08 Attn-IA top1={inv['top1']['value']:.4f} top10={inv['top10']['value']:.4f} "
              f"({round(time.time()-t0,1)}s, {out.get('invariant_dims')} invariant cols)", flush=True)
    return {"attn_ia": {k: v["value"] for k, v in inv.items()},
            "n_invariant_columns": out.get("invariant_dims"),
            "layers": [int(layer) for layer in out["layers"]],
            "missing_weight_layers": list(out.get("missing_weight_layers") or []),
            "skipped_gram_blocks": skipped_blocks,
            "skipped_gram_blocks_by_layer": skipped_by_layer,
            "positive_control": control,
            "positive_control_passed": control_passed}


def _linkage_control_passed(by_delta: dict) -> bool:
    """Plan §7.1 control for linkage: same-key/plaintext exact linking must be ~1.000.

    Matches the real producer output (``attacks/linkage/core.py``): buckets are
    labelled with the frozen ``DELTA_BUCKETS`` names and the rate is stored as
    ``link_rate``. The smallest gap (delta 1 -> label ``"1"``) is the same-key /
    adjacent-request control.
    """
    if not isinstance(by_delta, Mapping):
        return False
    record = by_delta.get(delta_bucket(1))
    if not isinstance(record, Mapping):
        return False
    rate = record.get("link_rate")
    try:
        # Protocol §7.1: the exactly-solvable same-key control must be ~1.000
        # (>=0.99), matching the pre-registered privacy-family gate.
        return bool(rate is not None and float(rate) >= 0.99)
    except (TypeError, ValueError, OverflowError):
        return False


def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if pos.size == 0 or neg.size == 0:
        return None
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # average ranks for ties
    sorted_scores = scores[order]
    start = 0
    for i in range(1, scores.size + 1):
        if i == scores.size or sorted_scores[i] != sorted_scores[start]:
            mean_rank = (start + 1 + i) / 2.0
            ranks[order[start:i]] = mean_rank
            start = i
    rank_sum = ranks[: pos.size].sum()
    u = rank_sum - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _rate_ci(hits: int, total: int) -> dict:
    if total == 0:
        return {"value": None, "numerator": 0, "denominator": 0, "lo": None, "hi": None}
    value = hits / total
    z = 1.959963984540054
    denom = 1.0 + z * z / total
    centre = (value + z * z / (2.0 * total)) / denom
    half = z * ((value * (1.0 - value) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
    return {
        "value": float(value),
        "numerator": int(hits),
        "denominator": int(total),
        "lo": float(max(0.0, centre - half)),
        "hi": float(min(1.0, centre + half)),
    }


def _session_identity(requests, index):
    request = requests[index]
    session = request.get("session_id")
    if session is None:
        session = ("unknown_session_request", int(index))
    return (request.get("stream_id"), session)


def _pair_clusters(requests, rows):
    """Endpoint session keys for each cross-request pair."""
    clusters = []
    for a, _pa, b, _pb, _token in rows:
        left = _session_identity(requests, a)
        right = _session_identity(requests, b)
        clusters.append((left, right))
    return clusters


def _session_pair_weights(counts, left, right):
    """Bootstrap weight of each pair under whole-session resampling.

    A cross-session pair (s,t) appears once per drawn copy of each endpoint, so
    its weight is ``counts[s] * counts[t]``. A within-session pair (s,s) exists
    once per drawn copy of that session, not once per ordered pair of copies, so
    its weight is ``counts[s]`` (the product would over-count it as m^2). The
    all-ones draw still reproduces the reported ``hits / total`` point exactly.
    """
    weights = counts[left] * counts[right]
    same = left == right
    if np.any(same):
        weights = np.array(weights, dtype=np.float64, copy=True)
        weights[same] = counts[left[same]]
    return weights


def _cluster_rate_ci(outcomes, clusters, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Rate of pair outcomes with a bootstrap over independent sessions.

    Linkage pairs are not independent observations: two pairs sharing a session
    share that session's randomness. Resampling whole sessions (and reweighting
    every pair by the product of its endpoints' resample multiplicities) keeps
    that dependence; resampling pair labels would treat it as independent.
    """
    outcomes = [bool(item) for item in outcomes]
    clusters = list(clusters)
    if len(outcomes) != len(clusters):
        raise ValueError("outcomes and clusters must align")
    total = len(outcomes)
    hits = int(sum(outcomes))
    value = (hits / total) if total else None
    session_pairs = bool(clusters) and all(
        isinstance(cluster, tuple) and len(cluster) == 2 for cluster in clusters
    )
    if total and (hits == 0 or hits == total):
        units = {session for pair in clusters for session in pair} if session_pairs else set(clusters)
        unknown_sessions = session_pairs and any(
            isinstance(session, tuple) and len(session) == 2
            and isinstance(session[1], tuple) and session[1][0] == "unknown_session_request"
            for session in units
        )
        return {"value": value, "numerator": hits, "denominator": total,
                "lo": None, "hi": None, "n_clusters": None if unknown_sessions else len(units), "n_boot": 0,
                "estimand": "degenerate outcomes; cluster interval unavailable",
                "empirical_resolution": 1.0 / total,
                "resolution_scope": "observed pair fraction; pairs are not independent samples",
                "reason": "no observed variation; bootstrap cannot establish zero risk"}
    if session_pairs:
        sessions = sorted({session for pair in clusters for session in pair}, key=str)
        if any(isinstance(session, tuple) and len(session) == 2
               and isinstance(session[1], tuple) and session[1][0] == "unknown_session_request"
               for session in sessions):
            return {"value": value, "numerator": hits, "denominator": total,
                    "lo": None, "hi": None, "n_clusters": None, "n_boot": 0,
                    "estimand": "session unavailable; no session confidence interval"}
        index = {session: position for position, session in enumerate(sessions)}
        left = np.array([index[a] for a, _ in clusters], dtype=np.int64)
        right = np.array([index[b] for _, b in clusters], dtype=np.int64)
        hit = np.array([int(outcome) for outcome in outcomes], dtype=np.float64)
        n_sessions = len(sessions)
        if total == 0 or n_sessions <= 1:
            # Unestimable design: no resampling variance, so no bounds -- a
            # certain [point, point] interval would overstate the evidence.
            return {"value": value, "numerator": hits, "denominator": total,
                    "lo": None, "hi": None, "n_clusters": n_sessions,
                    "n_boot": 0, "estimand": "session bootstrap"}
        rng = np.random.default_rng(seed)
        target = int(n_boot)
        # A draw can give an endpoint session multiplicity 0, making the pair
        # weight product 0 and the ratio undefined. Only finite replicates carry
        # information, so keep drawing (bounded at 20x the request) and report
        # the number of finite replicates actually used -- never a nominal 2000.
        max_attempts = max(target, 20 * target)
        finite_rates: list[float] = []
        attempts = 0
        while len(finite_rates) < target and attempts < max_attempts:
            attempts += 1
            counts = rng.multinomial(
                n_sessions, np.full(n_sessions, 1.0 / n_sessions)
            ).astype(np.float64)
            weights = _session_pair_weights(counts, left, right)
            denominator = float(weights.sum())
            if denominator <= 0:
                continue
            finite_rates.append(float((weights * hit).sum() / denominator))
        rates = np.asarray(finite_rates, dtype=np.float64)
        used_boot = int(rates.size)
        if used_boot < 2:
            # Fewer than two finite replicates cannot define a percentile
            # interval; report no interval and n_boot=0 (same rule as
            # metrics/intervals.py and ratio_ci).
            return {"value": value, "numerator": hits, "denominator": total,
                    "lo": None, "hi": None, "n_clusters": n_sessions,
                    "n_boot": 0, "estimand": "session bootstrap",
                    "bootstrap_attempts": int(attempts)}
        lo = float(np.percentile(rates, 2.5))
        hi = float(np.percentile(rates, 97.5))
        return {"value": value, "numerator": hits, "denominator": total,
                "lo": lo, "hi": hi, "n_clusters": n_sessions,
                "n_boot": used_boot, "estimand": "session bootstrap",
                "bootstrap_attempts": int(attempts)}
    unique = sorted(set(clusters))
    n_clusters = len(unique)
    if total == 0 or n_clusters <= 1:
        # Unestimable design: no bounds, consistent with metrics/intervals.py.
        return {"value": value, "numerator": hits, "denominator": total,
                "lo": None, "hi": None, "n_clusters": n_clusters,
                "n_boot": 0, "estimand": "pair-cluster bootstrap"}
    index = {cluster: position for position, cluster in enumerate(unique)}
    cluster_hits = np.zeros(n_clusters, dtype=np.float64)
    cluster_counts = np.zeros(n_clusters, dtype=np.float64)
    for outcome, cluster in zip(outcomes, clusters):
        position = index[cluster]
        cluster_hits[position] += int(outcome)
        cluster_counts[position] += 1
    target = int(n_boot)
    if target < 2:
        # Cannot form a percentile interval from fewer than two replicates.
        return {"value": value, "numerator": hits, "denominator": total,
                "lo": None, "hi": None, "n_clusters": n_clusters,
                "n_boot": 0, "estimand": "pair-cluster bootstrap"}
    rng = np.random.default_rng(seed)
    rates = np.empty(target, dtype=np.float64)
    for b in range(target):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        denominator = cluster_counts[drawn].sum()
        rates[b] = cluster_hits[drawn].sum() / denominator if denominator else np.nan
    finite = rates[np.isfinite(rates)]
    used_boot = int(finite.size)
    if used_boot < 2:
        return {"value": value, "numerator": hits, "denominator": total,
                "lo": None, "hi": None, "n_clusters": n_clusters,
                "n_boot": 0, "estimand": "pair-cluster bootstrap"}
    lo = float(np.percentile(finite, 2.5))
    hi = float(np.percentile(finite, 97.5))
    return {"value": value, "numerator": hits, "denominator": total,
            "lo": lo, "hi": hi, "n_clusters": n_clusters,
            "n_boot": used_boot, "estimand": "pair-cluster bootstrap"}


def _matched_linkage_pairs(requests, truth, *, seed: int = 0, max_pairs: int = 200_000) -> dict:
    """Sample position pairs uniformly without enumerating the pair population.

    Each repeated occurrence participates. For each positive pair, its matched
    negative replaces the right position by the nearest different token.
    Positive and negative populations are sampled independently without replacement.
    """
    import bisect

    capacity = int(max_pairs)
    if capacity < 0 or len(requests) != len(truth):
        raise ValueError("max_pairs must be non-negative and requests/truth must align")
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(row["plaintext_ids"], dtype=np.int64) for row in truth]
    postings: dict[int, list[tuple[int, list[int]]]] = {}
    for request, array in enumerate(arrays):
        positions: dict[int, list[int]] = {}
        for position, token in enumerate(array.tolist()):
            positions.setdefault(int(token), []).append(int(position))
        for token, offsets in positions.items():
            postings.setdefault(token, []).append((request, offsets))

    blocks = {"positive": [], "negative": []}
    cumulative = {"positive": [], "negative": []}
    totals = {"positive": 0, "negative": 0}
    for token, entries in sorted(postings.items()):
        prefix = [0]
        for _, offsets in entries:
            prefix.append(prefix[-1] + len(offsets))
        for index in range(1, len(entries)):
            request, offsets = entries[index]
            size = prefix[index] * len(offsets)
            for kind in ("positive", "negative"):
                if kind == "negative" and len(offsets) == len(arrays[request]):
                    continue
                totals[kind] += size
                blocks[kind].append((token, entries, prefix, index))
                cumulative[kind].append(totals[kind])

    def sample(kind):
        total = totals[kind]
        count = min(capacity, total)
        if count == total:
            selected = range(total)
        else:
            # Floyd's algorithm uses O(count) memory even when total is enormous.
            chosen = set()
            for end in range(total - count, total):
                draw = int(rng.integers(end + 1))
                chosen.add(end if draw in chosen else draw)
            selected = sorted(chosen)
        rows = []
        for rank in selected:
            block = bisect.bisect_right(cumulative[kind], rank)
            token, entries, prefix, index = blocks[kind][block]
            local = rank - (cumulative[kind][block - 1] if block else 0)
            b, offsets_b = entries[index]
            left_rank, right_rank = divmod(local, len(offsets_b))
            left_index = bisect.bisect_right(prefix, left_rank) - 1
            a, offsets_a = entries[left_index]
            pa = offsets_a[left_rank - prefix[left_index]]
            pb = offsets_b[right_rank]
            if kind == "negative":
                different = np.flatnonzero(arrays[b] != token)
                pb = int(different[np.argmin(np.abs(different - pb))])
            rows.append((a, pa, b, pb, token))
        return sorted(rows, key=lambda row: (row[0], row[2], row[4], row[1], row[3]))

    positives, negatives = sample("positive"), sample("negative")
    seen_positives, seen_negatives = totals["positive"], totals["negative"]
    return {
        "positives": positives,
        "negatives": negatives,
        "sampling": {
            "seed": int(seed),
            "capacity": capacity,
            "total_seen_positives": int(seen_positives),
            "total_seen_negatives": int(seen_negatives),
            "rule": (
                "uniform without replacement per class using indexed position-pair populations"
            ),
            "truncated_positives": bool(seen_positives > capacity),
            "truncated_negatives": bool(seen_negatives > capacity),
            "truncated": bool(seen_positives > capacity or seen_negatives > capacity),
            "positive_inclusion_probability": min(1.0, capacity / seen_positives) if seen_positives else None,
            "negative_inclusion_probability": min(1.0, capacity / seen_negatives) if seen_negatives else None,
            "population": "all same-token position pairs across distinct requests; negative uses nearest different-token position",
            "negative_duplicates": "distinct positive pairs can map to the same negative endpoints; their sampling multiplicity is retained",
        },
    }


def _cosine(req_a, pos_a, req_b, pos_b) -> float:
    va = np.asarray(req_a["vectors"][pos_a], dtype=np.float64)
    vb = np.asarray(req_b["vectors"][pos_b], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        dot = float(np.dot(va, vb))
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    # A non-finite score is undefined, not "not similar": return NaN so the
    # threshold selector fails loudly instead of silently reporting 0 TPR/FPR.
    if not (np.isfinite(dot) and np.isfinite(denom)):
        return float("nan")
    if denom <= 0:
        raise ValueError(
            "linkage cosine: zero-norm wire vector (cosine undefined); refusing "
            "to score it as 0.0"
        )
    return float(dot / denom)


def _mapping_path_block(rows, deployment_ids, mapping: dict[int, int], requests,
                        *, seed: int = 0) -> dict:
    """Mapping-dependent linkage scored on the **full** test split (S05).

    Pairs are never dropped because an endpoint has no recovered mapping:
    an uncovered positive endpoint is a non-detection (False) and an uncovered
    negative endpoint is not flagged (False); both stay in their denominators.
    Conditional metrics on covered pairs are reported separately.
    """
    outcomes: list[bool] = []
    covered: list[bool] = []
    for a, pos_a, b, pos_b, _token in rows:
        left_id = int(deployment_ids[a][pos_a])
        right_id = int(deployment_ids[b][pos_b])
        left = mapping.get(left_id)
        right = mapping.get(right_id)
        pair_covered = left is not None and right is not None
        covered.append(pair_covered)
        outcomes.append(bool(pair_covered and int(left) == int(right)))
    full = _cluster_rate_ci(outcomes, _pair_clusters(requests, rows), seed=seed)
    covered_rows = [row for row, flag in zip(rows, covered) if flag]
    conditional = _cluster_rate_ci(
        [outcome for outcome, flag in zip(outcomes, covered) if flag],
        _pair_clusters(requests, covered_rows), seed=seed,
    )
    return {
        "full_denominator": full,
        "conditional_on_covered": conditional,
        # Coverage is a per-pair outcome; cluster it like the rates (Wilson over
        # dependent pairs would report a falsely tight interval).
        "coverage": _cluster_rate_ci(
            covered, _pair_clusters(requests, rows), seed=seed
        ),
        "denominator_rule": (
            "uncovered positive endpoint = non-detection (False); uncovered negative "
            "endpoint = not flagged (False); both remain in the denominator"
        ),
    }


def _deployment_id_outcomes(rows, deployment_ids) -> list[bool]:
    """Exact equality of the stage-1 KNN deployment ids (no stage-2 mapping).

    ``-1`` means "no assignment" (zero/non-finite query vector): a pair touching
    an unassigned endpoint is a non-match, never a match of two ``-1`` values.
    """
    outcomes: list[bool] = []
    for a, pos_a, b, pos_b, _token in rows:
        left = int(deployment_ids[a][pos_a])
        right = int(deployment_ids[b][pos_b])
        outcomes.append(bool(left >= 0 and left == right))
    return outcomes


def _within_stream_gap(requests, row) -> int:
    a, _pos_a, b, _pos_b, _token = row
    left = int(requests[a].get("request_id", a))
    right = int(requests[b].get("request_id", b))
    return abs(right - left)


def _wire_bits_equal(requests, a: int, pa: int, b: int, pb: int) -> bool:
    left = np.ascontiguousarray(requests[a]["vectors"][pa])
    right = np.ascontiguousarray(requests[b]["vectors"][pb])
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _delta_bucket_blocks(requests, rows, *, seed: int = 0) -> tuple[dict, int]:
    """Per-delta-bucket linkage rates over same-stream test pairs.

    The pooled index gap mixes streams: boundary-adjacent requests from different
    streams would land in bucket "1". Buckets use the within-stream request gap,
    and cross-stream pairs are excluded from the buckets (and counted) instead.
    """
    same_stream: list = []
    excluded = 0
    for row in rows:
        a, _pos_a, b, _pos_b, _token = row
        if requests[a].get("stream_id") != requests[b].get("stream_id"):
            excluded += 1
            continue
        same_stream.append(row)
    buckets: dict = {}
    for low, high in DELTA_BUCKETS:
        label = (f"{low}" if high == low
                 else (f"{low}-{high}" if high is not None else f"{low}+"))
        subset = [row for row in same_stream
                  if delta_bucket(_within_stream_gap(requests, row)) == label]
        if not subset:
            continue
        outcomes = [
            int(_wire_bits_equal(requests, a, pa, b, pb))
            for a, pa, b, pb, _ in subset
        ]
        # Pairs sharing a session are dependent; use the session/pair cluster
        # bootstrap (a single held-out session yields no bounds / n_boot 0).
        buckets[label] = _cluster_rate_ci(
            outcomes, _pair_clusters(requests, subset), seed=seed
        )
    return buckets, int(excluded)


def _threshold_from_validation(val_neg_cos, target_fpr: float) -> dict:
    """Validation-split threshold with the selected FPR never above target (S05).

    When the allowed-count boundary falls inside a tie, ``>=`` would flag the
    whole tied mass and exceed the target. The threshold is then raised strictly
    above the tied value (``nextafter``), and the adjustment is recorded.
    """
    scores = np.sort(np.asarray(val_neg_cos, dtype=np.float64))[::-1]
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "validation negative cosines contain non-finite values (NaN/inf); "
            "refusing to select a threshold from an undefined score"
        )
    n = int(scores.size)
    if n == 0:
        return {"threshold": 1.0, "val_flagged": 0, "val_n": 0,
                "val_fpr_actual": 0.0, "tie_boundary_adjusted": False}
    allowed = int(np.floor(float(target_fpr) * n))
    preliminary = float(scores[0] if allowed <= 0 else scores[min(allowed - 1, n - 1)])
    flagged = int(np.sum(scores >= preliminary))
    adjusted = False
    threshold = preliminary
    if flagged > allowed:
        adjusted = True
        threshold = float(np.nextafter(preliminary, np.inf))
        flagged = int(np.sum(scores >= threshold))
    return {
        "threshold": threshold,
        "val_flagged": int(flagged),
        "val_n": n,
        "val_fpr_actual": float(flagged / n) if n else 0.0,
        "tie_boundary_adjusted": adjusted,
    }


def run_c12(dep: ABDeployment, texts: Optional[Sequence[str]] = None, *,
            text_sets: Optional[Sequence[Sequence[str]]] = None,
            session_sets: Optional[Sequence[Sequence[str]]] = None,
            n_keys: int = 16, rotate_every: int = 64, seed: int = 0,
            fpr_targets: Sequence[float] = (1e-2, 1e-3), verbose: bool = True,
            cache: Optional[dict] = None,
            stage2_mapping: Optional[dict] = None,
            aggregation_threshold: Optional[float] = None,
            aggregation_repeat_labels: Optional[Sequence[object]] = None,
            profile: str = "probe",
            protocol_sha256: str = "") -> dict:
    """Linkage with matched negatives, validation FPR, and multi-stream support (F11)."""
    t0 = time.time()
    sets = [list(texts or [])] if text_sets is None else [list(row) for row in text_sets]
    if not sets or all(not row for row in sets):
        raise ValueError("linkage needs at least one non-empty text set")
    if session_sets is not None and (len(session_sets) != len(sets) or any(
            len(session_set) != len(text_set) for session_set, text_set in zip(session_sets, sets))):
        raise ValueError("session_sets must match each stream's requests")
    if aggregation_threshold is not None:
        if profile != "probe":
            raise ValueError("public aggregation is a probe until threshold selection and formal controls are verified")
        if aggregation_repeat_labels is None or len(aggregation_repeat_labels) != sum(map(len, sets)):
            raise ValueError("aggregation requires one independent scorer-side repeat label per request")
        if any(label is None for label in aggregation_repeat_labels):
            raise ValueError("aggregation repeat labels must be explicit")
        try:
            if not 0 <= float(aggregation_threshold) <= 1:
                raise ValueError("aggregation threshold must be in [0, 1]")
            for label in aggregation_repeat_labels:
                hash(label)
        except TypeError as exc:
            raise ValueError("aggregation labels must be hashable") from exc
    # Content identity: equal request counts from different corpora or different
    # stream partitions must not collide in data_sha256/run_id (length-framed
    # text digest plus the stream structure, as in C10).
    corpus_texts = [text for row in sets for text in row]
    corpus_sha256 = _stream_sets_digest(sets)
    deployment_state_sha256 = _deployment_state_digest(dep).hex()
    client_secret_sha256 = _client_secret_commitment(dep)
    requests: list[dict] = []
    truth: list[dict] = []
    stream_meta: list[dict] = []
    for stream_id, stream_texts in enumerate(sets):
        stream = build_stream(
            dep, stream_texts, n_keys=n_keys, rotate_every=rotate_every,
            seed=seed + stream_id, profile=profile,
            protocol_sha256=protocol_sha256,
            cache=cache if (cache is not None and len(sets) == 1) else None,
            session_ids=session_sets[stream_id] if session_sets is not None else None,
        )
        for record, truth_row in zip(stream["requests"], stream["truth"]):
            public = dict(record)
            public["stream_id"] = stream_id
            requests.append(public)
            truth.append(truth_row)
        stream_meta.append({
            "stream_id": stream_id,
            "n_requests": stream["n_requests"],
            "n_positions": stream["n_positions"],
            "n_keys": stream["n_keys"],
            "n_epochs": stream["n_epochs"],
            "stream_key": stream["stream_key"],
            "profile": profile,
        })
    pairs = _matched_linkage_pairs(requests, truth, seed=seed)
    positives = pairs["positives"]
    negatives = pairs["negatives"]

    def _session_key(index):
        return _session_identity(requests, index)

    all_sessions = sorted({
        key for row in positives + negatives
        for key in (_session_key(row[0]), _session_key(row[2]))
    }, key=str)
    if len(all_sessions) < 2:
        validation = set(all_sessions)
        held_out = set()
    else:
        cut = max(1, int(round(len(all_sessions) * 0.5)))
        validation = set(all_sessions[:cut])
        held_out = set(all_sessions[cut:])

    def _split(rows):
        # Session-disjoint: validation pairs have both endpoints in the validation
        # sessions, test pairs have both endpoints in held-out sessions, and any
        # pair crossing the split is dropped. No session contributes to both.
        train = [row for row in rows
                 if _session_key(row[0]) in validation and _session_key(row[2]) in validation]
        test = [row for row in rows
                if _session_key(row[0]) in held_out and _session_key(row[2]) in held_out]
        return train, test

    val_pos, test_pos = _split(positives)
    val_neg, test_neg = _split(negatives)

    def _cosines(rows):
        return np.array([_cosine(requests[a], pa, requests[b], pb) for a, pa, b, pb, _ in rows])

    val_pos_cos = _cosines(val_pos)
    val_neg_cos = _cosines(val_neg)
    test_pos_cos = _cosines(test_pos)
    test_neg_cos = _cosines(test_neg)
    # A non-finite cosine is undefined, not "below threshold": NaN comparisons
    # would silently become False in the outcomes, the median and the AUC. Refuse
    # every split rather than absorbing it.
    for _label, _values in (
        ("validation positives", val_pos_cos),
        ("validation negatives", val_neg_cos),
        ("test positives", test_pos_cos),
        ("test negatives", test_neg_cos),
    ):
        _array = np.asarray(_values)
        if _array.size and not np.all(np.isfinite(_array)):
            raise ValueError(
                f"linkage cosines: {_label} contain non-finite values (NaN/inf); "
                "refusing to score an undefined similarity"
            )

    def _threshold_for(target_fpr: float) -> dict:
        # S05: selection keeps the *validation* FPR <= target; a tie that would
        # flag more than the allowed count pushes the threshold strictly above
        # the tied value.
        return _threshold_from_validation(val_neg_cos, float(target_fpr))

    approximate = {}
    pos_clusters = _pair_clusters(requests, test_pos)
    neg_clusters = _pair_clusters(requests, test_neg)
    for target in fpr_targets:
        selection = _threshold_for(float(target))
        threshold = selection["threshold"]
        pos_outcomes = [float(value) >= threshold for value in test_pos_cos]
        neg_outcomes = [float(value) >= threshold for value in test_neg_cos]
        approximate[f"fpr_target_{target:g}"] = {
            "threshold_from_validation": threshold,
            **selection,
            "tpr": _cluster_rate_ci(pos_outcomes, pos_clusters, seed=seed),
            "fpr": _cluster_rate_ci(neg_outcomes, neg_clusters, seed=seed),
        }

    exact_outcomes = [
        int(_wire_bits_equal(requests, a, pa, b, pb))
        for a, pa, b, pb, _ in test_pos
    ]
    exact_test = _cluster_rate_ci(
        exact_outcomes, _pair_clusters(requests, test_pos), seed=seed
    )
    rounded_outcomes = [int(requests[a]["symbols"][pa] == requests[b]["symbols"][pb])
                        for a, pa, b, pb, _ in test_pos]
    quantized_test = _cluster_rate_ci(
        rounded_outcomes, _pair_clusters(requests, test_pos), seed=seed
    )
    per_unit = [
        {"unit": index, "hits": int(outcome), "count": 1}
        for index, outcome in enumerate(exact_outcomes)
    ]
    same_group = [row for row in positives
                  if requests[row[0]]["group_id"] == requests[row[2]]["group_id"]]
    control_outcomes = [
        int(_wire_bits_equal(requests, a, pa, b, pb))
        for a, pa, b, pb, _ in same_group
    ]
    control = _cluster_rate_ci(
        control_outcomes, _pair_clusters(requests, same_group), seed=seed
    )
    control_pass = bool(control["value"] is not None and control["value"] >= 0.99)

    # --- stage-1 deployment ids: always available, no stage-2 mapping needed ---
    # S05: comparing the KNN deployment ids of both endpoints does not require
    # the stage-2 plaintext mapping (the deployment vocabulary is key-independent).
    deployment_table = dep.table("ab")
    deployment_ids = _knn_deployment_ids(requests, deployment_table)
    n_table_rows = int(np.asarray(deployment_table).shape[0])
    assigned_positions = total_positions = 0
    for ids in deployment_ids:
        flat = np.asarray(ids).reshape(-1)
        total_positions += int(flat.size)
        assigned_positions += int(np.sum((flat >= 0) & (flat < n_table_rows)))
    deployment_pos_outcomes = _deployment_id_outcomes(test_pos, deployment_ids)
    deployment_neg_outcomes = _deployment_id_outcomes(test_neg, deployment_ids)
    deployment_id_test = {
        "status": "implemented",
        "test": "exact equality of the stage-1 KNN deployment ids of both endpoints",
        "requires_stage2_mapping": False,
        "uses_secret": False,
        "tpr": _cluster_rate_ci(
            deployment_pos_outcomes, _pair_clusters(requests, test_pos), seed=seed
        ),
        "fpr": _cluster_rate_ci(
            deployment_neg_outcomes, _pair_clusters(requests, test_neg), seed=seed
        ),
        # Measured, not a literal: fraction of observed positions that a
        # stage-1 KNN actually assigned a valid deployment-table row.
        "stage1_coverage": {
            "value": (assigned_positions / total_positions) if total_positions else None,
            "numerator": int(assigned_positions),
            "denominator": int(total_positions),
        },
    }
    deployment_per_unit = {
        "positives": [
            {"unit": index, "hits": int(outcome), "count": 1}
            for index, outcome in enumerate(deployment_pos_outcomes)
        ],
        "negatives": [
            {"unit": index, "hits": int(outcome), "count": 1}
            for index, outcome in enumerate(deployment_neg_outcomes)
        ],
    }

    recovered_metrics = {
        "status": "not_run", "reason": "stage2_mapping not supplied",
        "requires_stage2_mapping": True,
    }
    if stage2_mapping is not None:
        # The deployment-id path that needs no mapping is ``deployment_id_test``
        # above; the mapping-dependent record is the end-to-end token-level
        # inversion. (The old ``knn_aligned`` block was the same computation on
        # the same pairs, so it reported duplicated numbers and was removed.)
        recovered_pos_block = _mapping_path_block(
            test_pos, deployment_ids, stage2_mapping, requests, seed=seed
        )
        recovered_neg_block = _mapping_path_block(
            test_neg, deployment_ids, stage2_mapping, requests, seed=seed
        )
        recovered_metrics = {
            "status": "implemented",
            "kind": "token-level inversion (stage-1 KNN -> stage-2 plaintext token), end-to-end",
            "requires_stage2_mapping": True,
            "tpr": recovered_pos_block["full_denominator"],
            "fpr": recovered_neg_block["full_denominator"],
            "conditional_tpr": recovered_pos_block["conditional_on_covered"],
            "conditional_fpr": recovered_neg_block["conditional_on_covered"],
            "coverage_positives": recovered_pos_block["coverage"],
            "coverage_negatives": recovered_neg_block["coverage"],
            "denominator_rule": recovered_pos_block["denominator_rule"],
            "text_normalisation": "none; token equality only",
        }

    # Frozen protocol bucket table (metrics/recovery.DELTA_BUCKETS), shared with
    # the linkage attack scorer: 1 / 2-3 / 4-15 / 16-63 / 64-127 / 128-511 / 512+.
    # Cross-stream pairs are excluded from the gap buckets and counted.
    delta_buckets, delta_excluded_cross_stream = _delta_bucket_blocks(
        requests, test_pos, seed=seed
    )

    aggregation = None
    aggregation_identity = None
    if aggregation_threshold is not None:
        from .aggregation import aggregate_public_stream, score_public_aggregation
        import hashlib

        labels = list(aggregation_repeat_labels)
        label_indices: dict[object, int] = {}
        normalized_labels = [label_indices.setdefault(label, len(label_indices)) for label in labels]
        label_hash = hashlib.sha256(json.dumps(normalized_labels, separators=(",", ":")).encode()).hexdigest()
        aggregation_identity = {"threshold": float(aggregation_threshold), "repeat_partition_sha256": label_hash}
        public_requests = [dict(request, request_id=index) for index, request in enumerate(requests)]
        try:
            inferred = aggregate_public_stream(
                dep.public_view("ab", "S-public"), public_requests,
                similarity_threshold=float(aggregation_threshold),
            )
            scored = score_public_aggregation(
                inferred, [row["permuted_ids"] for row in truth], labels,
            )
            aggregation = {"status": "executed", "score": scored,
                           "candidate_pairs": inferred["candidate_pairs"],
                           "pair_visits": inferred["pair_visits"],
                           "alignment_cells": inferred["alignment_cells"],
                           "seconds": inferred["seconds"],
                           "threshold": inferred["similarity_threshold"],
                           "threshold_provenance": inferred["threshold_provenance"],
                           "method": inferred["method"]}
        except (ValueError, TypeError) as exc:
            aggregation = {"status": "not_run", "reason": str(exc),
                           "threshold_provenance": "caller supplied; validation selection not verified"}

    payload = result_payload(
        attack_name="linkage.exact",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={
            "exact_test": exact_test,
            "quantized_4_decimal_test": quantized_test,
            "approximate": approximate,
            "auc": _auc(test_pos_cos, test_neg_cos),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "validation": {"positives": len(val_pos), "negatives": len(val_neg)},
            "test": {"positives": len(test_pos), "negatives": len(test_neg)},
            "negative_pair_rule": "position-matched, different-token endpoint in the same request",
            "delta_buckets": delta_buckets,
            "delta_buckets_excluded_cross_stream": int(delta_excluded_cross_stream),
            "validation_positive_cosine_median": (
                float(np.median(val_pos_cos)) if val_pos_cos.size else None
            ),
            "recovered_text": recovered_metrics,
            "deployment_id_test": deployment_id_test,
            "pair_sampling": pairs["sampling"],
            "session_source": "provided" if session_sets is not None else "unavailable",
            "session_split_verified": session_sets is not None,
            "n_streams": len(sets),
            "public_aggregation_probe": aggregation,
        },
        diagnostics={
            "candidate_mode": "full",
            "positive_control": {"name": "same_group_exact_link", "top1": control["value"],
                                 "required_top1": 0.99, "n": control["denominator"]},
            "positive_control_passed": control_pass,
            "rotation_grouping": "SH-P (epoch/group metadata visible)",
            "n_requests": len(requests),
            "n_keys": int(sum(meta["n_keys"] for meta in stream_meta)),
            "n_epochs": int(sum(meta["n_epochs"] for meta in stream_meta)),
            "per_stream": stream_meta,
            "deployment_id_per_unit": deployment_per_unit,
            "pair_sampling": pairs["sampling"],
            "note": ("Thresholds are selected on a validation split and applied to the test split; "
                     "a tie at the allowed-count boundary is resolved strictly above the tied "
                     "value so the validation FPR never exceeds the target. Negatives are "
                     "position-matched different-token endpoints, not another request's first "
                     "token. Deployment-id linkage needs no stage-2 mapping; mapping-dependent "
                     "paths keep uncovered pairs in their denominators."),
        },
        ci={"exact_test": exact_test},
        n_prompts=len(requests),
        n_tokens=int(sum(meta["n_positions"] for meta in stream_meta)),
        n_keys=int(sum(meta["n_keys"] for meta in stream_meta)),
        n_types=int(len({int(token) for row in truth for token in row["plaintext_ids"]})),
        reference_tokens=0,
        source={"artifact": str(dep.artifact), "stream": "ab"},
        data={"requests": len(requests), "rotate_every": rotate_every,
              "n_streams": len(sets),
              "n_texts": int(len(corpus_texts)),
              "corpus_sha256": corpus_sha256,
              "artifact": str(dep.artifact),
              "deployment_state_sha256": deployment_state_sha256,
              "client_secret_sha256": client_secret_sha256,
              "n_keys_requested": int(n_keys),
              "seed": int(seed),
              **({"aggregation": aggregation_identity} if aggregation_identity is not None else {})},
    )
    payload["attack_id"] = "linkage_exact"
    payload["n_positives"] = int(len(positives))
    payload["n_negatives"] = int(len(negatives))
    payload["n_streams"] = int(len(sets))
    write_payload("C12", "linkage.exact", "ab", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "per_unit_predictions": per_unit,
                   "profile": profile,
                   "protocol_sha256": protocol_sha256})
    if verbose:
        print(f"  C12 linkage exact={exact_test['value']} auc={payload['metrics']['auc']} "
              f"control={control['value']} streams={len(sets)} "
              f"({round(time.time() - t0, 1)}s)", flush=True)
    return {"exact_test": exact_test, "approximate": approximate, "control": control,
            "auc": payload["metrics"]["auc"],
            "recovered_text": recovered_metrics,
            "deployment_id_test": deployment_id_test,
            "pair_sampling": pairs["sampling"],
            "public_aggregation_probe": aggregation,
            "n_streams": len(sets),
            # Public aggregates only: the scorer-side truth rows (plaintext_ids /
            # permuted_ids) and the raw request records stay internal to this
            # function. permuted_ids is reversible client-secret material and
            # must never reach a serialized summary.
            "stream": {"per_stream": stream_meta}}


def run_c13_c14(dep: ABDeployment, *, prompts: Optional[Sequence[str]] = None,
                annotations: Optional[Sequence[dict]] = None,
                max_new_tokens: int = 256, seed: int = 0,
                output_mapping: Optional[dict[int, int]] = None,
                trace_root: Optional[Path] = None,
                verbose: bool = True) -> dict:
    """Real stepwise output generation + PII audit (F07).

    The old path uploaded ``prompt + scripted answer`` and called the tail of the
    input wire an output. This path generates token by token through the real
    deployment (client re-encodes/re-uploads the exact prefix each step), records
    every public step, and scores PII only when annotations are supplied. A
    scripted answer is never used as an output record.
    """
    import hashlib
    import uuid

    from .generation import PublicGenerationWriter, generate_record
    from .output_attack import recover_output_from_trace
    from .pii_spans import score_verified_entity_recovery

    t0 = time.time()
    if prompts is None:
        raise ValueError(
            "run_c13_c14 requires explicit prompts; a scripted PII record set is not "
            "an output-generation experiment"
        )
    prompt_list = list(prompts)
    if annotations is not None and len(annotations) != len(prompt_list):
        raise ValueError("output annotations must have one entry per prompt")
    prompt_set_sha256 = hashlib.sha256(json.dumps(prompt_list, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    annotation_sha256 = (hashlib.sha256(json.dumps(annotations, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":"), default=_json_default).encode("utf-8")).hexdigest()
                         if annotations is not None else None)
    mapping_sha256 = None
    if output_mapping is not None:
        mapping_items = sorted((int(key), int(value)) for key, value in output_mapping.items())
        if len({key for key, _ in mapping_items}) != len(mapping_items):
            raise ValueError("output mapping contains duplicate normalized IDs")
        mapping_sha256 = hashlib.sha256(json.dumps(mapping_items, separators=(",", ":")).encode("utf-8")).hexdigest()
    rows = []
    total_steps = 0
    continuation_lengths = []
    trace_directory = Path(trace_root or (OUT_ROOT / "C13" / "public_traces")) / uuid.uuid4().hex

    def public_entity_score(score):
        # Entity values and character spans are scorer-side truth.
        return {key: value for key, value in score.items() if key != "entities"}

    for index, prompt in enumerate(prompt_list):
        prompt_ids = dep.encode(prompt)
        writer = PublicGenerationWriter(trace_directory / f"request-{index:06d}")
        record = generate_record(
            dep, prompt_ids, arm="ab", max_new_tokens=max_new_tokens,
            greedy=True, seed=seed + index, key_index=index, public_writer=writer,
        )
        trace = writer.finish(record)
        total_steps += len(record.steps)
        continuation_lengths.append(len(record.continuation_ids))
        row = {
            **_prompt_identity(index, prompt),
            "public_trace": record.public_trace(),
            "public_trace_artifact": trace,
            "n_steps": len(record.steps),
            "input_pii_recovered": {"status": "not_run", "reason": "no input inversion supplied"},
        }
        recovered = None
        if output_mapping is not None:
            recovered = recover_output_from_trace(trace["path"], recovered_mapping=output_mapping,
                                                  expected_sha256=trace["sha256"])
            row["output_attack"] = {key: value for key, value in recovered.items() if key != "recovered_ids"}
            prediction = recovered["recovered_ids"]
            target = record.continuation_ids
            hits = sum(int(prediction[i] == value) for i, value in enumerate(target) if i < len(prediction))
            row["output_reconstruction"] = {
                "token_top1": {"numerator": hits, "denominator": len(target),
                               "value": hits / len(target) if target else None},
                "exact_sequence": prediction == target,
            }
        else:
            row["output_attack"] = {"status": "not_run", "reason": "no attacker-recovered mapping supplied"}
        if annotations is not None and index < len(annotations):
            annotation = annotations[index]
            output_entities = annotation.get("output_entities", [])
            try:
                output_text = (dep.tokenizer.decode(record.continuation_ids,
                                                   clean_up_tokenization_spaces=False)
                               if output_entities else "")
            except (TypeError, ValueError, NotImplementedError):
                output_text = None

            def output_score(prediction):
                if output_text is None:
                    return {"status": "not_run", "reason": "generated text could not be decoded for annotation verification"}
                return public_entity_score(score_verified_entity_recovery(
                    dep.tokenizer, output_text, record.continuation_ids, prediction,
                    output_entities, annotation.get("output_offsets", []),
                ))

            # The client holds the plaintext; comparing it to itself is the
            # protocol's client-side positive control, NOT an attack result.
            row["input_pii_plaintext_control"] = public_entity_score(score_verified_entity_recovery(
                dep.tokenizer, prompt, prompt_ids, prompt_ids,
                annotation.get("prompt_entities", []),
                annotation.get("prompt_offsets", []),
            ))
            row["output_pii_plaintext_control"] = output_score(record.continuation_ids)
            if recovered is None:
                row["output_pii_recovered"] = {"status": "not_run", "reason": "no attacker-recovered mapping supplied"}
            else:
                row["output_pii_recovered"] = output_score(recovered["recovered_ids"])
        else:
            row["output_pii_recovered"] = {"status": "not_run", "reason": "no independent output annotations supplied"}
        rows.append(row)
    payload = result_payload(
        attack_name="pii.extract",
        scheme_name="ab_deployment:ab_output",
        target="llama-3.2-1b/ab_output",
        metrics={
            "n_prompts": len(rows),
            "n_generated_tokens": int(sum(continuation_lengths)),
            "n_generation_steps": int(total_steps),
            "mean_continuation_len": (
                float(np.mean(continuation_lengths))
                if rows else None
            ),
            "rows": rows,
            "decoding": "greedy",
            "max_new_tokens": int(max_new_tokens),
        },
        diagnostics={
            "candidate_mode": "full",
            "generation_path": "full-prefix re-encode/re-upload, client sampling",
            "real_generation": True,
            "scripted_answer_used": False,
            "prompt_set_sha256": prompt_set_sha256,
            "annotation_sha256": annotation_sha256,
            "mapping_sha256": mapping_sha256,
            "mapping_provenance": "caller_supplied_unverified" if output_mapping is not None else "not_supplied",
            "annotation_provenance": "caller_supplied_unverified" if annotations is not None else "not_supplied",
            "pii_scoring": (
                "input/output entity namespaces are separate; the self-comparison "
                "rows are client plaintext positive controls, not recovery results; "
                "output-PII recovery uses only the stored server trace and caller-supplied "
                "mapping; mapping provenance must be independently verified"
            ),
            "positive_control": {"name": "not_run_in_this_entry"},
            "positive_control_passed": None,
            "attack_variant": "output_side_real_generation",
        },
        ci={},
        n_prompts=len(rows),
        n_tokens=int(sum(continuation_lengths)),
        n_keys=len(rows), n_types=0, reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "real_generation"},
        data={"prompts": len(rows), "prompt_set_sha256": prompt_set_sha256,
              "annotation_sha256": annotation_sha256, "mapping_sha256": mapping_sha256,
              "max_new_tokens": max_new_tokens},
    )
    payload["attack_id"] = "output_pii"
    payload["max_new_tokens"] = int(max_new_tokens)
    payload["generation_record"] = True
    write_payload("C13", "pii.extract", "ab_output", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "executed_steps": int(total_steps)})
    if verbose:
        print(f"  C13/C14 real generation: {len(rows)} prompts, {total_steps} steps "
              f"({round(time.time()-t0,1)}s)", flush=True)
    return {"rows": rows, "n_steps": total_steps}


def _entity_token_spans(dep: ABDeployment, text: str, entities):
    """Token index spans of each entity occurrence inside ``text``.

    Uses the tokenizer's offset mapping so BPE boundary tokens (``ĠMaria``) are matched
    against the *character* span of the entity, not against the ids of the isolated
    string -- the latter systematically under-counts (0/18 at token accuracy 1.0).
    """
    tok = dep.tokenizer
    out = []
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return [[int(t) for t in dep.encode(e)] for e in entities]
    offsets = list(offsets)
    for ent in entities:
        start = text.find(ent)
        if start < 0:
            out.append([int(t) for t in dep.encode(ent)])
            continue
        end = start + len(ent)
        span = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        out.append(span)
    return out


def _as_list(x):
    return list(x) if x is not None else []


def _contains_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    """True iff ``needle`` appears contiguously in ``haystack`` (entity-level PII rule)."""
    hay = [int(x) for x in _as_list(haystack) if x is not None and int(x) >= 0]
    ned = [int(x) for x in needle]
    if not ned:
        return False
    for start in range(0, len(hay) - len(ned) + 1):
        if hay[start:start + len(ned)] == ned:
            return True
    return False


# ---------------------------------------------------------------------------
# C05 - ISA: optimise an input embedding to match an observed hidden state (S2/K0)
# ---------------------------------------------------------------------------
_ISA_STRATA = (
    ("32-64", 32, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
)


def _isa_length_bucket(dep: ABDeployment, text: str) -> Optional[str]:
    length = len(dep.encode(text))
    for name, low, high in _ISA_STRATA:
        if low <= length <= high:
            return name
    return None


def select_isa_prompts(
    dep: ABDeployment, texts: Sequence[str], strata: Mapping[str, int],
) -> tuple[list[str], dict[str, int]]:
    """Select the protocol's stratified ISA prompt set by *token* length.

    The floor is 50 prompts in each of 32-64 / 65-128 / 129-256 / 257-512
    tokens. Character length is not token length; the selection uses the real
    tokenizer and refuses when a band cannot be filled.
    """
    buckets: dict[str, list[str]] = {name: [] for name, _, _ in _ISA_STRATA}
    for text in texts:
        name = _isa_length_bucket(dep, text)
        if name is not None:
            buckets[name].append(text)
    selected: list[str] = []
    observed: dict[str, int] = {}
    shortages: list[str] = []
    for name, _, _ in _ISA_STRATA:
        need = int(strata.get(name, 0))
        have = buckets[name][:need]
        observed[name] = len(have)
        selected.extend(have)
        if len(have) < need:
            shortages.append(f"{name}: {len(have)} < {need}")
    if shortages:
        raise ValueError(
            "ISA length strata cannot be filled from the supplied corpus: "
            + "; ".join(shortages)
        )
    return selected, observed


def _layer_control_summary(control_curve: Mapping[str, Mapping[str, Mapping]],
                           checkpoint_key: str, *, required: float = 0.90,
                           attack_curve: Optional[Mapping[str, Mapping[str, Mapping]]] = None,
                           ) -> dict:
    """Per-layer positive-control verdicts at one checkpoint (S04).

    A shallow layer's passing control must not mask a deep layer's failure: the
    overall control is the **minimum over every used layer**, and it passes only
    if every layer passes. Each layer's record also carries that layer's attack
    result (when ``attack_curve`` is supplied) so the attack number stays bound
    to the control measured on the same layer instead of to a global maximum.
    """
    by_layer: dict[str, dict] = {}
    for layer_name, layer in (control_curve or {}).items():
        entry = (layer or {}).get(checkpoint_key) or {}
        top_block = entry.get("top1") if isinstance(entry, Mapping) else None
        value = top_block.get("value") if isinstance(top_block, Mapping) else None
        try:
            numeric = None if value is None else float(value)
        except (TypeError, ValueError, OverflowError):
            numeric = None
        layer_passed = bool(numeric is not None and numeric >= float(required))
        record = {
            "top1": numeric,
            "numerator": (top_block.get("numerator")
                          if isinstance(top_block, Mapping) else None),
            "denominator": (top_block.get("denominator")
                            if isinstance(top_block, Mapping) else None),
            "required_top1": float(required),
            "passed": layer_passed,
            "checkpoint": str(checkpoint_key),
        }
        if attack_curve is not None:
            attack_entry = ((attack_curve.get(layer_name) or {}).get(checkpoint_key) or {})
            attack_block = attack_entry.get("top1") if isinstance(attack_entry, Mapping) else None
            attack_value = attack_block.get("value") if isinstance(attack_block, Mapping) else None
            record["attack_top1"] = (
                None if attack_value is None else float(attack_value)
            )
            record["attack_bound_to_this_layer_control"] = layer_passed
        by_layer[str(layer_name)] = record
    values = [record["top1"] for record in by_layer.values() if record["top1"] is not None]
    overall = float(min(values)) if values else None
    passed = bool(by_layer) and all(record["passed"] for record in by_layer.values())
    return {
        "top1": overall,
        "required_top1": float(required),
        "passed": passed,
        "aggregation": "min over used layers; a passing shallow layer cannot mask a failing deep layer",
        "checkpoint": str(checkpoint_key),
        "layers_used": sorted(by_layer),
        "by_layer": by_layer,
    }


def _saturation_record(curve: Mapping, primary_layer: str,
                       steps: Sequence[int]) -> Optional[dict]:
    """Signed gain between the primary layer's final two measured checkpoints.

    The gain is ``(last - first) * 100`` -- a regression is a negative number, so
    a large negative value never reads as a positive saturation gain.
    """
    ordered = sorted({int(step) for step in steps})
    if len(ordered) < 2:
        return None
    previous_step, final_step = ordered[-2], ordered[-1]

    def _top1(step: int):
        block = (curve.get(primary_layer) or {}).get(f"steps{int(step)}") or {}
        top = block.get("top1") if isinstance(block, Mapping) else None
        return top.get("value") if isinstance(top, Mapping) else None

    previous_top, final_top = _top1(previous_step), _top1(final_step)
    if previous_top is None or final_top is None:
        return None
    return {
        "points": [
            {"budget": int(previous_step), "top1": float(previous_top)},
            {"budget": int(final_step), "top1": float(final_top)},
        ],
        "gain_pp": (float(final_top) - float(previous_top)) * 100.0,
        "source": ("measured budget curve: primary layer's final two checkpoints "
                   "from this run (signed gain)"),
    }


# Literature-inspired defaults for probes without a validation selection.
ISA_TABLE_LR_SHALLOW = 1e-3
ISA_TABLE_LR_DEEP = 1e-2


def _isa_table_lr(layer: int) -> float:
    return ISA_TABLE_LR_SHALLOW if int(layer) < 8 else ISA_TABLE_LR_DEEP


def _isa_lr_by_layer(lr, layers: Sequence[int]) -> dict[int, float]:
    """Resolve the executed learning rate of every sampled layer.

    ``lr=None`` uses the literature-inspired defaults; an explicit scalar or
    complete mapping may supply the value selected on validation data.
    """
    table = {int(layer): _isa_table_lr(layer) for layer in layers}
    hint = "every sampled layer needs an explicit learning rate"
    if lr is None:
        return table
    if isinstance(lr, Mapping):
        resolved: dict[int, float] = {}
        for layer in table:
            value = lr.get(layer)
            if value is None:
                value = lr.get(str(layer))
            if value is None:
                raise ValueError(
                    f"ISA lr mapping omits sampled layer {layer}; every used "
                    f"layer needs an explicit value ({hint})"
                )
            try:
                resolved[layer] = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"ISA lr for layer {layer} is not numeric: {value!r}"
                ) from None
        for key in lr:
            try:
                named = int(key)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"ISA lr mapping key {key!r} is not a layer index"
                ) from None
            if named not in table:
                raise ValueError(
                    f"ISA lr mapping names layer {named}, which this run does "
                    f"not sample (used: {sorted(table)})"
                )
        for layer, value in resolved.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"ISA lr for layer {layer} must be finite and positive")
        return resolved
    try:
        scalar = float(lr)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"ISA lr is not numeric: {lr!r}") from None
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError("ISA lr must be finite and positive")
    return {layer: scalar for layer in table}


def run_c05(dep: ABDeployment, texts: Sequence[str], ref_counts: dict[int, int], *,
            prompts: int = 1, steps: Sequence[int] = (20, 100),
            lr: Optional[float | Mapping] = None,
            isa_layers: Sequence[int] = (1,), seed: int = 0, method: str = "er",
            restarts: int = 1, keys_per_prompt: int = 1,
            lambda_dm: Optional[float] = None,
            strata: Optional[Mapping[str, int]] = None,
            selected_on_validation: Optional[Mapping] = None,
            profile: str = "probe", protocol_sha256: str = "",
            scenario: str = "S-public",
            arm: str = "ab",
            verbose: bool = True) -> dict:
    """ISA deployment-row recovery, paired with plaintext control."""
    from .real_isa import isa_attack

    if method == "aloepri_state" and profile != "probe":
        raise ValueError("aloepri_state is an unregistered probe; formal acceptance is not established")
    if arm == "osnip":
        raise ValueError("C05 OSNIP-only requires an independently trained clean-backbone encoder")
    if arm != "ab" and (arm != "aloepri" or profile != "probe"):
        raise ValueError("C05 supports ab and the probe-only A engineering artifact")

    if int(keys_per_prompt) < 1:
        raise ValueError("keys_per_prompt must be >= 1")
    validation_block: Optional[dict] = None
    if selected_on_validation is not None:
        # A hyper-parameter selection is only credible if it was made on the
        # independent 40-prompt / 10-per-band validation study; never invent one.
        validation_block = dict(selected_on_validation)
        try:
            validation_prompts = int(validation_block.get("n_prompts"))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "selected_on_validation.n_prompts must be an integer >= 40"
            ) from None
        if validation_prompts < 40:
            raise ValueError(
                f"selected_on_validation.n_prompts {validation_prompts} < 40"
            )
        strata_block = validation_block.get("strata")
        if not isinstance(strata_block, Mapping):
            raise ValueError(
                "selected_on_validation.strata must map the four length bands"
            )
        for band in ("32-64", "65-128", "129-256", "257-512"):
            try:
                band_count = int(strata_block.get(band))
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"selected_on_validation.strata.{band} must be an integer >= 10"
                ) from None
            if band_count < 10:
                raise ValueError(
                    f"selected_on_validation.strata.{band} {band_count} < 10"
                )
    # The executed hyperparameters must be the ones the validation study chose:
    # default to them when supplied, and refuse a conflicting explicit value.
    if validation_block is not None:
        selected_lr = validation_block.get("lr")
        selected_lambda = validation_block.get("lambda_dm")
        if selected_lr is not None:
            if lr is None:
                lr = float(selected_lr)
            elif isinstance(lr, Mapping):
                try:
                    values = {float(value) for value in lr.values()}
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        "explicit lr mapping values must be numeric"
                    ) from None
                if values != {float(selected_lr)}:
                    raise ValueError(
                        f"explicit lr {lr!r} conflicts with "
                        f"selected_on_validation.lr {selected_lr}; the executed "
                        "value must equal the selection"
                    )
            elif float(lr) != float(selected_lr):
                raise ValueError(
                    f"explicit lr {lr} conflicts with selected_on_validation.lr "
                    f"{selected_lr}; the executed value must equal the selection"
                )
            lr = float(selected_lr) if not isinstance(lr, Mapping) else lr
        if selected_lambda is not None:
            if (lambda_dm is not None
                    and float(lambda_dm) != float(selected_lambda)):
                raise ValueError(
                    f"explicit lambda_dm {lambda_dm} conflicts with "
                    f"selected_on_validation.lambda_dm {selected_lambda}"
                )
            lambda_dm = float(selected_lambda)
    if lambda_dm is None:
        lambda_dm = 0.0
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        # The plaintext control arm observes the clean model and needs its
        # weights; refusal is clearer than silently running an S-public surface.
        raise ValueError(
            "run_c05 needs the public clean model for its plaintext control; "
            f"scenario {scenario!r} does not expose it (allowed: "
            f"{sorted(CLEAN_ACCESS_SCENARIOS)})"
        )
    t0 = time.time()
    layers = tuple(isa_layers) if isa_layers else (1, 5)
    observed_layers = getattr(dep, "layers", layers)
    if set(layers) - set(observed_layers):
        raise ValueError(f"requested ISA layers {layers} are outside observed layers {observed_layers}")
    # Use probe defaults only when no learning rate is supplied. ``executed_lr`` is the
    # single scalar only when every sampled layer shares one table value.
    if lr is None and method == "aloepri_state":
        lr = 5e-4  # configs/aloepri/isa_state.yaml, not a paper-specified value
    lr_by_layer = _isa_lr_by_layer(lr, layers)
    distinct_lr = sorted(set(lr_by_layer.values()))
    executed_lr = distinct_lr[0] if len(distinct_lr) == 1 else None
    attack_lr = executed_lr if executed_lr is not None else dict(lr_by_layer)
    if strata:
        selected, stratum_counts = select_isa_prompts(dep, texts, strata)
    else:
        selected = list(texts[:prompts])
        stratum_counts = {name: 0 for name, _, _ in _ISA_STRATA}
        for text in selected:
            name = _isa_length_bucket(dep, text)
            if name is not None:
                stratum_counts[name] += 1
    ab_wires = []
    plain_wires = []
    wire_prompt: list[int] = []
    wire_key: list[int] = []
    for prompt_index, text in enumerate(selected):
        for key_offset in range(int(keys_per_prompt)):
            wire_index = prompt_index * int(keys_per_prompt) + key_offset
            ab_wires.append(
                dep.observe(text, arm=arm, request_id=wire_index, key_index=wire_index)
            )
            plain_wires.append(
                dep.observe(text, arm="plaintext", request_id=wire_index,
                            key_index=wire_index)
            )
            wire_prompt.append(prompt_index)
            wire_key.append(key_offset)
    ab = isa_attack(
        dep.public_view(arm, scenario), [wire.public for wire in ab_wires],
        method=method, layers=layers, steps=steps, lr=attack_lr,
        lambda_dm=lambda_dm, restarts=restarts, seed=seed, verbose=verbose,
    )
    control = isa_attack(
        dep.public_view("plaintext", scenario),
        [wire.public for wire in plain_wires], method=method, layers=layers,
        steps=steps, lr=attack_lr, lambda_dm=lambda_dm, restarts=restarts,
        seed=seed, verbose=False,
    )

    def _asr(run: dict, truths: list[np.ndarray]) -> dict:
        out: dict = {}
        for key, layer in run["layers"].items():
            out[key] = {}
            for step in steps:
                entry = layer["checkpoints"].get(int(step)) or layer["best"]
                candidates = entry.get("candidates") or [
                    [[int(value)] for value in recovered]
                    for recovered in entry["recovered_token_ids"]
                ]
                metrics = {}
                for k in KS:
                    hits: list[bool] = []
                    for wire_candidates, truth in zip(candidates, truths):
                        truth_arr = np.asarray(truth, dtype=np.int64)
                        # Every truth position stays in the denominator; a position
                        # the attack never produced is a miss, never dropped.
                        for pos in range(int(truth_arr.size)):
                            hits.append(
                                pos < len(wire_candidates)
                                and int(truth_arr[pos]) in wire_candidates[pos][:k]
                            )
                    metrics[f"top{k}"] = {
                        "value": float(np.mean(hits)) if hits else None,
                        "numerator": int(sum(hits)),
                        "denominator": int(len(hits)),
                    }
                out[key][f"steps{int(step)}"] = metrics
        return out

    ab_truth = [np.asarray(wire.truth.permuted_ids, dtype=np.int64) for wire in ab_wires]
    plain_truth = [np.asarray(wire.truth.plaintext_ids, dtype=np.int64) for wire in plain_wires]
    ab_curve = _asr(ab, ab_truth)
    control_curve = _asr(control, plain_truth)
    last_key = f"steps{int(max(steps))}"
    # S04: a per-layer verdict, aggregated by min (all used layers must pass).
    # ``max(control_last)`` let a shallow layer's passing control hide a deep
    # layer's failure.
    control_summary = _layer_control_summary(
        control_curve, last_key, required=0.90, attack_curve=ab_curve
    )
    best_control = control_summary["top1"]
    def _layer_candidates(layer_name: str):
        entry = (ab["layers"][layer_name]["checkpoints"].get(int(max(steps)))
                 or ab["layers"][layer_name]["best"])
        return entry.get("candidates") or [
            [[int(value)] for value in recovered]
            for recovered in entry["recovered_token_ids"]
        ]

    def _per_unit_rows(candidates_by_wire) -> list[dict]:
        rows = []
        for wire_index, (candidates, truth_row) in enumerate(
            zip(candidates_by_wire, ab_truth)
        ):
            truth_ids = [int(value) for value in np.asarray(truth_row).reshape(-1).tolist()]
            ranks: list[int] = []
            predicted: list[int] = []
            for position, token in enumerate(truth_ids):
                cand_list = ([int(value) for value in candidates[position]]
                             if position < len(candidates) else [])
                predicted.append(int(cand_list[0]) if cand_list else -1)
                if token in cand_list:
                    ranks.append(int(cand_list.index(token)) + 1)
                else:
                    # Absent from the candidate set: rank > every reported k so
                    # the audit's top-k reconciliation matches _asr.
                    ranks.append(max(len(cand_list) + 1, max(KS) + 1))
            hits = sum(1 for value in ranks if value <= 1)
            rows.append({
                "unit": wire_index,
                "prompt": int(wire_prompt[wire_index]),
                "key": int(wire_key[wire_index]),
                "hits": int(hits),
                "count": int(len(truth_ids)),
                "rank": ranks,
                "predicted": predicted,
                "truth": truth_ids,
            })
        return rows

    # ISA reports every sampled layer. The headline table stays layers[0] for
    # _evidence_gaps; each layer's own curve reconciles against that layer's
    # rows (ISA-AUD-02 follow-up).
    layer_entries = {f"layer{layer}": _layer_candidates(f"layer{layer}")
                     for layer in layers}
    per_unit = _per_unit_rows(layer_entries[f"layer{layers[0]}"])
    per_unit_by_layer = {
        name: _per_unit_rows(candidates) for name, candidates in layer_entries.items()
    }
    token_total = sum(int(row["count"]) for row in per_unit)
    primary_layer = f"layer{layers[0]}"
    # Micro CI for the headline layer: resample prompts and recompute
    # sum(hits)/sum(positions). protocol_audit binds this interval to
    # ``payload["primary_layer"]``'s final-checkpoint top1 (never the max across
    # layers, which would describe a different estimate).
    headline_ci = ratio_ci(
        [row["hits"] for row in per_unit],
        [row["count"] for row in per_unit],
        clusters=[row["prompt"] for row in per_unit],
        seed=seed, unit="positions",
    )
    # Saturation is a measured quantity of this run: the primary layer's final
    # two budget checkpoints. Fewer than two steps cannot support a gain, so the
    # record is omitted then (never fabricated).
    measured_saturation = _saturation_record(ab_curve, primary_layer, steps)
    payload = result_payload(
        attack_name={"er": "isa.er", "tbs": "isa.tbs", "aloepri_state": "isa.aloepri_state"}[method],
        scheme_name=f"ab_deployment:{arm}",
        target=f"{dep.artifact}/model_server ({arm})",
        metrics={
            "budget_curve": ab_curve,
            "budget_curve_id_space": "permuted_deployment_token_ids",
            "end_to_end_plaintext_recovery": None,
            "plaintext_control_curve": control_curve,
            "steps": list(steps), "lr": executed_lr,
            "lr_by_layer": {int(layer): float(value)
                            for layer, value in lr_by_layer.items()},
            "lambda_dm": float(lambda_dm),
            "restarts": int(restarts),
            "optimizer_steps_total": ab.get("optimizer_steps_total"),
            "trajectory_mode": ab.get("trajectory_mode"),
            "layers": list(layers),
            "method": ab["method"],
            "optimizer": ab["optimizer"],
        },
        diagnostics={
            "candidate_mode": "full",
            "positive_control": {
                "name": "plaintext_hidden_states",
                "top1": best_control,
                "required_top1": 0.90,
                "passed": bool(control_summary["passed"]),
                "aggregation": control_summary["aggregation"],
                "checkpoint": last_key,
                "positive_control_by_layer": control_summary["by_layer"],
            },
            "positive_control_passed": bool(control_summary["passed"]),
            "per_layer_control": control_summary["by_layer"],
            "primary_layer": primary_layer,
            # Wire-level failure accounting (the audit's observation unit) plus
            # the raw failed-run count; never per-run under the wire name.
            "failed_observations": int(ab.get("failed_observations") or 0),
            "failed_runs": int(ab.get("failed_runs") or 0),
            "trajectory_failures": list(ab.get("trajectory_failures") or []),
            "attack_variant": ("isa_aloepri_hidden_state_adaptation" if method == "aloepri_state"
                               else f"isa_{method}_original_class"),
            "target_state": ab.get("target_state", "HiddenState"),
            "attnscore_implemented": bool(ab.get("attnscore_implemented", False)),
            "attnscore_status": ab.get("attnscore_status", ""),
            "uses_known_pairs": False,
            "executed_steps": int(max(steps)),
            "lr": executed_lr,
            "lr_by_layer": {int(layer): float(value)
                            for layer, value in lr_by_layer.items()},
            "lambda_dm": float(lambda_dm),
            "optimizer_steps_total": ab.get("optimizer_steps_total"),
            "restarts": int(restarts),
            "paper_working_point": {
                "paper_alpha_e": 1.0, "paper_alpha_h": 0.2,
                "artifact_alpha_e": dep.secret_metadata.get("alpha_e"),
                "artifact_alpha_h": dep.secret_metadata.get("alpha_h"),
            },
        },
        ci={"top1_micro": headline_ci},
        n_prompts=len(selected),
        n_tokens=int(token_total),
        n_keys=int(keys_per_prompt), n_types=0, reference_tokens=int(sum(ref_counts.values())),
        source={"artifact": str(dep.artifact), "attack": ("isa_aloepri_hidden_state_adaptation"
                if method == "aloepri_state" else f"isa_{method}_original"),
                "scenario": scenario},
        data={"prompts": len(selected), "steps": list(steps), "lr": executed_lr,
              "lr_by_layer": {int(layer): float(value)
                              for layer, value in lr_by_layer.items()},
              "keys_per_prompt": int(keys_per_prompt),
              "observations": int(len(ab_wires)),
              "scenario": scenario},
    )
    payload["attack_id"] = {"er": "isa_er", "tbs": "isa_tbs", "aloepri_state": "isa_aloepri_state"}[method]
    payload["adapter_scenario"] = scenario
    # Headline layer: the layer whose per-unit rows and micro CI the payload
    # reports (and which protocol_audit binds the CI point to).
    payload["primary_layer"] = primary_layer
    if measured_saturation is not None:
        payload["diagnostics"]["saturation"] = measured_saturation
    if validation_block is not None:
        # Measured hyper-parameter selection from the independent validation
        # study; absent when None so the audit fails honestly.
        payload["selected_on_validation"] = dict(validation_block)
        payload["diagnostics"]["selected_on_validation"] = dict(validation_block)
    payload["keys_per_prompt"] = int(keys_per_prompt)
    payload["n_observations"] = int(len(ab_wires))
    payload["length_strata"] = {
        "required": dict(strata or {}),
        "observed": dict(stratum_counts),
    }
    payload["layers"] = [int(layer) for layer in layers]
    payload["executed_steps"] = int(max(steps))
    payload["restarts"] = int(restarts)
    write_payload("C05", {"er": "isa.er", "tbs": "isa.tbs", "aloepri_state": "isa.aloepri_state"}[method], arm, payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "executed_steps": int(max(steps)), "profile": profile,
                   "protocol_sha256": protocol_sha256,
                   "per_unit_predictions": per_unit,
                   "per_unit_predictions_by_layer": per_unit_by_layer})
    if verbose:
        print(f"  C05 ISA-{ab['method']} control_top1={best_control} "
              f"({round(time.time()-t0,1)}s)", flush=True)
    return {arm: ab, "plaintext_control": control, "control_top1": best_control}


def _pii_token_ids(dep: ABDeployment, prompt: str) -> list[list[int]]:
    """Entity token ids for the synthetic PII in ``prompt`` (name/phone/email/address)."""
    import re

    patterns = [
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",              # person names
        r"\b\d{3}-\d{4}\b",                           # phone
        r"\b[\w.]+@[\w.]+\b",                         # email
        r"\b\d+ [A-Z][a-z]+ (Street|Avenue|Road)\b",  # address
    ]
    out = []
    for pat in patterns:
        for m in re.finditer(pat, prompt):
            ids = dep.encode(m.group(0))
            if ids.size:
                out.append([int(x) for x in ids])
    return out



# ---------------------------------------------------------------------------
# config-driven entry point for the runner's ``mode: full`` path
# ---------------------------------------------------------------------------
def run_c11_stream(dep: ABDeployment, texts: Sequence[str] | Mapping[int, str],
                   reference_texts: Sequence[str], *, method: str = "sda2023",
                   num_train: int = 256, epochs: int = 6, beam: int = 10,
                   seed: int = 0, n_val_sequences: int = 0,
                   max_requests: int | None = None, profile: str = "probe",
                   protocol_sha256: str = "",
                   training_params: dict | None = None, split_seed: int = 0,
                   device: str = "cpu", symbol_frontend: str = "rounded",
                   distance_threshold: float | None = None, group_joint: bool = False,
                   checkpoint_path: str | None = None, resume: bool = False,
                   control_keys: int = 1, validation_beams: Sequence[int] | None = None,
                   validation_noise_rate: float = 0.0,
                   validation_noise_kinds: Sequence[str] = ("sub",),
                   decode_cache: bool = False, verbose: bool = False,
                   scoring_clusters: Mapping[int, Mapping[str, str]] | None = None,
                   budget_checkpoints: Sequence[str] | None = None,
                   frozen_uploads: Sequence | None = None) -> dict:
    """Run the lookup-embedding SDA variant on the A+B observation stream."""
    from .sda_stream import run_sda_stream

    if profile != "probe":
        raise ValueError("C11 engineering entry requires profile=probe; the formal executor is not verified")
    if isinstance(n_val_sequences, bool) or not isinstance(n_val_sequences, int) or n_val_sequences < 0:
        raise ValueError("n_val_sequences must be a nonnegative integer")
    if frozen_uploads is not None:
        from ..schemes.adapters import UploadSurface
        frozen_uploads = list(frozen_uploads)
        if not frozen_uploads or any(upload.surface != UploadSurface.VECTORS for upload in frozen_uploads):
            raise ValueError("C11 A+B frozen uploads must be nonempty vector observations")
    t0 = time.time()
    out = run_sda_stream(
        dep, texts, reference_texts, method=method, num_train=num_train,
        epochs=epochs, beam_width=beam, seed=seed, max_requests=max_requests,
        positive_control=True, control_keys=control_keys, verbose=verbose, training_params=training_params,
        split_seed=split_seed, n_validation=n_val_sequences or None, device=device, symbol_frontend=symbol_frontend,
        distance_threshold=distance_threshold, group_joint=group_joint,
        checkpoint_path=checkpoint_path, resume=resume, validation_beams=validation_beams,
        validation_noise_rate=validation_noise_rate, validation_noise_kinds=validation_noise_kinds,
        decode_cache=decode_cache, scoring_clusters=scoring_clusters, budget_checkpoints=budget_checkpoints,
        frozen_uploads=frozen_uploads, require_public_groups=True,
    )
    control = out.get("positive_control") or {}
    if out["public_group_count"] is None:
        raise ValueError("SDA result cannot report key groups: public group identifiers are missing")
    payload = result_payload(
        attack_name="sda.sda2021" if method == "sda2021" else "sda.sda2023",
        scheme_name="ab_deployment:ab",
        target=str(getattr(dep, "public_artifact", dep.artifact)),
        metrics=out,
        diagnostics={
            "candidate_mode": "full",
            "positive_control": control,
            "positive_control_passed": control.get("passed"),
            "attack_variant": method + "_" + out["model_config"].get("representation", "lookup") + "_probe",
            "training_pairs": out["training_pairs_source"],
            "uses_known_pairs": False,
            "upload_provenance": ("caller_supplied_ab_unverified" if frozen_uploads is not None
                                  else "deployment_observation"),
        },
        ci={},
        n_prompts=out["n_requests"],
        n_tokens=out["token_accuracy"]["denominator"],
        n_keys=out["public_group_count"],
        n_types=out["cipher_alphabet_size"],
        reference_tokens=out["reference_tokens"],
        source={"artifact": str(dep.artifact), "attack": method},
        data={"requests": out["n_requests"], "epochs": epochs, "beam": beam},
        overrides={"rotation_aware": False},
    )
    payload["attack_id"] = "sda"
    payload["n_train_sequences"] = out["n_train_sequences"]
    payload["n_training_pairs"] = out["n_training_pairs"]
    validation = out["validation"]
    payload["n_val_sequences"] = validation.get("n_documents", 0)
    payload["validation"] = validation if validation["status"] == "selected" else {
        "status": "not_run", "requested": int(n_val_sequences),
        "reason": "validation beam selection was not requested; paper-scale SDA remains blocked",
    }
    payload["n_epochs"] = int(epochs)
    payload["beam"] = int(out["beam_width"])
    per_unit = [
        {
            "unit": int(row["request_id"]),
            "hits": int(row["accuracy"]["numerator"]),
            "count": int(row["accuracy"]["denominator"]),
        }
        for row in out["rows"]
    ]
    write_payload("C11", "sda.sda2021" if method == "sda2021" else "sda.sda2023",
                  "ab", payload,
                  {"seconds": round(time.time() - t0, 1), "seed": seed,
                   "executed_steps": out["optimizer_steps"], "profile": profile,
                   "protocol_sha256": protocol_sha256,
                   "per_unit_predictions": per_unit})
    return out


def _resolve_c11_probe_config(cfg: dict) -> dict:
    """Resolve one explicitly named engineering SDA variant without silent defaults."""
    for name, default, upper in (("seed", 0, 2**32), ("split_seed", 0, 2**64)):
        value = cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
            raise ValueError(f"{name} must be an unsigned integer less than {upper}")
    for name, default, minimum in (("requests", 1000, 1), ("sda_val", 0, 0)):
        value = cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    noise_rate = cfg.get("validation_noise_rate", 0.0)
    if (isinstance(noise_rate, bool) or not isinstance(noise_rate, (int, float)) or
            not np.isfinite(noise_rate) or not 0 <= noise_rate <= 1):
        raise ValueError("validation_noise_rate must be a number between 0 and 1")
    noise_kinds = cfg.get("validation_noise_kinds", ("sub",))
    if (not isinstance(noise_kinds, (tuple, list)) or not noise_kinds or
            any(not isinstance(kind, str) or kind not in {"sub", "ins", "del"} for kind in noise_kinds)):
        raise ValueError("validation_noise_kinds must be a nonempty list of sub, ins or del")
    attack = cfg.get("attack")
    if not isinstance(attack, dict):
        raise ValueError("C11 requires an explicit attack.name, attack.variant and attack.params")
    name = attack.get("name")
    variants = {"sda.sda2021": ("sda2021", {"frequency_rank_lookup_variant"}),
                "sda.sda2023": ("sda2023", {"recurrence_lookup_variant", "recurrence_decimal_char_cnn_variant"})}
    if name not in variants:
        raise ValueError(f"unsupported C11 attack name: {name!r}")
    method, expected_variants = variants[name]
    variant = attack.get("variant")
    if variant not in expected_variants:
        raise ValueError(f"C11 variant {variant!r} is not verified; use one of {sorted(expected_variants)!r} only for a probe")
    if cfg.get("profile") != "probe":
        raise ValueError("C11 bridge requires explicit profile=probe; formal executor remains blocked")
    params = attack.get("params")
    if not isinstance(params, dict):
        raise ValueError("C11 attack.params must be a mapping")
    if method == "sda2023":
        representation = params.get("representation", "lookup")
        expected_representation = ("lookup" if variant == "recurrence_lookup_variant" else "decimal_char_cnn")
        if representation != expected_representation:
            raise ValueError(f"C11 variant {variant!r} requires representation={expected_representation!r}")
    required = {"num_train", "epochs", "beam_width", "min_len", "max_len"}
    if required - params.keys():
        raise ValueError(f"C11 missing explicit attack parameters: {sorted(required - params.keys())}")
    for key in required:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"C11 {key} must be a positive integer")
    common = {"num_train", "epochs", "beam_width", "min_len", "max_len", "max_cipher_len",
              "batch_size", "lr", "d_model", "nhead", "num_layers", "dim_ff", "dropout",
              "checkpoint_path", "resume", "decode_mode", "decode_cache", "synthesis_mode", "sampling_mode", "sampling_seed", "pair_storage", "pair_storage_dir", "window_manifest_path", "cluster_storage", "cluster_temp_dir"}
    specific = {"noise_rate", "noise_kinds", "length_mode", "max_output_len", "space_token_id"} if method == "sda2021" else {"homophonic", "homophone_symbols", "representation"}
    unknown = set(params) - common - specific
    if unknown:
        raise ValueError(f"C11 unsupported attack parameters: {sorted(unknown)}")
    values = dict(params)
    num_train = values.pop("num_train")
    epochs = values.pop("epochs")
    beam = values.pop("beam_width")
    decode_cache = values.pop("decode_cache", False)
    if not isinstance(decode_cache, bool):
        raise ValueError("decode_cache must be a boolean")
    if decode_cache and method != "sda2023":
        raise ValueError("KV cache is currently supported only for SDA2023")
    return {"method": method, "num_train": int(num_train), "epochs": int(epochs),
            "beam": int(beam), "decode_cache": decode_cache, "training_params": values}


def _run_c11_bridge(dep: ABDeployment, corpus, ref_counts, cfg) -> dict:
    resolved = _resolve_c11_probe_config(cfg)
    return run_c11_stream(
        dep, corpus["private"].texts()[:cfg.get("requests", 1000)],
        corpus["reference"].texts(), **resolved, seed=cfg.get("seed", 0),
        n_val_sequences=cfg.get("sda_val", 0), split_seed=cfg.get("split_seed", 0),
        device=str(cfg.get("device", "cpu")), profile="probe",
        symbol_frontend=cfg.get("symbol_frontend", "rounded"),
        distance_threshold=cfg.get("distance_threshold"), group_joint=cfg.get("group_joint", False),
        validation_beams=cfg.get("validation_beams"), control_keys=cfg.get("control_keys", 1),
        validation_noise_rate=float(cfg.get("validation_noise_rate", 0.0)),
        validation_noise_kinds=tuple(cfg.get("validation_noise_kinds", ("sub",))),
    )


def _run_c10_bridge(dep: ABDeployment, corpus, ref_counts, cfg) -> dict:
    """C10 with protocol-scale streams and a measured saturation study.

    Texts are split into 3 disjoint near-equal streams only when the selected
    corpus reaches the protocol request floor (>=15000); below it the run stays
    honestly single-stream. ``cfg["saturation_study"]`` (a mapping with
    ``saturation``/``saturation_curve``/``saturation_variant``) is passed through
    when supplied and never fabricated.
    """
    texts = corpus["private"].texts()[: int(cfg.get("requests", 1000))]
    study = cfg.get("saturation_study") or {}
    return run_c10(
        dep, texts, ref_counts,
        n_keys=int(cfg.get("keys", 16)),
        rotate_every=int(cfg.get("rotate_every", 64)),
        seed=int(cfg.get("seed", 0)),
        profile=str(cfg.get("profile", "probe")),
        protocol_sha256=str(cfg.get("protocol_sha256", "")),
        text_sets=_partition_texts(texts),
        saturation=study.get("saturation") if isinstance(study, Mapping) else None,
        saturation_curve=(
            study.get("saturation_curve") if isinstance(study, Mapping) else None
        ),
        saturation_variant=(
            study.get("saturation_variant") if isinstance(study, Mapping) else None
        ),
    )


def _run_c12_bridge(dep: ABDeployment, corpus, ref_counts, cfg) -> dict:
    scenario = str(cfg.get("scenario", "S-public"))
    mapping = recover_mapping_gate_ia(
        dep, n_types=int(cfg.get("stage2_types", 4000)), top_k=1,
        seed=int(cfg.get("seed", 0)), scenario=scenario,
    )["mapping"]
    stage2 = {
        int(row): (int(values[0]) if isinstance(values, (list, tuple)) and values else values)
        for row, values in mapping.items()
    }
    return run_c12(
        dep, corpus["private"].texts()[: int(cfg.get("requests", 1000))],
        n_keys=int(cfg.get("keys", 16)),
        rotate_every=int(cfg.get("rotate_every", 64)),
        seed=int(cfg.get("seed", 0)), stage2_mapping=stage2,
        aggregation_threshold=cfg.get("aggregation_threshold"),
        aggregation_repeat_labels=cfg.get("aggregation_repeat_labels"),
        profile=str(cfg.get("profile", "probe")),
        protocol_sha256=str(cfg.get("protocol_sha256", "")),
    )


def _run_c14_bridge(dep: ABDeployment, corpus, ref_counts, cfg) -> dict:
    """C14 through the public view and an attack-estimated stage-2 mapping."""
    from .real_remaining import run_c14

    texts = corpus["vma"].texts()[: int(cfg.get("gen_prompts", 4))]
    scenario = str(cfg.get("scenario", "S-public"))
    wires = [
        dep.observe(text, arm="ab", request_id=i, key_index=i)
        for i, text in enumerate(texts)
    ]
    mapping = recover_mapping_gate_ia(
        dep, n_types=int(cfg.get("stage2_types", 4000)), top_k=1,
        seed=int(cfg.get("seed", 0)), scenario=scenario,
    )["mapping"]
    stage2 = {
        int(row): (int(values[0]) if isinstance(values, (list, tuple)) and values else values)
        for row, values in mapping.items()
    }
    return run_c14(
        dep.public_view("ab", scenario), [wire.public for wire in wires],
        stage2_mapping=stage2, truth=texts,
    )


CLASS_RUNNERS = {
    "C01": lambda dep, corpus, ref_counts, cfg: run_c01(
        dep, corpus["knn"].texts()[: cfg.get("knn_texts", 1000)], ref_counts),
    "C06W": lambda dep, corpus, ref_counts, cfg: run_c06w(
        dep, n_types=cfg.get("weight_types", 1000), seed=cfg.get("seed", 0),
        scenario=str(cfg.get("scenario", "S-public"))),
    "C08": lambda dep, corpus, ref_counts, cfg: run_c08(
        dep, n_types=cfg.get("weight_types", 1000), seed=cfg.get("seed", 0),
        scenario=str(cfg.get("scenario", "S-public"))),
    "D3": lambda dep, corpus, ref_counts, cfg: run_composed(
        dep, corpus["knn"].texts()[: cfg.get("knn_texts", 200)], ref_counts,
        n_types=cfg.get("stage2_types", 4000), seed=cfg.get("seed", 0),
        scenario=str(cfg.get("scenario", "S-public"))),
    "C10": _run_c10_bridge,
    "C12": _run_c12_bridge,
    "C13": lambda dep, corpus, ref_counts, cfg: run_c13_c14(
        dep, prompts=corpus["vma"].texts()[: cfg.get("gen_prompts", 3)],
        annotations=cfg.get("output_annotations"), output_mapping=cfg.get("output_mapping"),
        trace_root=cfg.get("output_trace_root"),
        max_new_tokens=cfg.get("max_new_tokens", 8), seed=cfg.get("seed", 0)),
    "C14": _run_c14_bridge,
    "C11": _run_c11_bridge,
    "C05": lambda dep, corpus, ref_counts, cfg: run_c05(
        dep, corpus["vma"].texts(), ref_counts,
        prompts=cfg.get("isa_prompts", 1),
        steps=tuple(cfg.get("isa_steps", [20, 100])),
        lr=cfg.get("isa_lr"),
        lambda_dm=cfg.get("isa_lambda_dm"),
        seed=cfg.get("seed", 0),
        isa_layers=tuple(cfg.get("isa_layers", (1,))),
        method=str(cfg.get("isa_method", "er")),
        restarts=int(cfg.get("isa_restarts", 1)),
        keys_per_prompt=cfg.get("keys", 1),
        strata=cfg.get("isa_length_strata") or None,
        selected_on_validation=cfg.get("isa_validation"),
        profile=str(cfg.get("profile", "probe")),
        protocol_sha256=str(cfg.get("protocol_sha256", "")),
        scenario=str(cfg.get("scenario", "S-public"))),
}


def check_backend_available(backend: dict) -> list[str]:
    """Return the *specific* missing resources for a ``full_backend`` block.

    The runner must not answer ``resource_blocked`` for a backend whose files are
    actually present (that is a fabricated blockage), and must not answer ``ok``
    for one whose files are missing (that would fabricate metrics). Both cases are
    decided here, with the offending path named.
    """
    from ..data.corpora import DEFAULT_ALPACA

    missing = []
    artifact = Path(backend.get("artifact", DEFAULT_ARTIFACT))
    clean = Path(backend.get("clean_model", DEFAULT_CLEAN))
    if not (artifact / "model_server" / "config.json").exists():
        missing.append(f"a+b artifact: {artifact}")
    if not (artifact / "client" / "client_secret.pt").exists():
        missing.append(f"client secret: {artifact / 'client' / 'client_secret.pt'}")
    if not (artifact / "osnip" / "encryptor.pt").exists():
        missing.append(f"osnip encoder: {artifact / 'osnip' / 'encryptor.pt'}")
    if not (clean / "config.json").exists():
        missing.append(f"clean reference model: {clean}")
    alpaca_path = Path(backend["alpaca"]) if backend.get("alpaca") else DEFAULT_ALPACA
    if not alpaca_path.exists():
        missing.append(f"alpaca snapshot: {alpaca_path}")
    return missing


def run_full_config(config: dict) -> dict:
    """Execute a ``mode: full`` config against the packaged artifact.

    Returns the same document shape the runner writes for smoke runs
    (``status`` / ``status_reason`` / ``metrics`` / ``result``), with one App.-A
    payload per executed cell already on disk under ``OUT_ROOT``.
    """
    attack_cfg = config.get("attack")
    if not isinstance(attack_cfg, dict):
        attack_cfg = {}
    requested_classes = [str(c).upper() for c in (config.get("classes") or [])]
    if attack_cfg.get("name") in {"sda.sda2021", "sda.sda2023"} or "C11" in requested_classes:
        try:
            _resolve_c11_probe_config(config)
        except ValueError as exc:
            return {"status": "resource_blocked", "status_reason": str(exc),
                    "metrics": {"status": "resource_blocked"}, "result": None}
        if "C11" not in requested_classes:
            return {"status": "resource_blocked",
                    "status_reason": "explicit classes: [C11] is required for the engineering probe",
                    "metrics": {"status": "resource_blocked"}, "result": None}
    backend = dict(config.get("full_backend") or {})
    if not backend:
        # A ``mode: full`` config that does not declare a backend has not opted into
        # the real-artifact path; say so, instead of silently using default paths (a
        # test or a CI box would otherwise start a 1B-model run).
        return {
            "status": "resource_blocked",
            "status_reason": (
                "mode=full but no `full_backend` declared: add "
                "{'artifact': ..., 'clean_model': ..., 'alpaca': ...} to run the real "
                "A+B battery (see scripts/run_full_battery.py)"
            ),
            "metrics": {"status": "resource_blocked"},
            "result": None,
        }
    missing = check_backend_available(backend)
    if missing:
        return {
            "status": "resource_blocked",
            "status_reason": "missing real backend resources: " + "; ".join(missing),
            "metrics": {"status": "resource_blocked", "missing": missing},
            "result": None,
        }
    classes = [str(c).upper() for c in (config.get("classes") or ["C01"])]
    seed = int(config.get("seed", 20261106))
    corpus = load_alpaca(Path(backend["alpaca"]) if backend.get("alpaca") else None,
                         seed=seed, n_knn=max(int(config.get("knn_texts", 1000)), 1))
    dep = ABDeployment(artifact=Path(backend.get("artifact", DEFAULT_ARTIFACT)),
                       clean_model=Path(backend.get("clean_model", DEFAULT_CLEAN)),
                       layers=tuple(config.get("layers", (1, 5, 10, 15))),
                       max_len=int(config.get("max_len", 64)))
    ref_counts, ref_tokens, _ = reference_counts(corpus["reference"].texts(), dep.encode,
                                                 vocab_size=dep.vocab_size)
    summary: dict = {
        "artifact": str(dep.artifact),
        "artifact_metadata": dep.secret_metadata,
        "classes": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for name in classes:
        runner = CLASS_RUNNERS.get(name)
        if runner is None:
            summary["classes"][name] = {"status": "not_applicable",
                                        "reason": f"no real-backend runner for class {name}"}
            continue
        summary["classes"][name] = runner(dep, corpus, ref_counts, config)
    return {
        "status": "ok",
        "status_reason": "",
        "metrics": {"reference_tokens": int(ref_tokens), "classes": classes},
        "result": summary,
    }
