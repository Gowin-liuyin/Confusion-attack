"""Stream a reviewable paper-v1 matrix without declaring any run complete.

The protocol remains authoritative. Specs describe planned units, not runnable
registrations; unsupported axes and unfrozen resources stay explicitly unresolved.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterator, Mapping

from .paper_profile import LAYERS_1B, PAPER_PROFILE_NAME, PRIMARY_MODEL, plan_jobs


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _defense_axes(arm: str, floors: dict) -> tuple[list[dict], list[str]]:
    seeds = floors["training_seeds"]
    grids = floors["baseline_grids"]
    noise = [dict(alpha_e=pair[0], alpha_h=pair[1]) for pair in floors["weight"]["noise_tiers"]]
    if arm == "plaintext":
        return [{}], []
    if arm == "aloepri":
        return [{"noise": tier, "random_replica": replica} for tier in noise for replica in range(3)], []
    if arm == "ab":
        return [{"noise": tier, "defense_training_seed": seed, "b_workpoint": None}
                for tier in noise for seed in seeds], ["B workpoint and matching trained artifact are not bound"]
    if arm == "osnip":
        return [{"defense_training_seed": seed, "b_workpoint": None} for seed in seeds], [
            "clean-model B workpoints and matching trained artifacts are not bound"]
    if arm == "obfuslm_plus":
        return [{"k": k, "epsilon": epsilon, "defense_training_seed": seed}
                for k, epsilon, seed in itertools.product(grids["obfuslm_k"], grids["obfuslm_epsilon"], seeds)], []
    if arm == "sentinellms":
        return [{"glide": glide, "replica": replica} for glide, replica in
                itertools.product(grids["sentinel_glide"], range(3))], [
            "training requirement and decoder adaptation require verified baseline evidence"]
    if arm == "dp_forward":
        return [{"epsilon": epsilon, "defense_training_seed": seed} for epsilon, seed in
                itertools.product(grids["epsilon"], seeds)], []
    if arm == "inferdpt":
        return [{"epsilon": epsilon, "random_replica": replica} for epsilon, replica in
                itertools.product(grids["epsilon"], range(3))], []
    raise ValueError(f"unknown arm: {arm}")


def _layers(model: str, manifest: Mapping | None) -> tuple[list[int | None], list[str]]:
    if model == PRIMARY_MODEL:
        return list(LAYERS_1B), []
    models = manifest.get("models", {}) if isinstance(manifest, Mapping) else {}
    info = models.get(model, {}) if isinstance(models, Mapping) else {}
    depth = info.get("num_hidden_layers") if isinstance(info, Mapping) else None
    if type(depth) is not int or depth < 3:
        return [None], ["model depth is not frozen; layer set cannot be expanded"]
    return sorted({1, math.ceil(depth / 3), math.ceil(2 * depth / 3), depth - 1}), [
        "model depth is manifest-declared and must be verified against config bytes"]


def _stream_axes(floors: dict) -> list[dict]:
    stream = floors["stream"]
    schedules = [{"mode": "fixed"}, {"mode": "fresh_per_request"}]
    schedules.extend({"mode": "pool", "key_pool": pool, "rotate_every": period}
                     for pool, period in itertools.product(stream["key_pool"], stream["rotate_every"]))
    return [{"rotation": schedule, "stream_index": index}
            for schedule, index in itertools.product(schedules, range(stream["n_streams"]))]


def _attack_axes(attack: str, model: str, scenario: str, floors: dict,
                 manifest: Mapping | None) -> tuple[list[dict], list[str]]:
    unresolved: list[str] = []
    if attack.startswith("vma_h_") or attack in {"isa_er", "isa_tbs"}:
        layers, unresolved = _layers(model, manifest)
        if attack.startswith("vma_h_"):
            return [{"layer": layer} for layer in layers], unresolved
        isa = floors["isa"]
        variants = [{"phase": "validation", "learning_rate": lr, "lambda_dm": weight,
                     "steps": isa["steps"]} for lr, weight in itertools.product(isa["lr_grid"], isa["lambda_dm"])]
        variants.append({"phase": "test", "configuration": "selected_on_validation", "steps": isa["steps"]})
        variants.extend({"phase": "test_extension", "configuration": "selected_on_validation", "steps": step,
                         "condition": "previous checkpoint not saturated"} for step in isa["extension_steps"])
        return [{"layer": layer, **variant} for layer, variant in itertools.product(layers, variants)], unresolved
    if attack in {"ima_ridge", "ima_neural"}:
        ima = floors["ima"]
        data_sizes = [{"train_docs": ima["train_docs"], "training_data_condition": "initial"}] + [
            {"train_docs": size, "training_data_condition": "token floor or saturation requires expansion"}
            for size in ima["expansion_docs"]]
        if attack == "ima_ridge":
            variants = [{"phase": "validation", "ridge_lambda": value} for value in ima["ridge_lambda"]]
            variants.append({"phase": "test", "ridge_lambda": "selected_on_validation"})
        else:
            variants = [{"phase": "validation", "attack_training_seed": seed, "learning_rate": lr,
                         "epoch_checkpoint": epoch,
                         "epoch_condition": "extension if unsaturated" if epoch == ima["epoch_extension"] else "scheduled checkpoint"}
                        for seed, lr, epoch in itertools.product(floors["training_seeds"], ima["lr"],
                                                                 [*ima["epochs"], ima["epoch_extension"]])]
            variants.extend({"phase": "test", "attack_training_seed": seed,
                             "learning_rate": "selected_on_validation", "epoch_checkpoint": epoch,
                             "epoch_condition": "extension if unsaturated" if epoch == ima["epoch_extension"] else "scheduled checkpoint"}
                            for seed, epoch in itertools.product(floors["training_seeds"],
                                                                  [*ima["epochs"], ima["epoch_extension"]]))
        conditions = ["attacker_owned_instance", "cross_key"]
        if scenario == "S-pair":
            conditions.append("target_pairs")
        return [{**size, **variant, "information_condition": condition, "attacker_instance": instance}
                for size, variant, condition in itertools.product(data_sizes, variants, conditions)
                for instance in ([None] if condition == "target_pairs" else
                                 [floors["training_seeds"].index(variant["attack_training_seed"])]
                                 if "attack_training_seed" in variant else range(3))], [
            "attack data, tokenizer, instance provenance and model architecture require frozen evidence"]
    stream_attacks = {"linkage_exact", "linkage_cosine", "linkage_knn_aligned", "linkage_recovered_text",
                      "aggregation", "tfma", "tfma_after_inversion", "sda"}
    if attack in stream_attacks:
        variants = [{}]
        if attack == "aggregation":
            variants = [{"observations": n} for n in floors["aggregation_m"]]
        elif attack in {"tfma", "tfma_after_inversion"}:
            variants = [{"prior": prior, "reference_tokens": tokens,
                         "condition": "20M tier when earlier reference tier is insufficient" if tokens == 20_000_000 else "scheduled tier"}
                        for prior, tokens in itertools.product(("general", "same_domain", "matched_proxy"), floors["tfma"]["curve"])]
        elif attack == "sda":
            variants = [{"algorithm": None, "prior": prior, "attack_training_seed": seed, "beam": beam,
                         "epoch_checkpoint": epoch, "attacker_instance": floors["training_seeds"].index(seed)}
                        for prior, seed, beam, epoch in itertools.product(
                            ("general", "same_domain", "matched_proxy"), floors["training_seeds"],
                            floors["sda"]["beam"], floors["sda"]["epochs"])]
        missing = ["stream and session data revisions and split evidence are not bound by this planner"]
        if attack in {"tfma", "tfma_after_inversion", "sda"}:
            missing.append("prior corpus revisions, licenses and disjointness require verification")
        if attack == "sda":
            missing.append("SDA algorithm, architecture and tokenizer must be frozen from source evidence")
        return [{**stream, **variant} for stream, variant in itertools.product(_stream_axes(floors), variants)], missing
    if attack == "text_bayes":
        return [{"attack_training_seed": seed, "beam": beam, "epoch_checkpoint": epoch}
                for seed, beam, epoch in itertools.product(floors["training_seeds"], floors["text"]["bayes_beam"],
                                                            floors["text"]["bayes_epochs"])], [
            "Bayesian attack model and data revisions must be frozen"]
    if attack == "efficiency":
        efficiency = floors["efficiency"]
        return [{"input_length": length, "concurrency": concurrent, "round_index": index}
                for length, concurrent, index in itertools.product(efficiency["input_lengths"],
                                                                     efficiency["concurrency"],
                                                                     range(efficiency["rounds"]))], [
            "hardware, precision, workload and complete service path must be frozen"]
    if attack == "utility_generation":
        return [{"task": task} for task in floors["utility"]["generation_tasks"]], []
    if attack == "utility_classification":
        key = "cross_family_classification_tasks" if model == "qwen2.5-7b-instruct" else "classification_tasks"
        return [{"task": task} for task in floors["utility"][key]], []
    if attack == "output_pii":
        return [{}], ["independent input/output entity annotations and generation traces are not bound"]
    if attack == "text_lm_rerank":
        return [{}], ["attacker model revision and readable observation source must be frozen"]
    return [{}], []


def iter_expanded_specs(profile, manifest: Mapping | None = None, *,
                        filters: Mapping[str, set[str]] | None = None) -> Iterator[dict]:
    """Yield content-addressed planned units; never mark a spec executed or verified."""
    if profile.name != PAPER_PROFILE_NAME:
        raise ValueError("expanded matrix requires paper-v1")
    filters = filters or {}
    if set(filters) - {"arms", "models", "scenarios", "attacks"}:
        raise ValueError("unknown matrix filter")
    manifest_digest = _digest(manifest) if manifest is not None else None
    carrier_digest = _digest(profile.floors)
    planner_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for job in plan_jobs(profile, dict(manifest) if manifest is not None else None)["jobs"]:
        if any(field in filters and job[key] not in filters[field] for field, key in
               (("arms", "arm"), ("models", "model"), ("scenarios", "scenario"), ("attacks", "attack"))):
            continue
        if not job["applicable"]:
            axes = [({}, {})]
            unresolved = []
        else:
            defenses, defense_gaps = _defense_axes(job["arm"], profile.floors)
            attacks, attack_gaps = _attack_axes(job["attack"], job["model"], job["scenario"], profile.floors, manifest)
            axes = itertools.product(defenses, attacks)
            unresolved = defense_gaps + attack_gaps
        for defense, attack in axes:
            identity = {"profile": profile.name, "protocol_sha256": profile.protocol_sha256,
                        "manifest_json_sha256": manifest_digest,
                        "execution_carrier_sha256": carrier_digest,
                        "planner_source_sha256": planner_digest,
                        "budget_sha256": _digest(job["budget"]),
                        "arm": job["arm"], "model": job["model"], "scenario": job["scenario"],
                        "attack": job["attack"], "defense": defense, "attack_axes": attack}
            pending = list(unresolved)
            if "attacker_instance" in attack and attack["attacker_instance"] is not None:
                pending.append("independent attacker-instance artifacts must be verified")
            if "random_replica" in defense:
                pending.append("three random replicas are a planning choice; the protocol requires independent retests")
            if manifest_digest is None:
                pending.append("identity manifest not supplied")
            else:
                pending.append("manifest file hashes are declared, not verified by this planner")
            yield {"plan_spec_id": "plan-" + _digest(identity), "identity": copy.deepcopy(identity),
                   "source_job_id": job["job_id"], "source_job_status": job["status"],
                   "applicable": job["applicable"],
                   "status": "not_applicable" if not job["applicable"] else "not_run",
                   "unresolved": pending, "budget": copy.deepcopy(job["budget"]),
                   "scope": "planning only; not a run ID, execution result, or verification"}


def validate_matrix(path: Path | str) -> dict:
    """Reject incomplete, modified or duplicate-ID matrix streams before use."""
    digest = hashlib.sha256()
    seen: set[str] = set()
    header = None
    trailer = None
    with Path(path).open("rb") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("matrix contains invalid JSON") from exc
            if not isinstance(row, dict) or trailer is not None:
                raise ValueError("matrix has invalid records or data after trailer")
            if header is None:
                if row.get("kind") != "matrix_header":
                    raise ValueError("matrix header missing")
                header = row
                digest.update(line)
            elif row.get("kind") == "plan_spec":
                spec = row.get("spec")
                if not isinstance(spec, dict) or not isinstance(spec.get("identity"), dict):
                    raise ValueError("invalid planned spec")
                spec_id = spec.get("plan_spec_id")
                if spec_id != "plan-" + _digest(spec["identity"]) or spec_id in seen:
                    raise ValueError("matrix contains a modified or duplicate spec identity")
                if spec["identity"].get("protocol_sha256") != header.get("protocol_sha256"):
                    raise ValueError("matrix protocol binding mismatch")
                seen.add(spec_id)
                digest.update(line)
            elif row.get("kind") == "matrix_end":
                trailer = row
            else:
                raise ValueError("unknown matrix record")
    if header is None or trailer is None:
        raise ValueError("matrix is incomplete: header or trailer missing")
    if trailer.get("spec_count") != len(seen) or trailer.get("preceding_lines_sha256") != digest.hexdigest():
        raise ValueError("matrix count or content digest mismatch")
    return {"header": header, "spec_count": len(seen), "preceding_lines_sha256": digest.hexdigest()}


def iter_coverage(path: Path | str, result_paths: list[Path | str]) -> Iterator[dict]:
    """Match declared spec identities to existing result files without certifying them."""
    validate_matrix(path)
    records: dict[str, list[tuple[str, dict]]] = {}
    for source in result_paths:
        source = Path(source)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("plan_spec_id"), str):
            records.setdefault(payload["plan_spec_id"], []).append((str(source), payload))
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") != "plan_spec":
                continue
            spec = row["spec"]
            candidates = records.get(spec["plan_spec_id"], [])
            matching = [path for path, payload in candidates if payload.get("plan_identity") == spec["identity"]]
            mismatched = [path for path, payload in candidates if payload.get("plan_identity") != spec["identity"]]
            if not spec["applicable"]:
                status = "not_applicable"
            elif matching:
                status = "recorded_unverified"
            elif mismatched:
                status = "identity_mismatch"
            else:
                status = "missing"
            yield {"plan_spec_id": spec["plan_spec_id"], "status": status,
                   "matching_results": matching, "identity_mismatches": mismatched,
                   "note": "file presence and declared identity are not an independent result audit"}
