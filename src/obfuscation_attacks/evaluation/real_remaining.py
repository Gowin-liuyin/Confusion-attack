"""Battery wiring for the remaining classes: C02, C03, C07, C09, C11, C14.

Kept in one module so the task cards stay readable: each ``run_cXX`` builds its plan App.-A payload
and states what it could and could not do. The common thread is that these are the *learned* or
*generative* attacks, whose cost is dominated by training/decoding rather than by matrix algebra.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..schemes.ab_deployment import ABDeployment
from .attack_model import result_payload
from .real_battery import _prompt_identity, write_payload
from .real_generative import lm_rewrite, text_metrics


def _decode_mapped(public, mapped) -> str:
    """Decode contiguous token runs together and retain each unmapped position."""
    parts, pending = [], []
    for token in mapped:
        if token is None:
            if pending:
                parts.append(public.decode(pending))
                pending = []
            parts.append("[UNK]")
        else:
            pending.append(int(token))
    if pending:
        parts.append(public.decode(pending))
    return "".join(parts)


def run_c09(dep: ABDeployment, texts: Sequence[str], *, ridge_classes: int = 2000,
            mlp_classes: int = 1000, mlp_epochs: int = 25, seed: int = 0,
            verbose: bool = True) -> dict:
    """IMA on the real wire: ridge + MLP inverter, scored per held-out *request*.

    Threat model: the pairs are minted by the **deployment** mechanism, i.e. the KP appendix variant
    (plan D8). The KI main-table variant needs an attacker-owned encoder instance, which this
    artifact cannot provide (the encoder is client-side and the key schedule is never shipped).
    """
    from .real_inversion import collect_pairs, mlp_inverter, ridge_inverter, split_by_request

    t0 = time.time()
    data = collect_pairs(dep, texts, arm="ab", layer=None)
    split = split_by_request(data, test_fraction=0.25, seed=seed)
    ridge = ridge_inverter(split, n_classes=ridge_classes, seed=seed)
    if verbose:
        print(f"  C09 ridge: top1={ridge['top1']['value']:.3f} top10={ridge['top10']['value']:.3f} "
              f"(classes {ridge['n_classes']}, coverage {ridge['coverage']:.2f})", flush=True)
    mlp = mlp_inverter(split, hidden=256, epochs=mlp_epochs, n_classes=mlp_classes,
                       lr=3e-3, seed=seed)
    if verbose:
        print(f"  C09 mlp:   top1={mlp['top1']['value']:.3f} top10={mlp['top10']['value']:.3f} "
              f"(classes {mlp['n_classes']})", flush=True)
    payload = result_payload(
        attack_name="ima.classify",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={"ridge": ridge, "mlp": mlp, "train_test_split": "by request, never by token"},
        diagnostics={
            "candidate_mode": "proposal",
            "positive_control": {"name": "in_domain_holdout", "ridge_top1": ridge["top1"]["value"],
                                 "required_top1": 0.95},
            "positive_control_passed": bool((ridge["top1"]["value"] or 0) >= 0.95),
            "attack_variant": "paired_inversion (KP)",
            "uses_known_pairs": True,
            "ki_unavailable_reason": ("the KI main-table variant needs the attacker's own OSNIP "
                                      "encoder instance; the shipped encoder is client-side and its "
                                      "key schedule is not in the artifact"),
            "note": "Appendix-only: the pairs come from the deployment key (plan D8).",
            "paper_working_point": {"paper_alpha_e": 1.0, "paper_alpha_h": 0.2,
                                    "artifact_alpha_e": dep.secret_metadata.get("alpha_e")},
        },
        ci={},
        n_prompts=len(texts), n_tokens=int(split["n_train"] + split["n_test"]),
        n_keys=len(texts), n_types=int(ridge["n_classes"]), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "ridge + mlp inverter"},
        data={"texts": len(texts), "ridge_classes": ridge_classes, "mlp_classes": mlp_classes},
        overrides={"uses_known_pairs": True, "main_table_eligible": False},
    )
    write_payload("C09", "ima.classify", "ab", payload, {"seconds": round(time.time() - t0, 1)})
    return {"ridge": ridge, "mlp": mlp}


def run_c02(dep: ABDeployment, texts: Sequence[str], *, mlp_epochs: int = 25, seed: int = 0,
            verbose: bool = True) -> dict:
    """Learned embedding inversion (C02) on the real wire, with its KI limitation stated.

    Same inverter machinery as C09 but reported under the learned-inversion class: the *learning*
    part is what distinguishes it (a trained inverse map rather than a nearest-neighbour lookup).
    """
    from .real_inversion import collect_pairs, mlp_inverter, split_by_request

    t0 = time.time()
    data = collect_pairs(dep, texts, arm="ab", layer=None)
    split = split_by_request(data, test_fraction=0.25, seed=seed)
    mlp = mlp_inverter(split, hidden=256, epochs=mlp_epochs, n_classes=800, lr=3e-3,
                       seed=seed)
    if verbose:
        print(f"  C02 learned inverter: top1={mlp['top1']['value']:.3f} "
              f"top10={mlp['top10']['value']:.3f} (classes {mlp['n_classes']})", flush=True)
    payload = result_payload(
        attack_name="ima.cross_instance",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={"learned_inverter": mlp},
        diagnostics={
            "candidate_mode": "proposal",
            "positive_control": {"name": "in_domain_holdout", "top1": mlp["top1"]["value"],
                                 "required_top1": 0.95},
            "positive_control_passed": bool((mlp["top1"]["value"] or 0) >= 0.95),
            "attack_variant": "learned inversion",
            "ki_unavailable_reason": ("KI requires an attacker-built instance of the mechanism; the "
                                      "packaged artifact exposes no encoder key, so the pairs here "
                                      "come from the deployment instance (KP)"),
            "note": "Appendix: learned inverse trained on deployment pairs.",
            "paper_working_point": {"paper_alpha_e": 1.0, "paper_alpha_h": 0.2,
                                    "artifact_alpha_e": dep.secret_metadata.get("alpha_e")},
        },
        ci={},
        n_prompts=len(texts), n_tokens=int(split["n_train"] + split["n_test"]),
        n_keys=len(texts), n_types=int(mlp["n_classes"]), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "mlp inverter"},
        data={"texts": len(texts)},
        overrides={"uses_known_pairs": True, "main_table_eligible": False},
    )
    write_payload("C02", "ima.cross_instance", "ab", payload, {"seconds": round(time.time() - t0, 1)})
    return {"learned_inverter": mlp}


def run_c07(dep: ABDeployment, texts: Sequence[str], *, max_len: int = 6, beam: int = 3,
            n_candidates: int = 5, seed: int = 0, stage2_mapping: dict | None = None,
            verbose: bool = True) -> dict:
    """BeamClean on the public wire; plaintext text only through an attack mapping."""
    from .real_generative import beamclean_recover, text_metrics

    t0 = time.time()
    public = dep.public_view("ab", "S-public")
    rows = []
    for i, text in enumerate(texts):
        wire = dep.observe(text, arm="ab", request_id=i, key_index=i)
        z = np.asarray(wire.public.wire_vectors, dtype=np.float32)
        rec = beamclean_recover(public, z, arm="ab", beam=beam, n_candidates=n_candidates,
                                max_len=max_len)
        got_ids = rec["ids"]
        truth_perm = [int(x) for x in wire.truth.permuted_ids]
        from ..metrics.recovery import bleu4 as _bleu4
        stage1_bleu = float(_bleu4(got_ids, truth_perm))
        got_text = None
        metrics = {}
        if stage2_mapping is not None:
            mapped = [stage2_mapping.get(int(d)) for d in got_ids]
            got_text = _decode_mapped(public, mapped)
            metrics = text_metrics(public, got_text, text)
        rows.append({**_prompt_identity(i, text), "recovered": got_text,
                     "stage1_bleu4": stage1_bleu,
                     "stage1_top1": (sum(a == b for a, b in zip(got_ids, truth_perm)) / len(truth_perm))
                     if truth_perm else None,
                     "stage1_scored_positions": len(truth_perm),
                     "stage1_predicted_positions": len(got_ids),
                     "oracle_text": None, **metrics})
    bleu = (float(np.mean([r["bleu4"] for r in rows]))
            if stage2_mapping is not None and rows else None)
    stage1 = float(np.mean([r["stage1_top1"] for r in rows if r["stage1_top1"] is not None]))
    if verbose:
        print(f"  C07 BeamClean: stage1_top1={stage1:.3f} text_bleu4={bleu} "
              f"(n={len(rows)}) in {round(time.time()-t0,1)}s", flush=True)
    payload = result_payload(
        attack_name="beamclean.gaussian",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={"bleu4": {"value": bleu, "denominator": len(rows)},
                 "stage1_token_accuracy": stage1,
                 "per_prompt": rows,
                 "beam": beam, "n_candidates": n_candidates, "max_len": max_len,
                 "lm_prior": "deployment model (teacher-forced next-token log-probs)"},
        diagnostics={
            "candidate_mode": "proposal",
            "positive_control": {"name": "plaintext_wire", "note": "wire observations are the "
                                 "deployment table rows; the control is the KNN recovery of C01"},
            "positive_control_passed": None,
            "attack_variant": "wire",
            "note": "BeamClean shape without a trained corrector; the LM prior is the real model.",
            "uses_known_pairs": False,
        },
        ci={}, n_prompts=len(texts), n_tokens=int(max_len * len(texts)), n_keys=len(texts),
        n_types=int(n_candidates), reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "beamclean"},
        data={"texts": len(texts), "beam": beam},
    )
    write_payload("C07", "beamclean.gaussian", "ab", payload, {"seconds": round(time.time() - t0, 1)})
    return {"bleu4": bleu, "rows": rows}


def run_c03(dep: ABDeployment, texts: Sequence[str], *, max_len: int = 6, rounds: int = 2,
            seed: int = 0, stage2_mapping: dict | None = None,
            verbose: bool = True) -> dict:
    """Iterative prior refinement; plaintext text only through an attack mapping."""
    from .real_generative import iterative_prior_refine, text_metrics

    t0 = time.time()
    public = dep.public_view("ab", "S-public")
    rows = []
    for i, text in enumerate(texts):
        wire = dep.observe(text, arm="ab", request_id=i, key_index=i)
        z = np.asarray(wire.public.wire_vectors, dtype=np.float32)[:max_len]
        from ..schemes.ab_deployment import knn_topk

        table = np.asarray(public.table("ab"), dtype=np.float32)
        init = knn_topk(z, table, k=1, metric="l2")[:, 0]
        out = iterative_prior_refine(public, init.tolist(), z, arm="ab", rounds=rounds)
        got_ids = out["ids"]
        truth_perm = [int(x) for x in wire.truth.permuted_ids]
        from ..metrics.recovery import bleu4 as _bleu4
        got_text = None
        metrics = {}
        if stage2_mapping is not None:
            mapped = [stage2_mapping.get(int(d)) for d in got_ids]
            got_text = _decode_mapped(public, mapped)
            metrics = text_metrics(public, got_text, text)
        rows.append({**_prompt_identity(i, text), "recovered": got_text,
                     "history": out["history"],
                     "stage1_bleu4": float(_bleu4(got_ids, truth_perm)),
                     "oracle_text": None, **metrics})
    bleu = (float(np.mean([r["bleu4"] for r in rows]))
            if stage2_mapping is not None and rows else None)
    if verbose:
        print(f"  C03 iterative prior: bleu4={bleu} rounds={rounds} "
              f"({round(time.time()-t0,1)}s)", flush=True)
    payload = result_payload(
        attack_name="vec2text.iterative",
        scheme_name="ab_deployment:ab",
        target="llama-3.2-1b/ab",
        metrics={"bleu4": {"value": bleu, "denominator": len(rows)}, "per_prompt": rows,
                 "rounds": rounds, "init": "KNN on the deployment table"},
        diagnostics={
            "candidate_mode": "proposal",
            "positive_control": {"name": "knn_initial_guess", "note": "the initial guess is C01's "
                                 "KNN result; the correction loop may only improve it"},
            "positive_control_passed": None,
            "attack_variant": "text-level iterative",
            "note": ("Vec2Text's loop shape without the trained corrector: the prior supplies "
                     "alternatives, the observation accepts or rejects them."),
        },
        ci={}, n_prompts=len(texts), n_tokens=int(max_len * len(texts)), n_keys=len(texts),
        n_types=0, reference_tokens=0,
        source={"artifact": str(dep.artifact), "attack": "vec2text"},
        data={"texts": len(texts), "rounds": rounds},
    )
    write_payload("C03", "vec2text.iterative", "ab", payload, {"seconds": round(time.time() - t0, 1)})
    return {"bleu4": bleu, "rows": rows}


def run_c11(dep: ABDeployment, texts: Sequence[str], *, n_sequences: int = 120, seq_len: int = 16,
            epochs: int = 40, seed: int = 0, verbose: bool = True) -> dict:
    """SDA-Transformer on a known substitution: the positive control the plan asks for."""
    from .real_sda import make_substitution_pairs, train_sda_transformer

    t0 = time.time()
    seqs = []
    for text in texts:
        ids = dep.encode(text)
        if len(ids) >= seq_len:
            seqs.append([int(t) for t in ids[:seq_len]])
        if len(seqs) >= n_sequences:
            break
    data = make_substitution_pairs(seqs, vocab_size=dep.vocab_size, seed=seed, n_classes=1500)
    out = train_sda_transformer(data, epochs=epochs, seed=seed, verbose=verbose)
    if verbose:
        print(f"  supervised substitution diagnostic: token_accuracy={out['token_accuracy']:.3f} "
              f"coverage={out['class_coverage']:.3f} ({out['seconds']}s)", flush=True)
    name = "sda.supervised_substitution_diagnostic"
    payload = result_payload(
        attack_name=name,
        scheme_name="synthetic:known_substitution",
        target="known_substitution",
        metrics=out,
        diagnostics={"candidate_mode": "training_classes_only",
                     "positive_control": {"status": "not_evaluated"},
                     "attack_variant": "supervised per-position classifier",
                     "pair_source": "synthetic same-substitution training pairs"},
        ci={}, n_prompts=len(seqs), n_tokens=sum(map(len, seqs)), n_keys=1,
        n_types=len(data["classes"]), reference_tokens=0,
        source={"attack": name, "seed": seed},
        data={"sequences": len(seqs), "seq_len": seq_len, "epochs": epochs},
        overrides={"main_table_eligible": False, "candidate_mode": "training_classes_only"},
    )
    write_payload("C11", name, "known_substitution", payload,
                  {"seconds": round(time.time() - t0, 1)})
    return out


def run_c14(public, wires, *, stage2_mapping: dict, max_len: int = 6, seed: int = 0,
            truth: Sequence[str] | None = None, verbose: bool = True) -> dict:
    """Output-side LM reconstruction from attack-estimated text.

    Stage-1 KNN returns deployment ids; stage-2 decodes them through the attack's
    own mapping estimate. A position the mapping does not cover stays in the
    denominator as ``[UNK]``. The secret inverse permutation is never used; an
    oracle score is a separately named field and only exists when a truth string
    is supplied by the scorer side.
    """
    if int(max_len) <= 0:
        raise ValueError("max_len must be positive")
    t0 = time.time()
    rows = []
    table = np.asarray(public.table(public.arm), dtype=np.float32)
    covered_total = 0
    positions_total = 0
    bleu_values = []
    for i, wire in enumerate(wires):
        vectors = np.asarray(wire.wire_vectors, dtype=np.float32)
        z = vectors[:max_len]
        if z.size == 0:
            rows.append({"read": "", "rewritten": "", "coverage": None})
            continue
        from ..schemes.ab_deployment import knn_topk

        ids = knn_topk(z, table, k=1, metric="l2")[:, 0]

        def _mapped(value):
            if isinstance(value, (list, tuple)):
                return int(value[0]) if value else None
            return None if value is None else int(value)

        mapped = [_mapped(stage2_mapping.get(int(d))) for d in ids]
        predicted_positions = len(mapped)
        mapped.extend([None] * (len(vectors) - predicted_positions))
        covered = [m is not None for m in mapped]
        covered_total += int(sum(covered))
        positions_total += len(mapped)
        read = _decode_mapped(public, mapped)
        rewritten = lm_rewrite(public, read)
        row = {
            "read": read,
            "rewritten": rewritten,
            "coverage": float(np.mean(covered)) if covered else None,
            "n_positions": len(mapped),
            "n_predicted_positions": predicted_positions,
        }
        if truth is not None and i < len(truth):
            metrics = text_metrics(public, rewritten, truth[i])
            row.update(metrics)
            bleu_values.append(metrics["bleu4"])
        rows.append(row)
    coverage = (covered_total / positions_total) if positions_total else None
    bleu4 = float(np.mean(bleu_values)) if bleu_values else None
    if verbose:
        print(f"  C14 LM recon: bleu4={bleu4} coverage={coverage} ({round(time.time()-t0,1)}s)",
              flush=True)
    return {
        "bleu4": bleu4,
        "coverage": coverage,
        "rows": rows,
        "input_source": "attack-estimated stage-2 mapping",
        "oracle_bleu4": None,
    }
