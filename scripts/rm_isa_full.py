"""Closed-loop ISA (ER / TBS) probe on the plaintext Llama-3.2-1B.

PROBE ONLY. This script runs one fixed hyper-parameter setting (lr 1e-3/1e-2,
AdamW, 5e4 steps) and is *not* the formal paper-v1 ISA executor: it has no
validation grid, restarts or convergence checkpoints. The formal ISA entry is
``real_battery.run_c05`` dispatched by ``scripts/run_full_battery.py
--profile paper-v1``. A non-probe ``--profile`` is refused here, and every
record written below carries ``probe_only: true``.

Runs the actual ``ERAttack`` / ``TBSAttack`` classes through the capability-gated
``AttackContext``, optimising the observed hidden state of 5 content tokens for
the paper's 5e4 AdamW steps (Sec. 5.1 / Table 3) on every prompt. ER targets
shallow layers (< 8, lr 1e-3), TBS deep layers (>= 8, lr 1e-2), exactly the
paper's split; TBS uses the full d_in basis (no truncation) and the paper's
``phi_z = alpha*arctan`` reparametrisation.

The model runs in fp32 with TF32 matmuls (Ampere tensor cores) so gradients are
fp32 while stepping at a usable rate on the 8 GB card. This is a speed setting,
not an algorithmic one; it is recorded in every result.

Artifacts: ``reports/closed_loop/isa_full.jsonl`` (job-level resumable; failed
records keep their ``layer``/``steps``/``prompt_idx``/``variant`` identity so
they remain in the per-setting denominator, and they are retried on resume).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rm_common import (
    OUTDIR, PROMPTS, append, load_done, load_model, log, status_value,
)

RESULTS = OUTDIR / "isa_full.jsonl"
N_CONTENT = 5
#: ER/TBS forward the embedding prefix from position 0 (no BOS prefix), so the
#: target and the scored truth must use exactly those positions.
_CONTENT_START = 0


def _content_span(n: int = N_CONTENT) -> slice:
    """Positions the attack forwards and scores (0..n-1, never a BOS prefix)."""
    start = int(_CONTENT_START)
    return slice(start, start + int(n))


def _content_slice(ids, n: int = N_CONTENT):
    """The positions the attack actually forwards and scores (0..n-1)."""
    return ids[_content_span(n)]


def tokenize_content(tok, text: str):
    """Token ids forwarded by the attack, with no BOS/special tokens.

    Llama tokenizers prepend BOS (128000) by default. The attack forwards the
    embedding prefix from position 0 and scores those same rows, so a default
    tokenization would score BOS+4 content tokens and inflate TTRSR.
    """
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def model_identity(model_dir: str) -> str:
    """Stable model identity for resume (normalized path + config.json hash).

    Hashing the 1B weights on every launch is too expensive for a probe, but a
    job completed on model A must never satisfy model B: the normalized path and
    the small config file are hashed into the job identity.
    """
    path = Path(model_dir).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    digest = hashlib.sha256(str(resolved).casefold().encode("utf-8"))
    config = resolved / "config.json"
    if config.is_file():
        digest.update(config.read_bytes())
    return digest.hexdigest()[:16]

PROBE_PROFILE = "probe"
#: Stamped on every record so a probe payload can never be mistaken for a
#: formal paper-v1 result.
PROBE_FIELDS = {"profile": PROBE_PROFILE, "probe_only": True}


def _failure_record(job: dict, status, reason, seconds: float | None = None) -> dict:
    """A failure row that keeps the job's identity for the denominator.

    A failure is not a completed unit, but it is still a sampled unit: dropping
    ``layer``/``steps``/``prompt_idx``/``variant`` would silently remove it from
    the per-setting denominator. ``load_done`` treats this row as pending, so
    the job is retried on the next run.
    """
    layer = int(job["layer"])
    rec = {
        "job_id": job["job_id"],
        "family": "isa",
        "variant": job["method"],
        "method": job["method"],
        "model": job.get("model"),
        "model_path": job.get("model_path"),
        "prompt_idx": job["pi"],
        "layer": layer,
        "steps": job["steps"],
        "lr": _probe_lr(job["method"], layer),
        "distance": "l2",
        "top_k": 5,
        "tf32": True,
        "n_positions": N_CONTENT,
        "status": status_value(status),
        "status_reason": reason,
        **PROBE_FIELDS,
    }
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


class LlamaHandle:
    """Differentiable view of the frozen model for ISA (fA).

    ``forward_from_embeddings`` stops after the requested decoder layer: the
    output of layer l is identical whether or not layers l+1..L are computed
    (verified bit-exact against HF's ``hidden_states[l]`` at startup), so the
    optimisation pays only for the layers it needs. Layer L also applies the
    final norm, matching ``hidden_states[L]``.
    """

    def __init__(self, m):
        self.model = m
        self.vocab_size = m.config.vocab_size
        self.hidden_size = m.config.hidden_size
        self.num_layers = m.config.num_hidden_layers
        self._E = m.get_input_embeddings().weight.detach()

    def embedding_matrix(self):
        return self._E

    def forward_from_embeddings(self, inputs_embeds, layer=None):
        from transformers.masking_utils import create_causal_mask

        m = self.model.model
        l = self.num_layers if layer is None else int(layer)
        h = inputs_embeds.unsqueeze(0)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        mask = create_causal_mask(config=m.config, inputs_embeds=h,
                                  attention_mask=None, past_key_values=None,
                                  position_ids=pos)
        pe = m.rotary_emb(h, position_ids=pos)
        for dl in m.layers[:l]:
            h = dl(h, attention_mask=mask, position_embeddings=pe,
                   position_ids=pos, past_key_values=None, use_cache=False)
        if l == self.num_layers:
            h = m.norm(h)
        return h[0]

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self


def verify_handle(model, handle, layer: int) -> float:
    """Max abs diff between early-exit forward and HF ``hidden_states[layer]``."""
    ids = torch.arange(2, 12, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        ref = model.model(input_ids=ids, use_cache=False, output_hidden_states=True,
                          return_dict=True).hidden_states[layer][0]
        E = model.get_input_embeddings()(ids)[0]
        got = handle.forward_from_embeddings(E, layer)
    return float((got - ref).abs().max())


def build_sample(tok, prompt_idx: int, layer: int, handle, model):
    from obfuscation_attacks.data.contracts import AttackSample

    ids = tokenize_content(tok, PROMPTS[prompt_idx])
    content = _content_slice(ids, N_CONTENT)
    with torch.inference_mode():
        hs = model.model(input_ids=ids.to("cuda").unsqueeze(0), use_cache=False,
                         output_hidden_states=True, return_dict=True).hidden_states
    target = hs[layer][0, _content_span()].float().cpu().numpy()
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b",
        seq_len=int(content.numel()), vocab_size=int(model.config.vocab_size),
        layer=layer, obfuscated_hidden_states=target,
        reference_embeddings=handle.embedding_matrix().detach().cpu().numpy(),
        plaintext_token_ids=content.numpy(), text_source="real", data_split="eval",
    )
    return sample, content, ids


def _probe_lr(method: str, layer: int) -> float:
    # HiddenState adaptation uses configs/aloepri/isa_state.yaml's working point.
    return 5e-4 if method == "astate" else (1e-3 if layer < 8 else 1e-2)


def run_job(model, tok, handle, job: dict) -> dict:
    from obfuscation_attacks.attacks.isa.er import ERAttack
    from obfuscation_attacks.attacks.isa.tbs import TBSAttack
    from obfuscation_attacks.attacks.isa.aloepri_state import AloePriStateAttack
    from obfuscation_attacks.data.contracts import Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    pi, layer, method = job["pi"], job["layer"], job["method"]
    steps = job["steps"]
    lr = _probe_lr(method, layer)
    sample, content, ids = build_sample(tok, pi, layer, handle, model)
    ctx = build_context(GenericScheme("isa", frozenset({Capability.HIDDEN_STATES})),
                        sample, attacker_model=handle)
    cls = {"er": ERAttack, "tbs": TBSAttack, "astate": AloePriStateAttack}[method]
    attack = cls(distance="l2", lr=lr, steps=steps, top_k=5, seed=0)
    t0 = time.time()
    res = attack.run(ctx)
    dt = time.time() - t0
    if status_value(res.status) != "ok" or res.recovered_token_ids is None:
        rec = _failure_record(job, res.status, res.status_reason, dt)
        rec["executed_steps"] = res.diagnostics.get("executed_steps", 0)
        return rec
    met = compute_metrics(res, sample)["token_recovery"]
    rec = [int(x) for x in np.asarray(res.recovered_token_ids).tolist()]
    original = tok.decode(content.tolist())
    recovered = tok.decode(rec)
    rec_dict = {
        "job_id": job["job_id"],
        "family": "isa",
        "variant": method,
        "method": method,
        "model": job.get("model"),
        "model_path": job.get("model_path"),
        "prompt_idx": pi,
        "layer": layer,
        "steps": steps,
        "lr": lr,
        "distance": "l2",
        "top_k": 5,
        "tf32": True,
        "n_positions": int(content.numel()),
        "ttrsr": met["ttrsr"],
        "asr1": met["asr@1"],
        "asr5": met["asr@5"],
        "state_distance": res.diagnostics.get("final_state_distance"),
        "final_loss": res.diagnostics.get("final_loss"),
        "original": original,
        "recovered": recovered,
        "recovered_ids": rec,
        "true_ids": content.tolist(),
        "seconds": round(dt, 1),
        **PROBE_FIELDS,
    }
    if method in ("astate", "er"):
        rec_dict["init"] = "random X2" if method == "astate" else "zeros"
    if method == "astate":
        for key in ("target_state", "attnscore_implemented", "init_scale"):
            rec_dict[key] = res.diagnostics[key]
    if method == "tbs":
        rec_dict["full_basis"] = res.diagnostics.get("full_basis")
        rec_dict["svd_reconstruction_error"] = res.diagnostics.get("svd_reconstruction_error")
    return rec_dict


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="PROBE ONLY: fixed-hyperparameter ISA closed loop. The formal "
                    "paper-v1 ISA entry is real_battery.run_c05.")
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--steps", type=int, default=50000, help="paper budget: 5e4")
    ap.add_argument("--methods", default="er:2,4;tbs:8,12",
                    help="semicolon-separated method:layer,list "
                         "(method in er|tbs|astate)")
    ap.add_argument("--prompts", default="", help="comma indices; default all")
    ap.add_argument("--profile", default=PROBE_PROFILE,
                    help="must be 'probe' (default); 'paper-v1' is refused because "
                         "this is a probe, not the formal run_c05 executor")
    return ap.parse_args(argv)


def profile_refusal(profile: str) -> str | None:
    """Fail closed on anything but the probe profile.

    The probe has none of the paper-v1 contract (no 200-sample x 5-key ISA set,
    no lr validation grid / restarts / convergence checkpoints), so accepting a
    formal profile here would produce a payload that looks paper-v1 but is not.
    """
    if profile == PROBE_PROFILE:
        return None
    return (
        f"rm_isa_full.py is a fixed-hyperparameter probe and refuses "
        f"--profile {profile!r}. The formal ISA entry is real_battery.run_c05, "
        f"dispatched by `python scripts/run_full_battery.py --profile paper-v1 "
        f"--manifest ...`. Probe runs must use --profile probe (the default); "
        f"their records carry probe_only=true."
    )


def _isa_job_id(method: str, prompt_idx: int, layer: int, steps: int,
                model_key: str) -> str:
    """Job identity including the model and the optimizer budget.

    A job completed on one ``--model`` must never mark another model as done,
    and a 1,000-step probe must never mark the 50,000-step budget as done.
    """
    job_id = f"isa_{method}:m{model_key}:{int(prompt_idx)}:L{int(layer)}:S{int(steps)}"
    # Historical astate jobs borrowed the ER/TBS learning rates.
    return f"{job_id}:lr{_probe_lr(method, layer):g}" if method == "astate" else job_id


def deployment_artifact_refusal(model_dir: str) -> str | None:
    """Refuse a deployment artifact as this probe's victim model.

    The closed loop scores recovered ids against *plaintext* content tokens.
    Pointing ``--model`` at a deployment artifact (e.g. ``ab_runnable/
    model_server``) would run the optimizer against the obfuscated deployment
    and then score deployment-vocabulary rows against plaintext ids -- a silent
    scoring-space mismatch that reads as a near-0 success rate while looking
    like a completed probe. The ab-arm ISA entry is ``real_battery.run_c05``
    (truth = permuted ids); the osnip arm is resource-blocked until its own
    clean-side encoder exists (plan G5).
    """
    path = Path(model_dir).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for dirpath in (resolved, *resolved.parents):
        if ((dirpath / "client" / "client_secret.pt").is_file()
                and (dirpath / "osnip").is_dir()):
            return (
                f"--model {model_dir!r} is inside a deployment artifact "
                f"({dirpath}); this probe scores the plaintext caliber and "
                "would silently mis-score a deployment arm. Use "
                "`run_full_battery.py --classes C05` (run_c05, ab arm, "
                "permuted-id truth) for deployment-arm ISA."
            )
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    refusal = profile_refusal(args.profile)
    if refusal is not None:
        print(f"[rm_isa_full] REFUSED: {refusal}", file=sys.stderr)
        return 2
    refusal = deployment_artifact_refusal(args.model)
    if refusal is not None:
        print(f"[rm_isa_full] REFUSED: {refusal}", file=sys.stderr)
        return 2

    methods = []
    for chunk in args.methods.split(";"):
        if not chunk.strip():
            continue
        name, layers = chunk.split(":")
        for l in layers.split(","):
            methods.append((name.strip(), int(l)))
    prompts = ([int(x) for x in args.prompts.split(",") if x.strip()]
               if args.prompts else list(range(len(PROMPTS))))

    # ``steps`` and the model identity are part of the job id: a completed
    # 1,000-step or model-A run must not mark another budget/model as done.
    mkey = model_identity(args.model)
    jobs = [{"job_id": _isa_job_id(m, pi, l, args.steps, mkey), "method": m,
             "pi": pi, "layer": l, "steps": args.steps, "model": mkey,
             "model_path": str(args.model)}
            for m, l in methods for pi in prompts]
    # A failed or partial row keeps its job pending; only a successful record
    # counts as done (see rm_common.is_success_record). Budget/model-defining
    # fields are required so a truncated/partial record is retried.
    done = load_done(
        RESULTS,
        required_fields=("layer", "steps", "ttrsr", "method", "prompt_idx",
                         "variant", "model"),
    )
    pending = [j for j in jobs if j["job_id"] not in done]
    log(f"ISA closed loop (PROBE ONLY, profile={PROBE_PROFILE}): "
        f"{len(pending)}/{len(jobs)} jobs pending "
        f"(steps={args.steps}, model={mkey}, methods={args.methods}, tf32=True)")
    if not pending:
        log("ALL DONE")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, tok = load_model(args.model, dtype="float32", device="cuda")
    handle = LlamaHandle(model)
    for l in sorted({j["layer"] for j in pending}):
        diff = verify_handle(model, handle, l)
        log(f"early-exit forward self-check layer {l}: max abs diff {diff:.3e}")
        if not math.isfinite(diff) or diff > 1e-4:
            raise SystemExit("early-exit forward differs from HF hidden_states; refusing to run")

    failed = 0
    for j in pending:
        rec = run_job(model, tok, handle, j)
        append(RESULTS, rec)
        if rec.get("ttrsr") is None:
            failed += 1
        log(f"{j['job_id']}: ttrsr={rec.get('ttrsr')} asr5={rec.get('asr5')} "
            f"dist={rec.get('state_distance')} ({rec.get('seconds')}s) "
            f"rec={rec.get('recovered')!r}")
        torch.cuda.empty_cache()
    if failed:
        log(f"{failed}/{len(pending)} jobs failed; retry pending jobs on resume")
        return 1
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
