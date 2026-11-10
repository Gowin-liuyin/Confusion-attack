"""Real-model ISA entry: the original ER/TBS classes behind a public adapter (F05).

The previous entry ran a private Adam loop and called it ``isa.er``. This module
builds an ``AttackContext`` over the public server model and invokes the original
``ERAttack``/``TBSAttack`` classes, so the maths, optimizer, initialization and
token recovery are the paper implementations. A differential test checks that the
entry reproduces the direct class output exactly.

Per-wire restart selection (S03)
--------------------------------
The selection unit is ``method x layer x wire(observation) x checkpoint``. A
*restart* is one optimizer trajectory with its own seed/prior; a *checkpoint* is
a prefix step count of that trajectory. For every layer and every checkpoint the
entry runs each restart on every wire and then keeps, **per wire independently**,
the restart whose visible reconstruction residual is smallest. Averaging losses
over all observations and picking one restart for the whole batch is explicitly
not done: it can discard a wire's much better legal candidate.

The visible residual is a statistic the attacker can compute from the public
observation alone. ``diagnostics["final_state_distance"]`` (the pure observed
state-matching term ``L_im``, no distribution-matching penalty) is preferred;
``diagnostics["final_loss"]`` (``L_im + lambda_dm * L_dm``) is used when the pure
term is absent or non-finite. With ``lambda_dm == 0`` the two coincide. The
output records which quantity was used in ``selection_metric``. Truth,
permutation, secret state and plaintext are never inputs to the choice.

Missing/nan handling: every wire stays in every output list. If at least one
restart produced a finite visible residual, that restart wins; if every restart
for a wire is non-finite, the first restart's result is retained and the row is
marked ``"selected_by": "fallback_first_restart"`` with a ``warning``. No
``np.nanmean`` is used to drop observations.

Output contract
---------------
``layers[layer_key]`` contains:

* ``"checkpoints"``: ``{steps: entry}``. Each ``entry`` is assembled from the
  per-wire best restarts:
  ``{"steps", "per_wire", "restart", "loss", "loss_n_finite", "loss_n_wires",
  "recovered_token_ids", "candidates", "selection_metric", "diagnostics"}``.
  ``per_wire`` is a list in wire order of ``{"restart", "loss",
  "selection_metric", "selection_value", "recovered_token_ids", "candidates",
  "selected_by", "warning"}``. ``recovered_token_ids`` and ``candidates`` are
  the per-wire best rows in wire order; ``restart`` is the matching per-wire
  restart-index list.
* ``"best"``: the checkpoint entry with the smallest **display-level** batch
  mean loss (mean of finite selected ``final_loss`` values). This is only a
  reporting convenience ("best checkpoint"); its per-wire rows are still the
  per-wire-selected restarts, not the output of a single restart run.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from ..attacks.isa.er import ERAttack
from ..attacks.isa.tbs import TBSAttack
from ..attacks.isa.aloepri_state import AloePriStateAttack
from ..data.contracts import AttackContext, AttackResult, AttackStatus, Capability
from ..schemes.ab_deployment import WirePublic
from ..schemes.public_view import PublicDeployment


class HFAttackerModel:
    """Duck-typed ``TorchToyModel`` adapter over the public server model."""

    def __init__(self, public: PublicDeployment) -> None:
        import torch

        self.public = public
        table = public.table(public.arm) if public.arm != "plaintext" else public.table("plaintext")
        self._embedding = torch.tensor(np.asarray(table, dtype=np.float32))

    def embedding_matrix(self):
        return self._embedding

    def forward_from_embeddings(self, what, layer: int):
        import torch

        out = self.public.forward(what.unsqueeze(0))
        hidden = torch.as_tensor(out.hidden_states[layer], dtype=torch.float32)
        # Keep autograd intact: ``.to`` is a differentiable copy, so the
        # attacker's ``what`` still receives gradients when the deployment
        # computes on another device than the attacker's inputs.
        if hidden.device != what.device:
            hidden = hidden.to(what.device)
        return hidden[0] if hidden.ndim == 3 else hidden

    def forward_attention_scores(self, what, layer: int):
        """Differentiable Table 4 AttnScore features, same layout as the victim log.

        ``layer`` is the ``hidden_states`` index (1 = first decoder block). The
        matching attention map is HuggingFace ``attentions[layer - 1]``.
        """
        import torch

        from ..attacks.isa.core import attention_token_features

        layer = int(layer)
        if layer < 1:
            raise ValueError("attention layer index must be >= 1")
        try:
            out = self.public.forward(what.unsqueeze(0), output_attentions=True)
        except TypeError as exc:
            raise RuntimeError(
                "public forward does not accept output_attentions=True"
            ) from exc
        attentions = getattr(out, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "forward returned no attention maps; eager attention is required"
            )
        index = layer - 1
        if index >= len(attentions) or attentions[index] is None:
            raise RuntimeError(f"no attention map for layer {layer}")
        attn = torch.as_tensor(attentions[index], dtype=torch.float32)
        if attn.device != what.device:
            attn = attn.to(what.device)
        return attention_token_features(attn)

    def parameters(self):
        return []

    def freeze(self) -> None:
        return None


def _target(wire: WirePublic, layer: int) -> np.ndarray:
    if wire.hidden_states is None or layer not in wire.layers:
        raise ValueError(f"wire {wire.request_id} has no observed hidden state for layer {layer}")
    hidden = np.asarray(wire.hidden_states[wire.layers.index(layer)], dtype=np.float32)
    if hidden.ndim == 3:
        hidden = hidden[0]
    if hidden.ndim != 2 or hidden.shape[0] != wire.seq_len:
        raise ValueError(f"wire {wire.request_id} layer {layer} has incompatible state shape {hidden.shape}")
    return hidden


def wire_from_states(arm: str, request_id: int, layers: Sequence[int], states: Sequence) -> WirePublic:
    """One public wire whose states ``isa_attack`` can index by layer.

    ``states`` is one ``[seq, feat]`` array per layer, in the same order as
    ``layers``. A leading batch of 1 is squeezed. This is the contract
    ``_target`` reads; a bare array without ``layers`` / ``seq_len`` /
    ``request_id`` is not a wire.
    """
    arrays = []
    seq = None
    for state in states:
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 3 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"state must be [seq, feat], got {arr.shape}")
        if seq is None:
            seq = int(arr.shape[0])
        elif int(arr.shape[0]) != seq:
            raise ValueError("states in one wire have different sequence lengths")
        arrays.append(np.ascontiguousarray(arr))
    if not arrays:
        raise ValueError("wire has no states")
    layer_list = tuple(int(layer) for layer in layers)
    if len(layer_list) != len(arrays):
        raise ValueError(
            f"layers ({len(layer_list)}) and states ({len(arrays)}) length differ"
        )
    return WirePublic(
        arm=str(arm),
        request_id=int(request_id),
        seq_len=int(seq),
        hidden_states=tuple(arrays),
        layers=layer_list,
    )


def invert_permutation(perm) -> np.ndarray:
    """Inverse of a 0..n-1 permutation. Scoring-only; never an attack input."""
    values = np.asarray(perm, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty permutation")
    if int(values.min()) < 0 or int(values.max()) >= int(values.size):
        raise ValueError("permutation values are outside 0..n-1")
    if int(np.unique(values).size) != int(values.size):
        raise ValueError("permutation has duplicate values")
    inverse = np.empty(values.size, dtype=np.int64)
    inverse[values] = np.arange(values.size, dtype=np.int64)
    return inverse


def map_readout_ids(ids, inverse_perm) -> list[int]:
    """Map model-table ids to plaintext ids. ``None`` leaves them unchanged."""
    if inverse_perm is None:
        return [int(value) for value in ids]
    inv = np.asarray(inverse_perm)
    out = []
    for value in ids:
        index = int(value)
        out.append(int(inv[index]) if 0 <= index < int(inv.size) else -1)
    return out


def map_prediction_lists(recovered, candidates, inverse_perm):
    """Apply :func:`map_readout_ids` to every wire and candidate list."""
    if inverse_perm is None:
        return recovered, candidates
    mapped_recovered = [map_readout_ids(row, inverse_perm) for row in recovered]
    mapped_candidates = [
        [map_readout_ids(position, inverse_perm) for position in wire]
        for wire in candidates
    ]
    return mapped_recovered, mapped_candidates


def score_recovery(recovered, candidates, truth, ks: Sequence[int] = (1, 5, 10, 100)) -> dict:
    """Top-k over every truth position. A missing prediction is a miss, not a drop.

    Top-1 uses the recovered id. Larger k use the ranked candidate list. Wires
    absent from ``recovered`` / ``candidates`` stay in the denominator.
    """
    ks = tuple(int(k) for k in ks)
    hits = {k: 0 for k in ks}
    total = 0
    n_wires = len(truth)
    for wire_index in range(n_wires):
        truth_row = [int(value) for value in truth[wire_index]]
        rec = [] if recovered is None or wire_index >= len(recovered) else list(recovered[wire_index])
        cand = [] if candidates is None or wire_index >= len(candidates) else list(candidates[wire_index])
        for pos, token in enumerate(truth_row):
            total += 1
            if 1 in hits and pos < len(rec) and int(rec[pos]) == token:
                hits[1] += 1
            ranked = [int(value) for value in cand[pos]] if pos < len(cand) else []
            for k in hits:
                if k == 1:
                    continue
                hits[k] += int(token in ranked[:k])
    return {
        **{f"top{k}": (hits[k] / total if total else None) for k in ks},
        "numerator": {f"top{k}": int(hits[k]) for k in ks},
        "denominator": int(total),
    }


def _public_init(public: PublicDeployment, seq_len: int, seed: int) -> np.ndarray:
    """Restart prior from the public table plus small noise (protocol §8.2)."""
    rng = np.random.default_rng(int(seed))
    table = np.asarray(public.table("plaintext" if public.arm == "plaintext" else public.arm),
                       dtype=np.float32)
    ids = rng.integers(0, table.shape[0], size=int(seq_len))
    base = table[ids]
    return (base + 0.05 * rng.standard_normal(base.shape)).astype(np.float32)


def _lr_for_layer(lr, layer: int) -> float:
    """Resolve the learning rate for one layer (scalar or per-layer mapping)."""
    if isinstance(lr, Mapping):
        value = lr.get(int(layer))
        if value is None:
            value = lr.get(str(int(layer)))
        if value is None:
            raise ValueError(f"lr mapping has no value for layer {layer}")
        return float(value)
    return float(lr)


def _attack_kwargs(method: str, *, lr: float, lambda_dm: float, top_k: int,
                   seed: int, basis: str, alpha: float,
                   target_state: str = "hidden") -> dict:
    common = dict(
        distance="l2", lr=lr, lambda_dm=lambda_dm, top_k=top_k, seed=seed,
        weight_decay=0.0,
    )
    if method == "er":
        if target_state != "hidden":
            raise ValueError("ER has no attention-score target")
        return common
    if method == "tbs":
        if target_state != "hidden":
            raise ValueError("TBS has no attention-score target")
        return {**common, "basis": basis, "alpha": alpha, "r": None}
    if method == "aloepri_state":
        if target_state not in {"hidden", "attention"}:
            raise ValueError(f"unknown aloepri_state target {target_state!r}")
        return {
            key: value for key, value in common.items() if key != "lambda_dm"
        } | {"target": target_state}
    raise ValueError(f"unknown ISA method {method!r}")


def _as_float(value) -> float:
    """Coerce a diagnostic to ``float``; unparseable/missing values become ``nan``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _visible_residual(diagnostics: dict) -> tuple[str, float]:
    """Attacker-visible residual used to rank restarts, with its metric name.

    ``final_state_distance`` is the pure observed-state matching term ``L_im``
    (distance to the public hidden state, no distribution-matching penalty), so
    it is the preferred selection quantity. ``final_loss`` (``L_im + lambda_dm *
    L_dm``) is the fallback when the pure term is absent or non-finite; with
    ``lambda_dm == 0`` the two are numerically identical. Both are computed by
    the attack from the public observation only.
    """
    for key in ("final_state_distance", "final_loss"):
        if key in diagnostics:
            value = _as_float(diagnostics.get(key))
            if np.isfinite(value):
                return key, value
    return "final_loss", float("nan")


def _assemble_entry(
    steps: int,
    chosen: Sequence[dict | None],
    first_run: Sequence[dict | None],
) -> dict:
    """Build one checkpoint entry from per-wire best-restart records.

    ``chosen[i]`` is the record with the smallest finite visible residual seen
    so far for wire ``i`` (or the first record if none was finite yet);
    ``first_run[i]`` is the first restart's record, used as the documented
    fallback. Every wire produces a row: nothing is dropped for a nan loss.
    """
    rows: list[dict] = []
    n_finite = 0
    metrics: set[str] = set()
    diagnostics: dict = {}
    for index, incumbent in enumerate(chosen):
        record = incumbent
        selected_by = "min_visible_residual"
        warning = None
        if record is None or not np.isfinite(record["selection_value"]):
            fallback = first_run[index] if index < len(first_run) else None
            if fallback is not None:
                record = fallback
                selected_by = "fallback_first_restart"
                warning = (
                    f"no restart produced a finite visible residual for wire {index}; "
                    f"kept restart {fallback['restart']}'s result"
                )
        if record is None:
            rows.append({
                "restart": None,
                "loss": float("nan"),
                "selection_metric": "final_loss",
                "selection_value": None,
                "recovered_token_ids": [],
                "candidates": [],
                "selected_by": "missing",
                "warning": f"wire {index} produced no attack result",
            })
            continue
        if index == 0:
            diagnostics = record["diagnostics"]
        loss = record["loss"]
        if np.isfinite(loss):
            n_finite += 1
        if selected_by == "min_visible_residual":
            metrics.add(record["selection_metric"])
        rows.append({
            "restart": int(record["restart"]),
            "loss": loss,
            "selection_metric": record["selection_metric"],
            "selection_value": record["selection_value"],
            "recovered_token_ids": list(record["recovered_token_ids"]),
            "candidates": [list(position) for position in record["candidates"]],
            "selected_by": selected_by,
            "warning": warning,
        })
    finite = [row["loss"] for row in rows if np.isfinite(row["loss"])]
    # Display-only batch mean over the selected rows; the per-wire rows above
    # stay the selection and reporting unit. None when no selected loss is
    # finite, never a silent drop of the nan rows.
    display_loss = float(np.mean(finite)) if finite else None
    if len(metrics) == 1:
        selection_metric = next(iter(metrics))
    elif len(metrics) > 1:
        selection_metric = "mixed"
    else:
        selection_metric = "none_finite"
    return {
        "steps": int(steps),
        "per_wire": rows,
        "restart": [row["restart"] for row in rows],
        "loss": display_loss,
        "loss_n_finite": int(n_finite),
        "loss_n_wires": int(len(rows)),
        "recovered_token_ids": [row["recovered_token_ids"] for row in rows],
        "candidates": [row["candidates"] for row in rows],
        "selection_metric": selection_metric,
        "diagnostics": diagnostics,
    }


def _entry_is_better(entry: dict, incumbent: dict | None) -> bool:
    """True when ``entry`` has the smaller display-level batch mean loss."""
    if incumbent is None:
        return True
    value = entry.get("loss")
    current = incumbent.get("loss")
    if value is None:
        return False
    if current is None:
        return True
    return bool(value < current)


def isa_attack(
    public: PublicDeployment,
    wires: Sequence[WirePublic],
    *,
    method: str = "er",
    layers: Sequence[int] = (1,),
    steps: Sequence[int] = (1000, 10000, 25000, 50000),
    lr: float | Mapping[int, float] = 1e-3,
    lambda_dm: float = 0.0,
    top_k: int = 100,
    restarts: int = 3,
    seed: int = 0,
    basis: str = "unbiased",
    alpha: float = 5.0 / np.pi,
    target_state: str = "hidden",
    verbose: bool = False,
) -> dict:
    """Run the original ER/TBS at each checkpoint and restart; select per wire.

    Selection is per ``method x layer x wire x checkpoint``: at every checkpoint
    each wire keeps the restart with the smallest visible residual (see the
    module docstring). ``per_wire`` rows document the chosen restart and the
    metric used; ``recovered_token_ids``/``candidates`` are assembled from those
    per-wire choices. ``best`` is only the checkpoint with the smallest display
    mean loss and still holds per-wire-selected rows.

    ``lr`` is either one scalar used for every layer or a mapping
    ``{layer: learning_rate}``; a missing layer raises ``ValueError``.

    ``optimizer_steps_total`` counts the optimizer steps actually executed. A
    run returns ``AttackResult.failed``/``not_applicable`` (e.g. a state
    dimension mismatch) *before* the optimizer loop, so it performs no step and
    contributes nothing. Each ``(layer, restart, wire)`` trajectory runs once
    to the final checkpoint, with intermediate snapshots. ``failed_runs`` counts failures;
    ``failed_observations`` counts **wires** (the audit's observation unit) that
    had no successful run in any layer, restart or checkpoint -- a layer that
    fails wholesale while another succeeds does not mark its wires failed. The
    affected wire still appears in the per-wire output under the documented
    fallback. A fully successful entry therefore reports ``max(checkpoints) *
    restarts * n_layers * n_wires`` steps and zero failures.
    """
    method_name = str(method).lower()
    attacks = {"er": ERAttack, "tbs": TBSAttack, "aloepri_state": AloePriStateAttack}
    if method_name not in attacks:
        raise ValueError(f"unknown ISA method {method!r}")
    if method_name == "aloepri_state" and lambda_dm != 0:
        raise ValueError("aloepri_state does not implement distribution matching")
    state_name = str(target_state)
    if method_name != "aloepri_state" and state_name != "hidden":
        raise ValueError(f"{method_name} has no attention-score target")
    if method_name == "aloepri_state" and state_name not in {"hidden", "attention"}:
        raise ValueError(f"unknown aloepri_state target {state_name!r}")
    attack_cls = attacks[method_name]
    layer_list = [int(layer) for layer in layers]
    checkpoints = sorted({int(step) for step in steps})
    if not checkpoints or checkpoints[0] <= 0 or int(restarts) < 1:
        raise ValueError("checkpoints and restarts must be positive")
    t0 = time.time()
    adapter = HFAttackerModel(public)
    out_layers: dict = {}
    actual_steps = 0
    failed_runs = 0
    trajectory_failures = []
    wire_ok = [False] * len(wires)
    for layer in layer_list:
        layer_lr = _lr_for_layer(lr, layer)
        targets = [_target(wire, layer) for wire in wires]
        n_wires = len(targets)
        # Per checkpoint and per wire: the best-restart record seen so far and
        # the first restart's record (restart 0) for the all-nan fallback.
        chosen: dict[int, list[dict | None]] = {
            checkpoint: [None] * n_wires for checkpoint in checkpoints
        }
        first_run: dict[int, list[dict | None]] = {
            checkpoint: [None] * n_wires for checkpoint in checkpoints
        }
        for restart in range(int(restarts)):
            run_seed = int(seed) + 1000 * restart
            for target_index, target in enumerate(targets):
                params = _attack_kwargs(
                    method_name, lr=layer_lr, lambda_dm=lambda_dm, top_k=top_k,
                    seed=run_seed, basis=basis, alpha=alpha, target_state=state_name,
                )
                params["steps"] = checkpoints[-1]
                ctx = AttackContext(
                        granted=frozenset({Capability.HIDDEN_STATES}),
                        descriptor={"layer": layer},
                        _obfuscated_hidden_states=target,
                        attacker_model=adapter,
                    )
                run_params = dict(params)
                if restart > 0 and method_name != "aloepri_state":
                    run_params["init_what"] = _public_init(
                        public, target.shape[0], run_seed + target_index
                    )
                if method_name == "aloepri_state":
                    run_params["seed"] = run_seed + target_index
                snapshots = {}
                run_params["checkpoints"] = checkpoints
                run_params["checkpoint_callback"] = lambda step, result: snapshots.__setitem__(int(step), result)
                result = attack_cls(**run_params).run(ctx)
                if getattr(result, "status", AttackStatus.OK) == AttackStatus.OK:
                    actual_steps += checkpoints[-1]
                    wire_ok[target_index] = True
                    snapshots.setdefault(checkpoints[-1], result)
                else:
                    failed_runs += 1
                    actual_steps += int((result.diagnostics or {}).get("executed_steps", 0))
                    trajectory_failures.append({
                        "layer": layer, "restart": restart, "wire_index": target_index,
                        "reason": getattr(result, "status_reason", ""),
                        "executed_steps": int((result.diagnostics or {}).get("executed_steps", 0)),
                    })
                for checkpoint in checkpoints:
                    checkpoint_result = snapshots.get(checkpoint)
                    if checkpoint_result is None:
                        checkpoint_result = result if getattr(result, "status", AttackStatus.OK) != AttackStatus.OK else AttackResult.failed("checkpoint not recorded")
                    diagnostics = dict(checkpoint_result.diagnostics or {})
                    metric, value = _visible_residual(diagnostics)
                    recovered_raw = checkpoint_result.recovered_token_ids
                    candidates_raw = checkpoint_result.candidates
                    record = {
                        "restart": int(restart),
                        "loss": _as_float(diagnostics.get("final_loss")),
                        "selection_metric": metric,
                        "selection_value": value,
                        "recovered_token_ids": (
                            [] if recovered_raw is None
                            else [int(x) for x in recovered_raw]
                        ),
                        "candidates": (
                            [] if candidates_raw is None
                            else [[int(x) for x in position] for position in candidates_raw]
                        ),
                        "diagnostics": diagnostics,
                    }
                    if first_run[checkpoint][target_index] is None:
                        first_run[checkpoint][target_index] = record
                    incumbent = chosen[checkpoint][target_index]
                    if incumbent is None:
                        chosen[checkpoint][target_index] = record
                    elif np.isfinite(value) and (
                        not np.isfinite(incumbent["selection_value"])
                        or value < incumbent["selection_value"]
                    ):
                        chosen[checkpoint][target_index] = record
        curve: dict[int, dict] = {}
        best: dict | None = None
        for checkpoint in checkpoints:
            entry = _assemble_entry(
                checkpoint, chosen[checkpoint], first_run[checkpoint]
            )
            curve[checkpoint] = entry
            if _entry_is_better(entry, best):
                best = entry
        out_layers[f"layer{layer}"] = {"checkpoints": curve, "best": best}
        if verbose:
            print(
                f"  ISA-{method_name.upper()} layer {layer}: "
                f"display mean loss {best['loss'] if best else None}",
                flush=True,
            )
    first = next(iter(out_layers.values()))["best"]
    diagnostics = first["diagnostics"] if first else {}
    failed_observations = int(sum(1 for ok in wire_ok if not ok))
    return {
        "method": {"er": "ER", "tbs": "TBS", "aloepri_state": "AloePri hidden-state adaptation"}[method_name],
        "optimizer": diagnostics.get("optimizer", "AdamW"),
        "final_budget_steps": int(max(checkpoints)),
        "optimizer_steps_total": int(actual_steps),
        "trajectory_mode": "continuous_checkpoints",
        "failed_observations": int(failed_observations),
        "failed_runs": int(failed_runs),
        "trajectory_failures": trajectory_failures,
        "restarts": int(restarts),
        "weight_decay": diagnostics.get("weight_decay", 0.0),
        "basis": basis if method_name == "tbs" else None,
        "full_basis": diagnostics.get("full_basis", True) if method_name == "tbs" else None,
        "target_state": "AttnScore" if state_name == "attention" else "HiddenState",
        "attnscore_implemented": method_name == "aloepri_state",
        "attnscore_status": diagnostics.get("attnscore_status", ""),
        "layers": out_layers,
        "seconds": round(time.time() - t0, 1),
    }
