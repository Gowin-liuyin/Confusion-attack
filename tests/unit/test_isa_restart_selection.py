"""Acceptance tests for S03: per-wire restart selection in the real ISA entry.

The old entry averaged ``final_loss`` over all observations (``np.nanmean``) and
picked a single restart for the whole batch. These tests replace ``ERAttack`` /
``TBSAttack`` with fakes whose visible residual is a function of
``(restart, wire)``, so a batch-level choice is provably wrong:

* restart 0 is much better on wire 0, restart 1 is much better on wire 1;
* the batch mean would pick restart 1 (0.15 < 0.55) and discard wire 0's 0.1.

The fakes also encode ``(restart, wire)`` into ``recovered_token_ids`` so the
assembled per-wire lists can be checked against the selected rows. A second case
covers nan losses: a wire whose first restart is nan must still switch to a
finite restart, and a wire whose every restart is nan must be kept (never
dropped) with the documented first-restart fallback and a warning.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.data.contracts import AttackResult
from obfuscation_attacks.evaluation import real_isa
from obfuscation_attacks.schemes.ab_deployment import WirePublic
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D, N = 8, 4, 3


def _public() -> PublicDeployment:
    rng = np.random.default_rng(0)
    E = rng.normal(size=(V, D)).astype(np.float32)
    return PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E,
        forward_fn=lambda embeds: SimpleNamespace(
            hidden_states=[np.asarray(embeds), np.asarray(embeds)]
        ),
        allow_clean_weights=True,
    )


def _wires(n_wires: int) -> list[WirePublic]:
    """One wire per observation; the fake attack reads its index from the state."""
    wires = []
    for index in range(n_wires):
        hidden = np.zeros((1, N, D), dtype=np.float32)
        hidden[0, 0, 0] = float(index)
        wires.append(
            WirePublic(
                arm="ab", request_id=index, seq_len=N,
                hidden_states=hidden, layers=(1,),
            )
        )
    return wires


class _FakeAttack:
    """Fake ER/TBS with a visible residual determined by ``(restart, wire)``."""

    residuals: np.ndarray
    runs: list[int]  # actual ``steps`` of every attack invocation
    successful_steps: list[int]  # steps of invocations that really optimized
    failing_wires: set[int]

    def __init__(self, **params) -> None:
        # ``real_isa`` seeds restart r with ``seed + 1000 * r`` and the test
        # passes seed=0, so the seed identifies the restart.
        self.restart = int(params["seed"]) // 1000
        self.steps = int(params["steps"])

    def run(self, ctx):
        type(self).runs.append(self.steps)
        target = np.asarray(ctx._obfuscated_hidden_states)
        wire = int(round(float(target[0, 0])))
        if wire in type(self).failing_wires:
            # Mirrors ER/TBS's early dimension-mismatch return: FAILED before
            # the optimizer loop, so zero steps are actually executed.
            return AttackResult.failed("fake dimension mismatch")
        type(self).successful_steps.append(self.steps)
        value = float(self.residuals[self.restart, wire])
        token = 100 + 10 * self.restart + wire
        return SimpleNamespace(
            recovered_token_ids=np.array([token], dtype=np.int64),
            candidates=[[token, 999]],
            diagnostics={"final_loss": value},
        )


def _run(monkeypatch, method: str, residuals, restarts: int = 2, failing_wires=()):
    matrix = np.asarray(residuals, dtype=np.float64)
    fake = type("FakeAttack", (_FakeAttack,), {
        "residuals": matrix,
        "runs": [],
        "successful_steps": [],
        "failing_wires": set(failing_wires),
    })
    attr = "ERAttack" if method == "er" else "TBSAttack"
    monkeypatch.setattr(real_isa, attr, fake)
    out = real_isa.isa_attack(
        _public(), _wires(matrix.shape[1]), method=method, layers=(1,),
        steps=[5], restarts=restarts, seed=0, verbose=False,
    )
    entry = out["layers"]["layer1"]["checkpoints"][5]
    return out, entry, fake


# [restart][wire]: restart 0 wins wire 0, restart 1 wins wire 1.
# Batch means are 5.05 (restart 0) and 0.6 (restart 1), so the old batch-level
# rule would pick restart 1 for both wires and discard wire 0's 0.1 candidate.
CROSSED = [[0.1, 10.0], [1.0, 0.2]]


@pytest.mark.parametrize("method", ["er", "tbs"])
def test_per_wire_selection_beats_the_batch_mean(monkeypatch, method):
    out, entry, fake = _run(monkeypatch, method, CROSSED)
    rows = entry["per_wire"]
    assert [row["restart"] for row in rows] == [0, 1]
    assert entry["restart"] == [0, 1]
    # Assembled from the per-wire best rows, not from one restart for the batch.
    assert entry["recovered_token_ids"] == [[100], [111]]
    assert entry["candidates"] == [[[100, 999]], [[111, 999]]]
    assert [row["loss"] for row in rows] == pytest.approx([0.1, 0.2])
    assert all(row["selected_by"] == "min_visible_residual" for row in rows)
    assert all(row["warning"] is None for row in rows)
    # Only final_loss is exposed by the fake, so that is the recorded metric.
    assert entry["selection_metric"] == "final_loss"
    assert [row["selection_metric"] for row in rows] == ["final_loss", "final_loss"]
    # Display-only batch mean of the selected rows.
    assert entry["loss"] == pytest.approx((0.1 + 0.2) / 2)
    assert entry["loss_n_finite"] == 2
    assert entry["loss_n_wires"] == 2
    # "best" is a display-level checkpoint choice; its rows stay per-wire.
    best = out["layers"]["layer1"]["best"]
    assert [row["restart"] for row in best["per_wire"]] == [0, 1]
    assert best["recovered_token_ids"] == [[100], [111]]
    # Instrumented: one attack invocation per (layer, restart, checkpoint, wire),
    # each running ``checkpoint`` optimizer steps: 2 restarts x 1 checkpoint x
    # 1 layer x 2 wires = 4 invocations x 5 steps = 20. An undercount would
    # report only 10.
    assert len(fake.runs) == 2 * 1 * 1 * 2
    assert out["optimizer_steps_total"] == sum(fake.successful_steps) == 5 * 2 * 1 * 2
    assert out["failed_observations"] == 0


def test_nan_losses_are_never_dropped(monkeypatch):
    # [restart][wire]: wire 0 has a nan first restart and a finite second one;
    # wire 1 is nan for every restart.
    residuals = [[np.nan, np.nan], [0.5, np.nan]]
    _, entry, _ = _run(monkeypatch, "er", residuals)
    rows = entry["per_wire"]
    assert len(rows) == 2  # the all-nan wire stays in the output
    # wire 0: the finite restart wins even though restart 0 was tried first.
    assert rows[0]["restart"] == 1
    assert rows[0]["selected_by"] == "min_visible_residual"
    assert rows[0]["warning"] is None
    assert rows[0]["loss"] == pytest.approx(0.5)
    # wire 1: every restart is nan -> first restart kept and marked.
    assert rows[1]["restart"] == 0
    assert rows[1]["selected_by"] == "fallback_first_restart"
    assert rows[1]["warning"]
    assert np.isnan(rows[1]["loss"])
    assert entry["recovered_token_ids"] == [[110], [101]]
    assert entry["loss"] == pytest.approx(0.5)
    assert entry["loss_n_finite"] == 1
    assert entry["loss_n_wires"] == 2


def test_optimizer_steps_exclude_failed_runs(monkeypatch):
    # Two wires, one restart, one layer. Wire 1's attack returns FAILED before
    # the optimizer loop (like ER/TBS's dimension mismatch), so it executes no
    # optimizer step; only wire 0's 5 steps are real.
    residuals = [[0.1, 1.0]]
    out, entry, fake = _run(
        monkeypatch, "er", residuals, restarts=1, failing_wires={1},
    )
    assert len(fake.runs) == 2  # both wires were attempted
    assert out["optimizer_steps_total"] == 5
    assert out["optimizer_steps_total"] == sum(fake.successful_steps)
    assert out["failed_observations"] == 1
    assert out["trajectory_failures"][0]["reason"] == "fake dimension mismatch"
    # The failed wire still appears, with the documented fallback.
    rows = entry["per_wire"]
    assert len(rows) == 2
    assert rows[0]["selected_by"] == "min_visible_residual"
    assert rows[1]["selected_by"] == "fallback_first_restart"
    assert rows[1]["warning"]
    assert np.isnan(rows[1]["loss"])


def _public_tensor_forward() -> PublicDeployment:
    """Public view whose forward keeps tensors on the input device and graph."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(V, D)).astype(np.float32)
    return PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E,
        forward_fn=lambda embeds: SimpleNamespace(hidden_states=[embeds, embeds]),
        allow_clean_weights=True,
    )


@pytest.mark.parametrize("method", ["er", "tbs"])
@pytest.mark.parametrize("lambda_dm", [0.0, 0.001])
def test_checkpoints_are_one_continuous_trajectory(method, lambda_dm):
    public = _public_tensor_forward()
    wire = _wires(1)
    combined = real_isa.isa_attack(public, wire, method=method, layers=(1,),
                                    steps=(2, 5), restarts=1, seed=7, lambda_dm=lambda_dm)
    assert combined["optimizer_steps_total"] == 5
    for budget in (2, 5):
        single = real_isa.isa_attack(public, wire, method=method, layers=(1,),
                                     steps=(budget,), restarts=1, seed=7, lambda_dm=lambda_dm)
        got = combined["layers"]["layer1"]["checkpoints"][budget]
        expected = single["layers"]["layer1"]["checkpoints"][budget]
        assert got["loss"] == pytest.approx(expected["loss"])
        assert got["recovered_token_ids"] == expected["recovered_token_ids"]


def test_forward_from_embeddings_returns_input_device_and_keeps_grad():
    model = real_isa.HFAttackerModel(_public_tensor_forward())
    what = torch.tensor(
        np.random.default_rng(1).normal(size=(N, D)), dtype=torch.float32,
        requires_grad=True,
    )
    out = model.forward_from_embeddings(what, 1)
    # On CPU ``what.device`` is what the deployment computes on; on a
    # GPU-resident deployment the adapter must move the state back to the
    # attacker's input device (see the ``.to`` branch). Either way the
    # returned state must live where ``what`` lives and keep its graph.
    assert out.device == what.device
    assert out.dtype == torch.float32
    assert out.requires_grad
    out.sum().backward()
    assert what.grad is not None
