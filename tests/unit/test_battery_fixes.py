"""Acceptance tests for the six shared-runner fixes (S02/S04/S05/S06/S09/S11).

Each test encodes the defect that was fixed and the new required behaviour, so a
regression fails here rather than silently re-entering a result payload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from obfuscation_attacks.evaluation import real_battery
from obfuscation_attacks.evaluation.real_battery import (
    _dense_mixing_control,
    _layer_control_summary,
    _mapping_path_block,
    _matched_linkage_pairs,
    _tfma_group_counts,
    _threshold_from_validation,
    run_c06w,
    run_c10,
    run_c12,
    run_composed,
)
from obfuscation_attacks.metrics.ratio_ci import ratio_ci
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit

TEXTS = [
    np.array([1, 2, 3, 4, 5]),
    np.array([1, 2, 6, 7, 8]),
    np.array([1, 3, 6, 9, 10]),
    np.array([2, 3, 7, 9, 11]),
    np.array([1, 4, 6, 8, 11]),
    np.array([2, 5, 7, 10, 11]),
    np.array([1, 2, 3, 6, 7]),
    np.array([1, 3, 5, 8, 9]),
    np.array([2, 4, 6, 9, 10]),
    np.array([1, 5, 7, 9, 11]),
]


class _PublicView:
    def __init__(self, arm: str, n_layers: int = 2, dep=None):
        self.arm = arm
        self.n_heads = 1
        self.n_layers = int(n_layers)
        self.vocab_size = int(getattr(dep, "vocab_size", 96))
        self._dep = dep

    def table(self, name, normalize: bool = False):
        if self._dep is None:
            raise AttributeError("fixture public view has no table")
        arm = "plaintext" if str(name) == "plaintext" else "ab"
        return self._dep.table(arm)

    def weight_tensor(self, name, side):
        if self._dep is None:
            raise AttributeError("fixture public view has no weights")
        return self._dep.weight_tensor(name, side)


class _FakeKeySchedule:
    """Minimal key-schedule identity holder (the cache binds to its root hash)."""

    def __init__(self, root: bytes):
        self._root = bytes(root)


class FakeDeployment:
    """Public half of a miniature A+B build: clean/deployment tables + wire."""

    artifact = Path("fixture-artifact")
    clean_model_path = Path("fixture-clean")
    max_len = 64

    def __init__(self, *, d: int = 32, vocab: int = 96, n_layers: int = 1,
                 seed: int = 11, duplicate_rows: tuple[int, int] | None = None,
                 drop_layers: tuple[int, ...] = (),
                 drop_layer_error: str = "key",
                 key_root: bytes | None = None,
                 artifact: Path | None = None,
                 encoder_state: dict | None = None,
                 zero_first_vector: bool = False,
                 zero_all_vectors: bool = False,
                 table_corruption: str | None = None,
                 nonfinite_vector: bool = False):
        rng = np.random.default_rng(seed)
        self.hidden = d
        self.vocab_size = vocab
        self.n_layers = n_layers
        self.drop_layers = {int(layer) for layer in drop_layers}
        self.drop_layer_error = str(drop_layer_error)
        self.zero_first_vector = bool(zero_first_vector)
        self.zero_all_vectors = bool(zero_all_vectors)
        self.table_corruption = table_corruption
        self.nonfinite_vector = bool(nonfinite_vector)
        if key_root is not None:
            self.key_schedule = _FakeKeySchedule(key_root)
        if artifact is not None:
            self.artifact = Path(artifact)
        if encoder_state is not None:
            self.encoder_state = dict(encoder_state)
        self.perm = rng.permutation(vocab).astype(np.int64)
        self.inv_perm = np.argsort(self.perm).astype(np.int64)
        self.E_clean = rng.standard_normal((vocab, d)).astype(np.float32)
        self.E_dep = np.empty_like(self.E_clean)
        self.E_dep[self.perm] = self.E_clean          # E_dep[perm[i]] = E_clean[i]
        if duplicate_rows is not None:
            self.E_dep[duplicate_rows[1]] = self.E_dep[duplicate_rows[0]]
        self.secret_metadata = {
            "alpha_e": 0.2, "alpha_h": 0.01,
            "transform_family": "dense", "paper_algorithm1_reproduced": False,
        }

    # --- weight-attack surface (dense mixing with M = I) --------------------
    def weight_tensor(self, name: str, side: str) -> np.ndarray:
        import re

        match = re.search(r"model\.layers\.(\d+)\.", str(name))
        if match and int(match.group(1)) in self.drop_layers:
            if self.drop_layer_error == "os":
                raise OSError(f"{name!r} is unreadable on the {side} side")
            if self.drop_layer_error == "shape":
                raise ValueError(
                    f"expect matching 2-D matrices for {name!r}; shape is invalid"
                )
            raise KeyError(f"{name!r} unavailable on the {side} side")
        return np.eye(self.hidden, dtype=np.float32)

    # --- wire ----------------------------------------------------------------
    def table(self, arm: str, *, normalize: bool = False) -> np.ndarray:
        if arm == "plaintext":
            return self.E_clean
        table = self.E_dep
        if self.table_corruption == "nan":
            table = table.copy()
            table[0, 0] = np.nan
        elif self.table_corruption == "zero":
            table = table.copy()
            table[0] = 0.0
        return table

    def encode(self, text) -> np.ndarray:
        return np.asarray(text, dtype=np.int64)

    def public_view(self, arm: str, scenario: str = "S-public") -> _PublicView:
        return _PublicView(arm, self.n_layers, self)

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None, forward=True):
        ids = np.asarray(text, dtype=np.int64)
        index = 0 if key_index is None else int(key_index)
        group = f"g-{index}"
        if arm == "plaintext":
            public = WirePublic(
                arm=arm, request_id=request_id, seq_len=len(ids),
                wire_token_ids=ids.copy(), group_id=group, epoch=epoch,
                session_id=session_id,
            )
        else:
            # Small key-dependent constant offset: symbols differ across keys
            # (rounding at 1e-4), while the deployment-row geometry -- and thus
            # the stage-1 KNN -- stays intact.
            offset = np.float32(0.05) + np.float32(0.40) * np.float32(
                (index % 65521) / 65521.0
            )
            z = self.E_dep[self.perm[ids]] + offset
            if self.zero_all_vectors:
                z = np.zeros_like(z)
            if self.nonfinite_vector:
                z = z.copy()
                z[:, 0] = np.inf          # cosine against any row is undefined
            if self.zero_first_vector and int(request_id) == 0:
                z = z.copy()
                z[0] = 0.0                     # a zero vector has no direction
            public = WirePublic(
                arm="ab", request_id=request_id, seq_len=len(ids),
                wire_vectors=z, wire_token_ids=self.perm[ids].copy(),
                group_id=group, epoch=epoch, session_id=session_id,
            )
        return Wire(
            public,
            WireTruth(plaintext_ids=ids.copy(), permuted_ids=self.perm[ids].copy()),
        )


def _payload(directory: Path, arm: str) -> dict:
    path = next(directory.glob(f"**/{arm}/**/result.json"))
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# S02 -- dense-mixing positive control is decided by an identifiable probe
# ---------------------------------------------------------------------------
def _s02_fixture():
    dep = FakeDeployment(d=32, vocab=96)
    rng = np.random.default_rng(3)
    X = dep.E_clean[rng.choice(len(dep.E_clean), size=16, replace=False)]
    return dep, X


def test_s02_dense_control_uses_a_column_full_rank_probe():
    dep, X = _s02_fixture()
    assert X.shape[0] < X.shape[1]          # the old diagnostic sample
    control = _dense_mixing_control(X, dep.E_clean, seed=7)
    assert control["probe"]["column_full_rank"] is True
    assert control["probe"]["n_rows"] >= control["probe"]["n_columns"]
    assert control["probe"]["relative_residual"] < 1e-6
    assert control["probe"]["heldout_relative_residual"] < 1e-6
    assert control["uses_recovered_M"] is True
    assert control["top1"] >= 0.99
    assert control["passed"] is True


def test_s02_underdetermined_probe_is_recorded_as_a_negative_and_does_not_veto():
    dep, X = _s02_fixture()
    control = _dense_mixing_control(X, dep.E_clean, seed=7)
    negative = control["underdetermined_probe"]
    assert negative["status"] == "not_identifiable"
    assert "rank deficiency means lstsq cannot identify M0" in negative["note"]
    assert negative["used_for_verdict"] is False
    # The identifiable construction passes; the underdetermined probe must not
    # turn that into a failed control.
    assert control["passed"] is True


def test_s02_run_c06w_control_follows_the_dense_variant(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    dep = FakeDeployment(d=32, vocab=96)
    out = run_c06w(dep, n_types=16, diag_types=16, seed=0, verbose=False)
    assert out["dense_mixing"]["accuracy"] == pytest.approx(1.0)
    payload = _payload(tmp_path, "aloepri_weights")
    diagnostics = payload["diagnostics"]
    assert diagnostics["positive_control_passed"] is True
    assert diagnostics["positive_control"]["top1"] >= 0.99
    assert diagnostics["positive_control"]["variant"] == "dense_mixing_inversion"
    assert diagnostics["rowsort_control"]["passed"] is True
    assert diagnostics["dense_control"]["passed"] is True
    assert diagnostics["positive_control"]["top1"] == pytest.approx(
        diagnostics["dense_control"]["top1"]
    )
    assert diagnostics["dense_control"]["underdetermined_probe"]["status"] == (
        "not_identifiable"
    )
    assert diagnostics["positive_control_passed"] == diagnostics["dense_control"]["passed"]


def test_s02_a_failing_dense_control_fails_the_cell_even_if_rowsort_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        real_battery, "_dense_mixing_control",
        lambda *args, **kwargs: {
            "top1": 0.0, "required_top1": 0.99, "passed": False,
            "underdetermined_probe": {"status": "not_identifiable",
                                      "used_for_verdict": False},
        },
    )
    run_c06w(FakeDeployment(d=32, vocab=96), n_types=16, diag_types=16,
             seed=0, verbose=False)
    diagnostics = _payload(tmp_path, "aloepri_weights")["diagnostics"]
    assert diagnostics["rowsort_control"]["passed"] is True
    assert diagnostics["positive_control_passed"] is False


def test_c06w_payload_has_engineering_variant_identity_and_actual_layers(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c06w(FakeDeployment(n_layers=3), n_types=16, diag_types=16,
                   seed=0, verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["attack_id"] == "least_squares"
    assert payload["attack"] == "vma.dense_mixing_weight_probe"
    assert "not AloePri Table 9" in payload["paper"]
    assert payload["layers"] == [0, 1, 2]
    assert payload["missing_weight_layers"] == []
    assert payload["diagnostics"]["mixing_layers"] == [0, 1, 2]
    # One variant decision; the paper-literal premise is the measured one.
    assert payload["metrics"]["applicable_variant"] == (
        payload["diagnostics"]["applicable_variant"]
    )
    assert payload["metrics"]["paper_literal_rowsort"]["mapping_recovery_rate"][
        "premise_holds"
    ] is True
    assert payload["diagnostics"]["attack_top1"] == pytest.approx(
        out["dense_mixing"]["accuracy"]
    )


def test_c06w_low_alignment_keeps_the_measured_dense_attempt(monkeypatch, tmp_path):
    import obfuscation_attacks.attacks.vma.dense_mixing as dense_mixing

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        real_battery, "_monomial_precondition",
        lambda *args, **kwargs: {"monomial_supported": False, "n_types": 16},
    )

    def zero_match_stats(rows, table, **kwargs):
        n = len(np.asarray(rows))
        return (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32))

    monkeypatch.setattr(dense_mixing, "forward_match_stats", zero_match_stats)
    out = run_c06w(FakeDeployment(), n_types=16, diag_types=16, seed=0,
                   verbose=True)
    assert out["dense_mixing"]["accuracy"] is not None
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["metrics"]["applicable_variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["applicable_variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["attack_top1"] == out["dense_mixing"]["accuracy"]
    literal = payload["metrics"]["paper_literal_rowsort"]["mapping_recovery_rate"]
    assert literal["premise_holds"] is False
    assert literal["value"] is None and "numerator" not in literal
    control = payload["diagnostics"]["positive_control"]
    assert control["variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["positive_control_passed"] is False
    assert payload["metrics"]["dense_mixing_inversion"]["premise_holds"] is None


def test_c06w_monomial_only_variant_is_oracle_aligned_diagnostic(monkeypatch, tmp_path):
    import obfuscation_attacks.attacks.vma.dense_mixing as dense_mixing

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def zero_match_stats(rows, table, **kwargs):
        n = len(np.asarray(rows))
        return (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32))

    # Dense premise fails while the fake table is monomial-aligned. The RowSort
    # variants score a secret-permutation-aligned pair, so they are oracle
    # diagnostics; the separate dense attempt remains visible and its control fails.
    monkeypatch.setattr(dense_mixing, "forward_match_stats", zero_match_stats)
    run_c06w(FakeDeployment(), n_types=16, diag_types=16, seed=0, verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["metrics"]["applicable_variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["applicable_variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["attack_top1"] is not None
    control = payload["diagnostics"]["positive_control"]
    assert control["variant"] == "dense_mixing_inversion"
    assert payload["diagnostics"]["positive_control_passed"] is False
    scaled_block = payload["metrics"]["scale_normalised_rowsort"]["mapping_recovery_rate"]
    assert scaled_block["status"] == "oracle_aligned_diagnostic"
    assert scaled_block["oracle_aligned_diagnostic"] is True
    assert scaled_block["uses_secret_permutation"] is True
    literal_block = payload["metrics"]["paper_literal_rowsort"]["mapping_recovery_rate"]
    assert literal_block["uses_secret_permutation"] is True
    assert payload["attack_uses_secret_permutation"] is False
    assert payload["diagnostic_variants_use_secret_permutation"] == [
        "paper_literal_rowsort", "scale_normalised_rowsort"
    ]


def test_c06w_unreadable_tensor_produces_a_coded_missing_layer(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.protocol_audit import _weight_layer_gaps

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "os")
    run_c06w(FakeDeployment(n_layers=16, drop_layers=(7,), drop_layer_error="os"),
             n_types=16, diag_types=16, seed=0, verbose=False)
    payload = _payload(tmp_path / "os", "aloepri_weights")
    records = payload["missing_weight_layers"]
    assert [record["layer"] for record in records] == [7]
    assert records[0]["code"] == "tensor_unreadable"
    assert payload["diagnostics"]["mixing_skipped_groups"]
    assert _weight_layer_gaps(
        payload, payload["diagnostics"],
        {int(layer) for layer in payload["layers"]}, 16,
    ) == []

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "shape")
    run_c06w(FakeDeployment(n_layers=16, drop_layers=(7,), drop_layer_error="shape"),
             n_types=16, diag_types=16, seed=0, verbose=False)
    shape_payload = _payload(tmp_path / "shape", "aloepri_weights")
    assert [record["layer"] for record in shape_payload["missing_weight_layers"]] == [7]
    assert shape_payload["missing_weight_layers"][0]["code"] in {
        "shape_invalid", "side_mismatch"
    }


# ---------------------------------------------------------------------------
# S04 -- ISA positive control is the minimum over used layers
# ---------------------------------------------------------------------------
def _layer_block(value, n=10):
    return {"steps100": {"top1": {"value": value, "numerator": int(value * n),
                                  "denominator": n}}}


def test_s04_layer_control_is_the_minimum_not_the_maximum():
    curve = {"layer1": _layer_block(1.0), "layer5": _layer_block(0.2)}
    attack = {"layer1": _layer_block(0.9), "layer5": _layer_block(0.1)}
    summary = _layer_control_summary(curve, "steps100", required=0.90,
                                     attack_curve=attack)
    assert summary["passed"] is False
    assert summary["top1"] == pytest.approx(0.2)
    assert summary["by_layer"]["layer1"]["passed"] is True
    assert summary["by_layer"]["layer5"]["passed"] is False
    assert summary["by_layer"]["layer5"]["attack_bound_to_this_layer_control"] is False
    assert summary["by_layer"]["layer1"]["attack_bound_to_this_layer_control"] is True


def test_s04_all_layers_must_pass():
    curve = {"layer1": _layer_block(1.0), "layer5": _layer_block(0.95)}
    summary = _layer_control_summary(curve, "steps100", required=0.90)
    assert summary["passed"] is True
    assert summary["top1"] == pytest.approx(0.95)


def _fake_isa_attack(public, wires, *, method, layers, steps, lr, restarts,
                     seed, verbose=False, **kwargs):
    """Run-shaped ISA: layers[0] perfect (ramps up over multi-step curves),
    deeper layers 20% (different patterns)."""
    step_list = sorted({int(step) for step in steps})
    run_layers = {}
    for layer_number in layers:
        first = int(layer_number) == int(layers[0])
        checkpoints = {}
        for step in step_list:
            limit = max(1, step // 10) if (first and len(step_list) > 1) else None
            cands = []
            for wire in wires:
                ids = getattr(wire, "wire_token_ids", None)
                if ids is None:
                    cands.append([[0]] * int(wire.seq_len))
                    continue
                row = []
                for position, token in enumerate(np.asarray(ids).tolist()):
                    correct = (position < limit) if limit is not None else first
                    if correct or (not first and position % 5 == 0):
                        row.append([int(token)])
                    else:
                        row.append([-1])
                cands.append(row)
            checkpoints[int(step)] = {"candidates": cands}
        name = f"layer{int(layer_number)}"
        run_layers[name] = {"checkpoints": checkpoints,
                            "best": checkpoints[step_list[-1]]}
    return {
        "method": method.upper(),
        "optimizer": "AdamW",
        "restarts": restarts,
        "optimizer_steps_total": int(max(steps)),
        "layers": run_layers,
    }


def test_s04_run_c05_payload_records_the_failing_layer(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
        {token: 10 for token in range(96)},
        prompts=1, steps=[100], isa_layers=(1, 5), restarts=1, verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    diagnostics = payload["diagnostics"]
    assert diagnostics["positive_control_passed"] is False
    assert diagnostics["positive_control"]["top1"] == pytest.approx(0.2)
    by_layer = diagnostics["positive_control"]["positive_control_by_layer"]
    assert by_layer["layer1"]["passed"] is True
    assert by_layer["layer5"]["passed"] is False
    # A single-checkpoint run cannot support a saturation gain: no record.
    assert "saturation" not in diagnostics
    # No validation study supplied: never fabricate a selection.
    assert "selected_on_validation" not in diagnostics


def test_c05_per_unit_ranks_reconcile_with_the_audit(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa
    from obfuscation_attacks.evaluation.protocol_audit import (
        _evidence_gaps,
        _rank_reconciliation_gaps,
    )

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
        {token: 10 for token in range(96)},
        prompts=1, steps=[100], isa_layers=(1,), restarts=1, verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert rows and all("rank" in row and len(row["rank"]) == row["count"] for row in rows)
    assert all(row["count"] > 0 for row in rows)
    assert all(row["hits"] == sum(1 for value in row["rank"] if value <= 1) for row in rows)
    assert _rank_reconciliation_gaps(
        "isa_er", payload["metrics"], payload["diagnostics"]
    ) == []
    gaps = _evidence_gaps(payload, payload["diagnostics"], "isa_er")
    assert len(gaps) == 2 and all("content hash" in gap for gap in gaps)


def test_c05_per_layer_rows_bind_each_layer_curve_to_its_own_rows(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa
    from obfuscation_attacks.evaluation.protocol_audit import _rank_reconciliation_gaps

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    real_battery.run_c05(
        FakeDeployment(),
        [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6])],
        {token: 10 for token in range(96)},
        prompts=2, steps=[100], isa_layers=(1, 5), restarts=1, verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    diagnostics = payload["diagnostics"]
    by_layer = diagnostics["per_unit_predictions_by_layer"]
    assert set(by_layer) == {"layer1", "layer5"}
    # The two fake layers hit different positions, so the binding is observable.
    assert by_layer["layer1"][0]["hits"] != by_layer["layer5"][0]["hits"]
    assert _rank_reconciliation_gaps(
        "isa_er", payload["metrics"], diagnostics
    ) == []
    swapped = {
        **diagnostics,
        "per_unit_predictions_by_layer": {
            "layer1": by_layer["layer5"],
            "layer5": by_layer["layer1"],
        },
    }
    assert _rank_reconciliation_gaps("isa_er", payload["metrics"], swapped) != []
    # The headline layer and its CI are bound to each other.
    assert payload["primary_layer"] == "layer1"
    final_top1 = payload["metrics"]["budget_curve"]["layer1"]["steps100"]["top1"]["value"]
    ci = payload["ci"]["top1_micro"]
    assert ci["estimand"] == "micro"
    assert ci["micro"] == pytest.approx(final_top1)
    assert ci["n_boot"] >= 2000


def test_c05_records_saturation_from_the_measured_curve(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
        {token: 10 for token in range(96)},
        prompts=1, steps=[10, 20, 30, 40], isa_layers=(1,), restarts=1,
        verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    saturation = payload["diagnostics"]["saturation"]
    curve = payload["metrics"]["budget_curve"]["layer1"]
    assert [point["budget"] for point in saturation["points"]] == [30, 40]
    assert saturation["points"][0]["top1"] == pytest.approx(
        curve["steps30"]["top1"]["value"]
    )
    assert saturation["points"][1]["top1"] == pytest.approx(
        curve["steps40"]["top1"]["value"]
    )
    assert saturation["gain_pp"] == pytest.approx(
        abs(saturation["points"][1]["top1"] - saturation["points"][0]["top1"]) * 100.0
    )


def test_c05_selected_on_validation_requires_a_real_study(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa
    from obfuscation_attacks.evaluation.protocol_audit import _isa_validation_gaps

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    valid = {
        "n_prompts": 40,
        "strata": {"32-64": 10, "65-128": 10, "129-256": 10, "257-512": 10},
        "lr": 1e-3,
        "lambda_dm": 0.0,
        "initialization": "paper", "split": "validation", "metric": "token_top1",
        "data_sha256": "b" * 64,
        "trials": [
            {"lr": rate, "lambda_dm": penalty, "initialization": "paper",
             "token_top1": .8 if rate == 1e-3 and penalty == 0 else .5}
            for rate in (1e-4, 5e-4, 1e-3, 1e-2) for penalty in (0., 1e-3)
        ],
    }
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
        {token: 10 for token in range(96)}, prompts=1, steps=[100],
        isa_layers=(1,), restarts=1, verbose=False,
        selected_on_validation=valid,
    )
    payload = _payload(tmp_path, "ab")
    assert payload["selected_on_validation"] == valid
    assert payload["diagnostics"]["selected_on_validation"] == valid
    assert payload["metrics"]["lr"] == pytest.approx(valid["lr"])
    assert payload["metrics"]["lambda_dm"] == pytest.approx(valid["lambda_dm"])
    assert payload["diagnostics"]["lr"] == pytest.approx(valid["lr"])
    assert _isa_validation_gaps(payload, payload["diagnostics"]) == []
    # An explicit lr that contradicts the selection is refused, never silently
    # executed instead of the validated value.
    with pytest.raises(ValueError, match="conflicts"):
        real_battery.run_c05(
            FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
            {token: 10 for token in range(96)}, prompts=1, steps=[100],
            lr=0.1, isa_layers=(1,), restarts=1, verbose=False,
            selected_on_validation=valid,
        )
    for broken in (
        {**valid, "n_prompts": 10},
        {**valid, "strata": {**valid["strata"], "32-64": 5}},
        {**valid, "strata": None},
    ):
        with pytest.raises(ValueError):
            real_battery.run_c05(
                FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
                {token: 10 for token in range(96)}, prompts=1, steps=[100],
                isa_layers=(1,), restarts=1, verbose=False,
                selected_on_validation=broken,
            )


def test_isa_failed_observations_counts_wires_not_runs(monkeypatch):
    from types import SimpleNamespace

    import obfuscation_attacks.evaluation.real_isa as real_isa
    from obfuscation_attacks.data.contracts import AttackStatus
    from obfuscation_attacks.evaluation import protocol_audit

    class _FakeER:
        def __init__(self, **kwargs):
            pass

        def run(self, ctx):
            if int(ctx.descriptor["layer"]) == 5:
                return SimpleNamespace(status=AttackStatus.FAILED, diagnostics={},
                                       recovered_token_ids=None, candidates=None)
            return SimpleNamespace(
                status=AttackStatus.OK,
                diagnostics={"final_loss": 1.0},
                recovered_token_ids=[1, 2, 3],
                candidates=[[1], [2], [3]],
            )

    class _Public:
        arm = "ab"
        layers = (1, 5)

        def table(self, name):
            return np.eye(8, 3, dtype=np.float32)

        def forward(self, value):
            raise AssertionError("the fake attack must not call forward")

    monkeypatch.setattr(real_isa, "ERAttack", _FakeER)
    wires = [
        WirePublic(arm="ab", request_id=index, seq_len=3,
                   hidden_states=np.zeros((2, 3, 3), dtype=np.float32),
                   layers=(1, 5))
        for index in range(2)
    ]
    out = real_isa.isa_attack(
        _Public(), wires, method="er", layers=(1, 5), steps=(10,), lr=0.1,
        restarts=1, seed=0, verbose=False,
    )
    assert out["failed_observations"] == 0    # every wire succeeded in layer1
    assert out["failed_runs"] == 2            # layer5 failed for both wires
    assert out["optimizer_steps_total"] == 2 * 10
    gaps = protocol_audit._attack_floors(
        "isa_er",
        {"failed_observations": out["failed_observations"], "n_observations": 2},
        {},
    )
    assert not any("every observation failed" in gap for gap in gaps)
    control = protocol_audit._attack_floors(
        "isa_er", {"failed_observations": 2, "n_observations": 2}, {},
    )
    assert any("every observation failed" in gap for gap in control)


# ---------------------------------------------------------------------------
# S05 -- linkage: deployment-id path and full-denominator mapping paths
# ---------------------------------------------------------------------------
def _fabricated_mapping_block():
    requests = [{"stream_id": 0, "session_id": f"s{index}"} for index in range(200)]
    rows = [(2 * index, 0, 2 * index + 1, 0, 0) for index in range(100)]
    deployment_ids = [[0] * 4 for _ in range(200)]
    mapping = {0: 7}
    # First 10 pairs: both endpoints covered and equal id -> hit.
    for index in range(10):
        deployment_ids[2 * index][0] = 0
        deployment_ids[2 * index + 1][0] = 0
    # Remaining 90: uncovered endpoints (id 999 not in the mapping).
    for index in range(10, 100):
        deployment_ids[2 * index][0] = 999
        deployment_ids[2 * index + 1][0] = 999
    return rows, deployment_ids, mapping, requests


def test_s05_full_denominator_tpr_and_conditional_tpr_are_separate():
    rows, deployment_ids, mapping, requests = _fabricated_mapping_block()
    block = _mapping_path_block(rows, deployment_ids, mapping, requests, seed=0)
    assert block["full_denominator"]["denominator"] == 100
    assert block["full_denominator"]["value"] == pytest.approx(0.1)
    assert block["conditional_on_covered"]["denominator"] == 10
    assert block["conditional_on_covered"]["value"] == pytest.approx(1.0)
    assert block["coverage"]["value"] == pytest.approx(0.1)


def test_s05_deployment_id_test_runs_without_stage2_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False, stage2_mapping=None)
    assert out["deployment_id_test"]["status"] == "implemented"
    assert out["deployment_id_test"]["requires_stage2_mapping"] is False
    assert out["deployment_id_test"]["tpr"]["denominator"] > 0
    assert out["deployment_id_test"]["fpr"]["denominator"] > 0
    assert "knn_aligned" not in out
    assert out["recovered_text"]["status"] == "not_run"
    payload = _payload(tmp_path, "ab")
    assert payload["metrics"]["deployment_id_test"]["status"] == "implemented"
    assert "knn_aligned" not in payload["metrics"]
    coverage = payload["metrics"]["deployment_id_test"]["stage1_coverage"]
    assert isinstance(coverage["numerator"], int)
    assert coverage["denominator"] > 0
    assert coverage["numerator"] <= coverage["denominator"]
    assert coverage["value"] == pytest.approx(
        coverage["numerator"] / coverage["denominator"]
    )
    assert out["deployment_id_test"]["stage1_coverage"]["numerator"] == (
        coverage["numerator"]
    )
    rows = payload["diagnostics"]["deployment_id_per_unit"]
    assert rows["positives"] and rows["negatives"]
    assert payload["diagnostics"]["pair_sampling"]["capacity"] > 0


def test_s05_recovered_text_keeps_full_denominators_without_a_duplicate_block(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    mapping = {index: index for index in range(96)}
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False, stage2_mapping=mapping)
    recovered = out["recovered_text"]
    assert "knn_aligned" not in out
    assert recovered["tpr"]["denominator"] == out["exact_test"]["denominator"]
    assert "conditional_tpr" in recovered and "conditional_fpr" in recovered
    assert "coverage_positives" in recovered and "coverage_negatives" in recovered
    assert "knn_aligned" not in _payload(tmp_path, "ab")["metrics"]


def _load_battery_script():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("run_full_battery_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_battery_summary_guard_rejects_prompt_and_secret_keys():
    guard = _load_battery_script()
    assert guard._forbidden_material_gaps({"classes": {"C12": {"prompt": "raw"}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"Plaintext_IDs": [1]}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"inv_perm": [1]}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"root_key": "a"}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"client_secret": "a"}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"encoder_state": [1]}}})
    assert guard._forbidden_material_gaps({"classes": {"x": {"raw_prompt_text": "raw"}}})
    # A 64-hex VALUE is not a blanket exemption: secret-named keys stay
    # forbidden even when they look like a hash commitment.
    assert guard._forbidden_material_gaps({"root_key": "a" * 64})
    assert guard._forbidden_material_gaps({"client_secret": "b" * 64})
    assert guard._forbidden_material_gaps({"inv_perm": "c" * 64})
    # Explicit commitment keys must carry a single 64-hex string.
    assert guard._forbidden_material_gaps({"data_sha256": "d" * 64}) == []
    assert guard._forbidden_material_gaps({"data_sha256": [0.1]})
    assert guard._forbidden_material_gaps({"study_sha256": "not-hex"})
    # Any *_sha256 key is a commitment when its value is a single 64-hex
    # string, even if the name contains a forbidden token (C13 logits).
    assert guard._forbidden_material_gaps({"logits_perm_sha256": "e" * 64}) == []
    assert guard._forbidden_material_gaps({"logits_sha256": "f" * 64}) == []
    assert guard._forbidden_material_gaps({"root_key_sha256": "a" * 64}) == []
    # ... but a key whose NAME is root_key/client_secret/... stays forbidden.
    assert guard._forbidden_material_gaps({"root_key": "b" * 64})
    # Both commitment spellings emitted by the identity/executor code.
    assert guard._forbidden_material_gaps(
        {"identity": {"key_schedule": {"key_root_commitment": "a" * 64}}}
    ) == []
    assert guard._forbidden_material_gaps(
        {"identity": {"key_schedule": {"root_commitment": "b" * 64}}}
    ) == []
    assert guard._forbidden_material_gaps(
        {"identity": {"key_schedule": {"key_root_commitment": "raw-bytes"}}}
    )
    assert guard._forbidden_material_gaps(
        {"identity": {"key_schedule": {"root_commitment": [1, 2]}}}
    )
    # Structural prompt identity and benign capability flags pass.
    assert guard._forbidden_material_gaps(
        {"classes": {"x": {"prompt_index": 0, "prompt_sha256": "a" * 64,
                           "prompt_id": 3, "n_prompts": 4, "keys_per_prompt": 1,
                           "attack_uses_encoder": False}}}
    ) == []
    # A numeric prompt COUNT is structural (plan budget), not raw prompt text.
    assert guard._forbidden_material_gaps(
        {"plan": {"jobs": [{"budget": {"prompts": 200}}]}}
    ) == []
    # ... but a list of prompts under the same key is still rejected.
    assert guard._forbidden_material_gaps(
        {"plan": {"jobs": [{"budget": {"prompts": ["raw text"]}}]}}
    )
    # Hash commitments and boolean availability flags are not material ...
    assert guard._forbidden_material_gaps(
        {"identity": {
            "artifact": {"files_sha256": {
                "client/client_secret.pt": "a" * 64,
                "osnip/encryptor.pt": "b" * 64,
            }},
            "key_schedule": {"root_key_available": True},
        }}
    ) == []
    # ... while raw bytes under the same names are rejected.
    assert guard._forbidden_material_gaps(
        {"identity": {"artifact": {"files_sha256": {
            "client/client_secret.pt": "raw-bytes",
        }}}}
    )
    # The deployment artifact's boolean permutation-validity verdict is benign
    # metadata (the summary echoes secret_metadata verbatim); a non-boolean
    # value under that name still looks like the permutation and is rejected.
    assert guard._forbidden_material_gaps(
        {"artifact_metadata": {"perm_is_valid": True}}
    ) == []
    assert guard._forbidden_material_gaps(
        {"artifact_metadata": {"perm_is_valid": [3, 1, 2]}}
    )


def test_permission_and_hyperparameter_keys_are_boolean_only():
    from obfuscation_attacks.evaluation.public_guard import forbidden_material_gaps

    for key in ("permission", "permissions", "permitted", "permit", "permits",
                "hyperparameter", "hyperparameters"):
        assert forbidden_material_gaps({key: True}) == []
        # A raw secret string or list under a capability-sounding name is not
        # material-free just because the name contains "perm".
        assert forbidden_material_gaps({key: "raw-secret-material"})
        assert forbidden_material_gaps({key: ["raw", "secret"]})


def test_summary_guard_scans_the_json_view_of_numpy_values():
    guard = _load_battery_script()
    corpus = ["evaluated prompt wrapped inside a numpy array"]
    payload = {"classes": {"C12": {"notes": np.array(corpus)}}}
    # An ndarray is invisible to a scan of the pre-serialisation object ...
    assert guard._corpus_text_leaks(payload, corpus) == []
    assert guard._forbidden_material_gaps(payload) == []
    # ... but json converts it to a list, and the writer serialises that view.
    with pytest.raises(SystemExit, match="truth/secret"):
        guard._guard_public_write(payload, label="summary.json",
                                  corpus_texts=corpus)
    nested = {"classes": {"C12": {"rows": np.array(
        [{"root_key": "raw-bytes"}], dtype=object
    )}}}
    with pytest.raises(SystemExit, match="truth/secret"):
        guard._guard_public_write(nested, label="summary.json")


def test_isa_init_generator_matches_the_tensor_device():
    import torch

    from obfuscation_attacks.attacks.isa.aloepri_state import _init_generator

    gen = _init_generator(torch.device("cpu"), 7)
    assert str(gen.device) == "cpu"
    first = torch.randn(5, generator=_init_generator(torch.device("cpu"), 7))
    second = torch.randn(5, generator=_init_generator(torch.device("cpu"), 7))
    assert torch.equal(first, second)


def test_isa_dm_feature_net_follows_the_state_device():
    import torch

    from obfuscation_attacks.attacks.isa import core as isa_core

    # ``meta`` stands in for any non-CPU attacker device: a CPU-built
    # RandomFeatureNet applied to it raises "Tensor on device cpu is not on the
    # expected device meta", so this fails without the device move.
    what = torch.zeros(4, 8, device="meta")
    embedding = torch.zeros(16, 8, device="meta")
    out = isa_core.distribution_matching_loss(
        what, embedding, torch.Generator().manual_seed(0),
        num_nets=1, hidden=4, depth=1,
    )
    assert out.device.type == "meta"


def test_c05_selects_the_table_learning_rate_by_depth(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    seen: list = []

    def _capture(public, wires, *, method, layers, steps, lr, restarts, seed,
                 verbose=False, **kwargs):
        seen.append(lr)
        return _fake_isa_attack(
            public, wires, method=method, layers=layers, steps=steps, lr=lr,
            restarts=restarts, seed=seed, verbose=verbose, **kwargs,
        )

    monkeypatch.setattr(real_isa, "isa_attack", _capture)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3]), np.array([2, 3, 4])],
        {token: 10 for token in range(96)}, prompts=2, steps=[100],
        isa_layers=(1, 5, 10, 15), restarts=1, verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    expected = {"1": pytest.approx(1e-3), "5": pytest.approx(1e-3),
                "10": pytest.approx(1e-2), "15": pytest.approx(1e-2)}
    assert payload["metrics"]["lr_by_layer"] == expected
    assert payload["diagnostics"]["lr_by_layer"] == expected
    # No single value describes a mixed-band run; the per-layer map does.
    assert payload["metrics"]["lr"] is None
    assert seen and all(
        item == {1: pytest.approx(1e-3), 5: pytest.approx(1e-3),
                 10: pytest.approx(1e-2), 15: pytest.approx(1e-2)}
        for item in seen
    )


def test_c05_accepts_explicit_lr_and_validates_layer_mapping(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3])],
        {token: 10 for token in range(96)}, prompts=1, steps=[100],
        isa_layers=(1,), lr=1e-3, restarts=1, verbose=False,
    )
    payload = _payload(tmp_path, "ab")
    assert payload["metrics"]["lr"] == pytest.approx(1e-3)
    assert payload["metrics"]["lr_by_layer"] == {"1": pytest.approx(1e-3)}

    # An explicit mapping that agrees with the table is accepted.
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "mixed")
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3])],
        {token: 10 for token in range(96)}, prompts=1, steps=[100],
        isa_layers=(1, 10), lr={1: 1e-3, 10: 1e-2}, restarts=1, verbose=False,
    )
    mixed = _payload(tmp_path / "mixed", "ab")
    assert mixed["metrics"]["lr_by_layer"] == {
        "1": pytest.approx(1e-3), "10": pytest.approx(1e-2),
    }

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "chosen")
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3])], {token: 10 for token in range(96)},
        prompts=1, steps=[100], isa_layers=(1, 10), lr=1e-4, restarts=1, verbose=False,
    )
    assert _payload(tmp_path / "chosen", "ab")["metrics"]["lr_by_layer"] == {"1": 1e-4, "10": 1e-4}
    for layers, lr in (
        ((1,), -0.1),
        ((1,), float("nan")),
        ((1, 10), {1: 1e-3}),                 # mapping omits a sampled layer
        ((1,), {1: 1e-3, 10: 1e-2}),          # mapping names an unused layer
    ):
        with pytest.raises(ValueError, match="finite and positive|omits sampled layer|does not sample"):
            real_battery.run_c05(
                FakeDeployment(), [np.array([1, 2, 3])],
                {token: 10 for token in range(96)}, prompts=1, steps=[100],
                isa_layers=layers, lr=lr, restarts=1, verbose=False,
            )


def test_c05_paper_bound_payload_carries_the_binding(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_isa as real_isa
    from obfuscation_attacks.evaluation.paper_profile import (
        protocol_sha256, require_profile)
    from obfuscation_attacks.evaluation.protocol_audit import audit_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_isa, "isa_attack", _fake_isa_attack)
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3, 4, 5])],
        {token: 10 for token in range(96)}, prompts=1, steps=[100],
        isa_layers=(1,), restarts=1, verbose=False,
        profile=profile.name, protocol_sha256=digest,
    )
    payload = _payload(tmp_path, "ab")
    assert payload["profile"] == "paper-v1"
    assert payload["protocol_sha256"] == digest
    audit = audit_payload(payload, profile=profile, protocol_sha256=digest)
    assert not any(
        gap.startswith("profile:") or gap.startswith("protocol:")
        for gap in audit["gaps"]
    )


def test_linkage_rejects_non_finite_cosines_and_scores():
    from obfuscation_attacks.attacks.linkage.core import cosine_scores
    from obfuscation_attacks.data.streams import RequestRecord
    from obfuscation_attacks.metrics.recovery import linkage_tpr_fpr

    good = RequestRecord(request_id=0, symbols=np.arange(2), epoch=0,
                         vectors=np.ones((2, 3), dtype=np.float32))
    bad = RequestRecord(
        request_id=1, symbols=np.arange(2), epoch=0,
        vectors=np.asarray([[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0]],
                           dtype=np.float32),
    )
    pairs = {
        "delta": np.array([1]), "left_req": np.array([0]),
        "right_req": np.array([1]), "left_pos": np.array([0]),
        "right_pos": np.array([0]),
    }
    with pytest.raises(ValueError, match="non-finite"):
        cosine_scores([good, bad], pairs)
    with pytest.raises(ValueError, match="non-finite"):
        linkage_tpr_fpr([0.5, float("nan")], [True, False], 0.5)
    clean = linkage_tpr_fpr([0.5, 0.1], [True, False], 0.5)
    assert clean["tpr"] == 1.0 and clean["fpr"] == 0.0


def test_battery_summary_guard_rejects_raw_corpus_text_under_a_benign_key():
    guard = _load_battery_script()
    secret_text = "evaluated prompt alpha beta gamma"
    escaped_text = 'say "hi" with a \\ backslash and\nnewline'
    corpus = [secret_text, escaped_text, "short"]
    clean = {"classes": {"C12": {"rows": [
        {"prompt_index": 0, "prompt_sha256": "a" * 64}
    ]}}}
    assert guard._corpus_text_leaks(clean, corpus) == []
    leaky = {"classes": {"C12": {"rows": [{"note": secret_text}]}}}
    gaps = guard._corpus_text_leaks(leaky, corpus)
    assert gaps and "appears verbatim" in gaps[0]
    assert secret_text not in gaps[0]     # the leak itself is not echoed
    assert "summary" in gaps[0]
    # Parsed-string scanning finds text that a raw json.dumps substring would
    # escape (quotes, backslashes, newlines).
    escaped = {"classes": {"C12": {"rows": [{"note": escaped_text}]}}}
    escaped_gaps = guard._corpus_text_leaks(escaped, corpus)
    assert escaped_gaps and "appears verbatim" in escaped_gaps[0]
    assert escaped_text not in escaped_gaps[0]


def test_write_payload_refuses_truth_and_secret_material(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.attack_model import result_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def _payload() -> dict:
        return result_payload(
            attack_name="ia.gate",
            scheme_name="ab_deployment:aloepri_weights",
            target="llama-3.2-1b/aloepri_weights",
            metrics={},
            diagnostics={"positive_control": {}, "positive_control_passed": None},
            n_prompts=0, n_tokens=0, n_keys=1, n_types=0, reference_tokens=0,
            ci={}, source={"x": 1}, data={"x": 1},
        )

    for index, key in enumerate(
        ("truth", "plaintext_ids", "permuted_ids", "root_key", "client_secret")
    ):
        payload = _payload()
        payload["diagnostics"][key] = [1, 2]
        with pytest.raises(ValueError, match="truth/secret"):
            real_battery.write_payload("C08", "ia.gate", f"arm{index}", payload)
    assert not list(tmp_path.rglob("result.json"))

    written = real_battery.write_payload("C08", "ia.gate", "clean", _payload())
    assert written["result_path"]
    assert list(tmp_path.rglob("result.json"))


def test_write_payload_guards_the_merged_extra_block(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.attack_model import result_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def _payload() -> dict:
        return result_payload(
            attack_name="ia.gate",
            scheme_name="ab_deployment:aloepri_weights",
            target="llama-3.2-1b/aloepri_weights",
            metrics={},
            diagnostics={"positive_control": {}, "positive_control_passed": None},
            n_prompts=0, n_tokens=0, n_keys=1, n_types=0, reference_tokens=0,
            ci={}, source={"x": 1}, data={"x": 1},
        )

    for index, extra in enumerate((
        {"run_outcome": {"permuted_ids": [1, 2]}},
        {"identity": {"client_secret": "raw-bytes"}},
        {"identity": {"key_schedule": {"root_key_available": 1}}},
    )):
        with pytest.raises(ValueError, match="truth/secret"):
            real_battery.write_payload(
                "C08", "ia.gate", f"extra{index}", _payload(), extra
            )
    assert not list(tmp_path.rglob("result.json"))

    # Sanctioned per-unit evidence rows are the only ground-truth exemption.
    rows = [{"unit": 0, "hits": 1, "count": 1, "rank": [1],
             "predicted": [7], "truth": [7]}]
    written = real_battery.write_payload(
        "C08", "ia.gate", "evidence", _payload(),
        {"per_unit_predictions": rows},
    )
    assert written["result_path"]


def test_write_payload_scans_the_serialised_json_view(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.attack_model import result_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    payload = result_payload(
        attack_name="ia.gate",
        scheme_name="ab_deployment:aloepri_weights",
        target="llama-3.2-1b/aloepri_weights",
        metrics={},
        diagnostics={"positive_control": {}, "positive_control_passed": None},
        n_prompts=0, n_tokens=0, n_keys=1, n_types=0, reference_tokens=0,
        ci={}, source={"x": 1}, data={"x": 1},
    )
    # An ndarray hides its element mappings from a scan of the pre-serialised
    # object; json converts it to a list before writing.
    payload["diagnostics"]["blocks"] = np.array(
        [{"root_key": "raw-bytes"}], dtype=object
    )
    with pytest.raises(ValueError, match="truth/secret"):
        real_battery.write_payload("C08", "ia.gate", "ndarray", payload)
    assert not list(tmp_path.rglob("result.json"))


def test_guard_rejects_loose_prompt_and_commitment_list_values():
    from obfuscation_attacks.evaluation.public_guard import forbidden_material_gaps

    assert forbidden_material_gaps(
        {"metrics": {"per_prompt": ["raw prompt text"]}}
    )
    assert forbidden_material_gaps(
        {"client_secret_sha256": [0.1, 0.2]}
    )
    assert forbidden_material_gaps(
        {"identity": {"key_schedule": {"root_key_available": 1}}}
    )
    # Strict evidence schema: an extra key inside a row is rejected.
    assert forbidden_material_gaps(
        {"diagnostics": {"per_unit_predictions": [
            {"unit": 0, "hits": 1, "count": 1, "rank": [1],
             "plaintext_ids": [7]}
        ]}}
    )
    # The sanctioned row schema itself passes.
    assert forbidden_material_gaps(
        {"diagnostics": {"per_unit_predictions": [
            {"unit": 0, "hits": 1, "count": 1, "rank": [1],
             "predicted": [7], "truth": [7]}
        ]}}
    ) == []


def test_c10_saturation_study_hash_binds_the_seed():
    saturation = {"points": [{"budget": 4_450_000, "top1": 0.5},
                             {"budget": 10_000_000, "top1": 0.5}],
                  "gain_pp": 0.0}
    curve = {4_450_000: 0.5, 10_000_000: 0.5}
    identity = dict(run_data_sha256="a" * 64, corpus_sha256="b" * 64,
                    deployment_state_sha256="c" * 64,
                    reference_counts_sha256="d" * 64)
    seed4 = real_battery._saturation_study_sha256(
        saturation, curve, seed=4, **identity
    )
    seed999 = real_battery._saturation_study_sha256(
        saturation, curve, seed=999, **identity
    )
    assert seed4 != seed999
    # A study carrying a study_sha256 that does not match the run is refused.
    with pytest.raises(ValueError, match="study_sha256"):
        real_battery.run_c10(
            FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False,
            saturation={**saturation, "study_sha256": seed999},
        )


def test_c10_sh_align_top100_ignores_unassigned_positions(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    # The fixture has 96 vocabulary rows, including zero-frequency candidates.
    ref_counts = {token: 1000 - token for token in range(96)}
    out = run_c10(FakeDeployment(zero_first_vector=True), TEXTS, ref_counts,
                  n_keys=2, rotate_every=4, seed=4, verbose=False)
    payload = _payload(tmp_path, "SH-ALIGN")
    diagnostics = payload["diagnostics"]
    assert diagnostics["alignment"]["unassigned_positions"] == 1
    assert payload["top100_distinct"] == 96
    assert diagnostics["candidate_lengths_min"] == 96
    assert payload["metrics"]["candidate_coverage"]["denominator"] == (
        out["stream"]["n_positions"]
    )


def test_guard_allows_numeric_structural_keys_from_public_traces():
    from obfuscation_attacks.evaluation.public_guard import forbidden_material_gaps

    # Exact C13 public_trace fixtures: lengths and shapes are metadata.
    assert forbidden_material_gaps(
        {"diagnostics": {"public_trace": {
            "prompt_len": 5,
            "logits_perm_shape": [5, 128],
            "prompt_count": 3,
            "n_prompts": 3,
        }}}
    ) == []
    # Prompt TEXT keys (or non-numeric values under them) are still rejected.
    assert forbidden_material_gaps({"prompt_text": "raw prompt"})
    assert forbidden_material_gaps({"prompt": "raw prompt"})
    assert forbidden_material_gaps({"prompt_len": "raw prompt"})


def test_c05_saturation_gain_is_signed():
    curve = {"layer1": {
        "steps10": {"top1": {"value": 0.95}},
        "steps20": {"top1": {"value": 0.90}},
        "steps30": {"top1": {"value": 0.96}},
    }}
    regression = real_battery._saturation_record(curve, "layer1", [10, 20])
    assert regression["gain_pp"] == pytest.approx(-5.0)
    improvement = real_battery._saturation_record(curve, "layer1", [20, 30])
    assert improvement["gain_pp"] == pytest.approx(6.0)
    assert real_battery._saturation_record(curve, "layer1", [10]) is None


def _load_script_module(filename: str, alias: str):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c06w_score_keeps_invalid_predictions_in_the_denominator():
    predicted = np.asarray([0, 1, -1, 2])
    truth = np.asarray([0, 1, 2, 3])
    rate = real_battery._mapping_recovery_rate(predicted, truth)
    assert rate == pytest.approx(2 / 4)     # NOT 2/3 (valid-only denominator)
    assert real_battery._mapping_recovery_rate([0, 1, 2, 3], truth) == 1.0
    assert real_battery._mapping_recovery_rate([-1, -1], truth) == 0.0
    assert real_battery._mapping_recovery_rate([], []) == 0.0


def test_weight_probe_scripts_refuse_non_clean_scenarios(monkeypatch):
    import sys as _sys

    for filename, alias in (("table9_probe.py", "table9_probe_guard_test"),
                            ("dense_mixing_probe.py", "dense_probe_guard_test"),
                            ("vma_profile.py", "vma_profile_guard_test")):
        module = _load_script_module(filename, alias)
        assert module.require_clean_scenario("S-public") == "S-public"
        assert module.require_clean_scenario("S-pair") == "S-pair"
        with pytest.raises(SystemExit, match="refusing to read clean"):
            module.require_clean_scenario("S-deploy")

        python_args = ["probe", "--scenario", "S-deploy"]
        monkeypatch.setattr(_sys, "argv", python_args)

        class _Bomb:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "ABDeployment must not be constructed for a non-clean scenario"
                )

        monkeypatch.setattr(module, "ABDeployment", _Bomb)
        with pytest.raises(SystemExit, match="refusing to read clean"):
            module.main()


def test_linkage_rejects_zero_norm_and_undefined_nearest_partner():
    from obfuscation_attacks.attacks.linkage.core import (
        cosine_scores,
        nearest_partner,
    )
    from obfuscation_attacks.data.streams import RequestRecord

    zero = RequestRecord(request_id=0, symbols=np.arange(2), epoch=0,
                         vectors=np.zeros((2, 3), dtype=np.float32))
    other = RequestRecord(request_id=1, symbols=np.arange(2), epoch=0,
                          vectors=np.ones((2, 3), dtype=np.float32))
    pairs = {
        "delta": np.array([1]), "left_req": np.array([0]),
        "right_req": np.array([1]), "left_pos": np.array([0]),
        "right_pos": np.array([0]),
    }
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_scores([zero, other], pairs)

    huge_vectors = np.asarray([[1e200, 0.0, 0.0], [0.0, 1.0, 0.0]],
                              dtype=np.float64)
    huge_a = RequestRecord(request_id=0, symbols=np.arange(2), epoch=0,
                           vectors=huge_vectors)
    huge_b = RequestRecord(request_id=1, symbols=np.arange(2), epoch=0,
                           vectors=huge_vectors)
    with pytest.raises(ValueError, match="non-finite|zero"):
        nearest_partner([huge_a, huge_b], max_queries=1)

    clean_a = RequestRecord(request_id=0, symbols=np.arange(2), epoch=0,
                            vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]],
                                               dtype=np.float32))
    clean_b = RequestRecord(request_id=1, symbols=np.arange(2), epoch=0,
                            vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]],
                                               dtype=np.float32))
    out = nearest_partner([clean_a, clean_b], max_queries=2)
    assert out["n_queries"] > 0
    assert all(np.isfinite(value) for value in out["best_cos"])
    assert max(out["best_cos"]) == pytest.approx(1.0)


def test_c06w_chance_baseline_is_per_variant(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c06w(FakeDeployment(), n_types=16, diag_types=16, seed=0, verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    baseline = payload["metrics"]["chance_baseline"]
    assert baseline["diagnostic_rows"] == pytest.approx(1.0 / 16)
    assert baseline["full_vocab"] == pytest.approx(1.0 / 96)
    chance_by_variant = payload["diagnostics"]["chance_by_variant"]
    assert chance_by_variant["paper_literal_rowsort"] == pytest.approx(1.0 / 16)
    assert chance_by_variant["scale_normalised_rowsort"] == pytest.approx(1.0 / 16)
    assert chance_by_variant["dense_mixing_inversion"] == pytest.approx(1.0 / 96)
    assert payload["diagnostics"]["chance"] == pytest.approx(1.0 / 96)


def test_c05_class_runner_forwards_the_configured_hyperparameters(monkeypatch):
    captured = {}

    def fake_run_c05(dep, texts, ref_counts, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(real_battery, "run_c05", fake_run_c05)

    class _Corpus:
        def __getitem__(self, name):
            return self

        def texts(self):
            return [np.array([1, 2, 3])]

    validation = {"n_prompts": 40,
                  "strata": {"32-64": 10, "65-128": 10, "129-256": 10,
                             "257-512": 10},
                  "lr": 1e-3, "lambda_dm": 0.0}
    cfg = {
        "isa_prompts": 2, "isa_steps": [1000, 50000], "isa_lr": 1e-3,
        "isa_lambda_dm": 0.0, "isa_layers": [1, 5, 10, 15], "isa_method": "tbs",
        "isa_restarts": 3, "keys": 5, "isa_validation": validation,
        "seed": 7, "scenario": "S-public",
    }
    real_battery.CLASS_RUNNERS["C05"](None, _Corpus(), {}, cfg)
    assert captured["method"] == "tbs"
    assert captured["restarts"] == 3
    assert captured["lr"] == pytest.approx(1e-3)
    assert captured["lambda_dm"] == pytest.approx(0.0)
    assert tuple(captured["isa_layers"]) == (1, 5, 10, 15)
    assert captured["selected_on_validation"] == validation
    assert captured["prompts"] == 2
    assert tuple(captured["steps"]) == (1000, 50000)
    assert captured["keys_per_prompt"] == 5


def test_isa_cli_flags_are_parsed(monkeypatch):
    module = _load_battery_script()
    parser = module._build_arg_parser()
    args = parser.parse_args([
        "--isa-method", "tbs", "--isa-restarts", "3", "--isa-keys", "5", "--isa-lr", "1e-3",
        "--isa-lambda-dm", "0.0", "--isa-layers", "1,5,10,15",
    ])
    assert args.isa_method == "tbs"
    assert args.isa_restarts == 3
    assert args.isa_keys == 5
    assert args.isa_lr == pytest.approx(1e-3)
    assert args.isa_lambda_dm == pytest.approx(0.0)
    assert args.isa_layers == "1,5,10,15"


def test_paper_branch_guard_runs_before_writing(monkeypatch, tmp_path):
    import sys as _sys

    import obfuscation_attacks.evaluation.paper_runner as paper_runner

    guard_module = _load_battery_script()
    monkeypatch.setattr(
        guard_module, "gate_battery_args",
        lambda *args, **kwargs: {"action": "run_paper"},
    )
    out = tmp_path / "paper_run_status.json"
    monkeypatch.setattr(
        _sys, "argv", ["run_full_battery.py", "--run-out", str(out)]
    )

    monkeypatch.setattr(
        paper_runner, "dispatch_configured",
        lambda *args, **kwargs: {"status": "executed", "configured": [],
                                 "truth": [1, 2]},
    )
    with pytest.raises(SystemExit, match="truth/secret material"):
        guard_module.main()
    assert not out.exists()

    monkeypatch.setattr(
        paper_runner, "dispatch_configured",
        lambda *args, **kwargs: {"status": "executed", "configured": []},
    )
    with pytest.raises(SystemExit) as excinfo:
        guard_module.main()
    assert excinfo.value.code == 0
    assert out.exists()


def test_linkage_control_helper_matches_the_producer_shape():
    from obfuscation_attacks.evaluation.real_battery import _linkage_control_passed
    from obfuscation_attacks.metrics.recovery import delta_bucket

    assert delta_bucket(1) == "1"
    produced = {delta_bucket(1): {"n_pairs": 3, "link_rate": 1.0}}
    assert _linkage_control_passed(produced) is True
    assert _linkage_control_passed({"1": {"n_pairs": 3, "link_rate": 0.5}}) is False
    # The old mislabelled shape must not be accepted as a control.
    assert _linkage_control_passed({"1-1": {"exact_link_rate": 1.0}}) is False


def test_c12_single_heldout_session_reports_no_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False)
    # Held-out set is one session: exact test and every delta bucket are
    # unestimable and must report no bounds / n_boot 0.
    assert out["exact_test"]["n_boot"] == 0
    assert out["exact_test"]["lo"] is None and out["exact_test"]["hi"] is None
    payload = _payload(tmp_path, "ab")
    assert payload["metrics"]["delta_buckets"]
    for label, block in payload["metrics"]["delta_buckets"].items():
        assert block["n_boot"] == 0, label
        assert block["lo"] is None and block["hi"] is None, label

    mapped_root = tmp_path / "mapped"
    monkeypatch.setattr(real_battery, "OUT_ROOT", mapped_root)
    mapping = {index: index for index in range(96)}
    mapped = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4,
                     seed=5, verbose=False, stage2_mapping=mapping)
    coverage = mapped["recovered_text"]["coverage_positives"]
    assert coverage["n_boot"] == 0
    assert coverage["lo"] is None and coverage["hi"] is None


def test_stream_cache_reacts_to_in_place_table_mutation():
    dep = FakeDeployment()
    cache: dict = {}
    first = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    dep.E_dep[0, 0] = float(dep.E_dep[0, 0]) + 1.0   # in-place mutation, same object
    second = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]


def test_symbol_invariance_is_not_measured_without_cross_key_occurrences():
    stream = real_battery.build_stream(FakeDeployment(), TEXTS[:2], n_keys=1,
                                       rotate_every=4, seed=1)
    premise = real_battery._symbol_invariance(stream)
    assert premise["measured"] is False
    assert premise["premise_holds"] is not True
    assert premise["reason"]
    assert premise["n_cross_key_token_pairs"] == 0


def test_stream_cache_reacts_to_post_construction_encoder_mutation():
    dep = FakeDeployment(encoder_state={"w": np.asarray([1.0], dtype=np.float32)})
    cache: dict = {}
    first = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    dep.encoder_state["w"][0] = 2.0            # in-memory mutation after construction
    second = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]


def test_stream_cache_reacts_to_post_construction_file_mutation(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "osnip").mkdir(parents=True)
    encoder = artifact / "osnip" / "encryptor.pt"
    encoder.write_bytes(b"A")
    dep = FakeDeployment(artifact=artifact)
    cache: dict = {}
    first = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    encoder.write_bytes(b"BB")                 # same dep, changed file/mtime/size
    second = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]


def test_c10_sh_align_skips_unassigned_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    small_ref = {token: 100 for token in range(5)}
    out = run_c10(FakeDeployment(zero_first_vector=True), TEXTS, small_ref,
                  n_keys=2, rotate_every=4, seed=4, verbose=False)
    payload = _payload(tmp_path, "SH-ALIGN")
    diagnostics = payload["diagnostics"]
    assert diagnostics["alignment"]["unassigned_positions"] == 1
    assert diagnostics["alignment"]["n_deployment_ids"] == out["SH-ALIGN"]["n_groups"]
    # The unassigned position has an empty candidate list, not row -1.
    rows = diagnostics["per_unit_predictions"]
    assert rows[0]["predicted"][0] == -1
    coverage = payload["metrics"]["candidate_coverage"]
    assert coverage["denominator"] == out["stream"]["n_positions"]
    assert coverage["value"] < 1.0


def test_clean_weight_runners_refuse_non_clean_scenarios():
    dep = FakeDeployment()
    gate = real_battery.run_c08(dep, n_types=5, verbose=False, scenario="S-deploy")
    assert gate["status"] == "not_run" and gate["scenario"] == "S-deploy"
    attn = real_battery.run_c08_attn(dep, n_types=5, verbose=False,
                                     scenario="S-deploy")
    assert attn["status"] == "not_run" and attn["scenario"] == "S-deploy"
    vma = real_battery.run_c06w(dep, verbose=False, scenario="S-deploy")
    assert vma["status"] == "not_run" and vma["scenario"] == "S-deploy"
    with pytest.raises(ValueError, match="clean model"):
        real_battery.run_c05(
            dep, [np.array([1, 2, 3])], {token: 10 for token in range(96)},
            prompts=1, steps=[2], verbose=False, scenario="S-deploy",
        )
    with pytest.raises(ValueError, match="clean model"):
        real_battery.run_c06h(dep, [np.array([1, 2, 3])], layers=(1,),
                              verbose=False, scenario="S-deploy")


def test_c10_saturation_study_binds_and_curve_is_copied(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.paper_profile import (
        protocol_sha256, require_profile)
    from obfuscation_attacks.evaluation.protocol_audit import audit_payload

    def constant_candidates(group_symbols, ref_counts, **kwargs):
        return {symbol: [0, 1, 2] for symbol in group_symbols}

    monkeypatch.setattr(real_battery, "_rank_window_candidates", constant_candidates)
    ref_counts = {token: 250_000 for token in range(40)}   # 10,000,000 total
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "probe")
    probe = run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2,
                    rotate_every=4, seed=4, verbose=False)
    top1 = probe["SH-P"]["top1"]
    assert probe["SH-B"]["top1"] == pytest.approx(top1)
    assert probe["SH-ALIGN"]["top1"] == pytest.approx(top1)
    probe_data_sha256 = _payload(tmp_path / "probe", "SH-P")["data_label_sha256"]

    saturation = {"points": [{"budget": 4_450_000, "top1": top1},
                             {"budget": 10_000_000, "top1": top1}],
                  "gain_pp": 0.0,
                  "data_sha256": probe_data_sha256}
    curve = {4_450_000: top1, 10_000_000: top1}
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "match")
    run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2, rotate_every=4,
            seed=4, verbose=False, profile=profile.name,
            protocol_sha256=digest, saturation=saturation,
            saturation_curve=curve, saturation_variant="SH-P")
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        payload = _payload(tmp_path / "match", mode)
        diagnostics = payload["diagnostics"]
        assert payload["metrics"]["occurrence_level"]["top1"]["value"] == pytest.approx(
            top1)
        if mode == "SH-P":
            assert diagnostics["saturation"] == saturation, mode
            assert diagnostics["saturation_curve"] == {
                str(key): value for key, value in curve.items()
            }, mode
            assert diagnostics["saturation_variant"] == mode, mode
            assert diagnostics["saturation_binding"]["status"] == "bound"
        else:
            # The study is bound to SH-P only; other variants carry neither the
            # study nor its curve (no cross-variant curve validation).
            assert "saturation" not in diagnostics, mode
            assert "saturation_curve" not in diagnostics, mode
            assert diagnostics["saturation_binding"]["status"] == "bound", mode

    payload = _payload(tmp_path / "match", "SH-P")
    payload["diagnostics"]["positive_control_passed"] = True
    payload["diagnostics"]["positive_control"]["top1"] = 0.95
    audit = audit_payload(payload, profile=profile, protocol_sha256=digest)
    assert any("saturation" in gap for gap in audit["gaps"])
    tampered = json.loads(json.dumps(payload))
    tampered["diagnostics"]["saturation"]["points"][-1]["top1"] += 0.5
    audit_bad = audit_payload(tampered, profile=profile, protocol_sha256=digest)
    assert any("saturation" in gap for gap in audit_bad["gaps"])

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "mismatch")
    bad = {"points": [{"budget": 4_450_000, "top1": top1 + 0.5},
                      {"budget": 10_000_000, "top1": top1 + 0.5}],
           "gain_pp": 0.0,
           "data_sha256": probe_data_sha256}
    with pytest.raises(ValueError, match="does not match"):
        run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2, rotate_every=4,
                seed=4, verbose=False, saturation=bad)
    assert not list((tmp_path / "mismatch").rglob("result.json"))

    # A different run identity (different seed) refuses the same study.
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "identity")
    with pytest.raises(ValueError, match="data_sha256"):
        run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2, rotate_every=4,
                seed=999, verbose=False, saturation=saturation)
    assert not list((tmp_path / "identity").rglob("result.json"))


def test_c10_saturation_tie_without_selection_binds_none(monkeypatch, tmp_path):
    def constant_candidates(group_symbols, ref_counts, **kwargs):
        return {symbol: [0, 1, 2] for symbol in group_symbols}

    monkeypatch.setattr(real_battery, "_rank_window_candidates", constant_candidates)
    ref_counts = {token: 250_000 for token in range(40)}
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "probe")
    probe = run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2,
                    rotate_every=4, seed=4, verbose=False)
    top1 = probe["SH-P"]["top1"]
    probe_data_sha256 = _payload(tmp_path / "probe", "SH-P")["data_label_sha256"]
    saturation = {"points": [{"budget": 4_450_000, "top1": top1},
                             {"budget": 10_000_000, "top1": top1}],
                  "gain_pp": 0.0,
                  "data_sha256": probe_data_sha256}
    curve = {4_450_000: top1, 10_000_000: top1}
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "tie")
    run_c10(FakeDeployment(), TEXTS, ref_counts, n_keys=2, rotate_every=4,
            seed=4, verbose=False, saturation=saturation,
            saturation_curve=curve)
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        diagnostics = _payload(tmp_path / "tie", mode)["diagnostics"]
        assert "saturation" not in diagnostics, mode
        assert "saturation_curve" not in diagnostics, mode
        assert diagnostics["saturation_binding"]["status"] == "ambiguous", mode
        assert set(diagnostics["saturation_binding"]["candidates"]) == {
            "SH-P", "SH-B", "SH-ALIGN"
        }


def test_partition_texts_respects_the_request_floor():
    texts = [str(index) for index in range(15000)]
    partitioned = real_battery._partition_texts(texts)
    assert partitioned is not None
    assert len(partitioned) == 3
    assert [len(row) for row in partitioned] == [5000, 5000, 5000]
    assert sorted(value for row in partitioned for value in row) == sorted(texts)
    assert real_battery._partition_texts(texts[:14999]) is None


def test_c12_rejects_zero_norm_wire_vectors(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    zero_request = {"vectors": np.zeros((2, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match="zero-norm"):
        real_battery._cosine(zero_request, 0, zero_request, 1)
    with pytest.raises(ValueError, match="zero-norm"):
        run_c12(FakeDeployment(zero_all_vectors=True), TEXTS, n_keys=2,
                rotate_every=4, seed=5, verbose=False)
    assert not list(tmp_path.rglob("result.json"))


def test_c12_rejects_invalid_deployment_table_rows(monkeypatch, tmp_path):
    for corruption, match in (("nan", "non-finite"), ("zero", "all-zero")):
        root = tmp_path / corruption
        monkeypatch.setattr(real_battery, "OUT_ROOT", root)
        with pytest.raises(ValueError, match=match):
            run_c12(FakeDeployment(table_corruption=corruption), TEXTS,
                    n_keys=2, rotate_every=4, seed=5, verbose=False)
        assert not list(root.rglob("result.json"))


def test_c12_rejects_non_finite_cosines(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        run_c12(FakeDeployment(nonfinite_vector=True), TEXTS, n_keys=2,
                rotate_every=4, seed=5, verbose=False)
    assert not list(tmp_path.rglob("result.json"))


def test_linkage_control_floor_is_099(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.real_battery import _linkage_control_passed

    assert _linkage_control_passed({"1": {"link_rate": 0.96}}) is False
    assert _linkage_control_passed({"1": {"link_rate": 0.995}}) is True
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
            verbose=False)
    payload = _payload(tmp_path, "ab")
    assert payload["diagnostics"]["positive_control"]["required_top1"] == 0.99


class _AttnFakePublic:
    """Tiny public view for the Attn-IA known-construction control."""

    arm = "ab"
    vocab_size = 32
    n_heads = 1
    n_kv_heads = 1
    n_layers = 2
    layers = (1,)

    def __init__(self, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((self.vocab_size, 4)).astype(np.float32)
        self.W = rng.standard_normal((4, 4)).astype(np.float32)

    def table(self, arm, normalize=False):
        return self.E

    def weight_tensor(self, name, side):
        return self.W


def test_c08a_known_construction_control_passes_and_can_fail():
    public = _AttnFakePublic()
    types = np.arange(10)
    passed = real_battery._attn_known_construction_control(
        public, types, layers=(1,), seed=0
    )
    assert passed["passed"] is True
    assert passed["top1"] >= 0.99
    assert passed["required_top1"] == 0.99
    assert "row permutation" in passed["construction"]


class _GateFakePublic:
    """Tiny public view for the Gate-IA known-construction control."""

    arm = "ab"
    vocab_size = 32
    n_heads = 1
    n_kv_heads = 1
    n_layers = 2
    layers = (1,)
    hidden_size = 4

    def __init__(self, seed: int = 5):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((self.vocab_size, 4)).astype(np.float32)
        # HF Linear layout (intermediate, hidden): (8, 4).
        self.W = rng.standard_normal((8, 4)).astype(np.float32)

    def table(self, arm, normalize=False):
        return self.E

    def weight_tensor(self, name, side):
        return self.W


def test_c08_gate_known_construction_control_passes_and_can_fail():
    public = _GateFakePublic()
    types = np.arange(10)
    control = real_battery._gate_known_construction_control(
        public, types, layers=(1,), seed=0
    )
    assert control["passed"] is True
    assert control["top1"] >= 0.99
    assert control["required_top1"] == 0.99
    assert "row permutation" in control["construction"]

    corrupted = real_battery._gate_known_construction_control(
        public, types, layers=(1,), seed=0, corrupt=True
    )
    assert corrupted["passed"] is False
    assert corrupted["top1"] < 0.99
    assert corrupted["matches_recovered_construction"] is False

    corrupted = real_battery._attn_known_construction_control(
        public, types, layers=(1,), seed=0, corrupt=True
    )
    assert corrupted["passed"] is False
    assert corrupted["top1"] < 0.99
    assert corrupted["matches_recovered_construction"] is False


def test_c08a_payload_has_an_in_cell_control_not_a_reference(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_weight_attacks as weight_attacks

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def fake_attn(public, types, layers=None, **kwargs):
        return {
            "candidates": {int(token): [0, 1] for token in types},
            "invariant_dims": 7,
            "layers": [1],
            "missing_weight_layers": [],
            "pairing": "half_split",
            "block_size": 2,
            "n_chunks": 1,
        }

    monkeypatch.setattr(weight_attacks, "attn_public_candidates", fake_attn)
    real_battery.run_c08_attn(FakeDeployment(n_layers=16), n_types=5,
                              seed=0, verbose=False)
    diagnostics = _payload(tmp_path, "aloepri_weights")["diagnostics"]
    control = diagnostics["positive_control"]
    assert control["name"] == "known_identity_weights_with_row_permutation"
    assert "reference" not in control
    assert control.get("status") != "reference"
    assert isinstance(control["top1"], float)
    assert isinstance(control["passed"], bool)
    assert diagnostics["positive_control_passed"] == control["passed"]


def test_c12_delta_buckets_exclude_cross_stream_pairs():
    from obfuscation_attacks.evaluation.real_battery import _delta_bucket_blocks

    requests = [
        {"stream_id": 0, "request_id": 0, "symbols": [7], "vectors": np.array([[7.]], dtype=np.float32)},
        {"stream_id": 1, "request_id": 0, "symbols": [7], "vectors": np.array([[7.]], dtype=np.float32)},
        {"stream_id": 0, "request_id": 2, "symbols": [9], "vectors": np.array([[9.]], dtype=np.float32)},
    ]
    rows = [
        (0, 0, 1, 0, 7),   # pooled gap 1 but cross-stream: excluded
        (0, 0, 2, 0, 9),   # same stream, within-stream gap 2
    ]
    buckets, excluded = _delta_bucket_blocks(requests, rows, seed=0)
    assert excluded == 1
    assert "1" not in buckets
    assert "2-3" in buckets
    assert buckets["2-3"]["denominator"] == 1
    assert buckets["2-3"]["numerator"] == 0    # symbols 7 vs 9 differ


def test_knn_ids_and_outcomes_treat_unassignable_vectors_as_unassigned():
    from obfuscation_attacks.evaluation.real_battery import (
        _deployment_id_outcomes,
        _knn_deployment_ids,
    )

    table = np.eye(3, dtype=np.float32)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]],
        dtype=np.float32,
    )
    ids = _knn_deployment_ids([{"vectors": vectors}], table)[0]
    assert ids[0] >= 0
    assert ids[1] == -1 and ids[2] == -1       # zero / inf vectors unassigned
    deployment_ids = [np.asarray([2, -1, -1]), np.asarray([2, -1, 3])]
    rows = [
        (0, 0, 1, 0, 0),   # 2 vs 2 -> match
        (0, 1, 1, 1, 0),   # -1 vs -1 -> NOT a match
        (0, 2, 1, 2, 0),   # -1 vs 3 -> NOT a match
    ]
    assert _deployment_id_outcomes(rows, deployment_ids) == [True, False, False]


def test_c12_stage1_coverage_counts_only_assigned_positions(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    real_knn = real_battery._knn_deployment_ids

    def _unassign_one(requests, table, **kwargs):
        out = real_knn(requests, table, **kwargs)
        ids = [np.array(entry, copy=True) for entry in out]
        ids[0][0] = -1          # a stage-1 position KNN could not assign
        return ids

    monkeypatch.setattr(real_battery, "_knn_deployment_ids", _unassign_one)
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False)
    coverage = out["deployment_id_test"]["stage1_coverage"]
    assert coverage["denominator"] == 5 * len(TEXTS)
    assert coverage["numerator"] == coverage["denominator"] - 1
    assert coverage["value"] < 1.0


def test_c12_delta_buckets_use_the_frozen_protocol_table(monkeypatch, tmp_path):
    from obfuscation_attacks.metrics.recovery import DELTA_BUCKETS

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=3, seed=5,
            verbose=False)
    payload = _payload(tmp_path, "ab")
    buckets = payload["metrics"]["delta_buckets"]
    frozen_labels = {
        (f"{low}" if high == low
         else (f"{low}-{high}" if high is not None else f"{low}+"))
        for low, high in DELTA_BUCKETS
    }
    assert set(buckets) <= frozen_labels
    assert "2-3" in buckets                       # a gap-2 pair lands in "2-3"
    assert not ({"1-15", "64-255", "256+"} & set(buckets))
    assert payload["metrics"]["delta_buckets_excluded_cross_stream"] == 0


def test_s05_threshold_tie_is_resolved_above_the_tied_value():
    selection = _threshold_from_validation([0.9, 0.9, 0.9, 0.1], 1e-2)
    assert selection["threshold"] > 0.9
    assert selection["val_flagged"] == 0
    assert selection["val_n"] == 4
    assert selection["val_fpr_actual"] == pytest.approx(0.0)
    assert selection["tie_boundary_adjusted"] is True

    interior = _threshold_from_validation([0.9, 0.9, 0.9, 0.1], 0.5)
    assert interior["threshold"] > 0.9
    assert interior["val_flagged"] <= interior["val_n"] * 0.5
    assert interior["tie_boundary_adjusted"] is True


# ---------------------------------------------------------------------------
# S06 -- TFMA grouping by key identity, SH-B premise, SH-ALIGN
# ---------------------------------------------------------------------------
REF_COUNTS = {token: 1000 - token for token in range(20)}


def test_s06_sh_p_groups_by_group_id_across_epochs():
    requests = [
        {"group_id": "A", "epoch": 0, "symbols": [10, 11]},
        {"group_id": "A", "epoch": 1, "symbols": [10, 12]},
        {"group_id": "B", "epoch": 2, "symbols": [10]},
    ]
    groups = _tfma_group_counts(requests, "SH-P")
    assert groups["A"] == {10: 2, 11: 1, 12: 1}      # epochs 0 and 1 merged
    assert groups["B"] == {10: 1}
    assert sum(sum(bucket.values()) for bucket in groups.values()) == 5


def test_s06_one_key_makes_sh_p_identical_to_the_merged_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=1, rotate_every=4,
                  seed=4, verbose=False)
    assert out["SH-P"]["n_groups"] == 1
    assert out["SH-P"]["top1"] == pytest.approx(out["SH-B"]["top1"])
    assert out["SH-P"]["top10"] == pytest.approx(out["SH-B"]["top10"])


def test_s06_attack_tables_do_not_read_truth(monkeypatch, tmp_path):
    dep = FakeDeployment()
    stream = real_battery.build_stream(dep, TEXTS, n_keys=2, rotate_every=4, seed=4)
    altered_truth = [
        {"plaintext_ids": np.roll(np.asarray(row["plaintext_ids"]), 1),
         "permuted_ids": row["permuted_ids"]}
        for row in stream["truth"]
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    monkeypatch.setattr(real_battery, "OUT_ROOT", first_dir)
    monkeypatch.setattr(real_battery, "build_stream", lambda *a, **k: stream)
    first = run_c10(dep, TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
                    seed=4, verbose=False)

    altered_stream = {**stream, "truth": altered_truth}
    monkeypatch.setattr(real_battery, "OUT_ROOT", second_dir)
    monkeypatch.setattr(real_battery, "build_stream", lambda *a, **k: altered_stream)
    second = run_c10(dep, TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
                     seed=4, verbose=False)

    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        assert first[mode]["group_candidates_sha256"] == (
            second[mode]["group_candidates_sha256"]
        ), mode


def test_s06_sh_b_carries_the_symbol_invariance_premise(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
                  seed=4, verbose=False)
    assert out["symbol_invariance"]["premise_holds"] is False
    assert out["symbol_invariance"]["measured"] is True
    assert out["symbol_invariance"]["symbol_agreement"] == pytest.approx(0.0)
    payload = _payload(tmp_path, "SH-B")
    premise = payload["diagnostics"]["premise"]
    assert premise["premise_holds"] is False
    assert "key-specific" in premise["premise"]
    assert payload["diagnostics"]["diagnostic_only"] is True


def test_s06_sh_align_runs_and_counts_every_position(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    dep = FakeDeployment(duplicate_rows=(5, 6))
    token = int(np.flatnonzero(dep.perm == 6)[0])     # deployment id 6 is duplicated
    texts = list(TEXTS) + [np.array([token, 1, 2])]
    out = run_c10(dep, texts, REF_COUNTS, n_keys=2, rotate_every=4, seed=4,
                  verbose=False)
    assert "SH-ALIGN" in out
    assert out["SH-ALIGN"]["n_positions"] == out["stream"]["n_positions"]
    assert out["SH-ALIGN"]["stage1_error_rate"] > 0
    payload = _payload(tmp_path, "SH-ALIGN")
    metrics = payload["metrics"]
    assert metrics["occurrence_level"]["top1"]["denominator"] == out["stream"]["n_positions"]
    assert metrics["stage1_deployment_id_top1"]["denominator"] == out["stream"]["n_positions"]
    assert payload["diagnostics"]["alignment"]["uses_secret"] is False


def test_s06_c10_top100_distinct_is_the_measured_minimum(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    # The 96-row vocabulary includes zero-frequency tokens as candidates.
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        payload = _payload(tmp_path, mode)
        assert payload["top100_distinct"] == 96, mode
        assert payload["diagnostics"]["top100_distinct"] == 96, mode


def test_s06_c10_payload_is_dispatched_as_tfma_after_inversion(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        payload = _payload(tmp_path, mode)
        assert payload["attack_id"] == "tfma_after_inversion", mode
        assert payload["attack"] == "tfma.independent", mode
        assert payload["attack_variant"] == mode, mode
        # The override is not applied by result_payload; the control verdict is
        # recorded explicitly (schema forbids self-certified eligibility).
        assert payload["main_table_eligible"] is False, mode
        assert payload["main_table_eligible_audit_hint"] == bool(
            payload["diagnostics"]["positive_control_passed"]
        ), mode


def test_s06_c10_per_unit_predictions_reconcile_with_occurrence_top1(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        payload = _payload(tmp_path, mode)
        rows = payload["diagnostics"]["per_unit_predictions"]
        top1 = payload["metrics"]["occurrence_level"]["top1"]
        assert len(rows) == payload["n_prompts"]
        assert payload["n_keys"] == 1            # one request = one prompt x one key
        assert payload["diagnostics"]["distinct_rotation_keys"] >= 1
        assert sum(row["count"] for row in rows) == top1["denominator"]
        assert sum(row["hits"] for row in rows) == top1["numerator"]
        for row in rows:
            assert row["count"] > 0 and len(row["rank"]) == row["count"]
            assert all(isinstance(value, int) and value >= 1 for value in row["rank"])
            assert row["hits"] == sum(1 for value in row["rank"] if value <= 1)
            for rank, predicted, truth in zip(
                row["rank"], row["predicted"], row["truth"]
            ):
                assert (rank == 1) == (predicted == truth)


# ---------------------------------------------------------------------------
# S09 -- ratio bootstrap: micro point and ratio replicates, not a macro mean
# ---------------------------------------------------------------------------
def test_s09_ratio_ci_point_is_micro_not_macro():
    ci = ratio_ci([1, 0], [1, 9], n_boot=2000, seed=0)
    assert ci["micro"] == pytest.approx(0.1)
    assert ci["macro_supplementary"] == pytest.approx(0.5)
    assert ci["estimand"] == "micro"
    assert ci["n_boot"] == 2000
    assert ci["lo"] <= ci["micro"] <= ci["hi"]


def test_s09_interval_is_centred_on_the_ratio_not_the_mean_of_rates():
    ci = ratio_ci([1] + [0] * 9, [1] + [9] * 9, n_boot=2000, seed=0)
    micro = 1.0 / 82.0
    assert ci["micro"] == pytest.approx(micro)
    assert ci["macro_supplementary"] == pytest.approx(0.1)
    midpoint = (ci["lo"] + ci["hi"]) / 2.0
    assert abs(midpoint - micro) < abs(midpoint - 0.1)


def test_s09_composed_and_tfma_payloads_use_ratio_intervals(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    composed = run_composed(FakeDeployment(vocab=128), TEXTS[:4], REF_COUNTS,
                            n_types=8, seed=0, verbose=False)
    ci = composed["composed_ci"]
    assert ci["estimand"] == "micro"
    assert ci["n_boot"] >= 2000
    assert ci["micro"] == pytest.approx(composed["composed_top1"])

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "tfma")
    out = run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
                  seed=4, verbose=False)
    for mode in ("SH-P", "SH-B", "SH-ALIGN"):
        block = out[mode]["ci_top1"]
        assert block["estimand"] == "micro"
        assert block["micro"] == pytest.approx(out[mode]["top1"])
        assert block["n_boot"] >= 2000
        assert block["n_clusters"] >= 2


def test_composed_attack_keeps_low_alignment_predictions_and_full_denominator(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    dep = FakeDeployment(vocab=128)
    dep.E_dep = np.random.default_rng(37).normal(size=dep.E_dep.shape).astype(np.float32)
    result = run_composed(dep, TEXTS[:4], REF_COUNTS, n_types=8, seed=2, verbose=False)
    payload = _payload(tmp_path, "ab_composed")
    assert result["composed_top1"] is not None
    assert payload["metrics"]["composed_end_to_end"]["top1"]["denominator"] == 20
    assert payload["diagnostics"]["stage2_premise"]["top1_cosine_mean"] < 0.9
    assert payload["attack"] == "vma.dense_mixing_composed"
    assert payload["attack_id"] == "least_squares"
    assert "not AloePri Table 9" in payload["paper"]
    assert payload["diagnostics"]["positive_control"]["name"] == "synthetic_dense_full_composition"
    assert payload["diagnostics"]["positive_control_passed"] is True


def test_composed_positive_control_rejects_a_broken_mapping_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    original = real_battery._compose_dense_candidates

    def broken(*args, **kwargs):
        candidates, evidence = original(*args, **kwargs)
        return np.zeros_like(candidates), evidence

    monkeypatch.setattr(real_battery, "_compose_dense_candidates", broken)
    run_composed(FakeDeployment(vocab=128), TEXTS[:4], REF_COUNTS, n_types=8, seed=0, verbose=False)
    diagnostics = _payload(tmp_path, "ab_composed")["diagnostics"]
    assert diagnostics["stage1_plaintext_control"]["passed"] is True
    assert diagnostics["positive_control"]["passed"] is False
    assert diagnostics["positive_control_passed"] is False


def test_composed_result_identity_changes_with_same_length_input_and_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    dep = FakeDeployment(vocab=128)
    texts = [item.copy() for item in TEXTS[:4]]
    run_composed(dep, texts, REF_COUNTS, n_types=8, seed=0, verbose=False)
    texts[0][0] = 12
    run_composed(dep, texts, REF_COUNTS, n_types=8, seed=0, verbose=False)
    run_composed(dep, texts, REF_COUNTS, n_types=8, seed=1, verbose=False)
    assert len(list(tmp_path.glob("**/ab_composed/**/result.json"))) == 3


def test_composed_refuses_a_scenario_without_clean_weights():
    dep = FakeDeployment(vocab=128)
    dep.weight_tensor = lambda *args: (_ for _ in ()).throw(AssertionError("read weights"))
    with pytest.raises(ValueError, match="requires public clean weights"):
        run_composed(dep, TEXTS[:1], REF_COUNTS, n_types=8, scenario="S-deploy", verbose=False)


def test_c10_sh_b_is_metadata_free_and_sh_p_is_protocol_aware(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    expected = {
        "SH-P": ("SH-P", True),
        "SH-B": ("SH-B", False),
        "SH-ALIGN": ("SH", False),
    }
    for mode, (knowledge, rotation_aware) in expected.items():
        payload = _payload(tmp_path, mode)
        assert payload["attacker_knowledge"] == knowledge, mode
        assert payload["rotation_aware"] is rotation_aware, mode


def test_c10_candidate_coverage_is_truth_coverage_not_presence(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    original = real_battery._rank_window_candidates
    monkeypatch.setattr(real_battery, "_rank_window_candidates", lambda *args, **kwargs: {
        symbol: candidates[:5] for symbol, candidates in original(*args, **kwargs).items()
    })
    small_ref = {token: 100 for token in range(5)}
    out = run_c10(FakeDeployment(), TEXTS, small_ref, n_keys=2, rotate_every=4,
                  seed=4, verbose=False)
    coverage = _payload(tmp_path, "SH-P")["metrics"]["candidate_coverage"]
    presence = _payload(tmp_path, "SH-P")["metrics"]["candidate_presence_coverage"]
    assert presence["value"] == pytest.approx(1.0)
    assert presence["numerator"] == presence["denominator"]
    assert coverage["value"] < 1.0
    assert coverage["value"] == pytest.approx(
        coverage["numerator"] / coverage["denominator"]
    )
    assert out["SH-P"]["candidate_presence_coverage"] == pytest.approx(1.0)


def test_c10_prior_uses_same_requests_and_paired_gain(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    payload = _payload(tmp_path, "SH-P")
    prior = payload["metrics"]["prior_baseline"]
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert prior["denominator"] == sum(row["count"] for row in rows)
    assert prior["numerator"] == sum(row["hits"] for row in prior["per_unit"])
    assert all(row["predicted"] == prior["predicted"] for row in prior["per_unit"])
    attack = payload["metrics"]["occurrence_level"]["top1"]["value"]
    assert prior["paired_gain"]["value"] == pytest.approx(attack - prior["value"])


def test_stream_cache_is_bound_to_the_deployment_key_schedule():
    cache: dict = {}
    dep_a = FakeDeployment(key_root=b"A" * 32)
    dep_b = FakeDeployment(key_root=b"B" * 32)
    first = real_battery.build_stream(dep_a, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    second = real_battery.build_stream(dep_b, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]


def test_stream_cache_is_bound_to_the_loaded_weights():
    cache: dict = {}
    dep_a = FakeDeployment(seed=11)
    dep_b = FakeDeployment(seed=11)
    dep_b.E_dep = dep_b.E_dep.copy()
    dep_b.E_dep[0] = dep_b.E_dep[0] + 1.0      # same paths/keys/perm, new weights
    first = real_battery.build_stream(dep_a, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    second = real_battery.build_stream(dep_b, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]
    assert getattr(dep_a, "_stream_state_digest", None) is not None


def test_text_digests_are_length_framed():
    from obfuscation_attacks.evaluation.real_battery import _stream_cache_key, _texts_digest

    assert _texts_digest(["a\x00b"]) != _texts_digest(["a", "b"])
    dep = FakeDeployment()
    key_one = _stream_cache_key(dep, ["a\x00b"], n_keys=2, rotate_every=4,
                                seed=1, profile="probe")
    key_two = _stream_cache_key(dep, ["a", "b"], n_keys=2, rotate_every=4,
                                seed=1, profile="probe")
    assert key_one != key_two


def test_c10_forwards_its_profile_to_the_stream_cache(monkeypatch, tmp_path):
    real_build = real_battery.build_stream
    seen = {}

    def recording(*args, **kwargs):
        seen["profile"] = kwargs.get("profile")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(real_battery, "build_stream", recording)
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False, profile="paper-v1", protocol_sha256="a" * 64)
    assert seen["profile"] == "paper-v1"


def test_stream_cache_is_bound_to_the_osnip_encoder(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "osnip").mkdir(parents=True)
    encoder = artifact / "osnip" / "encryptor.pt"
    encoder.write_bytes(b"encoder-A")
    cache: dict = {}
    dep_a = FakeDeployment(artifact=artifact)     # same path, E, perm, root
    first = real_battery.build_stream(dep_a, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    encoder.write_bytes(b"encoder-B")             # rebuilt artifact at the same path
    dep_b = FakeDeployment(artifact=artifact)
    second = real_battery.build_stream(dep_b, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]
    assert dep_a._stream_state_digest != dep_b._stream_state_digest


def test_stream_cache_is_bound_to_the_tokenizer(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "osnip").mkdir(parents=True)
    (artifact / "model_server").mkdir(parents=True)
    (artifact / "osnip" / "encryptor.pt").write_bytes(b"encoder")
    tokenizer = artifact / "model_server" / "tokenizer.json"
    tokenizer.write_text('{"model": {"vocab": {"a": 0}}}', encoding="utf-8")
    cache: dict = {}
    dep_a = FakeDeployment(artifact=artifact)
    first = real_battery.build_stream(dep_a, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    tokenizer.write_text('{"model": {"vocab": {"a": 1}}}', encoding="utf-8")
    dep_b = FakeDeployment(artifact=artifact)
    second = real_battery.build_stream(dep_b, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert second["stream_key"] != first["stream_key"]
    assert dep_a._stream_state_digest != dep_b._stream_state_digest


def test_stream_cache_is_bound_to_the_in_memory_encoder(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "osnip").mkdir(parents=True)
    encoder_file = artifact / "osnip" / "encryptor.pt"
    encoder_file.write_bytes(b"encoder-A")
    dep_a = FakeDeployment(
        artifact=artifact,
        encoder_state={"w": np.asarray([1.0], dtype=np.float32)},
    )
    # The on-disk encoder is replaced after dep_a was constructed; its lazy
    # digest must still reflect dep_a's in-memory parameters (what produced z).
    encoder_file.write_bytes(b"encoder-B")
    dep_b = FakeDeployment(
        artifact=artifact,
        encoder_state={"w": np.asarray([2.0], dtype=np.float32)},
    )
    cache: dict = {}
    first = real_battery.build_stream(dep_a, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    second = real_battery.build_stream(dep_b, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second is not first
    assert first["stream_key"] != second["stream_key"]
    assert dep_a._stream_state_digest != dep_b._stream_state_digest


def test_c10_run_identity_separates_deployments(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    dep_a = FakeDeployment(seed=11)
    dep_b = FakeDeployment(seed=11)
    dep_b.E_dep = dep_b.E_dep.copy()
    dep_b.E_dep[0] = dep_b.E_dep[0] + 1.0      # same texts/seed/profile, new weights
    for dep in (dep_a, dep_b):
        run_c10(dep, TEXTS, REF_COUNTS, n_keys=2, rotate_every=4, seed=4,
                verbose=False)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("**/SH-P/**/result.json")
    ]
    assert len(payloads) == 2
    assert payloads[0]["data_label_sha256"] != payloads[1]["data_label_sha256"]
    assert payloads[0]["run_id"] != payloads[1]["run_id"]


_FORBIDDEN_SECRET_KEYS = frozenset({
    "truth", "plaintext_ids", "permuted_ids", "perm", "inv_perm",
    "root_key", "secret", "encoder",
})


def _forbidden_key_paths(obj, path: str = "result") -> list[str]:
    gaps: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key)
            if name.lower() in _FORBIDDEN_SECRET_KEYS:
                gaps.append(f"{path}.{name}")
            gaps.extend(_forbidden_key_paths(value, f"{path}.{name}"))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            gaps.extend(_forbidden_key_paths(value, f"{path}[{index}]"))
    return gaps


class _RunnerPublic:
    arm = "ab"

    def table(self, name: str) -> np.ndarray:
        return np.eye(32, 4, dtype=np.float32)


class _RunnerTokeniser:
    @staticmethod
    def decode(ids) -> str:
        return "decoded"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}


class _RunnerFakeDep:
    """Minimal deployment for the per-prompt runner rows (C03/C07/C13)."""

    artifact = Path("fixture-artifact")

    def __init__(self):
        self.perm = np.arange(32)
        self.tokenizer = _RunnerTokeniser()

    def encode(self, text) -> np.ndarray:
        return np.asarray([1, 2, 3], dtype=np.int64)

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None, forward=True):
        ids = np.asarray([1, 2, 3], dtype=np.int64)
        return Wire(
            WirePublic(arm="ab", request_id=request_id, seq_len=3,
                       wire_vectors=np.ones((3, 4), dtype=np.float32),
                       group_id="g-0", epoch=epoch),
            WireTruth(plaintext_ids=ids.copy(), permuted_ids=ids.copy()),
        )

    def public_view(self, arm, scenario="S-public") -> _RunnerPublic:
        return _RunnerPublic()


def test_beamclean_counts_unpredicted_positions(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_generative, real_remaining

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(real_generative, "beamclean_recover",
                        lambda *args, **kwargs: {"ids": [1]})
    out = real_remaining.run_c07(_RunnerFakeDep(), ["sample"], max_len=1, verbose=False)
    row = out["rows"][0]
    assert row["stage1_top1"] == pytest.approx(1 / 3)
    assert row["stage1_scored_positions"] == 3
    assert row["stage1_predicted_positions"] == 1


def test_per_prompt_runner_rows_do_not_expose_raw_prompts(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.generation as generation
    import obfuscation_attacks.evaluation.real_generative as real_generative
    import obfuscation_attacks.evaluation.real_remaining as real_remaining

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    prompts = ["SECRET PROMPT ALPHA", "second secret beta"]
    monkeypatch.setattr(
        real_generative, "beamclean_recover",
        lambda public, z, **kwargs: {"ids": [1, 2, 3]},
    )
    monkeypatch.setattr(
        real_generative, "iterative_prior_refine",
        lambda public, ids, z, arm, rounds: {"ids": [1, 2, 3], "history": []},
    )
    monkeypatch.setattr(
        real_generative, "text_metrics",
        lambda public, got, truth: {"bleu4": 0.0},
    )
    def generate_fixture(_dep, prompt_ids, *, public_writer, **kwargs):
        steps = []
        for index in range(2):
            length = len(prompt_ids) + index
            logits = np.zeros((length, 32), dtype=np.float32)
            logits[-1, 5 + index] = 1
            wire = WirePublic(arm="ab", request_id=index, seq_len=length,
                              wire_vectors=np.ones((length, 4), dtype=np.float32), logits_perm=logits)
            step = generation.GenerationStep(index, wire, logits[-1], logits[-1], 5 + index, True)
            public_writer.add(step)
            steps.append(step)
        return generation.GenerationRecord("ab", list(prompt_ids), [5, 6], steps, True)

    monkeypatch.setattr(generation, "generate_record", generate_fixture)
    dep = _RunnerFakeDep()
    c07 = real_remaining.run_c07(dep, prompts, max_len=3, verbose=False)
    c03 = real_remaining.run_c03(dep, prompts, max_len=3, verbose=False)
    c13 = real_battery.run_c13_c14(dep, prompts=prompts, max_new_tokens=2,
                                   seed=0, verbose=False)
    for result in (c07, c03, c13):
        serialized = json.dumps(result, default=str)
        assert "SECRET PROMPT ALPHA" not in serialized
        assert "second secret beta" not in serialized
        assert _forbidden_key_paths(result) == []
    for rows in (c07["rows"], c03["rows"], c13["rows"]):
        for row in rows:
            assert "prompt" not in row
            assert row["prompt_index"] in (0, 1)
            assert len(row["prompt_sha256"]) == 64
    written = "".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.glob("**/result.json")
    )
    assert "SECRET PROMPT ALPHA" not in written
    assert "second secret beta" not in written


def test_c12_can_run_public_aggregation_without_exposing_repeat_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    labels = ["PRIVATE_REPEAT_LABEL"] * 2 + [f"other-{i}" for i in range(len(TEXTS) - 2)]
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  aggregation_threshold=0.8, aggregation_repeat_labels=labels, verbose=False)
    result = out["public_aggregation_probe"]
    assert result["status"] == "executed"
    assert result["score"]["aggregated"]["denominator"] == sum(map(len, TEXTS))
    assert "validation selection not verified" in result["threshold_provenance"]
    assert "PRIVATE_REPEAT_LABEL" not in json.dumps(out)
    payload = _payload(tmp_path, "ab")
    assert payload["metrics"]["public_aggregation_probe"]["status"] == "executed"
    assert "PRIVATE_REPEAT_LABEL" not in json.dumps(payload)


def test_c12_aggregation_identity_tracks_threshold_and_repeat_partition(monkeypatch, tmp_path):
    outputs = []
    for index, (threshold, labels) in enumerate((
        (0.7, list(range(len(TEXTS)))),
        (0.8, list(range(len(TEXTS)))),
        (0.7, [0, 0] + list(range(2, len(TEXTS)))),
    )):
        root = tmp_path / str(index)
        monkeypatch.setattr(real_battery, "OUT_ROOT", root)
        run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5, verbose=False,
                aggregation_threshold=threshold, aggregation_repeat_labels=labels)
        outputs.append(_payload(root, "ab")["data_label_sha256"])
    assert len(set(outputs)) == 3
    with pytest.raises(ValueError, match="probe"):
        run_c12(FakeDeployment(), TEXTS, profile="paper-v1", aggregation_threshold=0.7,
                aggregation_repeat_labels=list(range(len(TEXTS))), verbose=False)


def test_c12_return_and_payload_do_not_leak_truth_or_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False)
    assert _forbidden_key_paths(out) == []
    serialized = json.loads(json.dumps(out, default=str))
    assert _forbidden_key_paths(serialized) == []
    payload = _payload(tmp_path, "ab")
    assert _forbidden_key_paths(payload) == []
    # The per-stream aggregate stays available without the raw stream rows.
    assert out["stream"]["per_stream"]


def test_c10_run_identity_commits_to_reference_frequencies(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    ref_a = dict(REF_COUNTS)
    ref_b = dict(REF_COUNTS)
    ref_b[0], ref_b[1] = ref_b[1], ref_b[0]    # same total tokens, different table
    run_c10(FakeDeployment(), TEXTS, ref_a, n_keys=2, rotate_every=4, seed=4,
            verbose=False)
    run_c10(FakeDeployment(), TEXTS, ref_b, n_keys=2, rotate_every=4, seed=4,
            verbose=False)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("**/SH-P/**/result.json")
    ]
    assert len(payloads) == 2
    assert payloads[0]["data_label_sha256"] != payloads[1]["data_label_sha256"]
    assert payloads[0]["run_id"] != payloads[1]["run_id"]


def test_c10_run_identity_commits_to_the_client_secret(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "client").mkdir(parents=True)
    secret = artifact / "client" / "client_secret.pt"
    secret.write_bytes(b"client-secret-A")
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "out")
    dep_a = FakeDeployment(artifact=artifact)
    run_c10(dep_a, TEXTS, REF_COUNTS, n_keys=2, rotate_every=4, seed=4,
            verbose=False)
    secret.write_bytes(b"client-secret-B")     # same path, new secret
    dep_b = FakeDeployment(artifact=artifact)
    run_c10(dep_b, TEXTS, REF_COUNTS, n_keys=2, rotate_every=4, seed=4,
            verbose=False)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "out").glob("**/SH-P/**/result.json")
    ]
    assert len(payloads) == 2
    assert payloads[0]["data_label_sha256"] != payloads[1]["data_label_sha256"]
    assert payloads[0]["run_id"] != payloads[1]["run_id"]


def test_state_digest_hashes_all_small_tokenizer_files(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "model_server").mkdir(parents=True)
    vocab = artifact / "model_server" / "vocab.json"
    merges = artifact / "model_server" / "merges.txt"
    vocab.write_text("vocab-A", encoding="utf-8")
    merges.write_text("merges-A", encoding="utf-8")
    cache: dict = {}
    dep_one = FakeDeployment(artifact=artifact)
    first = real_battery.build_stream(dep_one, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    vocab.write_text("vocab-B", encoding="utf-8")
    dep_two = FakeDeployment(artifact=artifact)
    second = real_battery.build_stream(dep_two, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second["stream_key"] != first["stream_key"]
    assert dep_one._stream_state_digest != dep_two._stream_state_digest
    merges.unlink()                            # removal must change the digest
    dep_three = FakeDeployment(artifact=artifact)
    third = real_battery.build_stream(dep_three, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    assert third["stream_key"] != second["stream_key"]
    assert dep_three._stream_state_digest != dep_two._stream_state_digest


def test_state_digest_always_hashes_the_large_tokenizer_json(tmp_path):
    # Llama-3.2-1B's tokenizer.json is ~17 MB, above the small-file cap.
    artifact = tmp_path / "artifact"
    server = artifact / "model_server"
    server.mkdir(parents=True)
    tokenizer = server / "tokenizer.json"
    big = 10 * 1024 * 1024 + 1
    tokenizer.write_bytes(b"A" * big)
    cache: dict = {}
    dep_one = FakeDeployment(artifact=artifact)
    first = real_battery.build_stream(dep_one, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    tokenizer.write_bytes(b"B" * big)
    dep_two = FakeDeployment(artifact=artifact)
    second = real_battery.build_stream(dep_two, TEXTS, n_keys=2, rotate_every=4,
                                       seed=1, cache=cache)
    assert second["stream_key"] != first["stream_key"]
    assert dep_one._stream_state_digest != dep_two._stream_state_digest
    tokenizer.unlink()
    dep_three = FakeDeployment(artifact=artifact)
    third = real_battery.build_stream(dep_three, TEXTS, n_keys=2, rotate_every=4,
                                      seed=1, cache=cache)
    assert third["stream_key"] != second["stream_key"]
    assert dep_three._stream_state_digest != dep_two._stream_state_digest


def test_c12_forwards_its_profile_to_the_stream_cache(monkeypatch, tmp_path):
    real_build = real_battery.build_stream
    seen = {}

    def recording(*args, **kwargs):
        seen["profile"] = kwargs.get("profile")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(real_battery, "build_stream", recording)
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
                  verbose=False, profile="paper-v1")
    assert seen["profile"] == "paper-v1"
    assert out["stream"]["per_stream"][0]["profile"] == "paper-v1"


def test_c12_paper_bound_payload_carries_the_binding(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.paper_profile import (
        protocol_sha256,
        require_profile,
    )
    from obfuscation_attacks.evaluation.protocol_audit import audit_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
            verbose=False, profile=profile.name, protocol_sha256=digest)
    payload = _payload(tmp_path, "ab")
    assert payload["profile"] == "paper-v1"
    assert payload["protocol_sha256"] == digest
    audit = audit_payload(payload, profile=profile, protocol_sha256=digest)
    assert not any(
        gap.startswith("profile:") or gap.startswith("protocol:")
        for gap in audit["gaps"]
    )


def test_c12_data_identity_commits_to_corpus_content(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "a")
    run_c12(FakeDeployment(), TEXTS, n_keys=2, rotate_every=4, seed=5,
            verbose=False)
    other = [text.copy() for text in TEXTS]
    other[0][0] = 30
    assert [len(text) for text in other] == [len(text) for text in TEXTS]
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "b")
    run_c12(FakeDeployment(), other, n_keys=2, rotate_every=4, seed=5,
            verbose=False)
    first = _payload(tmp_path / "a", "ab")["data_label_sha256"]
    second = _payload(tmp_path / "b", "ab")["data_label_sha256"]
    assert first != second


def test_c12_data_identity_commits_to_the_stream_partition(monkeypatch, tmp_path):
    # Same flattened tokens, same counts, different partition [1,9] vs [2,8].
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "a")
    run_c12(FakeDeployment(), text_sets=[TEXTS[:1], TEXTS[1:]],
            n_keys=2, rotate_every=4, seed=5, verbose=False)
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "b")
    run_c12(FakeDeployment(), text_sets=[TEXTS[:2], TEXTS[2:]],
            n_keys=2, rotate_every=4, seed=5, verbose=False)
    first = _payload(tmp_path / "a", "ab")["data_label_sha256"]
    second = _payload(tmp_path / "b", "ab")["data_label_sha256"]
    assert first != second


def test_stream_cache_is_bound_to_the_protocol_hash():
    cache: dict = {}
    dep = FakeDeployment()
    first = real_battery.build_stream(
        dep, TEXTS, n_keys=2, rotate_every=4, seed=1, cache=cache,
        profile="paper-v1", protocol_sha256="a" * 64,
    )
    second = real_battery.build_stream(
        dep, TEXTS, n_keys=2, rotate_every=4, seed=1, cache=cache,
        profile="paper-v1", protocol_sha256="b" * 64,
    )
    assert second is not first
    assert second["stream_key"] != first["stream_key"]


def test_macro_bootstrap_reports_no_interval_below_two_clusters():
    from obfuscation_attacks.metrics.intervals import macro_bootstrap

    degenerate = macro_bootstrap([1.0, 0.0], ["a", "a"], n_boot=2000, seed=0)
    assert degenerate["macro"] == pytest.approx(0.5)
    assert degenerate["n_boot"] == 0
    assert degenerate["lo"] is None and degenerate["hi"] is None
    empty = macro_bootstrap([], [], n_boot=2000, seed=0)
    assert empty["macro"] is None and empty["lo"] is None and empty["hi"] is None
    assert empty["n_boot"] == 0


def test_session_pair_weights_reproduce_the_point_on_the_all_ones_draw():
    from obfuscation_attacks.evaluation.real_battery import _session_pair_weights

    left = np.array([0, 0])
    right = np.array([0, 1])                   # one same-session, one cross-session pair
    counts = np.ones(2)
    weights = _session_pair_weights(counts, left, right)
    assert np.array_equal(weights, np.ones(2))
    hits = np.array([1.0, 0.0])
    assert (weights * hits).sum() / weights.sum() == pytest.approx(0.5)


def test_session_pair_weights_use_multiplicity_for_within_session_pairs():
    from obfuscation_attacks.evaluation.real_battery import _session_pair_weights

    left = np.array([0, 0])
    right = np.array([0, 1])                   # same-session (A,A), cross-session (A,B)
    counts = np.array([2.0, 1.0])              # m_A = 2, m_B = 1
    weights = _session_pair_weights(counts, left, right)
    assert np.array_equal(weights, np.array([2.0, 2.0]))   # not [4, 2]
    hits = np.array([1.0, 0.0])
    assert (weights * hits).sum() / weights.sum() == pytest.approx(0.5)


def test_session_pair_interval_contains_the_reported_point():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    out = _cluster_rate_ci(
        [True, False], [("S1", "S1"), ("S1", "S2")], n_boot=2000, seed=0
    )
    assert out["value"] == pytest.approx(0.5)
    assert out["n_clusters"] == 2
    assert out["lo"] <= out["value"] <= out["hi"]


def test_session_pair_bootstrap_reports_only_finite_replicates():
    from obfuscation_attacks.evaluation.real_battery import _cluster_rate_ci

    # Three sessions and two cross-session pairs: some multinomial draws give an
    # endpoint multiplicity 0, so the weighted ratio is undefined for that draw.
    out = _cluster_rate_ci(
        [True, False], [("s1", "s2"), ("s1", "s3")], n_boot=2000, seed=0
    )
    assert out["n_boot"] == 2000          # finite replicates actually used
    assert out["bootstrap_attempts"] >= out["n_boot"]
    assert out["n_clusters"] == 3
    assert out["lo"] <= out["value"] <= out["hi"]


def test_c10_data_identity_commits_to_corpus_content(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "a")
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    other = [text.copy() for text in TEXTS]
    other[0][0] = 30
    assert [len(text) for text in other] == [len(text) for text in TEXTS]
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "b")
    run_c10(FakeDeployment(), other, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    first = _payload(tmp_path / "a", "SH-P")["data_label_sha256"]
    second = _payload(tmp_path / "b", "SH-P")["data_label_sha256"]
    assert first != second


def test_c10_supports_independent_streams(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    sets = [TEXTS[:3], TEXTS[3:6], TEXTS[6:]]
    out = run_c10(FakeDeployment(), [], REF_COUNTS, text_sets=sets,
                  n_keys=2, rotate_every=4, seed=4, verbose=False)
    assert out["stream"]["n_streams"] == 3
    assert out["stream"]["n_requests"] == len(TEXTS)
    assert out["stream"]["n_positions"] == 5 * len(TEXTS)
    assert out["SH-ALIGN"]["n_positions"] == 5 * len(TEXTS)
    payload = _payload(tmp_path, "SH-P")
    assert payload["n_streams"] == 3
    assert len(payload["diagnostics"]["per_stream"]) == 3
    expected_types = [
        len({int(token) for text in row for token in np.asarray(text).tolist()})
        for row in sets
    ]
    for entry, expected in zip(payload["diagnostics"]["per_stream"],
                               expected_types):
        assert entry["n_types"] == expected
        assert entry["n_symbols"] > 0
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert len(rows) == payload["n_prompts"] == len(TEXTS)
    assert [row["request"] for row in rows] == list(range(len(TEXTS)))
    top1 = payload["metrics"]["occurrence_level"]["top1"]
    assert sum(row["count"] for row in rows) == top1["denominator"]
    assert sum(row["hits"] for row in rows) == top1["numerator"]


def test_c10_saturation_is_passed_through_or_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "none")
    probe = run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
                    seed=4, verbose=False)
    assert "saturation" not in _payload(tmp_path / "none", "SH-P")["diagnostics"]
    top1 = probe["SH-P"]["top1"]
    probe_data_sha256 = _payload(tmp_path / "none", "SH-P")["data_label_sha256"]
    measured = {
        "points": [{"budget": 1000, "top1": top1}, {"budget": 2000, "top1": top1}],
        "gain_pp": 0.0,
        "data_sha256": probe_data_sha256,
    }
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "with")
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False, saturation=measured,
            saturation_variant="SH-P")
    payload = _payload(tmp_path / "with", "SH-P")
    assert payload["diagnostics"]["saturation"] == measured
    assert payload["diagnostics"]["saturation_variant"] == "SH-P"


def test_c10_candidate_list_distinctness_is_verified(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "normal")
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    payload = _payload(tmp_path / "normal", "SH-P")
    diagnostics = payload["diagnostics"]
    assert diagnostics["candidate_lists_unique"] is True
    assert diagnostics["candidate_lists_with_duplicates"] == 0
    assert diagnostics["candidate_lengths_min"] == payload["top100_distinct"]

    original = real_battery._rank_window_candidates

    def with_duplicates(group_symbols, ref_counts, **kwargs):
        table = original(group_symbols, ref_counts, **kwargs)
        return {
            key: (list(values) + [values[0]] if values else [])
            for key, values in table.items()
        }

    monkeypatch.setattr(real_battery, "_rank_window_candidates", with_duplicates)
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path / "duplicated")
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False)
    flagged = _payload(tmp_path / "duplicated", "SH-B")
    assert flagged["diagnostics"]["candidate_lists_unique"] is False
    assert flagged["diagnostics"]["candidate_lists_with_duplicates"] > 0


def test_micro_bootstrap_reports_no_interval_for_one_cluster():
    from obfuscation_attacks.metrics.intervals import micro_cluster_bootstrap

    degenerate = micro_cluster_bootstrap([1, 0], [1, 1], [0, 0],
                                         n_boot=2000, seed=0)
    assert degenerate["n_boot"] == 0
    assert degenerate["lo"] is None and degenerate["hi"] is None
    assert degenerate["micro"] == pytest.approx(0.5)

    normal = micro_cluster_bootstrap([1, 0], [1, 1], [0, 1],
                                     n_boot=2000, seed=0)
    assert normal["n_boot"] == 2000
    assert normal["lo"] is not None and normal["hi"] is not None


def test_ratio_ci_propagates_the_degenerate_interval():
    degenerate = ratio_ci([1, 0], [1, 9], clusters=[0, 0], n_boot=2000, seed=0)
    assert degenerate["micro"] == pytest.approx(0.1)
    assert degenerate["n_boot"] == 0
    assert degenerate["lo"] is None and degenerate["hi"] is None


# ---------------------------------------------------------------------------
# S08 caller side -- payloads record the layers that actually participated
# ---------------------------------------------------------------------------
def test_s08_run_c08_records_the_attacks_actual_layers(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_weight_attacks as weight_attacks

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def fake_gate(public, types, **kwargs):
        return {
            "candidates": {int(token): [0, 1] for token in types},
            "n_candidates": int(public.vocab_size),
            "layers": [2, 4],
            "missing_weight_layers": [0],
            "attack_id": "gate_ia",
        }

    monkeypatch.setattr(weight_attacks, "gate_public_candidates", fake_gate)
    out = real_battery.run_c08(FakeDeployment(), n_types=5, seed=0, verbose=False)
    assert out["layers"] == [2, 4]
    assert out["missing_weight_layers"] == [0]
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["diagnostics"]["layers"] == [2, 4]
    assert payload["diagnostics"]["missing_weight_layers"] == [0]


def test_s08_run_c08_attn_records_the_attacks_actual_layers(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_weight_attacks as weight_attacks

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)

    def fake_attn(public, types, layers=None, **kwargs):
        return {
            "candidates": {int(token): [0, 1] for token in types},
            "invariant_dims": 7,
            "layers": [1, 5],
            "missing_weight_layers": [10],
            "pairing": "half_split",
        }

    monkeypatch.setattr(weight_attacks, "attn_public_candidates", fake_attn)
    out = real_battery.run_c08_attn(
        FakeDeployment(), n_types=5, layers=(1, 5), seed=0, verbose=False,
    )
    assert out["layers"] == [1, 5]
    assert out["missing_weight_layers"] == [10]
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["metrics"]["layers"] == [1, 5]
    assert payload["metrics"]["missing_weight_layers"] == [10]
    assert payload["diagnostics"]["layers"] == [1, 5]


def test_s08_attn_payload_records_skipped_gram_blocks(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_weight_attacks as weight_attacks

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    skipped = [{
        "layer": 5, "group": "qk", "head": 0, "block": [0, 1],
        "condition": float("inf"), "code": "singular_gram",
        "reason": "non-finite Gram condition (singular block)",
    }]
    missing = [{"layer": 10, "code": "zero_columns",
                "reason": "all Gram blocks singular; no invariant column"}]

    def fake_attn(public, types, layers=None, **kwargs):
        return {
            "candidates": {int(token): [0, 1] for token in types},
            "invariant_dims": 7,
            "layers": [1, 5],
            "missing_weight_layers": missing,
            "skipped_gram_blocks": skipped,
            "pairing": "half_split",
            "block_size": 2,
            "n_chunks": 1,
        }

    monkeypatch.setattr(weight_attacks, "attn_public_candidates", fake_attn)
    out = real_battery.run_c08_attn(FakeDeployment(n_layers=16), n_types=5,
                                    seed=0, verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    diagnostics = payload["diagnostics"]
    assert diagnostics["skipped_gram_blocks"] == skipped
    by_layer = diagnostics["skipped_gram_blocks_by_layer"]
    assert by_layer["layer5"][0]["code"] == "singular_gram"
    assert diagnostics["layers"] == [1, 5]          # partially singular layer stays used
    assert [record["layer"] for record in payload["metrics"]["missing_weight_layers"]] == [10]
    assert out["skipped_gram_blocks"] == skipped
    assert out["layers"] == [1, 5]


def test_s08_run_c08_attn_default_enumerates_all_layers(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_weight_attacks as weight_attacks

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    seen = {}

    def fake_attn(public, types, layers=None, **kwargs):
        seen["layers"] = layers
        used = (list(range(int(public.n_layers))) if layers is None
                else [int(layer) for layer in layers])
        return {
            "candidates": {int(token): [0, 1] for token in types},
            "invariant_dims": 7,
            "layers": used,
            "missing_weight_layers": [],
            "pairing": "half_split",
        }

    monkeypatch.setattr(weight_attacks, "attn_public_candidates", fake_attn)
    out = real_battery.run_c08_attn(
        FakeDeployment(n_layers=16), n_types=5, seed=0, verbose=False,
    )
    assert seen["layers"] is None               # default enumerates every layer
    assert out["layers"] == list(range(16))
    payload = _payload(tmp_path, "aloepri_weights")
    assert payload["metrics"]["layers"] == list(range(16))
    assert payload["diagnostics"]["layers"] == list(range(16))


def test_c06w_weight_layer_evidence_covers_every_layer(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.protocol_audit import (
        _layer_gaps,
        _weight_layer_gaps,
    )

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c06w(FakeDeployment(n_layers=16), n_types=16, diag_types=16, seed=0,
             verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    used = {int(layer) for layer in payload["layers"]}
    assert used == set(range(16))
    assert payload["missing_weight_layers"] == []
    assert _weight_layer_gaps(payload, payload["diagnostics"], used, 16) == []
    assert _layer_gaps(payload, payload["diagnostics"], attack="vma_w") == []


def test_c06w_records_a_dropped_layer_as_a_coded_missing_record(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.protocol_audit import _weight_layer_gaps

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    run_c06w(FakeDeployment(n_layers=16, drop_layers=(7,)), n_types=16,
             diag_types=16, seed=0, verbose=False)
    payload = _payload(tmp_path, "aloepri_weights")
    used = {int(layer) for layer in payload["layers"]}
    assert 7 not in used and len(used) == 15
    records = payload["missing_weight_layers"]
    assert [record["layer"] for record in records] == [7]
    assert records[0]["code"] == "weight_missing"
    assert str(records[0]["reason"]).strip()
    assert _weight_layer_gaps(payload, payload["diagnostics"], used, 16) == []


# ---------------------------------------------------------------------------
# CI / threshold and paper-binding guards
# ---------------------------------------------------------------------------
def test_nonfinite_validation_cosines_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _threshold_from_validation([0.5, float("nan")], 1e-2)
    with pytest.raises(ValueError, match="non-finite"):
        _threshold_from_validation([0.5, float("inf")], 1e-2)


def test_infinite_norm_vector_makes_the_cosine_undefined():
    req = {"vectors": np.asarray([[1.0, 2.0], [np.inf, 0.0]], dtype=np.float32)}
    value = real_battery._cosine(req, 1, req, 0)
    assert np.isnan(value)
    with pytest.raises(ValueError, match="non-finite"):
        _threshold_from_validation([0.9, value], 1e-2)


def test_c10_paper_bound_payload_reaches_the_tfma_floors(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation.paper_profile import (
        protocol_sha256,
        require_profile,
    )
    from obfuscation_attacks.evaluation.protocol_audit import audit_payload

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    profile = require_profile("paper-v1")
    digest = protocol_sha256()
    run_c10(FakeDeployment(), TEXTS, REF_COUNTS, n_keys=2, rotate_every=4,
            seed=4, verbose=False, profile=profile.name, protocol_sha256=digest)
    payload = _payload(tmp_path, "SH-P")
    assert payload["profile"] == "paper-v1"
    assert payload["protocol_sha256"] == digest
    assert payload["diagnostics"]["candidate_lists_unique"] is True
    assert payload["diagnostics"]["candidate_lengths_min"] == payload["top100_distinct"]
    # The synthetic control fails at this corpus size; pass it only in memory so
    # the audit proceeds past the control gate and actually reaches the floors.
    payload["diagnostics"]["positive_control_passed"] = True
    payload["diagnostics"]["positive_control"]["top1"] = 0.95
    audit = audit_payload(payload, profile=profile, protocol_sha256=digest)
    assert not any(
        gap.startswith("profile:") or gap.startswith("protocol:")
        for gap in audit["gaps"]
    )
    assert audit["verdict"] == "below_budget"      # below the paper-scale floors
    assert any(
        "reference_tokens" in gap or "top100" in gap or "n_streams" in gap
        for gap in audit["gaps"]
    )
    assert not any("candidate_lists_unique" in gap for gap in audit["gaps"])


# ---------------------------------------------------------------------------
# S11 -- bounded, deterministic reservoir sampling during enumeration
# ---------------------------------------------------------------------------
def _brute_force_pairs(requests, truth, max_pairs):
    positives, negatives = [], []
    for a in range(len(requests)):
        ta = np.asarray(truth[a]["plaintext_ids"], dtype=np.int64)
        for b in range(a + 1, len(requests)):
            tb = np.asarray(truth[b]["plaintext_ids"], dtype=np.int64)
            common = sorted(set(int(x) for x in ta) & set(int(x) for x in tb))
            for token in common:
                for pa in np.flatnonzero(ta == token):
                    for pb in np.flatnonzero(tb == token):
                        positives.append((a, int(pa), b, int(pb), token))
                        different = np.flatnonzero(tb != token)
                        if different.size:
                            pb2 = int(different[np.argmin(np.abs(different - pb))])
                            negatives.append((a, int(pa), b, pb2, token))
    key = lambda row: (row[0], row[2], row[4], row[1], row[3])
    return sorted(positives, key=key)[:max_pairs], sorted(negatives, key=key)[:max_pairs]


def test_s11_under_capacity_is_the_exact_full_enumeration():
    truth = [{"plaintext_ids": text} for text in TEXTS]
    requests = [{} for _ in TEXTS]
    out = _matched_linkage_pairs(requests, truth, seed=3, max_pairs=10 ** 6)
    expected_pos, expected_neg = _brute_force_pairs(requests, truth, 10 ** 6)
    assert out["positives"] == expected_pos
    assert out["negatives"] == expected_neg
    assert out["sampling"]["truncated"] is False
    assert out["sampling"]["total_seen_positives"] == len(expected_pos)


def test_s11_over_capacity_is_bounded_deterministic_and_fast():
    rng = np.random.default_rng(0)
    n_requests = 300
    truth = [
        {"plaintext_ids": rng.integers(0, 20, size=20)}
        for _ in range(n_requests)
    ]
    requests = [{} for _ in range(n_requests)]
    started = time.time()
    first = _matched_linkage_pairs(requests, truth, seed=7, max_pairs=100)
    elapsed = time.time() - started
    second = _matched_linkage_pairs(requests, truth, seed=7, max_pairs=100)
    assert elapsed < 20.0
    assert len(first["positives"]) <= 100
    assert len(first["negatives"]) <= 100
    assert first["positives"] == second["positives"]
    assert first["negatives"] == second["negatives"]
    assert first["sampling"]["total_seen_positives"] > 100
    assert first["sampling"]["truncated"] is True
    assert first["sampling"]["capacity"] == 100
    assert first["sampling"]["seed"] == 7
    assert first["sampling"]["positive_inclusion_probability"] == pytest.approx(
        100 / first["sampling"]["total_seen_positives"]
    )


def test_linkage_pair_sampling_keeps_repeated_positions():
    truth = [{"plaintext_ids": [1, 1, 2]}, {"plaintext_ids": [1, 1, 3]}]
    out = _matched_linkage_pairs([{}, {}], truth, max_pairs=100)
    assert out["sampling"]["total_seen_positives"] == 4
    assert {(row[1], row[3]) for row in out["positives"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert out["sampling"]["total_seen_negatives"] == 4


def test_linkage_direct_sampling_counts_large_population_without_enumerating_it():
    count = 5000
    truth = [{"plaintext_ids": [1, 2]} for _ in range(count)]
    out = _matched_linkage_pairs([{}] * count, truth, seed=5, max_pairs=25)
    population = count * (count - 1)
    assert out["sampling"]["total_seen_positives"] == population
    assert out["sampling"]["total_seen_negatives"] == population
    assert len(out["positives"]) == len(out["negatives"]) == 25
    assert out["sampling"]["positive_inclusion_probability"] == pytest.approx(25 / population)
