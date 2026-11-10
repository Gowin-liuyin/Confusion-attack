"""End-to-end call-chain tests for the real C05/C06H entries (final review findings)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.evaluation import real_battery
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth
from obfuscation_attacks.schemes.public_view import PublicDeployment

pytestmark = pytest.mark.unit

V, D = 8, 4
E = np.eye(V, D, dtype=np.float32)
W = np.arange(D * D, dtype=np.float32).reshape(D, D) / 10.0 + np.eye(D, dtype=np.float32)


def _forward(inputs_embeds):
    x = inputs_embeds if isinstance(inputs_embeds, torch.Tensor) else torch.as_tensor(
        np.asarray(inputs_embeds), dtype=torch.float32
    )
    if x.ndim == 2:
        x = x[None]
    hidden = x @ torch.tensor(W)
    return SimpleNamespace(hidden_states=[x, hidden])


class FakeTokenizer:
    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class FakeDeployment:
    artifact = "fixture"
    vocab_size = V
    n_layers = 2
    layers = (1,)
    secret_metadata = {}

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def encode(self, text):
        return np.asarray(text, dtype=np.int64)

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None):
        ids = np.asarray(text, dtype=np.int64)
        vectors = E[ids]
        hidden = _forward(vectors).hidden_states[1].numpy()
        public = WirePublic(
            arm=arm, request_id=request_id, seq_len=len(ids),
            wire_vectors=vectors if arm == "ab" else None,
            wire_token_ids=ids.copy() if arm in {"plaintext", "aloepri"} else None,
            hidden_states=hidden, layers=(1,), logits_perm=np.zeros((len(ids), V), np.float32),
            group_id="g-test", epoch=epoch, session_id=session_id,
        )
        truth = WireTruth(
            plaintext_ids=ids.copy(),
            permuted_ids=None if arm == "plaintext" else ids.copy(),
        )
        return Wire(public, truth)

    def public_view(self, arm, scenario="S-public"):
        return PublicDeployment(
            arm=arm, scenario=scenario, artifact="fixture", vocab_size=V,
            hidden_size=D, n_layers=2, layers=(1,), max_len=64, tokenizer_id="fixture",
            E_server=E, E_clean=E, forward_fn=_forward, clean_forward_fn=_forward,
            encode_fn=self.encode, decode_fn=self.tokenizer.decode,
            allow_clean_weights=True,
        )


def test_run_c06h_passes_public_wires_and_scores_stage1(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c06h(
        FakeDeployment(), [np.array([1, 2, 3])], layers=(1,), max_positions=3,
        full_vocab=True, verbose=False,
    )
    scored = out["scored"]["layer1"]
    assert scored["total_top1"]["denominator"] == 3
    assert scored["total_top1"]["numerator"] == 3


def test_run_c05_calls_the_original_er_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3])], {token: 10 for token in range(V)},
        prompts=1, steps=[2], isa_layers=(1,), restarts=1, verbose=False,
    )
    assert out["ab"]["method"] == "ER"
    assert out["ab"]["optimizer"] == "AdamW"
    assert out["ab"]["restarts"] == 1
    assert out["ab"]["optimizer_steps_total"] == 2
    assert out["ab"]["layers"]["layer1"]["best"]["recovered_token_ids"]
    payload = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["metrics"]["budget_curve_id_space"] == "permuted_deployment_token_ids"
    assert payload["metrics"]["end_to_end_plaintext_recovery"] is None


def test_run_c05_aloepri_state_uses_real_entry_and_labels_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2, 3])], {token: 10 for token in range(V)},
        prompts=1, steps=[1, 2], isa_layers=(1,), restarts=1, keys_per_prompt=2,
        method="aloepri_state", verbose=False,
    )
    assert out["ab"]["method"] == "AloePri hidden-state adaptation"
    assert out["ab"]["optimizer_steps_total"] == 4
    payload = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["attack_id"] == "isa_aloepri_state"
    assert payload["diagnostics"]["attack_variant"] == "isa_aloepri_hidden_state_adaptation"
    assert payload["keys_per_prompt"] == 2
    with pytest.raises(ValueError, match="unregistered probe"):
        real_battery.run_c05(FakeDeployment(), [], {}, method="aloepri_state", profile="paper-v1")


def test_run_c06h_scores_ragged_multi_prompt_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c06h(
        FakeDeployment(), [np.array([1, 2, 3]), np.array([2, 1])],
        layers=(1,), max_positions=3, full_vocab=True, verbose=False,
    )
    scored = out["scored"]["layer1"]
    assert scored["total_top1"]["denominator"] == 5
    assert scored["total_top1"]["numerator"] == 5


def test_run_c06h_records_keys_per_prompt_not_prompt_count(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    real_battery.run_c06h(
        FakeDeployment(), [np.array([1, 2, 3]), np.array([2, 1])],
        layers=(1,), max_positions=3, full_vocab=True, keys_per_prompt=2,
        verbose=False,
    )
    payload = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["n_prompts"] == 2
    assert payload["n_keys"] == 2
    assert payload["keys_per_prompt"] == 2
    assert payload["n_observations"] == 4
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert [row["prompt"] for row in rows] == [0, 0, 1, 1]
    assert [row["key"] for row in rows] == [0, 1, 0, 1]


def test_select_isa_prompts_refuses_missing_bands():
    with pytest.raises(ValueError, match="length strata"):
        real_battery.select_isa_prompts(
            FakeDeployment(), [np.arange(40)],
            {"32-64": 50, "65-128": 50, "129-256": 50, "257-512": 50},
        )


def test_run_c05_repeats_each_prompt_under_fresh_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    real_battery.run_c05(
        FakeDeployment(), [np.array([1, 2]), np.array([2, 1])],
        {token: 10 for token in range(V)},
        prompts=2, steps=[2], isa_layers=(1,), restarts=1, keys_per_prompt=2,
        verbose=False,
    )
    payload = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["n_prompts"] == 2
    assert payload["n_keys"] == 2
    assert payload["keys_per_prompt"] == 2
    assert payload["n_observations"] == 4
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert [row["prompt"] for row in rows] == [0, 0, 1, 1]
    assert [row["key"] for row in rows] == [0, 1, 0, 1]


def test_run_c06w_refuses_without_public_clean_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c06w(
        FakeDeployment(), clean_model_public=False, verbose=False
    )
    assert out["status"] == "not_run"
    assert not list(tmp_path.rglob("result.json"))


def test_verify_fidelity_rejects_unknown_formula_refs_and_missing_declarations(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "verify_fidelity.py"
    spec = importlib.util.spec_from_file_location("verify_fidelity_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FORMULA_LEDGER.md").write_text(
        "# Formula Ledger\n\n"
        "| ID | Formula | Paper | Section | Variant | Code | Test |\n"
        "|----|---------|-------|---------|---------|------|------|\n"
        "| KNOWN-1 | x | p | s | original | `src/x.py` | `tests/x.py` |\n",
        encoding="utf-8",
    )
    for sub in ("original", "aloepri", "osnip"):
        (tmp_path / "configs" / sub).mkdir(parents=True)
    (tmp_path / "configs" / "original" / "bad.yaml").write_text(
        yaml.safe_dump({
            "mode": "full",
            "fidelity": {"approximations": {"truncation": False}},
            "formula_refs": ["MISSING-1"],
        }),
        encoding="utf-8",
    )
    module.FAILURES.clear()
    module.check_configs()
    assert any("formula_refs" in failure for failure in module.FAILURES)
    (tmp_path / "configs" / "original" / "bad.yaml").write_text(
        yaml.safe_dump({"mode": "full", "formula_refs": ["KNOWN-1"]}),
        encoding="utf-8",
    )
    module.FAILURES.clear()
    module.check_configs()
    assert any("must declare fidelity.approximations" in failure
               for failure in module.FAILURES)


def test_gate_summary_merge_does_not_drop_other_classes():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "privacy_family_gates.py"
    spec = importlib.util.spec_from_file_location("privacy_family_gates_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    existing = {
        "n_arms": 2,
        "n_failures": 0,
        "failures": [],
        "rows": ["C10  tfma.independent         x  ok", "C12  linkage.exact            y  ok"],
    }
    merged = module._merged_summary(
        existing, "C12", ["C12  linkage.exact            y  ok (rerun)"], [], 1
    )
    assert any("C10" in row for row in merged["rows"])
    assert sum(1 for row in merged["rows"] if row.startswith("C12")) == 1
    assert merged["n_arms"] == len(merged["rows"])


def test_sda_fidelity_checker_rejects_lookup_variant_and_unconsumed_config():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "verify_fidelity.py"
    spec = importlib.util.spec_from_file_location("verify_sda_fidelity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_sda_config_scale()
    failures = "\n".join(module.FAILURES)
    assert "num_tune/num_test" in failures
    assert "lookup embeddings" in failures
    assert "12/12" in failures
    assert "feed-forward dimension=96" in failures


def test_sda_fidelity_checker_builds_configured_character_encoder(tmp_path, monkeypatch):
    import importlib.util
    import yaml
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "verify_fidelity.py"
    spec = importlib.util.spec_from_file_location("verify_sda_char_fidelity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "configs" / "original"
    original.mkdir(parents=True)
    (original / "sda2021.yaml").write_text(yaml.safe_dump({"attack": {"params": {
        "beam_width": 100, "num_train": 2000000, "num_layers": 6, "d_model": 8, "nhead": 2}}}), encoding="utf-8")
    (original / "sda2023.yaml").write_text(yaml.safe_dump({"attack": {"params": {
        "beam_width": 200, "num_train": 1200000, "num_layers": 12, "d_model": 12, "nhead": 12,
        "dim_ff": 16, "representation": "decimal_char_cnn"}}}), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.check_sda_config_scale()
    failures = "\n".join(module.FAILURES)
    assert "lookup embeddings" not in failures and "12/12" not in failures
    assert "unverified engineering variant" in failures
    assert "feed-forward dimension=16" in failures
