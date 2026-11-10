"""Acceptance tests for the SDA real-stream bridge (W4/W6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obfuscation_attacks.attacks.sda.sda2021_seq2seq import SDA2021
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import SDA2023
from obfuscation_attacks.data.contracts import AttackContext, Capability
from obfuscation_attacks.evaluation.sda_stream import (
    decode_sda,
    fit_sda,
    run_sda_stream,
    wire_symbol_ids,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit

VOCAB = 12
REFERENCE = [
    np.array([1, 2, 3, 4, 5], dtype=np.int64),
    np.array([5, 4, 3, 2, 6], dtype=np.int64),
    np.array([7, 8, 9, 1, 2], dtype=np.int64),
    np.array([3, 5, 7, 9, 10], dtype=np.int64),
]
CIPHER = np.array([1, 2, 3, 4, 5], dtype=np.int64)

PARAMS_2021 = {
    "num_train": 16, "epochs": 1, "batch_size": 8,
    "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16,
}
PARAMS_2023 = {
    "num_train": 16, "epochs": 1, "batch_size": 8,
    "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16,
}


@pytest.mark.parametrize("cipher", [[], [True], [1.5], [-1], [[1, 2]], np.array([2**63], dtype=np.uint64)])
def test_decode_sda_rejects_invalid_cipher_without_silent_cast(cipher):
    bundle = {"method": "sda2021", "model": None, "cfg": None, "max_cipher_len": 8}
    with pytest.raises(ValueError, match="nonempty sequence"):
        decode_sda(bundle, cipher)


@pytest.mark.parametrize("beam", [0, -1, True, 1.5])
def test_decode_sda_rejects_invalid_beam(beam):
    with pytest.raises(ValueError, match="beam_width"):
        decode_sda({}, [1], beam_width=beam)


@pytest.mark.parametrize("document", [[1.5, 2], [True, False], [-1, 2], [[1, 2]],
                                     np.array([2**63], dtype=np.uint64)])
def test_fit_sda_rejects_invalid_reference_ids_without_silent_cast(document):
    with pytest.raises(ValueError, match="reference documents"):
        fit_sda([document], [[1]], vocab_size=8, params={"num_train": 1, "epochs": 1})


@pytest.mark.parametrize("name,value", [("epochs", 1.5), ("num_train", True), ("batch_size", 0),
                                         ("min_len", 2.3), ("max_cipher_len", -1), ("d_model", 4.5)])
def test_fit_sda_rejects_noninteger_or_nonpositive_training_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params={name: value})


@pytest.mark.parametrize("option,value", [("resume", "false"), ("homophonic", "false"),
                                          ("seed", 1.5), ("seed", True)])
def test_fit_sda_rejects_ambiguous_flags_before_creating_artifacts(tmp_path, option, value):
    path = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match=option):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, method="sda2023",
                params={option: value, "num_train": 1, "epochs": 1}, checkpoint_path=str(path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": 2**32}, {"validation_seed": -1},
                                   {"validation_seed": 1.5}, {"resume": "false"}])
def test_fit_sda_rejects_invalid_explicit_seed_and_resume(kwargs):
    with pytest.raises(ValueError):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params={"num_train": 1, "epochs": 1}, **kwargs)


def _context() -> AttackContext:
    return AttackContext(
        granted=frozenset(
            {Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}
        ),
        descriptor={"vocab_size": VOCAB},
        _ciphertext_token_ids=CIPHER.copy(),
        reference_corpus=[row.copy() for row in REFERENCE],
    )


def test_sda2021_bridge_matches_the_original_class():
    direct = SDA2021(**PARAMS_2021).run(_context())
    bundle = fit_sda(
        REFERENCE, [CIPHER], vocab_size=VOCAB, method="sda2021",
        params=PARAMS_2021, seed=0,
    )
    bridged = decode_sda(bundle, CIPHER, beam_width=5)
    assert bridged["tokens"] == [int(x) for x in direct.recovered_token_ids]


def test_sda2023_bridge_matches_the_original_class():
    direct = SDA2023(**PARAMS_2023).run(_context())
    bundle = fit_sda(
        REFERENCE, [CIPHER], vocab_size=VOCAB, method="sda2023",
        params=PARAMS_2023, seed=0,
    )
    bridged = decode_sda(bundle, CIPHER, beam_width=5)
    assert bridged["tokens"] == [int(x) for x in direct.recovered_token_ids]


@pytest.mark.parametrize("attack", [SDA2021, SDA2023])
@pytest.mark.parametrize("cipher", [[1.5, 2], [True, False], [-1, 2]])
def test_direct_sda_rejects_invalid_cipher_before_training(monkeypatch, attack, cipher):
    from obfuscation_attacks.evaluation import sda_stream
    monkeypatch.setattr(sda_stream, "fit_sda", lambda *args, **kwargs: pytest.fail("training started"))
    ctx = _context()
    ctx._ciphertext_token_ids = np.asarray(cipher)
    with pytest.raises(ValueError, match="non-negative int64"):
        attack(**PARAMS_2021)._run(ctx)


@pytest.mark.parametrize("attack", [SDA2021, SDA2023])
@pytest.mark.parametrize("beam", [True, 1.5, 0])
def test_direct_sda_rejects_invalid_beam(attack, beam):
    with pytest.raises(ValueError, match="beam_width"):
        attack(**PARAMS_2021, beam_width=beam)._run(_context())


def test_wire_symbols_are_stable_and_compact():
    vectors = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    symbols, mapping = wire_symbol_ids(vectors)
    assert symbols[0] == symbols[1]
    assert symbols[2] != symbols[0]
    assert sorted(mapping.values()) == [0, 1]


class FakeDeployment:
    artifact = Path("fixture")
    vocab_size = VOCAB

    def encode(self, text):
        return np.asarray([int(part) for part in str(text).split()], dtype=np.int64)

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                epoch=0, session_id=None, forward=True):
        ids = np.asarray([int(part) for part in str(text).split()], dtype=np.int64)
        vectors = np.zeros((len(ids), 4), dtype=np.float32)
        vectors[np.arange(len(ids)), ids % 4] = 1.0
        public = WirePublic(
            arm=arm, request_id=request_id, seq_len=len(ids), wire_vectors=vectors,
            group_id="g-test", epoch=epoch, session_id=session_id,
        )
        return Wire(public, WireTruth(plaintext_ids=ids.copy(), permuted_ids=ids.copy()))


def test_run_sda_stream_fits_once_and_reports_control():
    out = run_sda_stream(
        FakeDeployment(),
        ["2 7 8 10 6"],
        ["1 2 3 4 5", "5 4 3 2 6", "3 5 7 9 10"],
        method="sda2021", num_train=16, epochs=1, beam_width=2, seed=0,
        max_requests=1, positive_control=True,
    )
    assert out["method"] == "sda2021"
    assert out["n_requests"] == 1
    assert out["token_accuracy"]["denominator"] == 5
    assert "positive_control" in out
    control = out["positive_control"]
    assert "prior_baseline" in control
    assert "control_significance" in control
    assert control["control_significance"]["status"] == "not_evaluated"
    assert out["synthesis"]["mode"] == "invariant"


def test_single_document_control_is_not_a_significance_test():
    from obfuscation_attacks.evaluation.sda_stream import _control_significance
    result = _control_significance({"numerator": 600, "denominator": 1000},
                                   {"frequency_prior": {"value": 0.5}})
    assert result["status"] == "not_evaluated"
    assert result["beat_best_baseline"] is None


def test_sda_accuracy_keeps_uncovered_positions():
    from obfuscation_attacks.evaluation.sda_stream import _token_accuracy

    out = _token_accuracy([1, 3], [1, 3, 5, 6])
    assert out["denominator"] == 4
    assert out["numerator"] == 2
    assert out["value"] == 0.5


def test_sda_training_is_independent_of_the_target_cipher_stream():
    # Behavioural provenance check: fitting on the same attacker-owned reference
    # corpus but a different target cipher stream must yield identical weights.
    # A change that trains on the target pairs would break this equality.
    import numpy as np
    from obfuscation_attacks.evaluation.sda_stream import fit_sda

    reference = [
        np.asarray([1, 2, 3, 4], dtype=np.int64),
        np.asarray([2, 3, 4, 5], dtype=np.int64),
    ]
    params = {"num_train": 8, "epochs": 1, "batch_size": 4}
    a = fit_sda(reference, [np.asarray([1, 2, 3, 4, 5])], vocab_size=10,
                method="sda2021", params=params, seed=0)
    b = fit_sda(reference, [np.asarray([5, 4, 3, 2, 9])], vocab_size=10,
                method="sda2021", params=params, seed=0)
    first = a["model"].state_dict()
    second = b["model"].state_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(
            first[key].detach().cpu().numpy(), second[key].detach().cpu().numpy()
        ), key


def test_run_c11_stream_writes_a_result(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c11_stream(
        FakeDeployment(), ["2 7 8 10 6"],
        ["1 2 3 4 5", "5 4 3 2 6", "3 5 7 9 10"],
        method="sda2021", num_train=16, epochs=1, beam=2, seed=0, max_requests=1,
    )
    assert out["n_requests"] == 1
    written = next(tmp_path.glob("**/result.json"))
    import json as _json

    payload = _json.loads(written.read_text(encoding="utf-8"))
    reason = payload["validation"]["reason"]
    assert "not requested" in reason
    assert payload["validation"]["status"] == "not_run"


def test_stream_rejects_truncated_request():
    class Truncated(FakeDeployment):
        def observe(self, text, **kwargs):
            wire = super().observe(text, **kwargs)
            wire.public.truncated = True
            return wire

    import pytest
    with pytest.raises(ValueError, match="truncated"):
        run_sda_stream(Truncated(), ["1 2 3 4 5"], ["6 7 8 9 10"],
                       positive_control=False)


def test_fit_rejects_zero_epochs():
    import pytest
    with pytest.raises(ValueError, match="epochs"):
        fit_sda([[1, 2, 3, 4]], [[1, 2, 3, 4]], vocab_size=VOCAB,
                params={"epochs": 0})


def test_cipher_alphabet_does_not_change_model_and_coverage_is_reported():
    from obfuscation_attacks.evaluation.sda_stream import decode_sda
    reference = [[1, 2, 1, 2], [2, 1, 2, 1]]
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "max_cipher_len": 8}
    for method in ("sda2021", "sda2023"):
        a = fit_sda(reference, [[9, 9, 9, 9]], vocab_size=10,
                    method=method, params=params, seed=3)
        b = fit_sda(reference, [[1, 2, 3, 4]], vocab_size=10,
                    method=method, params=params, seed=3)
        for name, tensor in a["model"].state_dict().items():
            assert np.array_equal(tensor.detach().numpy(), b["model"].state_dict()[name].detach().numpy())
        coverage = decode_sda(a, [1, 2, 3, 4], beam_width=1)["source_code_coverage"]
        assert coverage["unseen_positions"] == 2
        assert coverage["total_positions"] == 4


def test_unsupported_options_and_variable_length_mode_are_rejected():
    import pytest
    with pytest.raises(ValueError, match="unsupported SDA"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=VOCAB,
                params={"num_tune": 3})
    with pytest.raises(ValueError, match="variable-length decoder"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=VOCAB,
                params={"noise_rate": 0.1, "noise_kinds": ["ins"]})


def test_window_provenance_distinguishes_repeated_augmentation_from_unique_data():
    from obfuscation_attacks.evaluation.sda_stream import document_sha256
    reference = [[1, 2, 3, 4]]
    for method in ("sda2021", "sda2023"):
        bundle = fit_sda(reference, [[5, 6]], vocab_size=VOCAB, method=method,
                         params={"num_train": 8, "epochs": 1, "batch_size": 4,
                                 "min_len": 4, "max_len": 4}, seed=2)
        evidence = bundle["window_evidence"]
        assert evidence["draws"] == 8
        assert evidence["unique_windows"] == evidence["unique_content_hashes"] == 1
        assert evidence["windows"] == [{"document_index": 0, "start": 0, "end": 4,
                                       "content_sha256": document_sha256(reference[0]), "draws": 8}]
        assert evidence["independence_status"].startswith("unverified")


def test_public_vector_frontend_nearest_uses_only_group_and_observations():
    from obfuscation_attacks.evaluation.sda_stream import PublicVectorSymbols
    frontend = PublicVectorSymbols("nearest_l2", distance_threshold=0.001)
    first = frontend.encode([[1.0, 2.0], [9.0, 9.0]], group_id="a", request_id=0)
    second = frontend.encode([[1.00006, 2.0], [9.1, 9.0]], group_id="a", request_id=1)
    other = frontend.encode([[1.0, 2.0]], group_id="b", request_id=2)
    unknown_a = frontend.encode([[1.0, 2.0]], group_id=None, request_id=3)
    unknown_b = frontend.encode([[1.0, 2.0]], group_id=None, request_id=4)
    assert second[0] == first[0]
    assert second[1] not in first
    assert other[0] not in first
    assert unknown_a != unknown_b


def test_public_vector_frontend_validates_threshold_and_dimensions():
    import pytest
    from obfuscation_attacks.evaluation.sda_stream import PublicVectorSymbols
    with pytest.raises(ValueError, match="threshold"):
        PublicVectorSymbols("nearest_l2", float("nan"))
    frontend = PublicVectorSymbols()
    frontend.encode([[1.0, 2.0]], group_id="a", request_id=0)
    with pytest.raises(ValueError, match="dimensions"):
        frontend.encode([[1.0]], group_id="a", request_id=1)


def test_joint_group_decode_shares_mapping_and_keeps_individual_results():
    out = run_sda_stream(
        FakeDeployment(), ["2 7 8 10 6", "6 10 8 7 2"],
        ["1 2 3 4 5", "5 4 3 2 6", "3 5 7 9 10"],
        method="sda2021", num_train=8, epochs=1, beam_width=2,
        positive_control=False, group_joint=True,
    )
    assert len(out["rows"]) == 2
    group = out["group_joint"]["groups"][0]
    assert group["request_ids"] == [0, 1]
    assert group["status"] == "ok"
    left, right = [row["recovered"] for row in group["requests"]]
    assert left == list(reversed(right))


def test_frontend_scoring_separates_splits_merges_and_public_groups():
    from obfuscation_attacks.evaluation.sda_stream import _score_symbol_frontend
    result = _score_symbol_frontend([[1, 2, 1], [1]], [[5, 5, 6], [5]], ["a", "b"])
    assert result["same_truth_pair_count"] == 1
    assert result["split_pairs"] == 1
    assert result["same_symbol_pairs"] == 1
    assert result["merge_pairs"] == 1
    assert result["split_rate"] == result["merge_rate"] == 1.0
    assert result["scope"] == "scoring_only"


def test_infeasible_joint_group_keeps_every_request_in_denominator(monkeypatch):
    from obfuscation_attacks.evaluation import sda_group
    monkeypatch.setattr(sda_group, "decode_sda_group", lambda *args, **kwargs: {"status": "infeasible"})
    out = run_sda_stream(
        FakeDeployment(), ["2 7 8 10 6", "6 10 8 7 2"],
        ["1 2 3 4 5", "5 4 3 2 6", "3 5 7 9 10"],
        method="sda2021", num_train=8, epochs=1, beam_width=2,
        positive_control=False, group_joint=True,
    )
    group = out["group_joint"]
    assert group["token_accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 10}
    assert [row["accuracy"]["denominator"] for row in group["groups"][0]["requests"]] == [5, 5]


def test_c11_bridge_resolves_declared_variant_and_parameters(monkeypatch):
    from types import SimpleNamespace
    from obfuscation_attacks.evaluation import real_battery
    seen = {}
    def capture(dep, texts, references, **kwargs):
        seen.update(kwargs)
        return {"status": "captured"}
    monkeypatch.setattr(real_battery, "run_c11_stream", capture)
    corpus = {"private": SimpleNamespace(texts=lambda: ["private"]),
              "reference": SimpleNamespace(texts=lambda: ["reference"])}
    cfg = {"profile": "probe", "seed": 17, "split_seed": 9,
           "symbol_frontend": "nearest_l2", "distance_threshold": 0.01, "group_joint": True,
           "attack": {"name": "sda.sda2023", "variant": "recurrence_decimal_char_cnn_variant",
                      "params": {"num_train": 12, "epochs": 3, "beam_width": 7,
                                 "min_len": 8, "max_len": 32, "d_model": 96,
                                 "representation": "decimal_char_cnn", "cluster_storage": "disk",
                                 "cluster_temp_dir": "temporary-dir"}}}
    real_battery._run_c11_bridge(None, corpus, None, cfg)
    assert seen["method"] == "sda2023"
    assert (seen["num_train"], seen["epochs"], seen["beam"]) == (12, 3, 7)
    assert seen["training_params"]["d_model"] == 96
    assert seen["training_params"]["representation"] == "decimal_char_cnn"
    assert seen["training_params"]["cluster_storage"] == "disk"
    assert seen["symbol_frontend"] == "nearest_l2" and seen["distance_threshold"] == 0.01
    assert seen["group_joint"] is True
    assert seen["split_seed"] == 9


@pytest.mark.parametrize("profile", ["paper-v1", "full", "", None])
def test_direct_c11_cannot_claim_formal_profile(profile):
    from obfuscation_attacks.evaluation.real_battery import run_c11_stream
    with pytest.raises(ValueError, match="profile=probe"):
        run_c11_stream(None, [], [], profile=profile)


def test_batch_c11_reads_explicit_configuration(tmp_path):
    import importlib.util
    import json
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("sda_battery_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "sda.json"
    config_path.write_text(json.dumps({"seed": 9, "requests": 12,
        "attack": {"name": "sda.sda2023", "variant": "recurrence_lookup_variant",
                   "params": {"num_train": 31, "epochs": 3, "beam_width": 7,
                              "min_len": 4, "max_len": 16}}}), encoding="utf-8")
    config = module._load_sda_probe_config(config_path, seed=42, requests=20)
    resolved = module._resolve_c11_probe_config(config)
    assert (config["seed"], config["requests"]) == (9, 12)
    assert (resolved["method"], resolved["num_train"], resolved["epochs"], resolved["beam"]) == (
        "sda2023", 31, 3, 7)
    original = json.loads(config_path.read_text(encoding="utf-8"))
    original["request"] = 99
    config_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported SDA configuration fields"):
        module._load_sda_probe_config(config_path, seed=42, requests=20)
    config_path.write_text("attack:\n  params:\n    epochs: 2\n    epochs: 20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate SDA configuration field: epochs"):
        module._load_sda_probe_config(config_path, seed=42, requests=20)


@pytest.mark.parametrize("field,value", [("validation_noise_rate", "0.1"),
                                          ("validation_noise_rate", True),
                                          ("validation_noise_kinds", "sub"),
                                          ("validation_noise_kinds", ["typo"])])
def test_c11_rejects_ambiguous_validation_noise(field, value):
    from obfuscation_attacks.evaluation.real_battery import _resolve_c11_probe_config
    config = {"profile": "probe", field: value, "attack": {"name": "sda.sda2021",
              "variant": "frequency_rank_lookup_variant", "params": {"num_train": 4,
              "epochs": 1, "beam_width": 1, "min_len": 4, "max_len": 8}}}
    with pytest.raises(ValueError, match=field):
        _resolve_c11_probe_config(config)


@pytest.mark.parametrize("year", ["2021", "2023"])
def test_batch_probe_example_has_explicit_executable_model_parameters(year):
    import yaml
    from obfuscation_attacks.evaluation.real_battery import _resolve_c11_probe_config
    path = Path(__file__).resolve().parents[2] / "configs" / "probe" / f"sda{year}_stream.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = _resolve_c11_probe_config(config)
    assert result["method"] == f"sda{year}"
    assert {"d_model", "nhead", "num_layers", "dim_ff", "min_len", "max_len"} <= set(result["training_params"])


@pytest.mark.parametrize("key,value", [("seed", 1.5), ("split_seed", True),
                                       ("requests", 2.5), ("requests", 0), ("sda_val", -1)])
def test_c11_bridge_rejects_coerced_configuration(key, value):
    from obfuscation_attacks.evaluation.real_battery import _run_c11_bridge
    cfg = {"profile": "probe", key: value,
           "attack": {"name": "sda.sda2021", "variant": "frequency_rank_lookup_variant",
                      "params": {"num_train": 4, "epochs": 1, "beam_width": 1,
                                 "min_len": 4, "max_len": 8}}}
    with pytest.raises(ValueError, match=key):
        _run_c11_bridge(None, None, None, cfg)


@pytest.mark.parametrize("key,value", [("seed", True), ("seed", -1),
                                       ("split_seed", 1.5), ("split_seed", 2**64)])
def test_stream_rejects_invalid_seeds_before_observation(key, value):
    from obfuscation_attacks.evaluation.sda_stream import run_sda_stream
    with pytest.raises(ValueError, match=key):
        run_sda_stream(None, [], [], **{key: value})


def test_stream_rejects_empty_private_requests_before_training():
    with pytest.raises(ValueError, match="at least one private request"):
        run_sda_stream(None, [], [])


@pytest.mark.parametrize("beams", [[], [0], [True], [1.5]])
def test_stream_rejects_invalid_validation_beams_before_observation(beams):
    with pytest.raises(ValueError, match="validation beam"):
        run_sda_stream(None, [], [], validation_beams=beams)


def test_original_c11_configs_do_not_silently_run_a_different_attack():
    import yaml
    from obfuscation_attacks.evaluation import real_battery
    root = Path(__file__).resolve().parents[2]
    for name in ("sda2021.yaml", "sda2023.yaml"):
        config = yaml.safe_load((root / "configs/original" / name).read_text(encoding="utf-8"))
        config["full_backend"] = {"artifact": "not_loaded"}
        result = real_battery.run_full_config(config)
        assert result["status"] == "resource_blocked"
        assert "variant" in result["status_reason"]


def test_c11_resolution_rejects_unknown_or_missing_parameters():
    import pytest
    from obfuscation_attacks.evaluation.real_battery import _resolve_c11_probe_config
    config = {"profile": "probe", "attack": {"name": "sda.sda2021", "variant": "frequency_rank_lookup_variant",
                         "params": {"num_train": 4, "epochs": 1, "beam_width": 1,
                                    "min_len": 4, "max_len": 8, "num_tune": 10}}}
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_c11_probe_config(config)
    del config["attack"]["params"]["num_tune"]
    del config["attack"]["params"]["min_len"]
    with pytest.raises(ValueError, match="missing"):
        _resolve_c11_probe_config(config)


@pytest.mark.parametrize("variant,representation", [("recurrence_lookup_variant", "decimal_char_cnn"),
                                                      ("recurrence_decimal_char_cnn_variant", "lookup")])
def test_c11_variant_must_match_actual_representation(variant, representation):
    from obfuscation_attacks.evaluation.real_battery import _resolve_c11_probe_config
    config = {"profile": "probe", "attack": {"name": "sda.sda2023", "variant": variant,
              "params": {"num_train": 4, "epochs": 1, "beam_width": 1, "min_len": 4,
                         "max_len": 8, "representation": representation}}}
    with pytest.raises(ValueError, match="requires representation"):
        _resolve_c11_probe_config(config)


@pytest.mark.parametrize("key,value", [("num_train", 1.5), ("epochs", True), ("beam_width", 0)])
def test_c11_resolution_never_silently_coerces_budget(key, value):
    from obfuscation_attacks.evaluation.real_battery import _resolve_c11_probe_config
    params = {"num_train": 4, "epochs": 1, "beam_width": 1, "min_len": 4, "max_len": 8}
    params[key] = value
    cfg = {"profile": "probe", "attack": {"name": "sda.sda2021",
           "variant": "frequency_rank_lookup_variant", "params": params}}
    with pytest.raises(ValueError, match=key):
        _resolve_c11_probe_config(cfg)


def test_control_uniform_prior_is_not_rounded_to_zero():
    from obfuscation_attacks.evaluation.sda_stream import _prior_baseline
    prior = _prior_baseline([1], [2], [[2]], vocab_size=10_000_000)
    assert prior["chance_uniform"] == pytest.approx(1e-7)


def test_frequency_prior_uses_distinct_unseen_vocabulary_for_missing_ranks():
    from obfuscation_attacks.evaluation.sda_stream import _prior_baseline
    result = _prior_baseline([8, 9, 10], [1, 0, 2], [[1, 1]], vocab_size=3)
    assert result["frequency_prior"]["prediction"] == [1, 0, 2]
    assert result["frequency_prior"]["numerator"] == 3
    exhausted = _prior_baseline([8, 9, 10, 11], [1, 0, 2, 1], [[1, 1]], vocab_size=3)
    assert exhausted["frequency_prior"]["prediction"] == [1, 0, 2, -1]
    assert exhausted["frequency_prior"]["denominator"] == 4


def test_nearest_frontend_matches_unoptimized_online_clustering():
    from obfuscation_attacks.evaluation.sda_stream import PublicVectorSymbols
    rng = np.random.default_rng(52)
    initial = rng.normal(size=(45, 8)).astype(np.float32)
    observations = [initial[:25], np.vstack([initial[1:12] + 0.001, initial[25:]])]
    threshold = 0.05
    reference_centers = []
    expected = []
    for request in observations:
        row = []
        for vector in request:
            if reference_centers:
                distances = np.linalg.norm(np.stack(reference_centers) - vector, axis=1)
                closest = int(np.argmin(distances))
                if float(distances[closest]) <= threshold:
                    row.append(closest)
                    continue
            row.append(len(reference_centers))
            reference_centers.append(vector.copy())
        expected.append(row)
    frontend = PublicVectorSymbols("nearest_l2", threshold)
    actual = [frontend.encode(vectors, group_id="shared", request_id=i)
              for i, vectors in enumerate(observations)]
    assert actual == expected


def test_vector_frontends_handle_signed_zero_and_large_finite_values():
    from obfuscation_attacks.evaluation.sda_stream import PublicVectorSymbols, wire_symbol_ids
    vectors = np.array([[0.0], [-0.0], [3e38], [-3e38]], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        rounded = PublicVectorSymbols("rounded").encode(vectors, group_id="g", request_id=1)
        legacy, _ = wire_symbol_ids(vectors)
        nearest = PublicVectorSymbols("nearest_l2", distance_threshold=1e30).encode(
            vectors, group_id="g", request_id=1)
    for output in (rounded, legacy, nearest):
        assert output[0] == output[1]
        assert len(set(output)) == 3
    with pytest.raises(ValueError, match="threshold"):
        PublicVectorSymbols("nearest_l2", True)
