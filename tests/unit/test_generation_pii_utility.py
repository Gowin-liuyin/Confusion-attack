"""Acceptance tests for W6 generation, PII spans and utility counting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obfuscation_attacks.evaluation.generation import generate_record
from obfuscation_attacks.evaluation.pii_spans import (
    char_spans_to_token_spans,
    score_entity_recovery,
)
from obfuscation_attacks.evaluation.utility import evaluate_ppl, sliding_windows
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit


class FakeTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False, **kwargs):
        import re
        matches = list(re.finditer(r"\S+", str(text)))
        result = {"input_ids": [int(match.group()) for match in matches]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [match.span() for match in matches]
        return result

    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)


class FakeDeployment:
    tokenizer = FakeTokenizer()
    inv_perm = np.arange(10, dtype=np.int64)
    artifact = "fixture"

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def observe_ids(self, ids, *, arm, request_id=0, key_index=None, key_seed=None,
                    epoch=0, session_id=None, text=""):
        ids = np.asarray(ids, dtype=np.int64)
        logits = np.zeros((len(ids), 10), dtype=np.float32)
        logits[-1, len(ids) % 10] = 10.0
        public = WirePublic(
            arm=arm, request_id=request_id, seq_len=len(ids), logits_perm=logits,
            group_id="g-test", epoch=epoch, session_id=session_id,
            wire_vectors=np.eye(10, dtype=np.float32)[ids] if arm == "ab" else None,
            wire_token_ids=ids.copy() if arm in {"plaintext", "aloepri"} else None,
        )
        return Wire(public, WireTruth(plaintext_ids=ids.copy(), text=text))


def test_generation_uses_the_exact_prefix_and_records_every_step():
    record = generate_record(
        FakeDeployment(), [1], arm="plaintext", max_new_tokens=3, greedy=True,
    )
    assert record.prompt_ids == [1]
    assert record.continuation_ids == [1, 2, 3]
    assert len(record.steps) == 3
    trace = record.public_trace()
    assert "continuation_ids" not in trace and "chosen_ids" not in trace
    assert trace["prompt_len"] == 1 and trace["n_steps"] == 3
    scoring = record.scoring_trace()
    assert scoring["continuation_ids"] == [1, 2, 3]


def test_generation_stops_at_eos():
    class EosDeployment(FakeDeployment):
        def observe_ids(self, ids, **kwargs):
            ids_arr = np.asarray(ids, dtype=np.int64)
            logits = np.zeros((len(ids_arr), 10), dtype=np.float32)
            logits[-1, 9] = 10.0
            return Wire(
                WirePublic(arm=kwargs["arm"], request_id=0, seq_len=len(ids_arr),
                           logits_perm=logits, group_id="g"),
                WireTruth(plaintext_ids=ids_arr.copy()),
            )

    record = generate_record(
        EosDeployment(), [1], arm="plaintext", max_new_tokens=5, greedy=True,
    )
    assert record.continuation_ids == []
    assert len(record.steps) == 1


def test_char_spans_map_to_offsets_and_keep_absent_entities():
    offsets = [(0, 5), (6, 11), (12, 14), (15, 20)]
    entities = [
        {"type": "name", "start": 0, "end": 5, "value": "Alice"},
        {"type": "address", "start": 15, "end": 20, "value": "Paris"},
        {"type": "email", "start": 25, "end": 40, "value": "missing@example.com"},
    ]
    spans = char_spans_to_token_spans(offsets, entities)
    assert spans[0]["token_span"] == [0]
    assert spans[1]["token_span"] == [3]
    assert spans[2]["present"] is False


def test_entity_scoring_counts_absent_and_partial_entities_as_failures():
    offsets = [(0, 5), (6, 11), (12, 14), (15, 20)]
    entities = [
        {"type": "name", "start": 0, "end": 5},
        {"type": "address", "start": 15, "end": 20},
        {"type": "email", "start": 25, "end": 40},
    ]
    truth = [1, 2, 3, 4]
    perfect = score_entity_recovery(truth, truth, entities, offsets)
    assert perfect["value"] == pytest.approx(2 / 3)
    assert perfect["denominator"] == 3
    partial = score_entity_recovery(truth, [1, 2, 3, 9], entities, offsets)
    assert partial["numerator"] == 1
    assert partial["denominator"] == 3


def test_empty_entity_set_is_null_not_zero():
    result = score_entity_recovery([1], [1], [], [(0, 1)])
    assert result["value"] is None and result["denominator"] == 0


def test_verified_entity_scoring_rejects_unbound_offsets_and_ids():
    from obfuscation_attacks.evaluation.pii_spans import score_verified_entity_recovery

    entity = [{"type": "id", "value": "1", "start": 0, "end": 1}]
    score = score_verified_entity_recovery(FakeTokenizer(), "1 2", [1, 2], [1, 2], entity, [(0, 1), (2, 3)])
    assert score["status"] == "scored" and score["numerator"] == 1
    bad_offsets = score_verified_entity_recovery(FakeTokenizer(), "1 2", [1, 2], [1, 2], entity, [(0, 2), (2, 3)])
    bad_ids = score_verified_entity_recovery(FakeTokenizer(), "1 2", [1, 3], [1, 3], entity, [(0, 1), (2, 3)])
    assert bad_offsets["status"] == bad_ids["status"] == "not_run"
    assert "numerator" not in bad_offsets and "numerator" not in bad_ids
    absent = [{"type": "id", "value": "9", "start": 4, "end": 5}]
    absent_score = score_verified_entity_recovery(FakeTokenizer(), "1 2", [1, 2], [1, 2], absent, [(0, 1), (2, 3)])
    assert absent_score["status"] == "scored"
    assert absent_score["numerator"] == 0 and absent_score["denominator"] == 1


def test_generation_pii_wrapper_verifies_output_annotations():
    from obfuscation_attacks.evaluation.pii_spans import score_generation_pii

    record = SimpleNamespace(prompt_ids=[1], continuation_ids=[2])
    result = score_generation_pii(
        record, tokenizer=FakeTokenizer(), prompt_text="1",
        prompt_entities=[], prompt_offsets=[(0, 1)],
        output_entities=[{"type": "id", "value": "9", "start": 0, "end": 1}],
        output_offsets=[(0, 1)], recovered_output_ids=[2],
    )
    assert result["generated_output"]["status"] == "not_run"
    assert "numerator" not in result["generated_output"]


def test_ppl_windows_count_every_token_exactly_once():
    windows = sliding_windows(list(range(10)), window=4, stride=2)
    targets = [position for _, positions, _ in windows for position in positions]
    assert targets == list(range(1, 10))
    assert len(targets) == len(set(targets))


@pytest.mark.parametrize("stride", [4, 5])
def test_ppl_rejects_stride_without_context(stride):
    with pytest.raises(ValueError):
        sliding_windows(list(range(10)), window=4, stride=stride)


def test_evaluate_ppl_counts_each_target_token_once():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False, **kwargs):
            return {"input_ids": list(range(6))}

    class Model:
        def __call__(self, input_ids):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=torch.zeros((batch, length, 8)))

    out = evaluate_ppl(Model(), Tokenizer(), ["one", "two"], window=4, stride=2)
    assert out["n_tokens"] == 10  # 5 targets per 6-token text
    assert out["ppl"] == pytest.approx(8.0, rel=1e-5)


def test_generation_trace_saves_public_arrays(tmp_path):
    record = generate_record(
        FakeDeployment(), [1], arm="plaintext", max_new_tokens=2, greedy=True,
    )
    paths = record.save_trace(tmp_path / "public" / "trace")
    public = __import__("json").loads(
        open(paths["public_trace"], encoding="utf-8").read()
    )
    assert "continuation_ids" not in public
    arrays = np.load(paths["arrays"])
    assert "step0_logits_perm" in arrays
    assert paths["scoring_trace"] is None
    with pytest.raises(ValueError, match="scoring_dir"):
        record.save_trace(tmp_path / "public" / "trace_truth", include_truth=True)
    with_truth = record.save_trace(tmp_path / "public" / "trace_truth", include_truth=True,
                                   scoring_dir=tmp_path / "scorer")
    scoring = __import__("json").loads(
        open(with_truth["scoring_trace"], encoding="utf-8").read()
    )
    assert scoring["continuation_ids"] == record.continuation_ids
    with pytest.raises(FileExistsError, match="already exists"):
        record.save_trace(tmp_path / "public" / "trace")


def test_output_pii_uses_a_real_generation_trace(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    out = real_battery.run_c13_c14(
        FakeDeployment(), prompts=["1 2"], max_new_tokens=2, verbose=False,
    )
    row = out["rows"][0]
    assert row["n_steps"] == 2
    assert row["public_trace"]["steps"]
    assert "continuation_ids" not in row["public_trace"]
    assert "scripted" not in row
    assert out["n_steps"] == 2
    assert "continuation" not in row
    assert __import__("pathlib").Path(row["public_trace_artifact"]["path"]).is_file()


def test_output_attack_replays_server_trace_without_scorer_truth(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery
    from obfuscation_attacks.evaluation.output_attack import recover_output_from_trace
    import json
    from pathlib import Path

    class Permuted(FakeDeployment):
        perm = np.array([2, 0, 3, 1, 4, 5, 6, 7, 8, 9])
        inv_perm = np.argsort(perm)

        def observe_ids(self, ids, *, arm, request_id=0, key_index=None, epoch=0, session_id=None):
            ids = np.asarray(ids, dtype=np.int64)
            logits = np.zeros((len(ids), 10), dtype=np.float32)
            logits[-1, self.perm[len(ids) % 10]] = 10.0
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_vectors=np.eye(10, dtype=np.float32)[self.perm[ids]],
                                logits_perm=logits, group_id="g", epoch=epoch)
            return Wire(public, WireTruth(plaintext_ids=ids.copy(), permuted_ids=self.perm[ids]))

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    annotation = {"prompt_entities": [], "prompt_offsets": [],
                  "output_entities": [{"type": "name", "value": "1", "start": 0, "end": 1}],
                  "output_offsets": [(0, 1), (2, 3)]}
    dep = Permuted()
    mapping = {int(dep.perm[i]): i for i in range(10)}
    result = real_battery.run_c13_c14(dep, prompts=["1"], annotations=[annotation],
                                     output_mapping=mapping, max_new_tokens=2, verbose=False)
    row = result["rows"][0]
    assert row["output_pii_recovered"]["numerator"] == row["output_pii_recovered"]["denominator"] == 1
    assert row["output_reconstruction"]["token_top1"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert row["output_reconstruction"]["exact_sequence"] is True
    partial = real_battery.run_c13_c14(dep, prompts=["1"], annotations=[annotation],
                                       output_mapping={int(dep.perm[1]): 1},
                                       max_new_tokens=2, verbose=False)["rows"][0]
    assert partial["output_reconstruction"]["token_top1"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert partial["output_attack"]["uncovered_steps"] == 1
    changed_annotation = {**annotation, "output_entities": [{"type": "name", "value": "OTHER_ENTITY", "start": 0, "end": 1}]}
    changed = real_battery.run_c13_c14(dep, prompts=["1"], annotations=[changed_annotation],
                                      output_mapping=mapping, max_new_tokens=2, verbose=False)
    assert changed["rows"][0]["output_pii_recovered"]["status"] == "not_run"
    assert "entity value" in changed["rows"][0]["output_pii_recovered"]["reason"]
    artifact = row["public_trace_artifact"]
    first = recover_output_from_trace(artifact["path"], recovered_mapping=mapping, expected_sha256=artifact["sha256"])
    assert first["recovered_ids"] == [1, 2]
    assert "SENSITIVE_ENTITY" not in json.dumps(result)
    assert all("SENSITIVE_ENTITY" not in p.read_text(encoding="utf-8")
               and "OTHER_ENTITY" not in p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    manifest = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert "continuation_ids" not in manifest and "continuation_len" not in manifest
    chunk = Path(artifact["path"]).parent / manifest["steps"][0]["arrays"]["server_logits"]["path"]
    with chunk.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="chunk hash mismatch"):
        recover_output_from_trace(artifact["path"], recovered_mapping=mapping)


def test_generation_refuses_truncated_prefix():
    class Truncated(FakeDeployment):
        def observe_ids(self, ids, **kwargs):
            wire = super().observe_ids(ids, **kwargs)
            wire.public.truncated = True
            return wire

    with pytest.raises(ValueError, match="truncated"):
        generate_record(Truncated(), [1, 2], arm="ab", max_new_tokens=1)


def test_public_guard_rejects_client_generation_truth():
    from obfuscation_attacks.evaluation.public_guard import forbidden_material_gaps

    for field in ("continuation", "continuation_ids", "chosen_ids"):
        assert forbidden_material_gaps({field: "private output"})
    assert not forbidden_material_gaps({"mean_continuation_len": 2.0})


def test_output_replay_abstains_on_unmapped_ties(tmp_path):
    from obfuscation_attacks.evaluation.generation import PublicGenerationWriter
    from obfuscation_attacks.evaluation.output_attack import recover_output_from_trace

    class Tied(FakeDeployment):
        perm = np.array([2, 0, 3, 1, 4, 5, 6, 7, 8, 9])
        inv_perm = np.argsort(perm)

        def observe_ids(self, ids, **kwargs):
            wire = super().observe_ids(ids, **kwargs)
            wire.public.logits_perm[-1] = 0
            wire.public.logits_perm[-1, [0, 1]] = 10
            return wire

    writer = PublicGenerationWriter(tmp_path / "trace")
    record = generate_record(Tied(), [1], arm="ab", max_new_tokens=1, public_writer=writer)
    artifact = writer.finish(record)
    with pytest.raises(ValueError, match="streamed generation"):
        record.save_trace(tmp_path / "incomplete")
    partial = recover_output_from_trace(artifact["path"], recovered_mapping={0: 1})
    assert partial["recovered_ids"] == [-1] and partial["uncovered_steps"] == 1
    complete = recover_output_from_trace(artifact["path"], recovered_mapping={0: 1, 1: 3})
    assert complete["recovered_ids"] == [1] and complete["tied_steps"] == 1


def test_classification_utility_scores_the_whole_choice_not_only_the_first_token():
    from obfuscation_attacks.evaluation.utility import classification_utility

    class Tok:
        def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
            ids = [int(part) for part in str(text).split()]
            if truncation and max_length:
                ids = ids[:max_length]
            return {"input_ids": ids}

        def decode(self, ids):
            return " ".join(str(int(i)) for i in ids)

    class Model:
        def __call__(self, input_ids):
            batch, length = input_ids.shape
            logits = torch.zeros(batch, length, 10)
            logits[:, :, 2] = 2.0
            logits[:, :, 3] = 0.0
            return SimpleNamespace(logits=logits)

    examples = [{"prompt": "1", "label": "2 2", "choices": [" 2 3", " 2 2"]}]
    out = classification_utility(Model(), Tok(), examples)
    assert out["accuracy"] == 1.0
    assert out["scoring"] == "sum_loglikelihood"


def test_choice_boundary_uses_full_tokenization_and_retains_all_choices():
    from obfuscation_attacks.evaluation.utility import _choice_loglikelihood

    class Tok:
        def __call__(self, text, **kwargs):
            # Last prompt character merges with the first candidate character.
            if text == "abcd":
                return {"input_ids": [0, 1, 2, 0],
                        "offset_mapping": [(0, 0), (0, 2), (2, 3), (3, 4)]}
            assert text == "abcdXY"
            return {"input_ids": [0, 1, 2, 3, 4],
                    "offset_mapping": [(0, 0), (0, 2), (2, 3), (3, 5), (5, 6)]}

    class Model:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids):
            self.ids = input_ids.tolist()[0]
            self.calls.append(self.ids)
            return SimpleNamespace(logits=torch.zeros(1, len(self.ids), 5))

    model = Model()
    result = _choice_loglikelihood(model, Tok(), "abcd", "XY", max_length=3)
    assert model.calls == [[2, 3, 4], [2, 0]]
    assert result == pytest.approx(-np.log(5))
    with pytest.raises(ValueError, match="preceding context"):
        _choice_loglikelihood(model, Tok(), "abcd", "XY", max_length=2)


def test_choice_without_offsets_rejects_unverifiable_boundary():
    from obfuscation_attacks.evaluation.utility import _choice_loglikelihood

    class Tok:
        def __call__(self, text, add_special_tokens=True, truncation=False):
            return {"input_ids": [1] if text == "a" else [2]}

    with pytest.raises(ValueError, match="boundary"):
        _choice_loglikelihood(None, Tok(), "a", "b", max_length=4)


def test_aloepri_generation_maps_server_logits_back_through_the_permutation():
    class PermutedDeployment(FakeDeployment):
        perm = np.array([2, 0, 3, 1, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(10)

        def observe_ids(self, ids, *, arm, request_id=0, key_index=None, key_seed=None,
                        epoch=0, session_id=None, text=""):
            ids = np.asarray(ids, dtype=np.int64)
            logits = np.zeros((len(ids), 10), dtype=np.float32)
            # the server sees permuted ids and emits permuted-space logits;
            # target clean id 1 maps to deployment id perm[1] = 0
            logits[-1, int(self.perm[1])] = 10.0
            public = WirePublic(
                arm=arm, request_id=request_id, seq_len=len(ids), logits_perm=logits,
                group_id="g-test", epoch=epoch, session_id=session_id,
                wire_token_ids=self.perm[ids],
            )
            return Wire(public, WireTruth(plaintext_ids=ids.copy(),
                                          permuted_ids=self.perm[ids].copy(), text=text))

    record = generate_record(
        PermutedDeployment(), [1], arm="aloepri", max_new_tokens=1, greedy=True,
    )
    assert record.continuation_ids == [1]


def test_c13_labels_self_comparison_as_plaintext_control(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    annotations = [{
        "prompt_entities": [], "prompt_offsets": [],
        "output_entities": [], "output_offsets": [],
    }]
    out = real_battery.run_c13_c14(
        FakeDeployment(), prompts=["1"], annotations=annotations,
        max_new_tokens=2, verbose=False,
    )
    row = out["rows"][0]
    assert "input_pii_plaintext_control" in row
    assert "output_pii_plaintext_control" in row
    assert row["output_pii_recovered"]["status"] == "not_run"
    assert "output_pii" not in row


def test_c13_refuses_incomplete_annotation_sets(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="one entry per prompt"):
        real_battery.run_c13_c14(FakeDeployment(), prompts=["1", "2"], annotations=[{}], verbose=False)
    assert not list(tmp_path.iterdir())


def test_output_cli_options_refuse_formal_profile(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("output_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["run_full_battery.py", "--profile", "paper-v1", "--classes", "C13",
                                      "--output-mapping", "mapping.json"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_config_driven_c13_supplies_prompts(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery

    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    corpus = {"vma": SimpleNamespace(texts=lambda: ["1"])}
    out = real_battery.CLASS_RUNNERS["C13"](
        FakeDeployment(), corpus, {}, {"max_new_tokens": 1, "output_mapping": {i: i for i in range(10)},
                                      "output_annotations": [{"output_entities": [], "output_offsets": []}]}
    )
    assert out["rows"][0]["output_attack"]["status"] == "executed"
    assert out["rows"][0]["output_pii_recovered"]["status"] == "scored"
