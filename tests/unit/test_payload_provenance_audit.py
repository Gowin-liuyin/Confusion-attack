import json

import pytest

from obfuscation_attacks.evaluation.attack_model import result_payload
from obfuscation_attacks.evaluation.real_battery import write_payload


def payload():
    return result_payload(attack_name="ia.gate", scheme_name="ab_deployment:aloepri_weights",
                          metrics={}, source={"name": "recipe"}, data={"name": "fixture"})


def test_missing_content_and_steps_never_look_completed(tmp_path):
    record = write_payload("C08", "ia.gate", "ab", payload(), result_root=tmp_path)
    saved = json.loads((tmp_path / "C08" / "ia.gate" / "ab" / record["run_id"] / "result.json").read_text())
    assert saved["status"] == "not_verified"
    assert "executed_steps" not in saved
    assert saved["source_sha256"] is None and saved["data_sha256"] is None
    assert len(saved["source_label_sha256"]) == 64 and len(saved["data_label_sha256"]) == 64


def test_content_hashes_track_file_bytes_and_missing_paths_fail(tmp_path):
    code = tmp_path / "attack.py"
    data = tmp_path / "corpus.json"
    code.write_text("a", encoding="utf-8")
    data.write_text("x", encoding="utf-8")
    extra = {"source_paths": [code], "data_paths": [data], "executed_steps": 2}
    first = write_payload("C08", "ia.gate", "a", payload(), extra, result_root=tmp_path)
    saved = json.loads((tmp_path / "C08" / "ia.gate" / "a" / first["run_id"] / "result.json").read_text())
    assert saved["status"] == "not_verified"
    code.write_text("b", encoding="utf-8")
    data.write_text("y", encoding="utf-8")
    second = write_payload("C08", "ia.gate", "b", payload(), extra, result_root=tmp_path)
    assert first["source_sha256"] != second["source_sha256"]
    assert first["data_sha256"] != second["data_sha256"]
    complete = write_payload("C08", "ia.gate", "done", payload(),
                             {**extra, "run_outcome": "completed"}, result_root=tmp_path)
    saved = json.loads((tmp_path / "C08" / "ia.gate" / "done" / complete["run_id"] / "result.json").read_text())
    assert saved["status"] == "completed"
    data.unlink()
    with pytest.raises(ValueError, match="data_paths"):
        write_payload("C08", "ia.gate", "c", payload(), extra, result_root=tmp_path)
