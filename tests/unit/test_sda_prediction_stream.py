import hashlib
import json
import numpy as np
import pytest

from obfuscation_attacks.evaluation import sda_public
from obfuscation_attacks.evaluation.sda_artifacts import score_sda_prediction_stream, write_sda_prediction_stream
from obfuscation_attacks.evaluation.sda_stream import fit_sda
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


def test_stream_keeps_failed_requests_in_scoring_denominator(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 3, 4, 5]], [[3, 4, 3]], vocab_size=16,
            params={"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8},
            checkpoint_path=str(checkpoint))
    uploads = [Upload(surface=UploadSurface.IDS, request_id=1, seq_len=3, full_len=3,
                      payload=np.asarray([3, 4, 3])),
               Upload(surface=UploadSurface.IDS, request_id=2, seq_len=3, full_len=3,
                      payload=np.asarray([6, 6, 6]))]
    observations = sda_public.prepare_sda_observations(uploads)
    def decode(bundle, symbols, **kwargs):
        if symbols[0] == 6:
            raise ValueError("ERROR DETAILS MUST NOT BE SAVED")
        return {"tokens": [1, 1, 1], "raw": {"beams": [[[1, 1, 1], -1.0]]},
                "constrained": {"status": "infeasible"}}
    monkeypatch.setattr(sda_public, "decode_sda", decode)
    path = tmp_path / "predictions.jsonl"
    written = write_sda_prediction_stream(observations, checkpoint, path, beam_width=2)
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [row["type"] for row in records] == ["header", "prediction", "prediction", "footer"]
    assert records[1]["decode"]["raw"]["beams"] == [[[1, 1, 1], -1.0]]
    assert records[2]["status"] == "failed" and records[2]["error_type"] == "ValueError"
    assert "ERROR DETAILS MUST NOT BE SAVED" not in "\n".join(lines)
    assert written["succeeded"] == written["failed"] == 1
    common = {"expected_artifact_sha256": written["artifact_sha256"],
              "expected_observations_sha256": observations["processed_observations_sha256"]}
    scored = score_sda_prediction_stream(path, {2: [9, 9, 9], 1: [1, 1, 1]}, **common)
    assert scored["scoring"]["token_accuracy"]["numerator"] == 3
    assert scored["scoring"]["token_accuracy"]["denominator"] == 6
    assert scored["scoring"]["missing_predictions"] == scored["failed_requests"] == 1
    with pytest.raises(ValueError, match="reference length"):
        score_sda_prediction_stream(path, {2: [9], 1: [1, 1, 1]}, **common)
    with pytest.raises(ValueError, match="exactly match"):
        score_sda_prediction_stream(path, {1: [1, 1, 1]}, **common)
    with pytest.raises(FileExistsError):
        write_sda_prediction_stream(observations, checkpoint, path, beam_width=2)
    with path.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="frozen expected fingerprint"):
        score_sda_prediction_stream(path, {1: [1, 1, 1], 2: [9, 9, 9]}, **common)
    changed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, json.JSONDecodeError)):
        score_sda_prediction_stream(path, {1: [1, 1, 1], 2: [9, 9, 9]},
                                    **{**common, "expected_artifact_sha256": changed_sha})


def test_stream_records_real_overlength_decode_failure_without_stopping(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 3, 4, 5]], [[3, 4]], vocab_size=16,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8},
            checkpoint_path=str(checkpoint))
    uploads = [Upload(surface=UploadSurface.IDS, request_id=1, seq_len=9, full_len=9,
                      payload=np.asarray([3] * 9)),
               Upload(surface=UploadSurface.IDS, request_id=2, seq_len=2, full_len=2,
                      payload=np.asarray([3, 4]))]
    observations = sda_public.prepare_sda_observations(uploads)
    path = tmp_path / "predictions.jsonl"
    result = write_sda_prediction_stream(observations, checkpoint, path, beam_width=1)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[1]["reason_code"] == "cipher_exceeds_configured_length"
    assert records[2]["status"] == "ok"
    score = score_sda_prediction_stream(path, {1: [1] * 9, 2: [1, 2]},
                                       expected_artifact_sha256=result["artifact_sha256"],
                                       expected_observations_sha256=result["processed_observations_sha256"])
    assert score["scoring"]["token_accuracy"]["denominator"] == 11
    assert score["failed_requests"] == 1


def test_stream_rejects_invalid_input_before_decoding(tmp_path):
    record = Upload(surface=UploadSurface.IDS, request_id=1, seq_len=2, full_len=2,
                    payload=np.asarray([3, 4]))
    observations = sda_public.prepare_sda_observations([record])
    observations["rows"][0]["symbols"][0] = -1
    observations["processed_observations_sha256"] = sda_public._observation_sha256(observations)
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError, match="symbols must be"):
        write_sda_prediction_stream(observations, tmp_path / "missing.pt", path, beam_width=1)
    assert not path.exists()


def test_stream_does_not_publish_cuda_out_of_memory_as_request_failure(tmp_path, monkeypatch):
    import torch
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 3, 4, 5]], [[3, 4]], vocab_size=16,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8},
            checkpoint_path=str(checkpoint))
    upload = Upload(surface=UploadSurface.IDS, request_id=1, seq_len=2, full_len=2,
                    payload=np.asarray([3, 4]))
    observations = sda_public.prepare_sda_observations([upload])
    def exhausted(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("fixture")
    monkeypatch.setattr(sda_public, "decode_sda", exhausted)
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(torch.cuda.OutOfMemoryError):
        write_sda_prediction_stream(observations, checkpoint, path, beam_width=1)
    assert not path.exists()
    assert not list(tmp_path.glob("predictions.jsonl.*.tmp"))


def test_joint_stream_saves_groups_and_scores_failed_requests(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 1, 2]], [[3, 4]], vocab_size=8,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 4},
            checkpoint_path=str(checkpoint))
    uploads = [Upload(surface=UploadSurface.IDS, request_id=i, seq_len=len(ids), full_len=len(ids),
                      payload=np.asarray(ids), group_id=group)
               for i, ids, group in ((1, [3, 4, 3], "a"), (2, [3] * 5, "b"), (3, [4, 3, 4], "a"))]
    observations = sda_public.prepare_sda_observations(uploads)
    path = tmp_path / "groups.jsonl"
    written = write_sda_prediction_stream(observations, checkpoint, path, beam_width=2, group_joint=True)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["header", "group_prediction", "group_prediction", "footer"]
    assert records[1]["request_ids"] == [1, 3] and records[1]["status"] == "ok"
    assert records[0]["decode_mode"] == "group_constrained"
    assert records[2]["request_ids"] == [2] and records[2]["status"] == "failed"
    assert records[1]["decode"]["context_reset"] == "per_request"
    references = {row["request_id"]: row["tokens"] for row in records[1]["rows"]}
    references[2] = [1] * 5
    scored = score_sda_prediction_stream(path, references,
        expected_artifact_sha256=written["artifact_sha256"],
        expected_observations_sha256=written["processed_observations_sha256"])
    assert scored["prediction_variant"] == "group_joint" and scored["failed_requests"] == 1
    assert scored["scoring"]["token_accuracy"]["numerator"] == 6
    assert scored["scoring"]["token_accuracy"]["denominator"] == 11
    assert scored["scoring"]["missing_predictions"] == 1
    records[1]["rows"][0]["tokens"][0] = 99
    content = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                       for row in records[:-1])
    records[-1]["content_sha256"] = hashlib.sha256(content).hexdigest()
    path.write_bytes(content + json.dumps(records[-1], ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError, match="do not match group decoding"):
        score_sda_prediction_stream(path, references, expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    expected_observations_sha256=written["processed_observations_sha256"])


def test_constrained_mode_is_recorded_and_used_in_stream(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 1, 2]], [[3, 4, 3]], vocab_size=8,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 4},
            checkpoint_path=str(checkpoint))
    upload = Upload(surface=UploadSurface.IDS, request_id=1, seq_len=3, full_len=3,
                    payload=np.array([3, 4, 3]))
    observations = sda_public.prepare_sda_observations([upload])
    path = tmp_path / "constrained.jsonl"
    write_sda_prediction_stream(observations, checkpoint, path, beam_width=2, decode_mode="constrained")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["decode_mode"] == "constrained"
    result = records[1]["decode"]
    assert result["mode"] == "constrained"
    assert result["tokens"] == result["constrained"]["tokens"]
