import copy
import json
import numpy as np
import pytest

from obfuscation_attacks.evaluation.sda_artifacts import _artifact_digest, write_sda_predictions, score_sda_prediction_artifact, score_sda_text_prediction_artifact
from obfuscation_attacks.evaluation.sda_public import prepare_sda_observations, decoded_sda_group_tokens
from obfuscation_attacks.evaluation.sda_stream import fit_sda
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_frozen_public_prediction_and_separate_scoring(tmp_path, method):
    checkpoint = tmp_path / "model.pt"
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8}
    fit_sda([[1, 2, 3, 4, 5]], [[3, 4, 3]], vocab_size=16, method=method,
            params=params, checkpoint_path=str(checkpoint))
    upload = Upload(surface=UploadSurface.IDS, request_id=42, seq_len=3, full_len=3,
                    payload=np.asarray([3, 4, 3]), metadata={"truth": "NEVER STORE THIS"})
    observations = prepare_sda_observations([upload])
    path = tmp_path / "prediction.json"
    written = write_sda_predictions(observations, checkpoint, path, beam_width=2)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert "NEVER STORE THIS" not in path.read_text(encoding="utf-8")
    assert artifact["artifact_sha256"] == written["artifact_sha256"]
    assert artifact["formal_status"] == "not_assessed"
    changed = copy.deepcopy(observations)
    changed["rows"][0]["symbols"][0] = 5
    with pytest.raises(ValueError, match="fingerprint"):
        write_sda_predictions(changed, checkpoint, tmp_path / "changed.json", beam_width=2)
    assert not (tmp_path / "changed.json").exists()
    scored = score_sda_prediction_artifact(path, {42: [7, 8, 7]},
                                           expected_artifact_sha256=written["artifact_sha256"],
                                           expected_observations_sha256=observations["processed_observations_sha256"])
    assert scored["scoring"]["token_accuracy"]["denominator"] == 3
    with pytest.raises(ValueError, match="reference length"):
        score_sda_prediction_artifact(path, {42: [7]}, expected_artifact_sha256=written["artifact_sha256"],
                                      expected_observations_sha256=observations["processed_observations_sha256"])
    assert "[7, 8, 7]" not in str(scored)
    with pytest.raises(FileExistsError):
        write_sda_predictions(observations, checkpoint, path, beam_width=2)
    with pytest.raises(ValueError, match="expected observations"):
        score_sda_prediction_artifact(path, {42: [7, 8, 7]}, expected_artifact_sha256=written["artifact_sha256"],
                                      expected_observations_sha256="0" * 64)
    tokens = artifact["decoded"]["rows"][0]["decode"]["tokens"]
    tokens[0] = (tokens[0] + 1) % 16
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        score_sda_prediction_artifact(path, {42: [7, 8, 7]},
                                      expected_artifact_sha256=written["artifact_sha256"],
                                      expected_observations_sha256=observations["processed_observations_sha256"])
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen expected fingerprint"):
        score_sda_prediction_artifact(path, {42: [7, 8, 7]},
                                      expected_artifact_sha256=written["artifact_sha256"],
                                      expected_observations_sha256=observations["processed_observations_sha256"])


def test_observation_artifact_rejects_extra_private_fields(tmp_path):
    record = Upload(surface=UploadSurface.IDS, request_id=1, seq_len=2, full_len=2,
                    payload=np.asarray([3, 4]))
    observations = prepare_sda_observations([record])
    observations["rows"][0]["truth"] = [1, 2]
    with pytest.raises(ValueError, match="unexpected fields"):
        write_sda_predictions(observations, tmp_path / "missing.pt", tmp_path / "out.json", beam_width=1)


def test_joint_predictions_are_frozen_and_scored_as_selected_variant(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 1, 2]], [[3, 4, 3]], vocab_size=8,
            params={"num_train": 3, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 4},
            checkpoint_path=str(checkpoint))
    uploads = [Upload(surface=UploadSurface.IDS, request_id=i, seq_len=3, full_len=3,
                      payload=np.asarray(ids), group_id="a")
               for i, ids in ((1, [3, 4, 3]), (2, [4, 3, 4]))]
    observations = prepare_sda_observations(uploads)
    path = tmp_path / "joint.json"
    written = write_sda_predictions(observations, checkpoint, path, beam_width=2, group_joint=True,
                                    decode_mode="constrained")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["decode_mode"] == artifact["decoded"]["decode_mode"] == "constrained"
    assert all(row["decode"]["tokens"] == row["decode"]["constrained"]["tokens"]
               for row in artifact["decoded"]["rows"])
    assert "group_decoder_sha256" in artifact["runtime_identity"]
    predictions = decoded_sda_group_tokens(artifact["decoded"])
    assert set(predictions) == {1, 2}
    scored = score_sda_prediction_artifact(path, predictions, prediction_variant="group_joint",
                                           expected_artifact_sha256=written["artifact_sha256"],
                                           expected_observations_sha256=observations["processed_observations_sha256"])
    assert scored["prediction_variant"] == "group_joint"
    assert scored["scoring"]["token_accuracy"]["numerator"] == 6
    assert scored["scoring"]["token_accuracy"]["denominator"] == 6
    bad = copy.deepcopy(artifact["decoded"])
    bad["group_joint"][0]["rows"][0]["tokens"][0] = 99
    with pytest.raises(ValueError, match="do not match group decoding"):
        decoded_sda_group_tokens(bad)


def test_infeasible_joint_group_is_counted_as_missing():
    from obfuscation_attacks.evaluation.sda_token_scoring import score_sda_token_rows
    decoded = {"mode": "neural", "rows": [{"request_id": 1, "decode": {"tokens": [0]}}],
               "group_joint": [{"request_ids": [1], "decode": {"status": "infeasible"}}]}
    scored = score_sda_token_rows({1: [1, 2]}, decoded_sda_group_tokens(decoded))
    assert scored["missing_predictions"] == 1
    assert scored["token_accuracy"]["denominator"] == 2
    decoded["group_joint"].append(copy.deepcopy(decoded["group_joint"][0]))
    with pytest.raises(ValueError, match="distinct request IDs"):
        decoded_sda_group_tokens(decoded)


def test_artifact_scoring_cannot_omit_infeasible_group_references(tmp_path):
    decoded = {"mode": "neural", "frontend": {"text_alignment": None},
               "processed_observations_sha256": "b" * 64,
               "rows": [{"request_id": 1, "decode": {"tokens": [0]}},
                        {"request_id": 2, "decode": {"tokens": [1]}}],
               "group_joint": [{"request_ids": [1], "decode": {"status": "infeasible"}},
                               {"request_ids": [2], "decode": {"status": "ok", "tokens": [1]},
                                "rows": [{"request_id": 2, "tokens": [1]}]}]}
    artifact = {"format": "sda_public_prediction_v1", "length_mode": "fixed",
                "source_lengths": [{"request_id": 1, "length": 1}, {"request_id": 2, "length": 1}],
                "group_joint": True, "decoded": decoded}
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    kwargs = {"prediction_variant": "group_joint", "expected_artifact_sha256": artifact["artifact_sha256"],
              "expected_observations_sha256": "b" * 64}
    with pytest.raises(ValueError, match="exactly match"):
        score_sda_prediction_artifact(path, {2: [1]}, **kwargs)
    result = score_sda_prediction_artifact(path, {1: [0], 2: [1]}, **kwargs)["scoring"]
    assert result["missing_predictions"] == 1
    assert result["token_accuracy"] == {"value": 0.5, "numerator": 1, "denominator": 2,
                                        "alignment": "positionwise"}
    del artifact["source_lengths"]
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    kwargs["expected_artifact_sha256"] = artifact["artifact_sha256"]
    with pytest.raises(ValueError, match="lacks complete frozen source lengths"):
        score_sda_prediction_artifact(path, {1: [0], 2: [1]}, **kwargs)


def test_text_observation_cannot_use_positional_token_scoring(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 2, 3, 4]], [[3, 4]], vocab_size=8,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8},
            checkpoint_path=str(checkpoint))
    record = Upload(surface=UploadSurface.TEXT, request_id=1, seq_len=2, full_len=2, payload="ab")
    observations = prepare_sda_observations([record], text_tokenizer=lambda text: [1, 2])
    path = tmp_path / "text.json"
    written = write_sda_predictions(observations, checkpoint, path, beam_width=1)
    with pytest.raises(ValueError, match="character-alignment"):
        score_sda_prediction_artifact(path, {1: [1, 2]},
                                      expected_artifact_sha256=written["artifact_sha256"],
                                      expected_observations_sha256=observations["processed_observations_sha256"])


def test_fixed_output_artifact_can_be_scored_against_isolated_text(tmp_path):
    checkpoint = tmp_path / "model.pt"
    fit_sda([[1, 0, 2, 0, 1]], [[3, 4, 3]], vocab_size=8,
            params={"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
                    "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8,
                    "min_len": 5, "max_len": 5, "space_token_id": 0,
                    "length_mode": "fixed_output", "max_output_len": 5}, checkpoint_path=str(checkpoint))
    record = Upload(surface=UploadSurface.IDS, request_id=1, seq_len=3, full_len=3, payload=np.array([3, 4, 3]))
    observations = prepare_sda_observations([record])
    path = tmp_path / "fixed.json"
    written = write_sda_predictions(observations, checkpoint, path, beam_width=1)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    decoder = lambda ids: "".join(chr(97 + token) for token in ids)
    prediction = decoder(artifact["decoded"]["rows"][0]["decode"]["tokens"])
    result = score_sda_text_prediction_artifact(path, {1: prediction}, decoder,
            expected_artifact_sha256=written["artifact_sha256"],
            expected_observations_sha256=written["processed_observations_sha256"])
    assert result["scoring"]["sda2021_character_ter"]["value"] == 0
    assert result["decoder_provenance"] == "caller_supplied_unverified"
    with pytest.raises(ValueError, match="expected observations"):
        score_sda_text_prediction_artifact(path, {1: prediction}, decoder,
            expected_artifact_sha256=written["artifact_sha256"], expected_observations_sha256="0" * 64)
