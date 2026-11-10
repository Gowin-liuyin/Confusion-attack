import hashlib
import copy

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
BertConfig = transformers.BertConfig
BertForSequenceClassification = transformers.BertForSequenceClassification

from obfuscation_attacks.attacks.sda.candidate_classifier import _examples, load_candidate_classifier, train_candidate_classifier
from obfuscation_attacks.attacks.sda.candidate_search import BertCandidateScorer
from obfuscation_attacks.evaluation.sda_artifacts import score_sda_prediction_artifact, write_sda_candidate_predictions
from obfuscation_attacks.evaluation.sda_public import decode_candidate_observations, prepare_sda_observations
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


def _tokenizer(texts, *, padding, truncation, return_tensors):
    assert padding and not truncation and return_tensors == "pt"
    rows = [[int(value) for value in text.split()] for text in texts]
    ids = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def _model():
    return BertForSequenceClassification(BertConfig(vocab_size=64, hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=32, num_labels=2))


def test_negative_samples_cannot_contradict_identical_positive():
    examples, corrected = _examples([[1, 2]], [[[1], [2, 3]]], np.random.default_rng(1))
    assert examples[0] == ([1, 2], 1)
    assert examples[1] == ([1, 3], 0)
    assert corrected in (0, 1)
    with pytest.raises(ValueError, match="cannot produce a distinct negative"):
        _examples([[1, 2]], [[[1], [2]]], np.random.default_rng(0))


def test_classifier_rejects_decoded_label_collision_within_role(tmp_path):
    train = [[1, 2, 3, 4, 5]]
    validation = [[10, 11, 12, 13, 14]]
    with pytest.raises(ValueError, match="conflicting labels"):
        train_candidate_classifier(
            _model(), _tokenizer, lambda ids: " ".join(map(str, ids[:-1])),
            train_documents=train, train_candidates=[[[x] if i < 4 else [x + 1] for i, x in enumerate(train[0])]],
            validation_documents=validation, validation_candidates=[[[x, x + 1] for x in validation[0]]],
            candidate_tokenizer_sha256="a" * 64, classifier_tokenizer_sha256="b" * 64,
            checkpoint_path=tmp_path / "collision.pt", epochs=1, batch_size=2, learning_rate=1e-3, seed=7)


def test_classifier_rejects_tokenized_label_collision(tmp_path):
    def collapse(texts, **kwargs):
        result = _tokenizer(texts, **kwargs)
        result["input_ids"][:] = 1
        return result
    with pytest.raises(ValueError, match="tokenized positive and negative"):
        _train(_model(), tmp_path / "token-collision.pt", tokenizer=collapse)


def test_classifier_rejects_tokenized_train_validation_overlap(tmp_path):
    def normalize(texts, **kwargs):
        result = _tokenizer(texts, **kwargs)
        result["input_ids"] %= 9
        return result
    with pytest.raises(ValueError, match="tokenized public training and validation examples overlap"):
        _train(_model(), tmp_path / "cross-role-token-collision.pt", tokenizer=normalize, validation_seed=7)


def _train(model, path, *, validation=None, seed=7, validation_seed=0, epochs=2,
           training_checkpoint_path=None, resume=False, tokenizer=_tokenizer):
    train = [[1, 2, 3, 4, 5]]
    validation = validation if validation is not None else [[10, 11, 12, 13, 14]]
    return train_candidate_classifier(
        model, tokenizer, lambda ids: " ".join(map(str, ids)),
        train_documents=train, train_candidates=[[[x, x + 1] for x in train[0]]],
        validation_documents=validation, validation_candidates=[[[x, x + 1] for x in validation[0]]],
        candidate_tokenizer_sha256="a" * 64, classifier_tokenizer_sha256="b" * 64,
        checkpoint_path=path, epochs=epochs, batch_size=2, learning_rate=1e-3, seed=seed,
        validation_seed=validation_seed, training_checkpoint_path=training_checkpoint_path, resume=resume)


def test_train_classifier_on_isolated_auxiliary_data_and_save_selected_weights(tmp_path):
    model = _model()
    path = tmp_path / "scorer.pt"
    result = _train(model, path)
    assert result["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["optimizer_steps"] == 2
    assert result["selected_epoch"] in (1, 2)
    assert result["formal_status"] == "not_assessed" and result["resume_supported"] is False
    assert all(row["train_examples"] == 2 and row["train_model_input_tokens"] == 10 for row in result["history"])
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["selected_epoch"] == result["selected_epoch"]
    assert checkpoint["provenance"]["source_ownership_status"] == "caller_supplied_unverified"
    scorer = BertCandidateScorer(model, _tokenizer, lambda ids: " ".join(map(str, ids)))
    scores = scorer([[1, 2, 3], [2, 3, 4]])
    restored = _model()
    loaded = load_candidate_classifier(restored, path, expected_checkpoint_sha256=result["checkpoint_sha256"],
                                       candidate_tokenizer_sha256="a" * 64, classifier_tokenizer_sha256="b" * 64)
    restored_scorer = BertCandidateScorer(restored, _tokenizer, lambda ids: " ".join(map(str, ids)))
    assert restored_scorer([[1, 2, 3], [2, 3, 4]]) == pytest.approx(scores)
    assert loaded["selected_epoch"] == result["selected_epoch"]
    upload = Upload(surface=UploadSurface.VECTORS, request_id=8, seq_len=2, full_len=2,
                    payload=np.array([[1.1], [2.1]], dtype=np.float32))
    observations = prepare_sda_observations([upload], vector_method="nearest_table",
                                            candidate_table=np.arange(16, dtype=np.float32).reshape(-1, 1),
                                            candidate_k=2)
    decoded = decode_candidate_observations(observations, restored_scorer, beam_width=2,
                                             scorer_sha256=loaded["checkpoint_sha256"],
                                             candidate_id_space="public_plaintext_token_ids")
    for token, choices in zip(decoded["rows"][0]["decode"]["tokens"], observations["rows"][0]["candidates"]):
        assert token in choices
    prediction_path = tmp_path / "candidate-prediction.json"
    written = write_sda_candidate_predictions(
        observations, _model(), _tokenizer, lambda ids: " ".join(map(str, ids)), path, prediction_path,
        expected_checkpoint_sha256=loaded["checkpoint_sha256"], candidate_tokenizer_sha256="a" * 64,
        classifier_tokenizer_sha256="b" * 64, candidate_id_space="public_plaintext_token_ids", beam_width=2)
    scored = score_sda_prediction_artifact(
        prediction_path, {8: [1, 2]}, expected_artifact_sha256=written["artifact_sha256"],
        expected_observations_sha256=observations["processed_observations_sha256"])
    assert scored["scoring"]["token_accuracy"]["denominator"] == 2
    assert scored["formal_status"] == "not_assessed"
    with pytest.raises(ValueError, match="tokenizer commitment mismatch"):
        load_candidate_classifier(restored, path, expected_checkpoint_sha256=result["checkpoint_sha256"],
                                  candidate_tokenizer_sha256="c" * 64, classifier_tokenizer_sha256="b" * 64)
    with pytest.raises(FileExistsError):
        _train(model, path)


def test_classifier_training_rejects_overlapping_validation_before_writing(tmp_path):
    path = tmp_path / "scorer.pt"
    with pytest.raises(ValueError, match="overlap or cannot be isolated"):
        _train(_model(), path, validation=[[1, 2, 3, 4, 5]])
    assert not path.exists()


def test_classifier_rejects_cross_role_collision_in_decoded_examples(tmp_path):
    path = tmp_path / "collision.pt"
    train, validation = [[1, 2, 3, 4, 5]], [[10, 11, 12, 13, 14]]
    with pytest.raises(ValueError, match="generated public training and validation examples overlap"):
        train_candidate_classifier(
            _model(), _tokenizer, lambda ids: "same decoded text",
            train_documents=train, train_candidates=[[[x, x + 1] for x in train[0]]],
            validation_documents=validation, validation_candidates=[[[x, x + 1] for x in validation[0]]],
            candidate_tokenizer_sha256="a" * 64, classifier_tokenizer_sha256="b" * 64,
            checkpoint_path=path, epochs=1, batch_size=2, learning_rate=1e-3, seed=7)
    assert not path.exists()


def test_classifier_validation_negatives_do_not_change_with_training_seed(tmp_path):
    first = _train(_model(), tmp_path / "one.pt", seed=1, validation_seed=123)
    second = _train(_model(), tmp_path / "two.pt", seed=2, validation_seed=123)
    assert first["provenance"]["validation_negative_sha256"] == second["provenance"]["validation_negative_sha256"]
    assert first["provenance"]["validation_seed"] == second["provenance"]["validation_seed"] == 123


def test_classifier_resume_matches_uninterrupted_training_and_preserves_parent(tmp_path):
    torch.manual_seed(123)
    initial = _model()
    full_model, partial_model, resumed_model = (copy.deepcopy(initial) for _ in range(3))
    full = _train(full_model, tmp_path / "full-inference.pt", seed=7, epochs=2)
    state_path = tmp_path / "training.pt"
    first = _train(partial_model, tmp_path / "partial-inference.pt", seed=7, epochs=1,
                   training_checkpoint_path=state_path)
    assert first["training_resume_supported"] is True and first["new_optimizer_steps"] == 1
    first_sha = first["training_checkpoint_sha256"]
    resumed = _train(resumed_model, tmp_path / "resumed-inference.pt", seed=7, epochs=2,
                     training_checkpoint_path=state_path, resume=True)
    assert resumed["training_start_epoch"] == 1
    assert resumed["optimizer_steps"] == 2 and resumed["new_optimizer_steps"] == 1
    assert resumed["history"] == full["history"]
    for name, value in full_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name]), name
    state = torch.load(state_path, weights_only=True)
    assert state["parent_sha256"] == first_sha
    assert hashlib.sha256((state_path.parent / state["parent_file"]).read_bytes()).hexdigest() == first_sha
    assert state["single_writer_required"] is True
    with pytest.raises(ValueError, match="identity mismatch"):
        _train(copy.deepcopy(initial), tmp_path / "bad-inference.pt", seed=8, epochs=2,
               training_checkpoint_path=state_path, resume=True)
    assert not (tmp_path / "bad-inference.pt").exists()
    def changed_tokenizer(*args, **kwargs):
        result = _tokenizer(*args, **kwargs)
        result["input_ids"] += 1
        return result
    with pytest.raises(ValueError, match="identity mismatch"):
        _train(copy.deepcopy(initial), tmp_path / "changed-tokenizer.pt", seed=7, epochs=2,
               training_checkpoint_path=state_path, resume=True, tokenizer=changed_tokenizer)
    assert not (tmp_path / "changed-tokenizer.pt").exists()
    state["history"][0]["optimizer_steps"] += 1
    invalid_state = tmp_path / "invalid-training.pt"
    torch.save(state, invalid_state)
    with pytest.raises(ValueError, match="invalid epoch or step history"):
        _train(copy.deepcopy(initial), tmp_path / "invalid-inference.pt", seed=7, epochs=2,
               training_checkpoint_path=invalid_state, resume=True)
    assert not (tmp_path / "invalid-inference.pt").exists()


def test_classifier_recovers_after_interrupted_epoch(tmp_path):
    torch.manual_seed(321)
    initial = _model()
    expected_model, interrupted_model, resumed_model = (copy.deepcopy(initial) for _ in range(3))
    expected = _train(expected_model, tmp_path / "full.pt", seed=4, epochs=2)
    calls = 0
    def interrupted_tokenizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated interruption after first epoch")
        return _tokenizer(*args, **kwargs)
    state_path = tmp_path / "training.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _train(interrupted_model, tmp_path / "interrupted.pt", seed=4, epochs=2,
               training_checkpoint_path=state_path, tokenizer=interrupted_tokenizer)
    assert not (tmp_path / "interrupted.pt").exists()
    assert torch.load(state_path, weights_only=True)["epoch"] == 1
    resumed = _train(resumed_model, tmp_path / "recovered.pt", seed=4, epochs=2,
                     training_checkpoint_path=state_path, resume=True)
    assert resumed["history"] == expected["history"]
    for name, tensor in expected_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name
