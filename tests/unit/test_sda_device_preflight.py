import pytest
import torch

from obfuscation_attacks.attacks.sda._neural import validate_sda_device
from obfuscation_attacks.attacks.sda.candidate_classifier import train_candidate_classifier
from obfuscation_attacks.evaluation.sda_stream import fit_sda


def test_unavailable_cuda_fails_before_creating_training_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "model.pt"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        fit_sda([], [], vocab_size=8, device="cuda", checkpoint_path=str(checkpoint),
                params={"num_train": 60000, "pair_storage": "disk"})
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        train_candidate_classifier(None, None, None, train_documents=[], train_candidates=[],
                                   validation_documents=[], validation_candidates=[],
                                   candidate_tokenizer_sha256="a" * 64,
                                   classifier_tokenizer_sha256="b" * 64,
                                   checkpoint_path=checkpoint, epochs=1, batch_size=1,
                                   learning_rate=0.01, seed=0, device="cuda")
    assert list(tmp_path.iterdir()) == []


def test_explicit_cuda_index_must_exist(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert validate_sda_device("cpu") == torch.device("cpu")
    assert validate_sda_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="only 1 CUDA devices"):
        validate_sda_device("cuda:1")
