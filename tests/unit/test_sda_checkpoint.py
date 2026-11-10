import numpy as np
import pytest
import torch

from obfuscation_attacks.evaluation.sda_stream import fit_sda, load_sda_checkpoint, decode_sda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_cuda_checkpoint_resume_restores_rng(tmp_path, method):
    path = tmp_path / "cuda.pt"
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1}
    common = dict(vocab_size=8, method=method, device="cuda")
    full = fit_sda([[1, 2, 3, 4]], [[1]], params={**params, "epochs": 2}, **common)
    common["checkpoint_path"] = str(path)
    fit_sda([[1, 2, 3, 4]], [[1]], params=params, **common)
    resumed = fit_sda([[1, 2, 3, 4]], [[1]], params={**params, "epochs": 2}, resume=True, **common)
    assert resumed["optimizer_steps"] == 2
    loaded = load_sda_checkpoint(path, device="cuda")
    assert loaded["epochs"] == 2
    assert next(loaded["model"].parameters()).is_cuda
    assert resumed["training_history"] == full["training_history"]


def test_first_checkpoint_does_not_overwrite_concurrent_writer(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16}
    save = torch.save

    def competing_save(state, temporary):
        save(state, temporary)
        path.write_bytes(b"another run's checkpoint")

    monkeypatch.setattr(torch, "save", competing_save)
    with pytest.raises(FileExistsError):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    assert path.read_bytes() == b"another run's checkpoint"


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_checkpoint_resume_matches_uninterrupted_training_and_reuses_weights(tmp_path, method):
    reference = [[1, 2, 3, 4, 2], [3, 2, 1, 4, 3]]
    params = {"num_train": 6, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1,
              "max_cipher_len": 8}
    full = fit_sda(reference, [[1, 2, 1]], vocab_size=8, method=method, params=params, seed=5)
    path = tmp_path / "model.pt"
    partial = fit_sda(reference, [[1, 2, 1]], vocab_size=8, method=method,
                      params={**params, "epochs": 1}, seed=5, checkpoint_path=str(path))
    assert partial["optimizer_steps"] == 3
    resumed = fit_sda(reference, [[2, 3, 2]], vocab_size=8, method=method,
                      params=params, seed=5, checkpoint_path=str(path), resume=True)
    loaded = load_sda_checkpoint(path)
    assert resumed["optimizer_steps"] == loaded["optimizer_steps"] == full["optimizer_steps"] == 6
    assert resumed["examples_seen"] == full["examples_seen"] == 12
    history = full["training_history"]
    assert history == resumed["training_history"] == loaded["training_history"]
    assert [row["epoch"] for row in history] == [1, 2]
    assert sum(row["examples"] for row in history) == full["examples_seen"]
    assert sum(row["optimizer_steps"] for row in history) == full["optimizer_steps"]
    assert all(row["target_tokens"] > 0 and np.isfinite(row["target_nll"]) for row in history)
    if method == "sda2023":
        assert all(row["source_tokens"] > 0 and np.isfinite(row["source_nll"]) for row in history)
    for key, tensor in full["model"].state_dict().items():
        assert torch.equal(tensor, resumed["model"].state_dict()[key]), key
        assert torch.equal(tensor, loaded["model"].state_dict()[key]), key
    for beam in (1, 2):
        assert decode_sda(loaded, [1, 2, 1], beam_width=beam)["tokens"] == decode_sda(full, [1, 2, 1], beam_width=beam)["tokens"]
    record = torch.load(path, weights_only=True)
    assert record["parent_sha256"] is not None
    import hashlib
    parent = path.parent / record["parent_file"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == record["parent_sha256"]


def test_checkpoint_rejects_changed_training_and_unrequested_overwrite(tmp_path):
    path = tmp_path / "model.pt"
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8, "nhead": 2,
              "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8}
    fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    with pytest.raises(FileExistsError):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    with pytest.raises(ValueError, match="mismatch"):
        fit_sda([[1, 2, 3, 5]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path), resume=True)


def test_checkpoint_rejects_implementation_change(tmp_path, monkeypatch):
    from obfuscation_attacks.evaluation import sda_stream
    path = tmp_path / "model.pt"
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8}
    sda_stream.fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    monkeypatch.setattr(sda_stream, "_sda_code_identity", lambda: {"changed": "sha256"})
    with pytest.raises(ValueError, match="identity mismatch"):
        sda_stream.load_sda_checkpoint(path)
    with pytest.raises(ValueError, match="mismatch"):
        sda_stream.fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params,
                           checkpoint_path=str(path), resume=True)


def test_checkpoint_rejects_altered_identity_material(tmp_path):
    path = tmp_path / "model.pt"
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "max_cipher_len": 8}
    fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    record = torch.load(path, weights_only=True)
    record["identity_material"]["seed"] += 1
    altered = tmp_path / "altered.pt"
    torch.save(record, altered)
    with pytest.raises(ValueError, match="training identity mismatch"):
        load_sda_checkpoint(altered)


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_validation_selection_survives_resume_and_is_bound_to_checkpoint(tmp_path, method):
    train = [[1, 2, 3, 4], [4, 3, 2, 1]]
    validation = [[5, 6, 5, 6], [0, 7, 0, 7]]
    params = {"num_train": 4, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1,
              "max_cipher_len": 8}
    common = dict(vocab_size=8, method=method, seed=5, validation_ids=validation, validation_seed=17)
    full = fit_sda(train, [[1, 2]], params=params, **common)
    path = tmp_path / "validation.pt"
    fit_sda(train, [[1, 2]], params={**params, "epochs": 1}, checkpoint_path=str(path), **common)
    resumed = fit_sda(train, [[1, 2]], params=params, checkpoint_path=str(path), resume=True, **common)
    loaded = load_sda_checkpoint(path)
    selection = full["validation_checkpoint"]
    assert selection == resumed["validation_checkpoint"] == loaded["validation_checkpoint"]
    assert selection["n_documents"] == 2
    assert selection["best_epoch"] == min(selection["history"], key=lambda row: row["target_nll"])["epoch"]
    for key, tensor in full["model"].state_dict().items():
        assert torch.equal(tensor, resumed["model"].state_dict()[key])
        assert torch.equal(tensor, loaded["model"].state_dict()[key])
    with pytest.raises(ValueError, match="mismatch"):
        fit_sda(train, [[1, 2]], params=params, checkpoint_path=str(path), resume=True,
                **{**common, "validation_seed": 18})
    with pytest.raises(ValueError, match="overlap"):
        fit_sda(train, [[1, 2]], params=params, **{**common, "validation_ids": [train[0]]})
