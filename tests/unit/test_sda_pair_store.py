from pathlib import Path
import numpy as np
import pytest
import torch

from obfuscation_attacks.evaluation.sda_pair_store import SQLitePairStore
from obfuscation_attacks.evaluation.sda_stream import fit_sda, decode_sda


def test_pair_store_roundtrip_and_cleanup(tmp_path):
    with SQLitePairStore(str(tmp_path)) as store:
        root = Path(store._tmp.name)
        store.append(([2, 1, 2], [8, 9]))
        store.append(([], [3]))
        store.finish()
        assert list(store) == [([2, 1, 2], [8, 9]), ([], [3])]
        assert store[np.int64(1)] == ([], [3])
        with pytest.raises(IndexError):
            _ = store[2]
    assert not root.exists()


@pytest.mark.parametrize("method,extra", [
    ("sda2021", {}),
    ("sda2023", {}),
    ("sda2023", {"homophonic": True, "homophone_symbols": 12}),
    ("sda2021", {"noise_rate": 0.2, "noise_kinds": ["ins"],
                 "length_mode": "variable", "max_output_len": 8}),
])
def test_disk_pairs_match_memory_model_and_decoding(tmp_path, method, extra):
    reference = [[1, 2, 3, 4, 2], [3, 2, 1, 4, 3]]
    params = {"num_train": 6, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1,
              "max_cipher_len": 8, **extra}
    memory = fit_sda(reference, [[1, 2, 1]], vocab_size=8, method=method,
                     params={**params, "pair_storage": "memory"}, seed=5)
    manifest = tmp_path / "windows.jsonl"
    disk = fit_sda(reference, [[1, 2, 1]], vocab_size=8, method=method,
                   params={**params, "pair_storage": "disk", "pair_storage_dir": str(tmp_path),
                           "window_manifest_path": str(manifest)}, seed=5)
    assert disk["pair_storage"] == "disk" and not any(p.is_dir() for p in tmp_path.iterdir())
    import json
    recorded = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert sorted(recorded, key=lambda x: (x["document_index"], x["start"], x["end"])) == sorted(
        memory["window_evidence"]["windows"], key=lambda x: (x["document_index"], x["start"], x["end"]))
    assert disk["window_evidence"]["unique_windows"] == memory["window_evidence"]["unique_windows"]
    assert disk["final_loss"] == memory["final_loss"]
    for name, value in memory["model"].state_dict().items():
        assert torch.equal(value, disk["model"].state_dict()[name]), name
    assert decode_sda(memory, [1, 2, 1], beam_width=2)["tokens"] == decode_sda(disk, [1, 2, 1], beam_width=2)["tokens"]


def test_disk_pairs_are_cleaned_after_failure(tmp_path):
    with pytest.raises(ValueError, match="empty ciphertext"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8,
                params={"num_train": 2, "epochs": 1, "noise_rate": 1.0,
                        "noise_kinds": ["del"], "length_mode": "variable", "max_output_len": 8,
                        "pair_storage": "disk", "pair_storage_dir": str(tmp_path),
                        "window_manifest_path": str(tmp_path / "windows.jsonl")})
    assert list(tmp_path.iterdir()) == []


def test_disk_training_resume_matches_uninterrupted(tmp_path):
    params = {"num_train": 4, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1,
              "pair_storage": "disk", "pair_storage_dir": str(tmp_path)}
    full = fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8,
                   params={**params, "window_manifest_path": str(tmp_path / "full.jsonl")}, seed=3)
    path = tmp_path / "checkpoint.pt"
    fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params={**params, "epochs": 1},
            seed=3, checkpoint_path=str(path))
    resumed = fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params,
                      seed=3, checkpoint_path=str(path), resume=True)
    for name, value in full["model"].state_dict().items():
        assert torch.equal(value, resumed["model"].state_dict()[name]), name
    assert not any(p.is_dir() for p in tmp_path.iterdir())


def test_disk_manifest_is_required_and_tampering_is_detected(tmp_path):
    from obfuscation_attacks.evaluation.sda_stream import load_sda_checkpoint
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16,
              "pair_storage": "disk", "pair_storage_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="window_manifest_path"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params)
    path = tmp_path / "checkpoint.pt"
    result = fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    manifest = Path(result["window_evidence"]["manifest_path"])
    manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sda_checkpoint(path)
    with pytest.raises(ValueError, match="manifest does not match"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path), resume=True)
    assert manifest.read_text(encoding="utf-8") == "tampered\n"


def test_storage_refuses_manifest_checkpoint_collision_without_writing(tmp_path):
    path = tmp_path / "same.pt"
    params = {"num_train": 2, "epochs": 1, "pair_storage": "disk",
              "window_manifest_path": str(path)}
    with pytest.raises(ValueError, match="different files"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params, checkpoint_path=str(path))
    assert not path.exists()


def test_storage_refuses_to_overwrite_existing_manifest(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("original", encoding="utf-8")
    params = {"num_train": 2, "epochs": 1, "pair_storage": "disk",
              "window_manifest_path": str(path)}
    with pytest.raises(FileExistsError):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=8, params=params)
    assert path.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
def test_disk_storage_preserves_validation_checkpoint_selection(tmp_path, method):
    params = {"num_train": 4, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.0}
    train, val = [[1, 2, 3, 4, 1]], [[2, 4, 2, 3, 4]]
    common = {"vocab_size": 8, "method": method, "seed": 5,
              "validation_ids": val, "validation_seed": 17}
    memory = fit_sda(train, [[1, 2]], params={**params, "pair_storage": "memory"}, **common)
    disk = fit_sda(train, [[1, 2]], params={**params, "pair_storage": "disk",
                   "window_manifest_path": str(tmp_path / "windows.jsonl")}, **common)
    assert memory["validation_checkpoint"] == disk["validation_checkpoint"]
    for name, value in memory["model"].state_dict().items():
        assert torch.equal(value, disk["model"].state_dict()[name]), name


def test_disk_resume_preserves_validation_selected_checkpoint(tmp_path):
    params = {"num_train": 4, "epochs": 2, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "dropout": 0.1,
              "pair_storage": "disk", "pair_storage_dir": str(tmp_path)}
    common = {"vocab_size": 8, "seed": 3,
              "validation_ids": [[2, 4, 2, 3, 4]], "validation_seed": 11}
    train = [[1, 2, 3, 4, 1]]
    full = fit_sda(train, [[1]], params={**params, "window_manifest_path": str(tmp_path / "full.jsonl")}, **common)
    checkpoint = tmp_path / "resume.pt"
    fit_sda(train, [[1]], params={**params, "epochs": 1}, checkpoint_path=str(checkpoint), **common)
    resumed = fit_sda(train, [[1]], params=params, checkpoint_path=str(checkpoint), resume=True, **common)
    assert full["validation_checkpoint"] == resumed["validation_checkpoint"]
    for name, value in full["model"].state_dict().items():
        assert torch.equal(value, resumed["model"].state_dict()[name]), name
