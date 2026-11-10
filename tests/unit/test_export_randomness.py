from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obfuscation_attacks.evaluation import export_randomness


@pytest.fixture
def secure_export(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    (root / "client").mkdir(parents=True)
    (root / "model_server").mkdir()
    (root / "manifest.json").write_text(json.dumps({"components": {"B": {"seed": 42}}}))
    (root / "model_server" / "config.json").write_text(json.dumps({"vocab_size": 12}))
    monkeypatch.setattr(export_randomness, "_load_tokenizer",
                        lambda path: SimpleNamespace(vocab_size=10, all_special_ids=[8, 9]))
    # Golden output from the exporter's independently implemented CryptoRandom.
    permutation = torch.tensor([4, 2, 1, 3, 5, 0, 7, 6, 8, 9, 10, 11])
    inverse = torch.argsort(permutation)
    randomness = {"random_source": "ChaCha20_HMAC_SHA256", "randomness_mode": "system_entropy"}
    secret = {"perm_vocab": permutation, "inv_perm_vocab": inverse,
              "metadata": {**randomness, "seed": 12345, "seed_source": "system_random"},
              "randomness": randomness, "random_master_key": bytes(range(32))}
    torch.save(secret, root / "client" / "client_secret.pt")
    return root, secret


def test_export_randomness_replays_golden_stream_and_fixed_rows(secure_export):
    root, _ = secure_export
    result = export_randomness.verify_export_randomness(root)
    assert result["verified"] is True
    assert "not independently provable" in result["limitation"]
    assert np.array_equal(
        export_randomness._replay_permutation(10, list(range(10)), bytes(range(32)), 12345),
        [6, 8, 0, 1, 5, 3, 4, 9, 7, 2],
    )
    assert "12345" not in json.dumps(result)


def test_export_randomness_accepts_standalone_server_layout(secure_export):
    root, _ = secure_export
    (root / "model_server").rename(root / "server")
    assert export_randomness.verify_export_randomness(root)["verified"] is True


@pytest.mark.parametrize("mutation", ["no_master", "seeded_mode", "different_map", "wrong_inverse"])
def test_export_randomness_rejects_missing_or_false_evidence(secure_export, mutation):
    root, secret = secure_export
    if mutation == "no_master":
        del secret["random_master_key"]
    elif mutation == "seeded_mode":
        secret["randomness"]["randomness_mode"] = "explicit_seed_experiment"
    elif mutation == "different_map":
        secret["perm_vocab"] = torch.arange(12)
        secret["inv_perm_vocab"] = torch.arange(12)
    else:
        secret["inv_perm_vocab"] = torch.arange(12)
    torch.save(secret, root / "client" / "client_secret.pt")
    with pytest.raises(ValueError):
        export_randomness.verify_export_randomness(root)


@pytest.mark.parametrize("leak", ["seed", "master", "permutation", "server_file", "readme", "tensor", "nested_client"])
def test_export_randomness_rejects_public_secret_material(secure_export, leak):
    root, _ = secure_export
    if leak == "server_file":
        (root / "model_server" / "client_secret.pt").write_bytes(b"bad")
    elif leak == "readme":
        (root / "README.md").write_text("export seed: 12345", encoding="utf-8")
    elif leak == "nested_client":
        folder = root / "model_server" / "client"
        folder.mkdir()
        (folder / "details.json").write_text(json.dumps({"seed": 12345}), encoding="utf-8")
    elif leak == "tensor":
        from safetensors.torch import save_file
        save_file({"perm_vocab": torch.arange(12)}, str(root / "model_server" / "weights.safetensors"))
    else:
        value = {"seed": 12345} if leak == "seed" else (
            {"copied": bytes(range(32)).hex()} if leak == "master" else {"perm_vocab": [1, 0]})
        (root / "manifest.json").write_text(json.dumps({"export": value}))
    with pytest.raises(ValueError, match="public export metadata|server export|weight header"):
        export_randomness.verify_export_randomness(root)
