import hashlib

import pytest
import torch

from obfuscation_attacks.evaluation.encoder_lineage import _commitment, verify_encoder_binding


@pytest.fixture
def bound_artifact(tmp_path):
    for part in ("model_server", "osnip", "client"):
        (tmp_path / part).mkdir()
    server = tmp_path / "model_server/config.json"
    server.write_text('{"vocab_size":3}')
    perm = torch.tensor([2, 0, 1])
    state = {"weight": torch.tensor([[1.0, 2.0]])}
    torch.save({"perm_vocab": perm}, tmp_path / "client/client_secret.pt")
    torch.save(state, tmp_path / "osnip/encryptor.pt")
    torch.save({"format": "osnip_encoder_binding_v1", "global_step": 1, "encoder": state,
                "model_files_sha256": {"config.json": hashlib.sha256(server.read_bytes()).hexdigest()},
                "token_map_commitment_sha256": _commitment(perm)}, tmp_path / "osnip/training_binding.pt")
    return tmp_path


def test_matching_content_passes(bound_artifact):
    assert verify_encoder_binding(bound_artifact)["verified"]
    assert _commitment(torch.tensor([2, 0, 1])) == "5237b8b02ae2d6b2e162c1634ee63c6460efc398b22ba79503fa33c5b28ee251"


@pytest.mark.parametrize("changed", ["server", "encoder", "permutation", "binding"])
def test_content_drift_fails(bound_artifact, changed):
    if changed == "server":
        (bound_artifact / "model_server/config.json").write_text("changed")
    elif changed == "encoder":
        torch.save({"weight": torch.tensor([[1.0, 3.0]])}, bound_artifact / "osnip/encryptor.pt")
    elif changed == "permutation":
        torch.save({"perm_vocab": torch.tensor([1, 0, 2])}, bound_artifact / "client/client_secret.pt")
    else:
        (bound_artifact / "osnip/training_binding.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_encoder_binding(bound_artifact)
