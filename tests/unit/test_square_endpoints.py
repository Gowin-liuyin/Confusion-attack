from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from obfuscation_attacks.evaluation import square_endpoints


@pytest.fixture
def endpoint_export(tmp_path, monkeypatch):
    base, artifact = tmp_path / "base", tmp_path / "artifact"
    server, client = artifact / "server", artifact / "client"
    base.mkdir()
    server.mkdir(parents=True)
    client.mkdir()
    (base / "config.json").write_text(json.dumps({"hidden_size": 4, "vocab_size": 8, "tie_word_embeddings": True}))
    monkeypatch.setattr(square_endpoints, "_load_tokenizer",
                        lambda server: SimpleNamespace(vocab_size=6, all_special_ids=[5]))
    inverse = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7])
    torch.save({"random_master_key": bytes(range(32)), "inv_perm_vocab": inverse,
                "metadata": {"seed": 12345, "implementation_variant": "square_monomial",
                             "alpha_e": 0.2, "alpha_h": 0.01, "dtype": "float32"}}, client / "client_secret.pt")
    clean = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    save_file({"model.embed_tokens.weight": clean}, str(base / "model.safetensors"))
    embed_noise = [0.3610319197177887, -0.5740956664085388, 1.1693397760391235, 2.4366166591644287,
                   -0.45555227994918823, -0.23490777611732483, 0.007960822433233261, 0.15865200757980347,
                   1.431437373161316, -0.31768542528152466, 0.5458176732063293, 0.24673621356487274,
                   -2.0649590492248535, -0.020620174705982208, -0.5411965847015381, 1.9854555130004883,
                   2.7418642044067383, -0.4524390995502472, -0.46881696581840515, -2.0070157051086426]
    head_noise = [-0.12387064099311829, -0.5762953758239746, -0.05065608769655228, -2.094184160232544,
                  0.5123472213745117, 0.13201379776000977, 0.797030508518219, -1.0837173461914062,
                  -1.2982624769210815, 0.8959342837333679, -0.42955511808395386, 0.7151647210121155,
                  0.6011942028999329, 0.49607425928115845, -1.0420362949371338, -0.9779539704322815,
                  0.09257584810256958, 0.625717282295227, -1.8302711248397827, -1.0847246646881104]
    permuted = clean[inverse]
    sigma = float(permuted[:5].std(unbiased=False))
    hidden_inverse, signs = torch.tensor([1, 3, 2, 0]), torch.tensor([-1., 1., 1., 1.])
    targets = {}
    for name, alpha, noise in (("model.embed_tokens.weight", 0.2, embed_noise),
                                ("lm_head.weight", 0.01, head_noise)):
        noisy = permuted.clone()
        noisy[:5] += alpha * (torch.tensor(noise).reshape(5, 4) * sigma)
        targets[name] = noisy[:, hidden_inverse] * signs
    save_file(targets, str(server / "model.safetensors"))
    return base, artifact, targets


def test_endpoint_replay_checks_all_noisy_and_fixed_rows(endpoint_export):
    base, artifact, _ = endpoint_export
    result = square_endpoints.verify_square_endpoints(artifact, base)
    assert result["verified"] is True and result["checked_rows"] == 16


def test_endpoint_replay_rejects_misreported_dtype(endpoint_export):
    base, artifact, targets = endpoint_export
    targets["lm_head.weight"] = targets["lm_head.weight"].to(torch.float16)
    save_file(targets, str(artifact / "server" / "model.safetensors"))
    with pytest.raises(ValueError, match="dtype disagrees"):
        square_endpoints.verify_square_endpoints(artifact, base)


@pytest.mark.parametrize("name,row", [("model.embed_tokens.weight", 2), ("lm_head.weight", 7)])
def test_endpoint_replay_rejects_changed_noisy_or_fixed_row(endpoint_export, name, row):
    base, artifact, targets = endpoint_export
    targets[name] = targets[name].clone()
    targets[name][row, 0] += 1
    save_file(targets, str(artifact / "server" / "model.safetensors"))
    with pytest.raises(ValueError, match="endpoint.*(?:does not|do not) match"):
        square_endpoints.verify_square_endpoints(artifact, base)
