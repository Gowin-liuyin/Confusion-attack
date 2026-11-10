from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from obfuscation_attacks.evaluation.square_lineage import _square_axes, verify_square_lineage


@pytest.fixture
def square_export(tmp_path):
    base = tmp_path / "base"
    artifact = tmp_path / "export"
    server = artifact / "server"
    client = artifact / "client"
    base.mkdir()
    server.mkdir(parents=True)
    client.mkdir()
    config = {"model_type": "llama", "hidden_size": 4, "vocab_size": 8,
              "intermediate_size": 6, "num_hidden_layers": 1,
              "num_attention_heads": 2, "num_key_value_heads": 2,
              "head_dim": 2, "hidden_act": "silu", "rms_norm_eps": 1e-5,
              "max_position_embeddings": 32, "attention_bias": False, "mlp_bias": False,
              "rope_theta": 10000.0, "rope_scaling": {"rope_type": "default"},
              "tie_word_embeddings": True}
    (base / "config.json").write_text(json.dumps(config))
    deployed_config = {**config, "tie_word_embeddings": False,
                       "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}
    del deployed_config["rope_theta"], deployed_config["rope_scaling"]
    (server / "config.json").write_text(json.dumps(deployed_config))
    torch.save({"random_master_key": bytes(range(32)),
                "metadata": {"seed": 12345, "implementation_variant": "square_monomial",
                             "num_layers": 1, "transform_dim": 4, "dtype": "float32"}}, client / "client_secret.pt")
    inverse = torch.tensor([1, 3, 2, 0])
    signs = torch.tensor([-1., 1., 1., 1.])
    source, target = {}, {}
    shapes = {"model.norm.weight": (4,),
              "model.layers.0.input_layernorm.weight": (4,),
              "model.layers.0.post_attention_layernorm.weight": (4,)}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        shapes[f"model.layers.0.self_attn.{proj}.weight"] = (4, 4)
    for proj in ("gate_proj", "up_proj"):
        shapes[f"model.layers.0.mlp.{proj}.weight"] = (6, 4)
    shapes["model.layers.0.mlp.down_proj.weight"] = (4, 6)
    for index, (name, shape) in enumerate(shapes.items()):
        tensor = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) + index
        source[name] = tensor
        if "norm" in name:
            target[name] = tensor[inverse]
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            target[name] = tensor[inverse] * signs[:, None]
        else:
            target[name] = tensor[:, inverse] * signs[None, :]
    source["model.embed_tokens.weight"] = torch.zeros((8, 4))
    target["model.embed_tokens.weight"] = torch.zeros((8, 4))
    target["lm_head.weight"] = torch.zeros((8, 4))
    save_file(source, str(base / "model.safetensors"))
    save_file(target, str(server / "model.safetensors"))
    return base, artifact, source, target


def test_square_axes_match_independent_exporter_golden():
    inverse, signs = _square_axes(bytes(range(32)), 12345, 4)
    assert inverse.tolist() == [1, 3, 2, 0]
    assert signs.tolist() == [-1, 1, 1, 1]


def test_square_lineage_checks_every_decoder_tensor(square_export):
    base, artifact, _, _ = square_export
    result = verify_square_lineage(artifact, base)
    assert result["verified"] is True and result["checked_tensors"] == 10
    assert "embedding and output-head" in result["scope"]


@pytest.mark.parametrize("field", ["decoder", "norm", "rope", "extra_config", "tied_head"])
def test_square_lineage_rejects_forged_matching_claim(square_export, field):
    base, artifact, _, target = square_export
    if field in {"rope", "extra_config", "tied_head"}:
        path = artifact / "server" / "config.json"
        config = json.loads(path.read_text())
        if field == "rope":
            config["rope_parameters"]["rope_theta"] = 20000.0
        elif field == "extra_config":
            config["sliding_window"] = 128
        else:
            config["tie_word_embeddings"] = True
        path.write_text(json.dumps(config))
    else:
        name = "model.layers.0.self_attn.k_proj.weight" if field == "decoder" else "model.norm.weight"
        target[name] = target[name].clone()
        target[name].view(-1)[0] += 1
        save_file(target, str(artifact / "server" / "model.safetensors"))
    with pytest.raises(ValueError, match="configurations|decoder tensor"):
        verify_square_lineage(artifact, base)
