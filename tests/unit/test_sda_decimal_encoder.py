import torch

from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import (
    CausalCipherLM, CausalLMConfig, DecimalRecurrenceEncoder,
)
from obfuscation_attacks.evaluation.sda_stream import fit_sda, decode_sda, load_sda_checkpoint


def test_decimal_recurrence_encoder_shapes_and_gradient():
    encoder = DecimalRecurrenceEncoder(max_rec=128, d_model=12)
    codes = torch.tensor([1, 12, 128, 12])
    embeddings = encoder(codes)
    assert embeddings.shape == (4, 12)
    assert torch.equal(embeddings[1], embeddings[3])
    embeddings.square().sum().backward()
    assert encoder.characters.weight.grad is not None
    assert encoder.convs[0].weight.grad.abs().sum() > 0
    assert len(encoder.highway_gate) == 2
    assert [(layer.kernel_size[0], layer.out_channels) for layer in encoder.convs] == [
        (1, 64), (2, 128), (3, 192), (4, 256)]


def test_decimal_model_is_causal_and_can_encode_unseen_codes():
    torch.manual_seed(2)
    cfg = CausalLMConfig(vocab_size=8, max_rec=30, d_model=12, nhead=3,
                         num_layers=1, dim_ff=24, dropout=0, max_len=32,
                         representation="decimal_char_cnn")
    model = CausalCipherLM(cfg).eval()
    first = torch.tensor([[cfg.rec_tag, cfg.rec_id(1), cfg.rec_id(27), cfg.plain_tag, 1]])
    second = first.clone()
    second[0, -1] = 2
    with torch.no_grad():
        a, b = model(first), model(second)
    assert a.shape == (1, 5, cfg.total_vocab)
    torch.testing.assert_close(a[:, :-1], b[:, :-1])
    assert torch.isfinite(a).all()


def test_decimal_variant_trains_saves_and_decodes(tmp_path):
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16,
              "max_cipher_len": 8, "representation": "decimal_char_cnn"}
    path = tmp_path / "decimal.pt"
    bundle = fit_sda([[1, 2, 1, 2]], [[4, 5, 6]], vocab_size=8,
                     method="sda2023", params=params, checkpoint_path=str(path))
    assert bundle["cfg"].representation == "decimal_char_cnn"
    assert bundle["model"].recurrence_encoder is not None
    loaded = load_sda_checkpoint(path)
    assert decode_sda(loaded, [4, 5, 6], beam_width=1)["tokens"] == decode_sda(bundle, [4, 5, 6], beam_width=1)["tokens"]
