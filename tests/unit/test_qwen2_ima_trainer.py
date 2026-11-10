import numpy as np
import pytest
import torch

from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter, Qwen2InversionModel
from obfuscation_attacks.attacks.ima.qwen2_trainer import _nearest_ids, fit_qwen2_inverter, select_qwen2_inverter


def _model():
    return Qwen2InversionModel(8, 8, hidden_size=8, intermediate_size=16,
                               max_position_embeddings=32, plain_embed_dim=8, seed=3)


def _pairs(sequences):
    table = np.eye(8, dtype=np.float32)
    return [(table[np.asarray(seq)], np.asarray(seq)) for seq in sequences]


@pytest.mark.parametrize("objective", ["embedding", "token_index"])
def test_batched_validation_selection_and_resume_are_reproducible(tmp_path, objective):
    train = _pairs([[0, 1, 2], [2, 3], [4, 5, 6, 7], [1, 6]])
    validation = _pairs([[0, 2], [3, 5, 7]])
    kwargs = dict(objective=objective, reference_embeddings=np.eye(8, dtype=np.float32),
                  validation_pairs=validation, batch_size=2, lr=0.002, seed=7)
    full = fit_qwen2_inverter(_model(), train, epochs=3, **kwargs)
    path = tmp_path / "training.pt"
    fit_qwen2_inverter(_model(), train, epochs=1, checkpoint_path=path, **kwargs)
    resumed = fit_qwen2_inverter(_model(), train, epochs=3, checkpoint_path=path, resume=True, **kwargs)
    assert full["steps"] == resumed["steps"] == 6
    assert full["examples_seen"] == 12
    assert full["tokens_seen"] == 33
    assert full["n_train_tokens"] == 11 and full["n_validation_tokens"] == 5
    assert full["instantiated_config"]["num_hidden_layers"] == 2
    assert full["instantiated_config"]["num_attention_heads"] == 8
    assert full["history"] == resumed["history"]
    assert full["best_epoch"] == max(full["history"],
        key=lambda row: (row["validation_top1"], -row["validation_loss"]))["epoch"]
    assert all(row["validation_tokens"] == 5 for row in full["history"])
    for name, tensor in full["model"].state_dict().items():
        assert torch.equal(tensor, resumed["model"].state_dict()[name]), name
    with pytest.raises(ValueError, match="mismatch"):
        fit_qwen2_inverter(_model(), train, epochs=3, checkpoint_path=path, resume=True,
                           **{**kwargs, "seed": 8})
    with pytest.raises(ValueError, match="overlap"):
        fit_qwen2_inverter(_model(), train, epochs=1, **{**kwargs, "validation_pairs": [train[0]]})


def test_attention_mask_preserves_short_document_predictions():
    model = _model().eval()
    short = torch.eye(8)[:2].unsqueeze(0)
    batch = torch.zeros(2, 4, 8)
    batch[0, :2] = short[0]
    batch[1] = torch.eye(8)[2:6]
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        alone = model.predict_token_logits(short)[0]
        padded = model.predict_token_logits(batch, attention_mask=mask)[0, :2]
    torch.testing.assert_close(alone, padded, rtol=1e-5, atol=1e-6)


def test_learning_rate_selection_uses_only_validation_results():
    train = _pairs([[0, 1, 2], [2, 3], [4, 5]])
    validation = _pairs([[0, 2], [3, 5]])
    result = select_qwen2_inverter(
        _model, train, validation, learning_rates=[0.001, 0.004], epochs=1,
        objective="token_index", batch_size=2, seed=7,
    )
    best = max(result["learning_rate_grid"], key=lambda row: (row["validation_top1"], -row["validation_loss"]))
    assert result["learning_rate"] == best["learning_rate"]
    assert result["best_epoch"] == best["best_epoch"]
    with pytest.raises(ValueError, match="requires isolated validation"):
        select_qwen2_inverter(_model, train, [], learning_rates=[0.001], epochs=1, objective="token_index")


def test_training_rejects_noninteger_targets_instead_of_truncating():
    with pytest.raises(ValueError, match="invalid training document pair"):
        fit_qwen2_inverter(_model(), [(np.eye(8, dtype=np.float32)[:2], [0.0, 1.5])],
                           objective="token_index", epochs=1)


def test_inverter_exposes_validated_grid_training():
    inverter = Qwen2AloePriInverter(objective="token_index", epochs=1, batch_size=2,
                                   hidden_size=8, intermediate_size=16, seed=3)
    result = inverter.fit_with_validation(_pairs([[0, 1], [2, 3]]), _pairs([[0, 2]]),
                                          vocab_size=8, learning_rates=[0.001, 0.003])
    assert result["n_validation_documents"] == 1
    assert result["validation_selection"] == "token_top1_then_loss"
    assert len(result["learning_rate_grid"]) == 2
    assert all(row["optimizer_steps"] == 1 for row in result["learning_rate_grid"])


def test_full_vocabulary_validation_nearest_matches_brute_force_with_ties():
    rng = np.random.default_rng(13)
    table = rng.normal(size=(11, 4)).astype(np.float32)
    table[8] = table[2]
    queries = rng.normal(size=(9, 4)).astype(np.float32)
    queries[0] = table[2]
    expected = np.square(queries[:, None] - table[None, :]).sum(axis=2).argmin(axis=1)
    actual = _nearest_ids(torch.from_numpy(queries), table, block=3, query_block=2)
    np.testing.assert_array_equal(actual.numpy(), expected)
