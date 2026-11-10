import pytest
import torch

from obfuscation_attacks.evaluation.sda_stream import compare_sda_checkpoint_budgets, fit_sda


@pytest.mark.parametrize("method,homophonic", [("sda2021", False), ("sda2023", False), ("sda2023", True)])
def test_epoch_budget_comparison_uses_same_frozen_validation_windows(tmp_path, method, homophonic):
    training = [[0, 1, 2, 3, 4]]
    validation = [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    params = {"num_train": 4, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4, "max_len": 4,
              "max_cipher_len": 8}
    if homophonic:
        params["homophonic"] = True
    checkpoint = tmp_path / "model.pt"
    common = dict(vocab_size=16, method=method, seed=4, validation_ids=validation, validation_seed=11,
                  checkpoint_path=str(checkpoint))
    fit_sda(training, [[1, 2]], params=params, **common)
    fit_sda(training, [[1, 2]], params={**params, "epochs": 2}, resume=True, **common)
    record = torch.load(checkpoint, weights_only=True)
    parent = checkpoint.parent / record["parent_file"]
    result = compare_sda_checkpoint_budgets([str(checkpoint), str(parent)], validation, beam_width=1)
    snapshots = result["checkpoints"]
    assert [item["epoch_budget"] for item in snapshots] == [1, 2]
    assert snapshots[0]["checkpoint_sha256"] == record["parent_sha256"]
    old, new = [item["result"] for item in snapshots]
    assert [r["source_sha256"] for r in old["rows"]] == [r["source_sha256"] for r in new["rows"]]
    assert [r["cipher_sha256"] for r in old["rows"]] == [r["cipher_sha256"] for r in new["rows"]]
    assert result["validation_surface"] == ("synthetic_homophonic_substitution" if homophonic
                                             else "synthetic_fixed_substitution")
    assert result["key_policy"] == ("one auxiliary key per validation window" if homophonic
                                     else "one auxiliary fixed key")
    comparison = result["comparisons"][0]
    assert comparison["difference"] == pytest.approx((new["hits"] - old["hits"]) / old["count"])
    assert comparison["n_boot"] == 2000
    assert result["saturation_status"] == "not_assessed"
    assert result["checkpoint_lineage"] == "hash_links_verified"
    assert result["checkpoint_lineage_checks"][0]["links_checked"] == 1
    with pytest.raises(ValueError, match="validation documents"):
        compare_sda_checkpoint_budgets([str(parent), str(checkpoint)], validation[::-1], beam_width=1)
    with pytest.raises(ValueError, match="budgets must be distinct"):
        compare_sda_checkpoint_budgets([str(parent), str(parent)], validation, beam_width=1)


def test_epoch_budget_comparison_rejects_different_training_identities(tmp_path):
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4, "max_len": 4,
              "max_cipher_len": 8}
    validation = [[5, 6, 7, 8]]
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for seed, path in enumerate(paths):
        fit_sda([[0, 1, 2, 3]], [[1]], vocab_size=10, params=params, seed=seed,
                checkpoint_path=str(path), validation_ids=validation)
    with pytest.raises(ValueError, match="training identities differ"):
        compare_sda_checkpoint_budgets([str(p) for p in paths], validation, beam_width=1)


def test_lineage_finds_missing_or_modified_intermediate_archive(tmp_path):
    import hashlib
    validation = [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4, "max_len": 4,
              "max_cipher_len": 8}
    path = tmp_path / "model.pt"
    kwargs = dict(vocab_size=16, seed=4, validation_ids=validation, validation_seed=11,
                  checkpoint_path=str(path))
    fit_sda([[0, 1, 2, 3, 4]], [[1]], params=params, **kwargs)
    first_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fit_sda([[0, 1, 2, 3, 4]], [[1]], params={**params, "epochs": 3}, resume=True, **kwargs)
    first = path.parent / (path.name + ".parent-" + first_sha)
    intermediate = path.parent / torch.load(path, weights_only=True)["parent_file"]
    result = compare_sda_checkpoint_budgets([str(first), str(path)], validation, beam_width=1)
    assert result["checkpoint_lineage"] == "hash_links_verified"
    assert result["checkpoint_lineage_checks"][0]["links_checked"] == 2
    saved = intermediate.read_bytes()
    intermediate.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="lineage hash mismatch"):
        compare_sda_checkpoint_budgets([str(first), str(path)], validation, beam_width=1)
    intermediate.unlink()
    result = compare_sda_checkpoint_budgets([str(first), str(path)], validation, beam_width=1)
    assert result["checkpoint_lineage"] == "incomplete_or_unlinked"
    assert result["checkpoint_lineage_checks"][0]["status"] == "incomplete"
    intermediate.write_bytes(saved)
    invalid = torch.load(intermediate, weights_only=True)
    invalid["training_history"][0]["examples"] += 1
    candidate = tmp_path / "candidate.pt"
    torch.save(invalid, candidate)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    invalid_parent = tmp_path / (path.name + ".parent-" + digest)
    candidate.replace(invalid_parent)
    head = torch.load(path, weights_only=True)
    head["parent_sha256"], head["parent_file"] = digest, invalid_parent.name
    invalid_head = tmp_path / "invalid-head.pt"
    torch.save(head, invalid_head)
    with pytest.raises(ValueError, match="invalid training history"):
        compare_sda_checkpoint_budgets([str(first), str(invalid_head)], validation, beam_width=1)


def test_lineage_does_not_infer_ancestry_from_matching_identity(tmp_path):
    validation = [[5, 6, 7, 8]]
    params = {"num_train": 2, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4, "max_len": 4,
              "max_cipher_len": 8}
    path = tmp_path / "model.pt"
    kwargs = dict(vocab_size=10, seed=4, validation_ids=validation, checkpoint_path=str(path))
    fit_sda([[0, 1, 2, 3]], [[1]], params=params, **kwargs)
    unrelated = tmp_path / "unrelated.pt"
    altered = torch.load(path, weights_only=True)
    altered["final_loss"] = 0.0
    torch.save(altered, unrelated)
    fit_sda([[0, 1, 2, 3]], [[1]], params={**params, "epochs": 2}, resume=True, **kwargs)
    result = compare_sda_checkpoint_budgets([str(unrelated), str(path)], validation, beam_width=1)
    assert result["checkpoint_lineage"] == "incomplete_or_unlinked"
    assert result["checkpoint_lineage_checks"][0]["status"] == "not_linked"
