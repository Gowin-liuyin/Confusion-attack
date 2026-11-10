import pytest

from obfuscation_attacks.evaluation.sda_stream import fit_sda, load_sda_checkpoint


@pytest.mark.parametrize("method", ["sda2021", "sda2023"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_distinct_windows_are_frozen_across_training_seeds_and_resume(tmp_path, method, storage):
    docs = [list(range(start, start + 8)) for start in (1, 11, 21, 31)]
    validation = [list(range(start, start + 8)) for start in (41, 51)]
    params = {"num_train": 3, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 4, "max_len": 4,
              "sampling_mode": "distinct_document_windows", "sampling_seed": 19,
              "pair_storage": storage}
    path = tmp_path / "first.pt"
    first = fit_sda(docs, [[1, 2]], vocab_size=64, method=method, params=params,
                    seed=1, checkpoint_path=str(path), validation_ids=validation, validation_seed=7)
    second_params = {**params}
    if storage == "disk":
        second_params["window_manifest_path"] = str(tmp_path / "second.windows.jsonl")
    second = fit_sda(docs, [[1, 2]], vocab_size=64, method=method, params=second_params,
                     seed=2, validation_ids=validation, validation_seed=7)
    def windows(bundle):
        evidence = bundle["window_evidence"]
        if storage == "disk":
            return open(evidence["manifest_path"], encoding="utf-8").read().splitlines()
        return sorted((row["document_index"], row["start"], row["end"], row["content_sha256"])
                      for row in evidence["windows"])
    assert windows(first) == windows(second)
    assert first["window_evidence"]["unique_windows"] == 3
    assert first["window_evidence"]["unique_content_hashes"] == 3
    assert first["validation_sampling"]["selected_documents"] == 2
    assert first["sampling_evidence"]["cross_role_isolation"]["passed"] is True
    assert all(row["end"] - row["start"] == 4 for row in first["validation_sampling"]["windows"])
    resumed = fit_sda(docs, [[1, 2]], vocab_size=64, method=method,
                      params={**params, "epochs": 2}, seed=1, checkpoint_path=str(path),
                      resume=True, validation_ids=validation, validation_seed=7)
    assert resumed["sampling_evidence"] == load_sda_checkpoint(path)["sampling_evidence"]
    assert resumed["validation_sampling"] == first["validation_sampling"]


def test_distinct_sampling_refuses_short_repeated_or_unfrozen_data():
    base = {"num_train": 2, "epochs": 1, "min_len": 4, "max_len": 4,
            "sampling_mode": "distinct_document_windows"}
    for docs in ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4, 5]] * 2,
                 [list(range(1, 11)), list(range(1, 10)) + [11]]):
        with pytest.raises(ValueError, match="insufficient distinct"):
            fit_sda(docs, [[1]], vocab_size=16, params=base)
    with pytest.raises(ValueError, match="fixed window length"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=16, params={**base, "min_len": 3})


@pytest.mark.parametrize("sampling_mode", ["distinct_document_windows", "with_replacement"])
def test_sampling_refuses_train_validation_near_duplicates(sampling_mode):
    params = {"num_train": 1, "epochs": 1, "min_len": 5, "max_len": 5,
              "sampling_mode": sampling_mode}
    train = [list(range(1, 11))]
    validation = [list(range(1, 10)) + [11]]
    with pytest.raises(ValueError, match="near overlap"):
        fit_sda(train, [[1]], vocab_size=16, params=params, validation_ids=validation)
    with pytest.raises(ValueError, match="unverifiable short-document"):
        fit_sda([[1, 2, 3, 4]], [[1]], vocab_size=16,
                params={**params, "min_len": 4, "max_len": 4}, validation_ids=[[4, 5, 6, 7]])


def test_disk_cluster_window_plan_matches_memory_and_is_used_by_fit(tmp_path):
    from obfuscation_attacks.evaluation.sda_stream import _distinct_document_window_plan
    docs = [list(range(start, start + 9)) for start in (1, 21, 41, 61, 81)]
    memory_plan, memory = _distinct_document_window_plan(docs, count=3, width=5, seed=19)
    disk_plan, disk = _distinct_document_window_plan(docs, count=3, width=5, seed=19,
                                                    cluster_storage="disk", cluster_temp_dir=tmp_path)
    assert disk_plan == memory_plan
    assert disk["clustering"]["n_clusters"] == memory["clustering"]["n_clusters"]
    assert disk["window_clustering"]["n_clusters"] == memory["window_clustering"]["n_clusters"]
    params = {"num_train": 3, "epochs": 1, "batch_size": 2, "d_model": 8,
              "nhead": 2, "num_layers": 1, "dim_ff": 16, "min_len": 5, "max_len": 5,
              "sampling_mode": "distinct_document_windows", "sampling_seed": 19,
              "cluster_storage": "disk", "cluster_temp_dir": str(tmp_path)}
    bundle = fit_sda(docs, [[1, 2]], vocab_size=128, params=params, seed=1,
                     validation_ids=[list(range(101, 110))])
    assert bundle["sampling_evidence"]["clustering"]["storage"] == "temporary_sqlite_index"
    assert bundle["sampling_evidence"]["cross_role_isolation"]["storage"] == "temporary_sqlite_index"
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="SHINGLE_SIZE"):
        _distinct_document_window_plan(docs, count=2, width=4, seed=19, cluster_storage="disk")
