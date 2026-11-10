"""Acceptance tests for S07: the SDA positive control must be held out.

``run_sda_stream`` used to train on every reference document and then use
``reference_ids[-1]`` as the known-substitution positive control. These tests
pin the replacement contract: disjoint training / validation / control roles
frozen before pair synthesis, ``fit_sda`` receives only the training role, a
cross-role exact or near-duplicate document is refused, and the split is
reproducible for a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.evaluation import sda_stream
from obfuscation_attacks.evaluation.sda_stream import (
    _document_record,
    assign_document_roles,
    check_document_isolation,
    document_sha256,
    run_sda_stream,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit

VOCAB = 64
N_DOCS = 20
DOC_LEN = 24


def _corpus(seed: int = 1234, n_docs: int = N_DOCS, length: int = DOC_LEN) -> list[str]:
    rng = np.random.default_rng(seed)
    return [
        " ".join(str(int(token)) for token in rng.integers(1, VOCAB, size=length))
        for _ in range(n_docs)
    ]


def _private_texts(seed: int = 999, n_docs: int = 1, length: int = DOC_LEN) -> list[str]:
    """Evaluated private texts over a token range disjoint from ``_corpus``."""
    rng = np.random.default_rng(seed)
    return [
        " ".join(str(int(token)) for token in rng.integers(1000, 2000, size=length))
        for _ in range(n_docs)
    ]


def _ids(text: str) -> np.ndarray:
    return np.asarray([int(part) for part in str(text).split()], dtype=np.int64)


def _observe(dep, text, *, arm, request_id=0, key_index=None, key_seed=None,
             epoch=0, session_id=None, forward=True):
    ids = _ids(text)
    vectors = np.zeros((len(ids), 4), dtype=np.float32)
    vectors[np.arange(len(ids)), ids % 4] = 1.0
    public = WirePublic(
        arm=arm, request_id=request_id, seq_len=len(ids), wire_vectors=vectors,
        group_id="g-test", epoch=epoch, session_id=session_id,
    )
    return Wire(public, WireTruth(plaintext_ids=ids.copy(), permuted_ids=ids.copy()))


class FakeDeployment:
    vocab_size = VOCAB

    def encode(self, text):
        return _ids(text)

    def observe(self, text, **kwargs):
        return _observe(self, text, **kwargs)


def _run(deployment, texts, references, **kwargs):
    defaults = dict(
        method="sda2021", num_train=4, epochs=1, beam_width=2, seed=0,
    )
    defaults.update(kwargs)
    return run_sda_stream(deployment, texts, references, **defaults)


def test_control_is_held_out_and_fit_never_sees_it(monkeypatch):
    corpus = _corpus()
    captured = {}
    real_fit = sda_stream.fit_sda

    def spy(reference_ids, cipher_sequences, **kwargs):
        captured["reference_ids"] = [
            np.asarray(entry, dtype=np.int64) for entry in reference_ids
        ]
        return real_fit(reference_ids, cipher_sequences, **kwargs)

    monkeypatch.setattr(sda_stream, "fit_sda", spy)
    out = _run(FakeDeployment(), _private_texts(n_docs=2), corpus)

    counts = out["corpus_roles"]["counts"]
    assert counts["train"] + counts["validation"] + counts["control"] == len(corpus)
    assert counts["train"] >= 1 and counts["control"] >= 1
    assert len(captured["reference_ids"]) == counts["train"]

    fit_hashes = {document_sha256(entry) for entry in captured["reference_ids"]}
    control_hashes = {
        document["sha256"]
        for document in out["corpus_roles"]["hashes"]["control"]["documents"]
    }
    assert fit_hashes.isdisjoint(control_hashes)

    control = out["positive_control"]
    assert control["source_role"] == "control"
    assert control["source_sha256"] in control_hashes
    assert control["source_sha256"] not in fit_hashes
    assert control["isolated_from_training"] is True
    assert out["overlap_check"]["passed"] is True
    assert out["overlap_check"]["required"] is True

    # every document handed to fit_sda is exactly the training role
    listed_train = {
        document["sha256"]
        for document in out["corpus_roles"]["hashes"]["train"]["documents"]
    }
    assert fit_hashes == listed_train

    # control record keeps the established shape and gains a prior baseline
    assert {"value", "numerator", "denominator", "passed"} <= set(control)
    assert control["status"] == "not_evaluated"
    baseline = control["baseline"]
    assert {"identity", "frequency_prior", "chance_uniform"} <= set(baseline)
    assert baseline["chance_uniform"] == pytest.approx(1.0 / VOCAB)


def test_disk_isolation_in_stream_preserves_role_results_and_predictions(tmp_path):
    corpus, private = _corpus(n_docs=12), _private_texts(n_docs=2)
    params = {"min_len": 5, "max_len": 5, "sampling_mode": "distinct_document_windows",
              "sampling_seed": 3, "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16}
    memory = _run(FakeDeployment(), private, corpus, positive_control=False, training_params=params)
    disk = _run(FakeDeployment(), private, corpus, positive_control=False,
                training_params={**params, "cluster_storage": "disk", "cluster_temp_dir": str(tmp_path)})
    assert disk["overlap_check"]["storage"] == "temporary_sqlite_index"
    assert disk["overlap_check"]["by_role_pair"] == memory["overlap_check"]["by_role_pair"]
    assert disk["corpus_roles"]["hashes"] == memory["corpus_roles"]["hashes"]
    assert [row["recovered"] for row in disk["rows"]] == [row["recovered"] for row in memory["rows"]]
    assert not list(tmp_path.iterdir())


def test_stream_uses_constrained_mode_for_validation_control_and_private(monkeypatch):
    observed = []
    original = sda_stream.decode_sda
    def capture(*args, **kwargs):
        observed.append(kwargs.get("mode"))
        return original(*args, **kwargs)
    monkeypatch.setattr(sda_stream, "decode_sda", capture)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), _corpus(n_docs=12),
               training_params={"decode_mode": "constrained", "d_model": 8,
                                "nhead": 2, "num_layers": 1, "dim_ff": 16},
               validation_beams=[1])
    assert observed and all(mode == "constrained" for mode in observed)
    assert out["decode_mode"] == out["validation"]["decode_mode"] == "constrained"
    assert out["rows"][0]["recovered"] == out["rows"][0]["decode"]["constrained"]["tokens"]

def test_exact_duplicate_in_explicit_control_is_refused():
    corpus = _corpus(seed=11)
    with pytest.raises(ValueError, match="cross-role document overlap") as excinfo:
        _run(
            FakeDeployment(), _private_texts(n_docs=1), corpus[10:],
            control_texts=[corpus[10]],
        )
    assert "1 exact and 0 near-duplicate" in str(excinfo.value)


def test_near_duplicate_in_explicit_control_is_refused():
    corpus = _corpus(seed=12)
    near_duplicate = corpus[10] + " 5"  # one appended token, same 5-token windows
    with pytest.raises(ValueError, match="near-duplicate") as excinfo:
        _run(
            FakeDeployment(), _private_texts(n_docs=1), corpus[10:],
            control_texts=[near_duplicate],
        )
    assert "0 exact and 1 near-duplicate" in str(excinfo.value)


def test_single_document_corpus_is_refused_not_silently_trained_on():
    with pytest.raises(ValueError, match="isolated SDA positive control"):
        _run(FakeDeployment(), _private_texts(n_docs=1), ["1 2 3 4 5"])


def test_roles_are_deterministic_for_the_same_seed():
    corpus = _corpus(seed=13)
    documents = [_ids(text) for text in corpus]

    first = assign_document_roles(documents, seed=3)
    second = assign_document_roles(documents, seed=3)
    assert first == second
    assert set(first["train"]).isdisjoint(first["control"])
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["validation"]).isdisjoint(first["control"])
    assert assign_document_roles(documents, seed=4) != first

    a = _run(FakeDeployment(), _private_texts(n_docs=1), corpus, seed=3)
    b = _run(FakeDeployment(), _private_texts(n_docs=1), corpus, seed=3)
    assert a["corpus_roles"]["hashes"] == b["corpus_roles"]["hashes"]
    assert (
        a["positive_control"]["source_sha256"]
        == b["positive_control"]["source_sha256"]
    )


def test_short_control_documents_are_unverifiable_and_refused():
    # S07-short-window-containment-gap: a 4-token control document is a
    # verbatim contiguous window of a training document, but at width 5 the
    # two shingle sets never intersect. The check must fail closed.
    short_window = [10, 11, 12, 13]
    containing_doc = [10, 11, 12, 13, 14, 15, 16, 17]
    long_corpus = [
        " ".join(str(token) for token in ids)
        for ids in [
            containing_doc,
            [20, 21, 22, 23, 24, 25, 26, 27],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]
    ]
    short_text = " ".join(str(token) for token in short_window)

    # direct report: short cross-role pairs are unverifiable, not passed
    short_record = _document_record(
        {"source": "reference", "index": 0, "tokens": np.asarray(short_window)},
        role="control",
    )
    long_record = _document_record(
        {"source": "reference", "index": 1, "tokens": np.asarray(containing_doc)},
        role="train",
    )
    report = check_document_isolation(
        {"train": [long_record], "validation": [], "control": [short_record]}
    )
    assert report["passed"] is False
    assert report["exact"]["collisions"] == 0
    assert report["near_duplicate"]["collisions"] == 0
    assert report["unverifiable_short_documents"]["pairs"] == 1
    assert report["unverifiable_short_documents"]["passed"] is False
    assert report["unverifiable_short_documents"]["examples"][0]["short_document"][
        "length"
    ] == len(short_window)

    # a pool-selected 4-token control document is refused, not silently used
    rng = np.random.default_rng(777)
    pool = [short_text, " ".join(str(t) for t in containing_doc)]
    pool += [
        " ".join(str(int(token)) for token in rng.integers(20, 63, size=12))
        for _ in range(10)
    ]
    roles = assign_document_roles([_ids(text) for text in pool], seed=5)
    assert roles["control"] == [0]  # the short document is the control role
    assert 1 in roles["train"]  # the containing document is a training document
    with pytest.raises(ValueError, match="too short to verify"):
        _run(FakeDeployment(), _private_texts(n_docs=1), pool, seed=5, split_seed=5)

    # explicit 4-token control windows are refused the same way
    with pytest.raises(ValueError, match="too short to verify"):
        _run(FakeDeployment(), _private_texts(n_docs=1), long_corpus, control_texts=[short_text])

    # Disjoint token sets prove there can be no shared nonempty token span.
    out = _run(FakeDeployment(), _private_texts(n_docs=1), long_corpus,
               control_texts=["40 41 42 43"])
    assert out["overlap_check"]["passed"] is True
    assert out["positive_control"]["status"] == "not_evaluated"

    # Diagnostics-only overlap stays visible and cannot certify the control.
    out = _run(FakeDeployment(), _private_texts(n_docs=1), long_corpus,
               control_texts=[short_text], require_isolated_control=False)
    assert out["overlap_check"]["passed"] is False
    assert out["positive_control"]["passed"] is None


def test_five_token_window_control_is_still_flagged_as_near_duplicate():
    containing_doc = [10, 11, 12, 13, 14, 15, 16, 17]
    long_corpus = [
        " ".join(str(token) for token in ids)
        for ids in [
            containing_doc,
            [20, 21, 22, 23, 24, 25, 26, 27],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]
    ]
    window = " ".join(str(token) for token in containing_doc[:5])
    with pytest.raises(ValueError, match="near-duplicate") as excinfo:
        _run(FakeDeployment(), _private_texts(n_docs=1), long_corpus, control_texts=[window])
    assert "0 exact and 1 near-duplicate" in str(excinfo.value)
    assert "too short to verify" not in str(excinfo.value)


def test_ordinary_long_document_split_still_passes():
    corpus = _corpus(seed=21)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus)
    assert out["overlap_check"]["passed"] is True
    assert out["overlap_check"]["unverifiable_short_documents"]["pairs"] == 0


def test_private_texts_overlapping_the_auxiliary_corpus_are_refused():
    corpus = _corpus(seed=31)

    # exact overlap: the evaluated private text is a reference document
    with pytest.raises(ValueError, match="private") as excinfo:
        _run(FakeDeployment(), corpus[:1], corpus)
    assert "cross-role document overlap" not in str(excinfo.value)
    assert "overlaps the SDA auxiliary corpus" in str(excinfo.value)

    # private contamination is fail-closed even in diagnostics-only mode
    with pytest.raises(ValueError, match="private"):
        _run(FakeDeployment(), corpus[:1], corpus, require_isolated_control=False)

    # a near-duplicate private text is refused as well
    with pytest.raises(ValueError, match="private"):
        _run(FakeDeployment(), [corpus[3] + " 5"], corpus)

    # a short private text cannot be verified against the auxiliary corpus
    with pytest.raises(ValueError, match="too short to verify"):
        _run(FakeDeployment(), ["10 11 12"], corpus)

    # disjoint private texts run normally and the role is recorded
    out = _run(FakeDeployment(), _private_texts(n_docs=2), corpus)
    assert out["overlap_check"]["passed"] is True
    counts = out["corpus_roles"]["counts"]
    assert counts["private"] == 2
    assert out["corpus_roles"]["hashes"]["private"]["count"] == 2
    assert out["corpus_roles"]["private_usage"]["status"] == "evaluated"
    assert "train~private" in out["overlap_check"]["by_role_pair"]


def test_overlap_is_recorded_not_raised_in_diagnostics_only_mode():
    corpus = _corpus(seed=15)
    out = _run(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:],
        control_texts=[corpus[10]], require_isolated_control=False,
    )
    assert out["overlap_check"]["passed"] is False
    assert out["overlap_check"]["required"] is False
    assert out["positive_control"]["isolated_from_training"] is False
    # still sourced from the control role: no silent fallback to training docs
    assert out["positive_control"]["source"] == "control_texts"


def test_positive_control_cannot_pass_when_not_isolated(monkeypatch):
    corpus = _corpus(seed=15)

    def perfect_accuracy(recovered, truth):
        n = int(np.asarray(truth).size)
        return {"value": 1.0, "numerator": n, "denominator": n}

    monkeypatch.setattr(sda_stream, "_token_accuracy", perfect_accuracy)
    monkeypatch.setattr(
        sda_stream, "_control_significance",
        lambda accuracy, baseline: {"beat_best_baseline": True},
    )

    # Overlapping control in diagnostics-only mode: accuracy and significance
    # both "pass", but isolation does not, so the control must not pass.
    out = _run(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:],
        control_texts=[corpus[10]], require_isolated_control=False,
    )
    control = out["positive_control"]
    assert control["value"] == 1.0
    assert control["control_significance"]["beat_best_baseline"] is True
    assert control["isolated_from_training"] is False
    assert control["isolation_enforced"] is False
    assert control["passed"] is None

    # The same perfect control passes when it really is held out.
    out_iso = _run(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
        control_texts=corpus[16:18], validation_texts=corpus[18:20],
    )
    control_iso = out_iso["positive_control"]
    assert control_iso["isolated_from_training"] is True
    assert control_iso["passed"] is None


def test_explicit_disjoint_control_and_validation_corpora_are_used():
    corpus = _corpus(seed=14)
    out = _run(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
        control_texts=corpus[16:18], validation_texts=corpus[18:20],
    )
    counts = out["corpus_roles"]["counts"]
    assert counts == {
        "train": 6, "validation": 2, "control": 2, "private": 1,
        "ignored_empty": 0,
    }
    assert out["overlap_check"]["passed"] is True
    control = out["positive_control"]
    assert control["source"] == "control_texts"
    assert control["source_index"] == 0
    assert out["corpus_roles"]["explicit_corpora"] == {
        "control_texts": True, "validation_texts": True,
    }

    # a one-document reference pool is still trainable when the control is
    # supplied explicitly and disjoint from it
    single = _run(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:11],
        control_texts=corpus[16:17],
    )
    assert single["corpus_roles"]["counts"]["train"] == 1
    assert single["overlap_check"]["passed"] is True


def test_control_uses_all_documents_and_paired_lm_baseline():
    corpus = _corpus(seed=14)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
               control_texts=corpus[16:18], validation_texts=corpus[18:20])
    control = out["positive_control"]
    assert len(control["rows"]) == 2
    assert control["denominator"] == sum(row["accuracy"]["denominator"] for row in control["rows"])
    assert all("unconditional_bigram_lm" in row["baselines"] for row in control["rows"])
    for name in ("frequency_prior", "unconditional_bigram_lm"):
        comparison = control["comparisons"][name]
        assert comparison["n_documents"] == 2
        assert comparison["n_boot"] == 2000
        assert comparison["conditional_on"] == "one fixed key"
    assert control["passed"] is None


def test_repeating_one_control_document_cannot_create_independent_clusters():
    from obfuscation_attacks.evaluation.sda_stream import _paired_control_interval
    row = {"source_sha256": "same", "accuracy": {"numerator": 6, "denominator": 10},
           "baselines": {"frequency_prior": {"numerator": 5}}}
    result = _paired_control_interval([row] * 100, "frequency_prior")
    assert result["status"] == "not_evaluated"
    assert result["n_documents"] == 1
    assert result["n_boot"] == 0


def test_validation_beam_selection_uses_isolated_auxiliary_documents():
    corpus = _corpus(seed=21)
    kwargs = dict(control_texts=corpus[16:18], validation_texts=corpus[18:20],
                  validation_beams=[2, 1], split_seed=8)
    first = _run(FakeDeployment(), _private_texts(n_docs=1, seed=901), corpus[10:16], **kwargs)
    second = _run(FakeDeployment(), _private_texts(n_docs=1, seed=903), corpus[10:16], **kwargs)
    selection = first["validation"]
    assert selection == second["validation"]
    assert selection["status"] == "selected"
    assert selection["surface"] == "synthetic_fixed_substitution"
    assert selection["n_documents"] == 2
    assert len(selection["candidates"]) == 2
    assert selection["selected_beam"] in [1, 2]
    assert first["beam_width"] == selection["selected_beam"]
    assert selection["saturation_status"] == "not_assessed"
    assert selection["beam_comparisons"][0]["n_boot"] == 2000
    assert selection["beam_comparisons"][0]["conditional_on"] == "one fixed key"


@pytest.mark.parametrize("sampling_mode", ["with_replacement", "distinct_document_windows"])
def test_beam_selection_uses_same_frozen_validation_windows_as_training(sampling_mode):
    corpus = _corpus(seed=101)
    training_params = {"min_len": 8, "max_len": 8, "max_cipher_len": 32,
                       "sampling_mode": sampling_mode}
    if sampling_mode == "distinct_document_windows":
        training_params["sampling_seed"] = 3
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
               control_texts=corpus[16:18], validation_texts=corpus[18:20],
               validation_beams=[1, 2], split_seed=8, training_params=training_params)
    windows = out["validation_sampling"]["windows"]
    for candidate in out["validation"]["candidates"]:
        assert [row["source_sha256"] for row in candidate["rows"]] == [row["content_sha256"] for row in windows]
        assert all(row["reference_length"] == 8 for row in candidate["rows"])


def test_stream_runs_epoch_curve_before_private_scoring(tmp_path):
    import hashlib
    corpus = _corpus(seed=222)
    path = tmp_path / "budget.pt"
    kwargs = dict(control_texts=corpus[16:18], validation_texts=corpus[18:20],
                  validation_beams=[1], split_seed=8, checkpoint_path=str(path),
                  training_params={"min_len": 8, "max_len": 8, "max_cipher_len": 32,
                                   "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16})
    _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16], **kwargs)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    parent = path.parent / (path.name + ".parent-" + digest)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16], epochs=2,
               resume=True, budget_checkpoints=[str(parent), str(path)], **kwargs)
    assert [row["epoch_budget"] for row in out["epoch_budget_curve"]["checkpoints"]] == [1, 2]
    assert out["epoch_budget_curve"]["beam_width"] == out["beam_width"]
    assert out["epoch_budget_curve"]["saturation_status"] == "not_assessed"


@pytest.mark.parametrize("length_mode", ["fixed", "variable"])
def test_validation_beam_curve_uses_paired_cluster_micro_interval(monkeypatch, length_mode):
    documents = [
        _document_record({"tokens": np.asarray(tokens), "source": "validation", "index": index},
                         role="validation")
        for index, tokens in enumerate(([1, 2, 3, 4, 5], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [1, 2, 3, 4, 5]))
    ]
    def decode(bundle, cipher, *, beam_width, use_cache, mode):
        tokens = [1, 2, 3, 4, 5] if len(cipher) == 5 and beam_width == 2 else [0] * len(cipher)
        return {"tokens": tokens}
    monkeypatch.setattr(sda_stream, "decode_sda", decode)
    result = sda_stream._select_validation_beam(
        {"length_mode": length_mode}, documents, vocab_size=32, candidates=[2, 1], split_seed=3)
    comparison = result["beam_comparisons"][0]
    assert comparison["difference"] == pytest.approx(0.5)
    assert comparison["n_documents"] == comparison["n_clusters"] == 2
    assert comparison["lo"] == pytest.approx(0)
    assert comparison["hi"] == pytest.approx(1)
    assert comparison["metric"] == ("token_edit_error" if length_mode == "variable" else "token_accuracy")
    assert result["saturation_status"] == "not_assessed"
    one_cluster = sda_stream._select_validation_beam(
        {"length_mode": length_mode}, [documents[0], documents[2]], vocab_size=32,
        candidates=[1, 2], split_seed=3)["beam_comparisons"][0]
    assert one_cluster["status"] == "not_evaluated"
    assert one_cluster["n_boot"] == 0


def test_homophonic_validation_uses_heldout_frequencies_and_same_ciphers_for_each_beam(monkeypatch):
    documents = [_document_record({"tokens": np.asarray(tokens), "source": "validation", "index": index},
                                  role="validation")
                 for index, tokens in enumerate(([1, 1, 1, 2], [2, 3, 3, 3]))]
    original_key = sda_stream.enc.make_substitution_key
    keys, observed = [], []
    def capture_key(symbols, **kwargs):
        keys.append((dict(kwargs["plain_counts"]), kwargs["approx_num_symbols"]))
        return original_key(symbols, **kwargs)
    def decode(bundle, cipher, *, beam_width, use_cache, mode):
        observed.append((beam_width, list(cipher)))
        return {"tokens": [0] * len(cipher)}
    monkeypatch.setattr(sda_stream.enc, "make_substitution_key", capture_key)
    monkeypatch.setattr(sda_stream, "decode_sda", decode)
    result = sda_stream._select_validation_beam(
        {"homophonic": True, "homophone_symbols": 9}, documents,
        vocab_size=1000, candidates=[1, 2], split_seed=3)
    assert keys == [({1: 3, 2: 1}, 9), ({2: 1, 3: 3}, 9)]
    assert [row[1] for row in observed[:2]] == [row[1] for row in observed[2:]]
    assert result["surface"] == "synthetic_homophonic_substitution"


def test_validation_selection_rejects_missing_or_overlapping_roles():
    corpus = _corpus(seed=31)
    with pytest.raises(ValueError, match="non-empty isolated validation"):
        _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
             validation_fraction=0, validation_beams=[1, 2])
    with pytest.raises(ValueError, match="selection requires isolated"):
        _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
             validation_texts=[corpus[10]], require_isolated_control=False,
             validation_beams=[1, 2])


def test_missing_validation_fails_before_training(monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("training must not start without validation data")
    monkeypatch.setattr(sda_stream, "fit_sda", unexpected_fit)
    corpus = _corpus(seed=32)
    with pytest.raises(ValueError, match="non-empty isolated validation"):
        _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
             validation_fraction=0, validation_beams=[1, 2])


def test_battery_records_actual_validation_selection(monkeypatch, tmp_path):
    import json
    from obfuscation_attacks.evaluation import real_battery
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FakeDeployment, "artifact", tmp_path, raising=False)
    corpus = _corpus(seed=45)
    out = real_battery.run_c11_stream(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:20],
        method="sda2021", num_train=4, epochs=1, beam=2,
        validation_beams=[1, 2], split_seed=5, n_val_sequences=2,
    )
    path = next(tmp_path.glob("**/result.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["validation"]["status"] == "selected"
    assert payload["n_val_sequences"] == out["validation"]["n_documents"]
    assert payload["n_val_sequences"] == 2
    assert out["corpus_roles"]["requested_validation_documents"] == 2
    assert payload["beam"] == out["beam_width"]


def test_battery_uses_frozen_vector_upload_without_reobserving(monkeypatch, tmp_path):
    import json
    from obfuscation_attacks.evaluation import real_battery
    from obfuscation_attacks.schemes.adapters import Upload, UploadSurface
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FakeDeployment, "artifact", tmp_path, raising=False)
    monkeypatch.setattr(FakeDeployment, "observe", lambda *a, **kw: pytest.fail("reobserved private request"))
    text = _private_texts(n_docs=1)[0]
    tokens = _ids(text)
    upload = Upload(surface=UploadSurface.VECTORS, request_id=42, seq_len=len(tokens), full_len=len(tokens),
                    payload=np.eye(4, dtype=np.float32)[tokens % 4], group_id="public-group")
    out = real_battery.run_c11_stream(FakeDeployment(), {42: text}, _corpus(seed=46)[10:20],
        method="sda2021", num_train=4, epochs=1, beam=2, frozen_uploads=[upload])
    payload = json.loads(next(tmp_path.glob("**/result.json")).read_text(encoding="utf-8"))
    assert out["scoring_join"] == "request_id" and out["rows"][0]["request_id"] == 42
    assert out["public_uploads_sha256"] and out["processed_observations_sha256"]
    assert payload["diagnostics"]["upload_provenance"] == "caller_supplied_ab_unverified"


def test_battery_rejects_missing_groups_before_model_training(monkeypatch):
    from obfuscation_attacks.evaluation import real_battery
    from obfuscation_attacks.schemes.adapters import Upload, UploadSurface
    monkeypatch.setattr(sda_stream, "fit_sda", lambda *a, **kw: pytest.fail("training started"))
    text = _private_texts(n_docs=1)[0]
    tokens = _ids(text)
    upload = Upload(surface=UploadSurface.VECTORS, request_id=42, seq_len=len(tokens), full_len=len(tokens),
                    payload=np.eye(4, dtype=np.float32)[tokens % 4], group_id=None)
    with pytest.raises(ValueError, match="public group identifiers are missing"):
        real_battery.run_c11_stream(FakeDeployment(), {42: text}, _corpus(seed=46)[10:20],
                                   method="sda2021", num_train=4, epochs=1, beam=2, frozen_uploads=[upload])


@pytest.mark.parametrize("year,variant", [("2021", "frequency_rank_lookup_variant"),
                                           ("2023", "recurrence_lookup_variant")])
def test_batch_config_reaches_real_c11_model_and_result(monkeypatch, tmp_path, year, variant):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from obfuscation_attacks.evaluation import real_battery
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FakeDeployment, "artifact", tmp_path, raising=False)
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("sda_batch_real_config_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"requests": 1, "seed": 17, "split_seed": 9, "attack": {
        "name": f"sda.sda{year}", "variant": variant,
        "params": {"num_train": 4, "epochs": 1, "beam_width": 2, "min_len": 4, "max_len": 8,
                   "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16}}}), encoding="utf-8")
    config = module._load_sda_probe_config(path, seed=99, requests=5)
    corpus = {"private": SimpleNamespace(texts=lambda: _private_texts(n_docs=2)),
              "reference": SimpleNamespace(texts=lambda: _corpus(seed=47)[10:20])}
    out = module._run_c11_bridge(FakeDeployment(), corpus, None, config)
    payload = json.loads(next(tmp_path.glob("**/result.json")).read_text(encoding="utf-8"))
    assert out["n_requests"] == 1 and out["seed"] == 17
    assert out["corpus_roles"]["seed"] == 9 and out["n_training_pairs"] == 4
    assert (out["model_config"]["d_model"], out["model_config"]["nhead"],
            out["model_config"]["num_layers"], out["model_config"]["dim_ff"]) == (8, 2, 1, 16)
    assert payload["attack"] == f"sda.sda{year}" and payload["beam"] == 2


def test_validation_count_is_not_silently_clamped():
    corpus = _corpus(seed=46)
    with pytest.raises(ValueError, match="requested n_validation"):
        _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:14], n_validation=4)
    with pytest.raises(ValueError, match="explicit validation_texts"):
        _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:14],
             n_validation=1, validation_texts=corpus[14:15])


def test_variable_length_validation_uses_noisy_cipher_and_edit_selection():
    corpus = _corpus(seed=54)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
               control_texts=corpus[16:18], validation_texts=corpus[18:20],
               training_params={"length_mode": "variable", "max_output_len": 32,
                                "noise_rate": 0.2, "noise_kinds": ["ins"]},
               validation_beams=[1], validation_noise_rate=1.0,
               validation_noise_kinds=["ins"])
    selection = out["validation"]
    assert selection["noise"] == {"rate": 1.0, "kinds": ["ins"]}
    assert "Levenshtein" in selection["rule"]
    assert all(row["cipher_length"] == 2 * row["reference_length"]
               for row in selection["candidates"][0]["rows"])
    assert out["token_accuracy"]["alignment"] == "positionwise_diagnostic"
    assert out["token_edit_error"]["metric"] == "token_levenshtein_error_not_TER"


def test_control_crossed_bootstrap_keeps_key_and_document_clusters():
    from obfuscation_attacks.evaluation.sda_stream import _crossed_control_interval
    rows = []
    for key in range(3):
        for doc, a, b in [("doc-a", 8, 4), ("doc-b", 6, 5)]:
            rows.append({"source_sha256": doc, "control_replica": key,
                         "accuracy": {"numerator": a, "denominator": 10},
                         "baselines": {"frequency_prior": {"numerator": b}}})
    result = _crossed_control_interval(rows, "frequency_prior", seed=1)
    duplicated = _crossed_control_interval(rows * 4, "frequency_prior", seed=1)
    assert result == duplicated
    assert result["n_keys"] == 3 and result["n_documents"] == 2
    assert result["n_boot"] == 2000
    assert result["difference"] == pytest.approx(0.25)
    assert result["lo"] <= result["difference"] <= result["hi"]
    with pytest.raises(ValueError, match="incomplete"):
        _crossed_control_interval(rows[:-1], "frequency_prior")
    single = _crossed_control_interval([row for row in rows if row["source_sha256"] == "doc-a"] * 20,
                                       "frequency_prior")
    assert single["status"] == "not_evaluated" and single["n_documents"] == 1


def test_multiple_control_keys_share_documents_and_are_not_formally_certified():
    corpus = _corpus(seed=14)
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
               control_texts=corpus[16:18], validation_texts=corpus[18:20],
               control_keys=3)
    control = out["positive_control"]
    assert control["control_key_count"] == 3
    assert len(control["rows"]) == 6
    for replica in range(3):
        rows = [row for row in control["rows"] if row["control_replica"] == replica]
        assert len(rows) == 2
    for comparison in control["comparisons"].values():
        assert comparison["n_keys"] == 3
        assert comparison["n_documents"] == 2
        assert comparison["n_boot"] == 2000
        assert comparison["status"] == "descriptive"
    assert control["passed"] is None


def test_control_comparison_for_variable_outputs_uses_edit_errors():
    from obfuscation_attacks.evaluation.sda_stream import _crossed_control_interval
    rows = []
    for key in range(2):
        for document in ("doc-a", "doc-b"):
            rows.append({"source_sha256": document, "control_replica": key,
                         "edit_error": {"edits": 2, "reference_tokens": 3},
                         "baselines": {"frequency_prior": {
                             "edit_error": {"edits": 5, "reference_tokens": 3}}}})
    result = _crossed_control_interval(rows, "frequency_prior", metric="token_edit_error")
    assert result["difference"] == pytest.approx(1.0)
    assert result["metric"] == "token_edit_error"
    assert result["lo"] == result["hi"] == pytest.approx(1.0)


def test_battery_writes_crossed_controls_without_secret_key_material(monkeypatch, tmp_path):
    import json
    from obfuscation_attacks.evaluation import real_battery
    monkeypatch.setattr(real_battery, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FakeDeployment, "artifact", tmp_path, raising=False)
    corpus = _corpus(seed=45)
    out = real_battery.run_c11_stream(
        FakeDeployment(), _private_texts(n_docs=1), corpus[10:20],
        method="sda2021", num_train=4, epochs=1, beam=1, control_keys=2,
    )
    payload = json.loads(next(tmp_path.glob("**/result.json")).read_text(encoding="utf-8"))
    control = payload["metrics"]["positive_control"]
    assert control["control_key_count"] == 2
    assert control["control_surface"] == "synthetic_direct_symbols_clean"
    assert all(row["control_replica"] in (0, 1) for row in control["rows"])
    assert "plain_to_cipher" not in json.dumps(control)
    assert out["positive_control"]["passed"] is None


def test_control_document_clustering_groups_transitive_near_duplicates_and_short_overlap():
    from obfuscation_attacks.evaluation.sda_stream import cluster_control_documents
    base = np.arange(40, dtype=np.int64)
    middle = base.copy(); middle[10] = 200
    last = middle.copy(); last[25] = 201
    distinct = np.arange(300, 340, dtype=np.int64)
    short = np.array([300, 999], dtype=np.int64)
    records = [_document_record({"source": "control", "index": i, "tokens": tokens}, role="control")
               for i, tokens in enumerate((base, middle, last, distinct, short))]
    clusters, report = cluster_control_documents(records)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] == clusters[4] and clusters[3] != clusters[0]
    assert report["n_documents"] == 5
    assert report["n_clusters"] == 2
    assert "does not prove" in report["independence"]


def test_near_duplicate_controls_are_one_resampling_cluster():
    from obfuscation_attacks.evaluation.sda_stream import _crossed_control_interval
    rows = []
    for key in range(3):
        for doc in ("doc-a", "doc-b"):
            rows.append({"source_sha256": doc, "source_cluster": "near-cluster",
                         "control_replica": key, "accuracy": {"numerator": 8, "denominator": 10},
                         "baselines": {"frequency_prior": {"numerator": 5}}})
    result = _crossed_control_interval(rows, "frequency_prior")
    assert result["status"] == "not_evaluated"
    assert result["n_documents"] == 2 and result["n_clusters"] == 1
    assert result["n_boot"] == 0


def test_crossed_control_clusters_do_not_require_identical_documents_to_score_equally():
    from obfuscation_attacks.evaluation.sda_stream import _crossed_control_interval
    rows = []
    for key in range(2):
        for doc, cluster, hits in (("a", "c1", 8), ("b", "c1", 6), ("c", "c2", 9)):
            rows.append({"source_sha256": doc, "source_cluster": cluster,
                         "control_replica": key, "accuracy": {"numerator": hits, "denominator": 10},
                         "baselines": {"frequency_prior": {"numerator": 5}}})
    result = _crossed_control_interval(rows, "frequency_prior", seed=3)
    assert result["n_documents"] == 3 and result["n_clusters"] == 2
    assert result["difference"] == pytest.approx((8 + 6 + 9 - 15) / 30)
    assert "heuristic" in result["document_independence"]


def test_control_runner_groups_near_duplicate_documents_before_bootstrap():
    corpus = _corpus(seed=67)
    original = _ids(corpus[16]).tolist()
    edited = original.copy(); edited[-1] = (edited[-1] + 1) % VOCAB
    similar = " ".join(map(str, edited))
    out = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
               control_texts=[corpus[16], similar], validation_texts=corpus[18:20],
               control_keys=2)
    control = out["positive_control"]
    assert control["document_clustering"]["n_documents"] == 2
    assert control["document_clustering"]["n_clusters"] == 1
    assert all(item["status"] == "not_evaluated" and item["n_boot"] == 0
               for item in control["comparisons"].values())
    assert control["passed"] is None


def test_stream_uses_same_cache_choice_for_validation_control_and_target():
    corpus = _corpus(seed=73)
    kwargs = dict(method="sda2023", control_texts=corpus[16:18],
                  validation_texts=corpus[18:20], validation_beams=[1, 2],
                  control_keys=2, split_seed=4)
    full = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16], **kwargs)
    cached = _run(FakeDeployment(), _private_texts(n_docs=1), corpus[10:16],
                  decode_cache=True, **kwargs)
    assert full["validation"]["selected_beam"] == cached["validation"]["selected_beam"]
    assert full["rows"][0]["recovered"] == cached["rows"][0]["recovered"]
    assert cached["rows"][0]["decode"]["cache_mode"] == "kv_incremental"
    assert [row["recovered"] for row in full["positive_control"]["rows"]] == [
        row["recovered"] for row in cached["positive_control"]["rows"]]
    assert cached["decode_cache"] == "kv_incremental"
