import numpy as np
import pytest
import copy
import json

from obfuscation_attacks.evaluation.sda_public import prepare_sda_observations, decode_sda_observations, decode_lm_key_observations, decoded_sda_tokens
from obfuscation_attacks.evaluation.sda_stream import fit_sda
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


def upload(surface, payload, rid, group=None, truncated=False):
    length = len(payload)
    return Upload(surface=surface, payload=payload, request_id=rid, seq_len=length,
                  full_len=length, group_id=group, truncated=truncated,
                  metadata={"truth": "SHOULD NEVER BE READ"})


def test_public_ids_are_directly_read_without_truth_or_reobservation():
    records = [upload(UploadSurface.IDS, np.array([4, 7, 4]), 3, "g")]
    prepared = prepare_sda_observations(records)
    assert prepared["rows"][0]["symbols"] == [4, 7, 4]
    result = decode_sda_observations({}, prepared, visible_plaintext=True)
    assert result["rows"] == [{"request_id": 3, "tokens": [4, 7, 4]}]
    assert "SHOULD NEVER BE READ" not in str(prepared)


def test_lm_key_search_uses_only_fingerprinted_public_observations():
    prepared = prepare_sda_observations([upload(UploadSurface.IDS, np.array([4, 7, 4]), 3, "g")])
    result = decode_lm_key_observations(prepared, [0, 1], lambda prefix: np.log([0.7, 0.3]),
                                        beam_width=2, seed=5)
    assert result["mode"].endswith("probe") and result["group_joint"] is False
    assert result["rows"][0]["tokens"][0] == result["rows"][0]["tokens"][2]
    assert result["processed_observations_sha256"] == prepared["processed_observations_sha256"]
    assert decoded_sda_tokens(result) == {3: result["rows"][0]["tokens"]}
    assert "SHOULD NEVER BE READ" not in str(result)
    altered = copy.deepcopy(prepared)
    altered["rows"][0]["symbols"][0] = 8
    with pytest.raises(ValueError, match="fingerprint"):
        decode_lm_key_observations(altered, [0, 1], lambda prefix: np.log([0.7, 0.3]),
                                   beam_width=2, seed=5)


def test_lm_key_search_joint_uses_visible_groups_and_resets_context():
    records = [upload(UploadSurface.IDS, np.array([4, 4, 7]), 1, "a"),
               upload(UploadSurface.IDS, np.array([7, 7]), 2, "b"),
               upload(UploadSurface.IDS, np.array([4, 4]), 3, "a"),
               upload(UploadSurface.IDS, np.array([4]), 4),
               upload(UploadSurface.IDS, np.array([4]), 5)]
    prepared = prepare_sda_observations(records)
    def lm(prefix):
        return np.log([0.8, 0.2] if not prefix or prefix[-1] == 0 else [0.1, 0.9])
    result = decode_lm_key_observations(prepared, [0, 1], lm, beam_width=4, seed=5, group_joint=True)
    assert [row["request_id"] for row in result["rows"]] == [1, 2, 3, 4, 5]
    assert [group["request_ids"] for group in result["groups"]] == [[1, 3], [2], [4], [5]]
    assert result["rows"][0]["tokens"][0] == result["rows"][2]["tokens"][0]
    assert result["groups"][0]["decode"]["sequence_lengths"] == [3, 2]
    assert result["groups"][0]["decode"]["context_reset"] == "per_sequence"
    assert result["rows"][0]["tokens"][-1] != result["rows"][1]["tokens"][0]
    assert decoded_sda_tokens(result) == {row["request_id"]: row["tokens"] for row in result["rows"]}


def test_public_vector_knn_keeps_real_topk_and_full_table():
    table = np.array([[0., 0.], [2., 0.], [4., 0.]], dtype=np.float32)
    vectors = np.array([[0.1, 0.], [3.9, 0.]], dtype=np.float32)
    prepared = prepare_sda_observations([upload(UploadSurface.VECTORS, vectors, 1)],
                                        vector_method="nearest_table", candidate_table=table,
                                        candidate_k=2, query_chunk=1)
    row = prepared["rows"][0]
    assert row["candidates"] == [[0, 1], [2, 1]]
    assert row["symbols"] == [0, 2]
    assert prepared["frontend"]["candidate_k"] == 2


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_public_vector_knn_handles_extreme_values(metric):
    huge = np.finfo(np.float32).max / 2
    record = upload(UploadSurface.VECTORS, np.array([[huge, huge]], dtype=np.float32), 1)
    table = np.array([[huge, huge], [0, 0]], dtype=np.float32)
    result = prepare_sda_observations([record], vector_method="nearest_table", candidate_table=table,
                                      candidate_k=1, metric=metric)
    assert result["rows"][0]["candidates"] == [[0]]


def test_public_vector_l2_does_not_lose_small_differences():
    record = upload(UploadSurface.VECTORS, np.array([[10000., 10000.]], dtype=np.float32), 1)
    table = np.array([[10001., 10000.], [10000., 10000.], [9999., 10000.]], dtype=np.float32)
    result = prepare_sda_observations([record], vector_method="nearest_table", candidate_table=table,
                                      candidate_k=3, metric="l2")
    assert result["rows"][0]["candidates"] == [[1, 0, 2]]


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_public_vector_knn_matches_full_precision_across_table_blocks(metric):
    rng = np.random.default_rng(14)
    table = rng.normal(size=(2051, 4)).astype(np.float32)
    table[-1] = table[0]
    vectors = np.stack([table[0], rng.normal(size=4).astype(np.float32)])
    result = prepare_sda_observations([upload(UploadSurface.VECTORS, vectors, 1)],
             vector_method="nearest_table", candidate_table=table, candidate_k=5, metric=metric)
    expected = []
    for query in vectors.astype(np.float64):
        data = table.astype(np.float64)
        if metric == "l2":
            distances = np.sum((data - query) ** 2, axis=1)
        else:
            distances = -(data / np.maximum(np.linalg.norm(data, axis=1), 1e-12)[:, None]) @ (
                query / max(np.linalg.norm(query), 1e-12))
        expected.append(np.lexsort((np.arange(len(table)), distances))[:5].tolist())
    assert result["rows"][0]["candidates"] == expected


def test_public_vector_groups_do_not_merge_and_joint_decode_is_consistent():
    records = [upload(UploadSurface.VECTORS, np.array([[1., 0.], [0., 1.]]), 1, "a"),
               upload(UploadSurface.VECTORS, np.array([[0., 1.], [1., 0.]]), 2, "a"),
               upload(UploadSurface.VECTORS, np.array([[1., 0.]]), 3, "b")]
    prepared = prepare_sda_observations(records)
    a, b, c = [r["symbols"] for r in prepared["rows"]]
    assert a == b[::-1] and c[0] not in a
    bundle = fit_sda([[1, 2, 1, 2]], [a, b, c], vocab_size=8,
                     params={"num_train": 4, "epochs": 1, "batch_size": 2,
                             "d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16}, seed=1)
    decoded = decode_sda_observations(bundle, prepared, beam_width=1, group_joint=True)
    first, second = decoded["group_joint"]
    assert first["request_ids"] == [1, 2] and second["request_ids"] == [3]
    assert first["rows"][0]["tokens"] == first["rows"][1]["tokens"][::-1]


def test_text_requires_explicit_public_tokenizer_and_alignment_is_unverified():
    record = upload(UploadSurface.TEXT, "12 9", 1)
    with pytest.raises(ValueError, match="tokenizer"):
        prepare_sda_observations([record])
    result = prepare_sda_observations([record], text_tokenizer=lambda value: [int(x) for x in value.split()])
    assert result["rows"][0]["symbols"] == [12, 9]
    assert result["frontend"]["text_alignment"] == "unverified"


def test_rejects_wrong_or_partial_surfaces():
    with pytest.raises(ValueError, match="truncated"):
        prepare_sda_observations([upload(UploadSurface.IDS, np.array([1]), 1, truncated=True)])
    with pytest.raises(ValueError, match="mix"):
        prepare_sda_observations([upload(UploadSurface.IDS, np.array([1]), 1),
                                  upload(UploadSurface.VECTORS, np.array([[0.]]), 2)])
    with pytest.raises(ValueError, match="request IDs must be integers"):
        prepare_sda_observations([upload(UploadSurface.IDS, np.array([1]), 1),
                                  upload(UploadSurface.IDS, np.array([2]), "1")])
    with pytest.raises(ValueError, match="integer"):
        prepare_sda_observations([upload(UploadSurface.IDS, np.array([1.1]), 1)])
    with pytest.raises(ValueError, match="requires a public candidate table"):
        prepare_sda_observations([upload(UploadSurface.VECTORS, np.array([[1.]]), 1)],
                                 vector_method="nearest_table")


def test_public_upload_rejects_impossible_lengths_and_out_of_range_ids():
    record = upload(UploadSurface.IDS, np.array([1, 2]), 1)
    record.full_len = 1
    with pytest.raises(ValueError, match="metadata"):
        prepare_sda_observations([record])
    record.full_len = 2
    record.seq_len = True
    with pytest.raises(ValueError, match="metadata"):
        prepare_sda_observations([record])
    with pytest.raises(ValueError, match="int64 range"):
        prepare_sda_observations([upload(UploadSurface.IDS, np.array([2**63], dtype=np.uint64), 2)])


def test_frozen_upload_fingerprint_ignores_private_metadata_but_tracks_wire():
    one = upload(UploadSurface.IDS, np.array([1, 2, 3]), 9, "g")
    first = prepare_sda_observations([one])["public_uploads_sha256"]
    one.metadata["truth"] = "different unused value"
    assert prepare_sda_observations([one])["public_uploads_sha256"] == first
    one.payload[0] = 4
    assert prepare_sda_observations([one])["public_uploads_sha256"] != first


def test_processed_observation_fingerprint_binds_symbols_candidates_and_frontend():
    table = np.array([[0.], [2.]], dtype=np.float32)
    record = upload(UploadSurface.VECTORS, np.array([[0.1], [1.9]]), 9, "g")
    prepared = prepare_sda_observations([record], vector_method="nearest_table",
                                        candidate_table=table, candidate_k=2)
    changed_table = prepare_sda_observations([record], vector_method="nearest_table",
                                            candidate_table=table + 0.25, candidate_k=2)
    assert changed_table["public_uploads_sha256"] == prepared["public_uploads_sha256"]
    assert changed_table["processed_observations_sha256"] != prepared["processed_observations_sha256"]
    restored = json.loads(json.dumps(prepared))
    assert restored == prepared
    mutations = (
        lambda row: row["rows"][0]["symbols"].__setitem__(0, 1),
        lambda row: row["rows"][0]["candidates"][0].reverse(),
        lambda row: row["frontend"].__setitem__("metric", "cosine"),
        lambda row: row["rows"][0].__setitem__("group_id", "other"),
        lambda row: row.__setitem__("public_uploads_sha256", "0" * 64),
    )
    for mutate in mutations:
        altered = copy.deepcopy(restored)
        mutate(altered)
        with pytest.raises(ValueError, match="fingerprint"):
            decode_sda_observations({}, altered, visible_plaintext=True)
    clean = prepare_sda_observations([upload(UploadSurface.IDS, np.array([1, 2]), 1)])
    decoded = decode_sda_observations({}, json.loads(json.dumps(clean)), visible_plaintext=True)
    assert decoded["processed_observations_sha256"] == clean["processed_observations_sha256"]


def test_stream_consumes_frozen_upload_without_reobserving():
    from obfuscation_attacks.evaluation.sda_stream import run_sda_stream
    class ScoringSide:
        vocab_size = 20
        def encode(self, text):
            return np.array([int(x) for x in text.split()], dtype=np.int64)
        def observe(self, *args, **kwargs):
            raise AssertionError("must not generate another observation")
    records = [upload(UploadSurface.IDS, np.array([8, 9, 8, 9, 8]), 42, "fixed")]
    prepared = prepare_sda_observations(records)
    result = run_sda_stream(
        ScoringSide(), {42: "1 2 3 4 5"}, ["6 7 8 9 10", "10 11 12 13 14", "14 15 16 17 18"],
        frozen_uploads=records, positive_control=False, num_train=4, epochs=1,
        beam_width=1, training_params={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16})
    assert result["public_uploads_sha256"] == prepared["public_uploads_sha256"]
    assert result["processed_observations_sha256"] == prepared["processed_observations_sha256"]
    assert result["rows"][0]["request_id"] == 42
    assert result["token_accuracy"]["denominator"] == 5
    assert result["scoring_join"] == "request_id"


def test_frozen_stream_scores_by_id_instead_of_mapping_order(monkeypatch):
    from obfuscation_attacks.evaluation import sda_stream
    class ScoringSide:
        vocab_size = 32
        def encode(self, text):
            return np.array([int(x) for x in text.split()], dtype=np.int64)
    records = [upload(UploadSurface.IDS, np.array([8] * 5), 42, "g"),
               upload(UploadSurface.IDS, np.array([9] * 5), 75, "g")]
    monkeypatch.setattr(sda_stream, "decode_sda", lambda bundle, cipher, **kwargs: {
        "tokens": [1, 2, 3, 4, 5] if cipher[0] == 8 else [21, 22, 23, 24, 25]})
    result = sda_stream.run_sda_stream(
        ScoringSide(), {75: "21 22 23 24 25", 42: "1 2 3 4 5"},
        ["11 12 13 14 15", "16 17 18 19 20"], frozen_uploads=records,
        positive_control=False, num_train=4, epochs=1, beam_width=1,
        scoring_clusters={42: {"document": "d1", "session": "s1", "key_group": "k1"},
                          75: {"document": "d2", "session": "s2", "key_group": "k2"}},
        training_params={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16})
    assert result["token_accuracy"]["value"] == 1.0
    assert result["clustered_token_scoring"]["token_accuracy"]["value"] == 1.0
    assert result["clustered_token_scoring"]["confidence_interval"]["n_boot"] == 2000
    assert [row["request_id"] for row in result["rows"]] == [42, 75]
    for bad in (["1 2 3 4 5", "21 22 23 24 25"], {42: "1 2 3 4 5"},
                {42: "1 2 3 4 5", 75: "21 22 23 24 25", 76: "1"}):
        with pytest.raises(ValueError, match="keyed by request ID|exactly match"):
            sda_stream.run_sda_stream(ScoringSide(), bad, [], frozen_uploads=records)


def test_stream_rejects_unaligned_text_scoring():
    from obfuscation_attacks.evaluation.sda_stream import run_sda_stream
    class ScoringSide:
        vocab_size = 20
    with pytest.raises(ValueError, match="character-alignment"):
        run_sda_stream(ScoringSide(), {0: "1 2"}, ["3 4"],
                       frozen_uploads=[upload(UploadSurface.TEXT, "1 2", 0)],
                       text_tokenizer=lambda text: [int(x) for x in text.split()])


def test_frozen_vector_stream_reuses_same_topk_public_observations():
    from obfuscation_attacks.evaluation.sda_stream import run_sda_stream
    class ScoringSide:
        vocab_size = 20
        def encode(self, text):
            return np.array([int(x) for x in text.split()], dtype=np.int64)
        def observe(self, *args, **kwargs):
            raise AssertionError("must not regenerate the frozen upload")
    table = np.array([[0., 0.], [2., 0.], [4., 0.]], dtype=np.float32)
    vectors = np.array([[0.1, 0.], [3.9, 0.], [2.1, 0.], [3.8, 0.], [0.2, 0.]], dtype=np.float32)
    records = [upload(UploadSurface.VECTORS, vectors, 75, "g")]
    prepared = prepare_sda_observations(records, vector_method="nearest_table",
                                        candidate_table=table, candidate_k=2)
    result = run_sda_stream(
        ScoringSide(), {75: "1 2 3 4 5"}, ["6 7 8 9 10", "10 11 12 13 14", "14 15 16 17 18"],
        frozen_uploads=records, symbol_frontend="nearest_table", candidate_table=table,
        candidate_k=2, positive_control=False, num_train=4, epochs=1, beam_width=1,
        training_params={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16})
    assert result["public_uploads_sha256"] == prepared["public_uploads_sha256"]
    assert result["rows"][0]["frontend_candidates"] == prepared["rows"][0]["candidates"]
    assert result["observation_frontend"]["candidate_table_sha256"] == prepared["frontend"]["candidate_table_sha256"]
    assert result["frontend_scoring"]["scope"] == "scoring_only"


def test_text_surface_scores_characters_without_inventing_token_alignment():
    from obfuscation_attacks.evaluation.sda_stream import run_sda_stream
    class ScoringSide:
        vocab_size = 30
        def encode(self, text):
            return np.array([ord(char) - 96 for char in text], dtype=np.int64)
        def observe(self, *args, **kwargs):
            raise AssertionError("must reuse frozen uploads")
    records = [upload(UploadSurface.TEXT, "gghhi", 42, "g")]
    result = run_sda_stream(
        ScoringSide(), {42: "abcde"}, ["fghij", "klmno", "pqrst"],
        frozen_uploads=records, text_tokenizer=lambda text: [ord(char) - 96 for char in text],
        text_decoder=lambda ids: "".join(chr(96 + value) if 1 <= value <= 26 else "?" for value in ids),
        text_entities={42: [{"start": 0, "end": 2, "type": "person"}]},
        positive_control=False, num_train=4, epochs=1, beam_width=1,
        training_params={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_ff": 16},
    )
    assert result["token_accuracy"]["status"] == "not_evaluated"
    assert result["token_accuracy"]["value"] is None
    assert result["frontend_scoring"]["status"] == "not_evaluated"
    assert result["text_scoring"]["n_requests"] == 1
    assert result["text_scoring"]["word_recovery"]["denominator"] == 1
    assert result["text_scoring"]["entity_recovery"]["denominator"] == 1
    assert result["text_scoring"]["reference_token_topk"]["status"] == "not_evaluated"
    assert "abcde" not in str(result["text_scoring"])
