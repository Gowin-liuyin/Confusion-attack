"""Acceptance tests for F02: the LM rewrite never sees secret-decoded text."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.evaluation import real_remaining
from obfuscation_attacks.schemes.ab_deployment import WirePublic
from obfuscation_attacks.schemes.public_view import PublicDeployment, assert_public

pytestmark = pytest.mark.unit

V, D = 8, 4
IDS = np.array([2, 5, 1, 7], dtype=np.int64)


def _fixture():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(V, D)).astype(np.float32)
    public = PublicDeployment(
        arm="ab", scenario="S-public", artifact="fixture",
        vocab_size=V, hidden_size=D, n_layers=2, layers=(1,), max_len=None,
        tokenizer_id="fixture", E_server=E, E_clean=E,
        decode_fn=lambda ids: " ".join(str(int(i)) for i in ids),
        allow_clean_weights=True,
    )
    wire = WirePublic(
        arm="ab", request_id=0, seq_len=len(IDS), wire_vectors=E[IDS], layers=(1,),
    )
    return public, wire, E


def test_lm_input_uses_the_attack_mapping_not_the_secret_inverse(monkeypatch):
    public, wire, E = _fixture()
    assert assert_public(public)
    assert not hasattr(public, "inv_perm")
    captured: list[str] = []

    def fake_lm(pub, text, **kwargs):
        captured.append(text)
        return text

    monkeypatch.setattr(real_remaining, "lm_rewrite", fake_lm)
    mapping = {int(i): int(i) for i in range(V)}
    out = real_remaining.run_c14(
        public, [wire], stage2_mapping=mapping, max_len=len(IDS), verbose=False
    )
    assert captured == ["2 5 1 7"]
    assert out["rows"][0]["read"] == "2 5 1 7"
    assert not any(key.startswith("oracle") for key in out["rows"][0])
    assert out["oracle_bleu4"] is None


def test_uncovered_mapping_stays_in_the_denominator(monkeypatch):
    public, wire, E = _fixture()
    captured: list[str] = []
    monkeypatch.setattr(
        real_remaining, "lm_rewrite",
        lambda pub, text, **kwargs: captured.append(text) or text,
    )
    mapping = {int(i): int(i) for i in range(V)}
    del mapping[5]
    out = real_remaining.run_c14(
        public, [wire], stage2_mapping=mapping, max_len=len(IDS), verbose=False
    )
    assert out["rows"][0]["coverage"] < 1.0
    assert "UNK" in captured[0] or "?" in captured[0]


def test_unprocessed_tail_stays_in_lm_input_and_denominator(monkeypatch):
    public, wire, _ = _fixture()
    monkeypatch.setattr(real_remaining, "lm_rewrite", lambda pub, text, **kwargs: text)
    mapping = {int(i): int(i) for i in range(V)}
    out = real_remaining.run_c14(public, [wire], stage2_mapping=mapping, max_len=2, verbose=False)
    row = out["rows"][0]
    assert row["coverage"] == out["coverage"] == 0.5
    assert row["n_positions"] == 4 and row["n_predicted_positions"] == 2
    assert row["read"].count("[UNK]") == 2


def test_identical_public_input_gives_identical_lm_input(monkeypatch):
    public, wire, E = _fixture()
    captured: list[str] = []
    monkeypatch.setattr(
        real_remaining, "lm_rewrite",
        lambda pub, text, **kwargs: captured.append(text) or text,
    )
    mapping = {int(i): int(i) for i in range(V)}
    real_remaining.run_c14(public, [wire], stage2_mapping=mapping, max_len=4, verbose=False)
    real_remaining.run_c14(public, [wire], stage2_mapping=mapping, max_len=4, verbose=False)
    assert captured[0] == captured[1] == "2 5 1 7"


def test_decoding_keeps_subword_runs_and_missing_positions():
    from types import SimpleNamespace

    pieces = {1: "Al", 2: "ice", 3: " lives"}
    public = SimpleNamespace(decode=lambda ids: "".join(pieces[i] for i in ids))
    assert real_remaining._decode_mapped(public, [1, 2, 3]) == "Alice lives"
    assert real_remaining._decode_mapped(public, [1, None, 3]) == "Al[UNK] lives"


def test_text_cosine_is_labeled_as_proxy_and_uses_the_whole_sequence():
    from types import SimpleNamespace
    from obfuscation_attacks.evaluation.real_generative import text_metrics

    table = np.asarray([[1., 0.], [0., 1.], [0., -1.]], dtype=np.float32)
    public = SimpleNamespace(encode_fn=True, encode=lambda text: [int(x) for x in text.split()],
                             table=lambda arm: table)
    first = " ".join(["0"] * 32 + ["1"])
    second = " ".join(["0"] * 32 + ["2"])
    result = text_metrics(public, first, second)
    assert "cosine" not in result
    assert result["mean_embedding_cosine_proxy"] < 1.0
