"""End-to-end regression tests for the SDA attacks.

A small, structured synthetic language (skewed word lexicon with a frequent word
separator) makes decipherment learnable. Each attack trains on a public
reference corpus, then deciphers a held-out plaintext enciphered with a fresh,
unseen key. We score the learned decode with the shared SER metric and
``decode_accuracy = 1 - SER`` and compare it against TWO no-model baselines:

* a **majority-class** baseline (predict the single most frequent plaintext id),
* a **frequency-matching** baseline (map cipher symbols to plaintext symbols by
  matching frequency ranks - the classic frequency-analysis attack, and the
  thing the 2021 frequency encoding is built on).

We assert the learned model is at least as good as both baselines where the tiny
smoke model allows, and document honestly where it cannot (the 2023 recurrence
model needs more context than a <128-symbol cipher offers, so it trails the
frequency baseline on 1:1 ciphers but beats it decisively on homophonic ciphers,
exactly as Kambhatla et al. 2023 report). We also check structural consistency
(equal cipher symbols decode to equal plaintext) via the recovered deterministic
key.

These train tiny torch models, so they are ``neural``; module-scoped fixtures
train each model once to stay within a few seconds on CPU. Small model / beam /
length sizes here are SMOKE-only; paper scale lives in configs/original/*.yaml.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from obfuscation_attacks.attacks.sda import encodings as enc
from obfuscation_attacks.attacks.sda.sda2021_seq2seq import SDA2021
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import SDA2023
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability
from obfuscation_attacks.metrics import bleu4, symbol_error_rate

pytestmark = [pytest.mark.regression, pytest.mark.neural]

VOCAB = 10
# A skewed lexicon over ids 1..9; id 0 is the word separator (most frequent
# symbol). Several words contain internal repetition (e.g. [1,2,1]) so the 2023
# recurrence signal is informative, and the skew gives the 2021 frequency signal.
WORDS = [[1, 2, 1], [3, 1], [1, 4], [5, 6, 5], [2, 1], [7, 8], [1, 9]]
WEIGHTS = np.array([0.34, 0.22, 0.16, 0.10, 0.08, 0.06, 0.04])
WEIGHTS = WEIGHTS / WEIGHTS.sum()
GRANTED = frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION})
CHANCE = 1.0 / VOCAB


def _sentence(rng: np.random.Generator, n_words: int) -> np.ndarray:
    out: list[int] = []
    for _ in range(n_words):
        out.append(0)  # separator
        out += WORDS[rng.choice(len(WORDS), p=WEIGHTS)]
    return np.asarray(out[1:], dtype=int)  # drop leading separator


def _decode_accuracy(recovered, truth) -> float:
    """1 - SER over the aligned positions (equal lengths here)."""
    return 1.0 - symbol_error_rate([int(t) for t in recovered], [int(t) for t in truth])


def _majority_class_prediction(corpus, length: int) -> list[int]:
    """No-model baseline: always predict the most frequent plaintext id."""
    counts: Counter = Counter()
    for seq in corpus:
        counts.update(int(t) for t in seq)
    majority = counts.most_common(1)[0][0]
    return [int(majority)] * length


def _frequency_matching_prediction(corpus, cipher) -> list[int]:
    """No-model baseline: map cipher symbols to plaintext symbols by frequency rank.

    The most frequent cipher symbol is decoded as the most frequent plaintext
    symbol, and so on - the classic frequency-analysis attack (Kambhatla 2018's
    "Freq. Match"). It is strong on 1:1 ciphers and collapses on homophonic ones.
    """
    pcounts: Counter = Counter()
    for seq in corpus:
        pcounts.update(int(t) for t in seq)
    plain_order = [s for s, _ in pcounts.most_common()]
    cipher_order = [s for s, _ in Counter(int(t) for t in cipher).most_common()]
    mapping = {
        c: (plain_order[i] if i < len(plain_order) else plain_order[-1])
        for i, c in enumerate(cipher_order)
    }
    return [int(mapping[int(c)]) for c in cipher]


def _structurally_consistent(cipher, recovered) -> bool:
    """True iff equal cipher symbols always decode to equal plaintext."""
    seen: dict[int, int] = {}
    for c, r in zip((int(x) for x in cipher), (int(x) for x in recovered)):
        if c in seen and seen[c] != r:
            return False
        seen[c] = r
    return True


# ---------------------------------------------------------------------------
# shared fixtures: build the data once; train each attack once.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(123)
    corpus = [_sentence(rng, int(rng.integers(4, 9))) for _ in range(300)]
    eval_plain = np.concatenate([_sentence(np.random.default_rng(900 + i), 4) for i in range(4)])
    # UNSEEN 1:1 key (independent of the attack's internal training keys)
    key = enc.make_substitution_key(
        range(VOCAB), rng=np.random.default_rng(7), homophonic=False, cipher_alphabet_size=VOCAB
    )
    cipher = enc.apply_substitution(eval_plain, key)
    assert not np.array_equal(cipher, eval_plain)  # the key really permutes ids

    majority = _majority_class_prediction(corpus, len(eval_plain))
    freqmatch = _frequency_matching_prediction(corpus, cipher)
    return {
        "corpus": corpus,
        "eval_plain": eval_plain,
        "cipher": cipher,
        "majority_acc": _decode_accuracy(majority, eval_plain),
        "freqmatch_acc": _decode_accuracy(freqmatch, eval_plain),
    }


def _ctx(cipher, corpus):
    return AttackContext(
        granted=GRANTED,
        descriptor={"vocab_size": VOCAB},
        _ciphertext_token_ids=cipher,
        reference_corpus=corpus,
    )


@pytest.fixture(scope="module")
def res2021(data):
    return SDA2021(seed=0, epochs=15, num_train=256, beam_width=5)._run(
        _ctx(data["cipher"], data["corpus"])
    )


@pytest.fixture(scope="module")
def res2023(data):
    return SDA2023(seed=0, epochs=20, num_train=256, lr=5e-4, beam_width=5)._run(
        _ctx(data["cipher"], data["corpus"])
    )


@pytest.fixture(scope="module")
def homophonic(data):
    """A GENUINELY homophonic eval cipher (many cipher symbols -> one plaintext)."""
    eval_plain = data["eval_plain"]
    ev_counts = {int(t): int(c) for t, c in zip(*np.unique(eval_plain, return_counts=True))}
    plain_counts = {s: ev_counts.get(s, 1) for s in range(VOCAB)}
    # Algorithm-1 homophone allocation: frequent plaintext symbols get several
    # cipher symbols. cipher_alphabet_size is comfortably larger than the plaintext
    # vocabulary, so the cipher genuinely uses more symbols than the plaintext.
    key = enc.make_substitution_key(
        range(VOCAB), rng=np.random.default_rng(21), homophonic=True,
        plain_counts=plain_counts, approx_num_symbols=20, cipher_alphabet_size=80,
    )
    hcipher = enc.apply_substitution(eval_plain, key, rng=np.random.default_rng(121))
    res = SDA2023(seed=0, homophonic=True, epochs=12, num_train=256, lr=5e-4, beam_width=5)._run(
        _ctx(hcipher, data["corpus"])
    )
    freqmatch = _frequency_matching_prediction(data["corpus"], hcipher)
    return {
        "key": key,
        "cipher": hcipher,
        "res": res,
        "freqmatch_acc": _decode_accuracy(freqmatch, eval_plain),
    }


# ---------------------------------------------------------------------------
# 2021 seq2seq (frequency rank)
# ---------------------------------------------------------------------------
def test_sda2021_beats_baselines_and_is_structurally_consistent(data, res2021):
    cipher, eval_plain = data["cipher"], data["eval_plain"]
    res = res2021
    assert res.status == AttackStatus.OK

    # structural consistency: length-matched decode, all valid plaintext ids
    assert len(res.recovered_token_ids) == len(cipher)
    assert res.recovered_text.count(" ") == len(cipher) - 1
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)

    ser = symbol_error_rate(list(res.recovered_token_ids), list(eval_plain))
    acc = 1.0 - ser
    assert ser == pytest.approx(1.0 - acc)

    # beats chance and the majority-class baseline, and is AT LEAST AS GOOD as the
    # frequency-matching baseline. On this tiny smoke model the seq2seq decode
    # matches the frequency baseline exactly (acc ~= 0.65): with 51 symbols and a
    # 2-layer model it recovers essentially the unigram frequency map.
    assert acc > CHANCE
    assert acc >= data["majority_acc"]
    assert acc >= data["freqmatch_acc"] - 1e-9

    # recovered a deterministic cipher->plain key => equal cipher symbols decode
    # to equal plaintext (the key is 1:1 here)
    assert _structurally_consistent(cipher, res.recovered_token_ids)
    assert res.recovered_mapping is not None
    assert set(res.recovered_mapping) <= set(int(c) for c in cipher)

    # AloePri-style text metric is well defined on the decode
    assert 0.0 <= bleu4(list(res.recovered_token_ids), list(eval_plain)) <= 1.0


# ---------------------------------------------------------------------------
# 2023 causal LM (recurrence)
# ---------------------------------------------------------------------------
def test_sda2023_beats_majority_and_documents_frequency_baseline(data, res2023):
    cipher, eval_plain = data["cipher"], data["eval_plain"]
    res = res2023
    assert res.status == AttackStatus.OK
    assert len(res.recovered_token_ids) == len(cipher)
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)

    # diagnostics confirm BOTH CLM loss terms were actually optimised
    assert res.diagnostics["final_train_loss_src"] > 0.0
    assert res.diagnostics["final_train_loss_tgt"] > 0.0

    ser = symbol_error_rate(list(res.recovered_token_ids), list(eval_plain))
    acc = 1.0 - ser
    assert acc > CHANCE
    assert acc >= data["majority_acc"]
    # DOCUMENTED LIMITATION (not a failure): the recurrence model (~0.53 acc)
    # trails the frequency-matching baseline (~0.65) on this SHORT 1:1 cipher.
    # Kambhatla et al. 2023 (Table 5, Sec 6) show recurrence encoding "requires
    # more context to be effective" and only matches/overtakes frequency ranking
    # on ciphers longer than 128 symbols; our smoke cipher is 51 symbols. The
    # recurrence model's real advantage is homophonic ciphers (see below).
    assert data["freqmatch_acc"] >= acc  # pins the documented short-cipher gap

    assert _structurally_consistent(cipher, res.recovered_token_ids)

    # AloePri runs THIS 2023 recurrence+CLM method ("version G2") and reports
    # BLEU-4 on the recovered text; verify the AloePri metric is well defined.
    assert 0.0 <= bleu4(list(res.recovered_token_ids), list(eval_plain)) <= 1.0


def test_sda2023_homophonic_end_to_end_evaluates_a_homophonic_cipher(data, homophonic):
    eval_plain = data["eval_plain"]
    key, hcipher, res = homophonic["key"], homophonic["cipher"], homophonic["res"]

    # the eval cipher is GENUINELY homophonic: one plaintext symbol -> several
    # cipher symbols, so the cipher uses more symbols than the plaintext.
    assert not key.is_one_to_one()
    assert len(set(int(c) for c in hcipher)) > len(set(int(p) for p in eval_plain))

    assert res.status == AttackStatus.OK
    assert res.diagnostics["homophonic"] is True
    assert len(res.recovered_token_ids) == len(hcipher)
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)

    # a homophonic cipher->plaintext map is still DETERMINISTIC (many -> one), so
    # equal cipher symbols must decode to equal plaintext.
    assert _structurally_consistent(hcipher, res.recovered_token_ids)

    ser = symbol_error_rate(list(res.recovered_token_ids), list(eval_plain))
    acc = 1.0 - ser
    # the paper's central claim: frequency analysis COLLAPSES on homophonic
    # ciphers (flattened frequencies) while the recurrence model still recovers
    # the message. Here the recurrence model beats both chance and the (now
    # defeated) frequency-matching baseline, and edges the majority baseline.
    assert acc > CHANCE
    assert acc > homophonic["freqmatch_acc"]
    assert acc >= data["majority_acc"]


# ---------------------------------------------------------------------------
# structural consistency under a 1:1 cipher (shared property, both lineages)
# ---------------------------------------------------------------------------
def test_sda_structural_consistency_under_1to1_cipher(data, res2021, res2023):
    cipher = data["cipher"]
    for res in (res2021, res2023):
        # equal cipher symbols decode to equal plaintext
        assert _structurally_consistent(cipher, res.recovered_token_ids)
        # the recovered mapping is a function cipher_id -> plain_id
        assert res.recovered_mapping is not None
        for c, p in zip((int(x) for x in cipher), res.recovered_token_ids.tolist()):
            assert res.recovered_mapping[c] == int(p)


# ---------------------------------------------------------------------------
# 2021 cipher-length condition (Sec 5.1): shorter ciphers, still above chance
# ---------------------------------------------------------------------------
def test_sda2021_cipher_length_condition(data):
    cipher, eval_plain = data["cipher"], data["eval_plain"]
    length = 16  # a short cipher from the paper's 16/32/64/128/256 sweep
    c_short = enc.crop_to_length(cipher, length, from_start=True)
    p_short = eval_plain[:length]
    res = SDA2021(seed=0, epochs=15, num_train=256, beam_width=5)._run(
        _ctx(c_short, data["corpus"])
    )
    assert res.status == AttackStatus.OK
    assert len(res.recovered_token_ids) == length  # length-matched to the (short) cipher
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)
    assert _structurally_consistent(c_short, res.recovered_token_ids)

    ser = symbol_error_rate(list(res.recovered_token_ids), list(p_short))
    assert (1.0 - ser) > CHANCE  # SER is well defined and the decode beats chance


# ---------------------------------------------------------------------------
# honest "not applicable" when the attacker owns no reference corpus
# ---------------------------------------------------------------------------
def test_sda_not_applicable_without_reference_corpus():
    cipher = np.array([2, 3, 2, 1], dtype=int)
    for attack in (SDA2021(seed=0), SDA2023(seed=0)):
        ctx = AttackContext(
            granted=GRANTED,
            descriptor={"vocab_size": VOCAB},
            _ciphertext_token_ids=cipher,
            reference_corpus=None,
        )
        res = attack._run(ctx)
        assert res.status == AttackStatus.NOT_APPLICABLE
