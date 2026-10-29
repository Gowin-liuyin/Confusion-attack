"""End-to-end regression tests for the SDA attacks.

A small, structured synthetic language (skewed word lexicon with a frequent word
separator) makes decipherment learnable. Each attack trains on a public
reference corpus, then deciphers a held-out plaintext that was enciphered with a
fresh, unseen key. We assert the recovery beats chance and that the output is
structurally consistent with the ciphertext (same length, valid plaintext ids).

These train tiny torch models, so they are ``neural``; they stay within a few
seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.sda import encodings as enc
from obfuscation_attacks.attacks.sda.sda2021_seq2seq import SDA2021
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import SDA2023
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability

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


def _per_position_accuracy(recovered: np.ndarray, truth: np.ndarray) -> float:
    rec = np.asarray(recovered)
    m = min(len(rec), len(truth))
    return float((rec[:m] == truth[:m]).mean()) if m else 0.0


@pytest.fixture(scope="module")
def cipher_fixture():
    """Public training corpus plus a held-out plaintext enciphered with a fresh key."""
    rng = np.random.default_rng(123)
    corpus = [_sentence(rng, int(rng.integers(4, 9))) for _ in range(300)]
    # held-out plaintext (not a member of the training corpus list)
    eval_plain = np.concatenate([_sentence(np.random.default_rng(900 + i), 4) for i in range(4)])
    # UNSEEN 1:1 key (independent of the attack's internal training keys)
    key = enc.make_substitution_key(
        range(VOCAB), rng=np.random.default_rng(7), homophonic=False, cipher_alphabet_size=VOCAB
    )
    cipher = enc.apply_substitution(eval_plain, key)
    assert not np.array_equal(cipher, eval_plain)  # the key really permutes ids
    return corpus, eval_plain, cipher


def test_sda2021_decipher_beats_chance(cipher_fixture):
    corpus, eval_plain, cipher = cipher_fixture
    ctx = AttackContext(
        granted=GRANTED,
        descriptor={"vocab_size": VOCAB},
        _ciphertext_token_ids=cipher,
        reference_corpus=corpus,
    )
    res = SDA2021(seed=0, epochs=15, num_train=256, beam_width=5)._run(ctx)

    assert res.status == AttackStatus.OK
    # structural consistency: length-matched decode, all valid plaintext ids
    assert len(res.recovered_token_ids) == len(cipher)
    assert res.recovered_text.count(" ") == len(cipher) - 1
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)

    acc = _per_position_accuracy(res.recovered_token_ids, eval_plain)
    assert acc > CHANCE
    assert acc >= 0.17  # comfortably above the 0.10 chance rate


def test_sda2023_decipher_beats_chance(cipher_fixture):
    corpus, eval_plain, cipher = cipher_fixture
    ctx = AttackContext(
        granted=GRANTED,
        descriptor={"vocab_size": VOCAB},
        _ciphertext_token_ids=cipher,
        reference_corpus=corpus,
    )
    res = SDA2023(seed=0, epochs=20, num_train=256, lr=5e-4, beam_width=5)._run(ctx)

    assert res.status == AttackStatus.OK
    assert len(res.recovered_token_ids) == len(cipher)
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)
    # diagnostics confirm BOTH loss terms were actually optimised
    assert res.diagnostics["final_train_loss_src"] > 0.0
    assert res.diagnostics["final_train_loss_tgt"] > 0.0

    acc = _per_position_accuracy(res.recovered_token_ids, eval_plain)
    assert acc > CHANCE
    assert acc >= 0.17


def test_sda2023_homophonic_runs_and_is_structurally_consistent(cipher_fixture):
    corpus, _eval_plain, cipher = cipher_fixture
    ctx = AttackContext(
        granted=GRANTED,
        descriptor={"vocab_size": VOCAB},
        _ciphertext_token_ids=cipher,
        reference_corpus=corpus,
    )
    # homophonic training augmentation (Algorithm 1). We only require that the
    # method runs end-to-end and produces a structurally valid decipherment.
    res = SDA2023(
        seed=0, homophonic=True, epochs=6, num_train=128, lr=5e-4, beam_width=3
    )._run(ctx)

    assert res.status == AttackStatus.OK
    assert res.diagnostics["homophonic"] is True
    assert len(res.recovered_token_ids) == len(cipher)
    assert np.all(res.recovered_token_ids >= 0) and np.all(res.recovered_token_ids < VOCAB)


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
