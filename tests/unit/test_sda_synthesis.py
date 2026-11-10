import numpy as np
import pytest

from obfuscation_attacks.attacks.sda import encodings as enc
from obfuscation_attacks.attacks.sda.sda2021_seq2seq import synthesize_pairs as make_2021
from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import synthesize_pairs as make_2023
from obfuscation_attacks.evaluation.sda_stream import fit_sda


@pytest.mark.parametrize("make,encode", [(make_2021, enc.frequency_rank_encode),
                                         (make_2023, enc.recurrence_encode)])
def test_invariant_encoding_matches_independent_permutations(make, encode):
    corpus = [np.array([4, 1, 4, 2, 1, 7, 8, 4, 7])]
    for seed in range(30):
        rng = np.random.default_rng(seed)
        key = enc.make_substitution_key(range(12), rng=rng, homophonic=False,
                                        cipher_alphabet_size=12)
        assert encode(enc.apply_substitution(corpus[0], key)) == encode(corpus[0])
    pairs = make(corpus, 12, 8, np.random.default_rng(3), min_len=4, max_len=9,
                 synthesis_mode="invariant")
    assert all(source == encode(plain) for source, plain in pairs)
    repeat = make(corpus, 12, 8, np.random.default_rng(3), min_len=4, max_len=9,
                  synthesis_mode="invariant")
    assert pairs == repeat


def test_legacy_rng_consumption_is_preserved_and_modes_are_distinct():
    corpus = [np.arange(24) % 10, np.arange(30) % 10]
    for make in (make_2021, make_2023):
        default = make(corpus, 10, 10, np.random.default_rng(2), min_len=4, max_len=10)
        legacy = make(corpus, 10, 10, np.random.default_rng(2), min_len=4, max_len=10,
                      synthesis_mode="legacy")
        fast = make(corpus, 10, 10, np.random.default_rng(2), min_len=4, max_len=10,
                    synthesis_mode="invariant")
        assert default == legacy
        assert fast != legacy


def test_invariant_synthesis_rejects_unsupported_cases():
    corpus = [np.array([1, 2, 1, 2])]
    with pytest.raises(ValueError, match="zero noise"):
        make_2021(corpus, 8, 1, np.random.default_rng(0), noise_rate=0.1,
                  synthesis_mode="invariant")
    with pytest.raises(ValueError, match="homophonic"):
        make_2023(corpus, 8, 1, np.random.default_rng(0), homophonic=True,
                  synthesis_mode="invariant")
    with pytest.raises(ValueError, match="vocabulary"):
        make_2021([np.array([1, 9])], 8, 1, np.random.default_rng(0), synthesis_mode="invariant")
    with pytest.raises(ValueError, match="noiseless one-to-one"):
        fit_sda(corpus, [[1]], vocab_size=8, method="sda2023",
                params={"homophonic": True, "synthesis_mode": "invariant"})


def test_homophonic_allocation_uses_only_observed_frequencies(monkeypatch):
    original = enc.make_substitution_key
    calls = []
    def capture(symbols, **kwargs):
        calls.append((list(symbols), kwargs["plain_counts"], kwargs["approx_num_symbols"]))
        return original(symbols, **kwargs)
    monkeypatch.setattr(enc, "make_substitution_key", capture)
    corpus = [np.array([1, 1, 1, 99999])]
    pairs = make_2023(corpus, 100000, 1, np.random.default_rng(0), homophonic=True,
                      min_len=4, max_len=4)
    assert calls == [([1, 99999], {1: 3, 99999: 1}, 4)]
    assert pairs[0][1] == corpus[0].tolist()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_homophonic_alphabet_size_must_be_positive_integer(value):
    with pytest.raises(ValueError, match="approx_num_symbols"):
        make_2023([np.array([0, 1, 0, 1])], 2, 1, np.random.default_rng(0),
                  homophonic=True, approx_num_symbols=value)
