import numpy as np
import pytest

from obfuscation_attacks.schemes.mechanism import AloePriToyMechanism


def test_noise_is_static_per_row_and_changes_with_permutation():
    table = np.arange(24, dtype=float).reshape(6, 4)
    mechanism = AloePriToyMechanism(table, alpha_e=1, A=np.eye(4), seed=19)
    key_a = np.arange(6)
    key_b = np.roll(key_a, 1)
    first = mechanism.obfuscate_embeddings([1, 2, 1], key_a)
    reordered = mechanism.obfuscate_embeddings([2, 1], key_a)
    assert np.array_equal(first[0], first[2])
    assert np.array_equal(first[0], reordered[1])
    assert np.array_equal(first[1], reordered[0])
    noise_a = mechanism.obfuscate_embeddings([0], key_a)[0] - table[key_a[0]]
    noise_b = mechanism.obfuscate_embeddings([1], key_b)[0] - table[key_b[1]]
    assert key_a[0] == key_b[1]
    assert not np.array_equal(noise_a, noise_b)


def test_invalid_key_and_token_are_rejected():
    mechanism = AloePriToyMechanism(np.eye(3), A=np.eye(3))
    with pytest.raises(ValueError, match="permutation"):
        mechanism.obfuscate_embeddings([0], [0, 0, 2])
    with pytest.raises(ValueError, match="outside"):
        mechanism.obfuscate_embeddings([-1], [0, 1, 2])
