"""Acceptance tests for the W1 key schedule (protocol §1, §8.1).

The server may see an opaque group id, never a seed, a root or a key it can
replay. These tests fail if any of those properties regress.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from obfuscation_attacks.schemes.key_schedule import KeySchedule

pytestmark = pytest.mark.unit

ROOT = bytes(range(32))


def test_derive_is_deterministic_and_domain_separated():
    a = KeySchedule(ROOT)
    b = KeySchedule(ROOT)
    assert np.array_equal(a.derive(0, 0), b.derive(0, 0))
    assert not np.array_equal(a.derive(0, 0), a.derive(1, 0))
    # protocol §8.2: the same key pool index is one key across rotation epochs
    assert np.array_equal(a.derive(0, 0), a.derive(0, 1))
    vec = a.derive(3, 2)
    assert vec.shape == (32,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    # Non-degenerate and close to the encoder's training distribution (standard normal).
    assert 0.5 < float(np.std(vec)) < 2.0


def test_root_must_be_256_bits():
    with pytest.raises(ValueError):
        KeySchedule(b"short")
    with pytest.raises(ValueError):
        KeySchedule(bytes(64))


def test_default_root_is_secret_and_not_shared():
    a, b = KeySchedule(), KeySchedule()
    assert not np.array_equal(a.derive(0, 0), b.derive(0, 0))


def test_group_id_is_opaque_stable_and_key_bound():
    ks = KeySchedule(ROOT)
    gid = ks.group_id(7, 0)
    assert gid.startswith("g-")
    assert gid != "7" and "index=7" not in gid
    assert gid != ks.derive(7, 0).tobytes().hex()
    assert gid == ks.group_id(7, 0)
    assert gid != ks.group_id(8, 0)
    # the epoch is rotation metadata, not part of key identity
    assert gid == ks.group_id(7, 1)
    assert len(gid) >= 16


def test_public_metadata_has_only_group_and_epoch():
    ks = KeySchedule(ROOT)
    meta = ks.public_metadata(2, 5)
    assert set(meta) == {"group_id", "epoch"}
    assert meta["group_id"] == ks.group_id(2, 5) == ks.group_id(2, 0)
    assert meta["epoch"] == 5
    blob = json.dumps(meta, sort_keys=True)
    assert ROOT.hex() not in blob


def test_repr_does_not_print_the_root_or_a_derived_key():
    ks = KeySchedule(ROOT)
    text = repr(ks) + str(ks)
    assert ROOT.hex() not in text
    assert ks.derive(0, 0).tobytes().hex() not in text


def test_next_index_is_monotone_per_schedule():
    a = KeySchedule(ROOT)
    assert [a.next_index(), a.next_index(), a.next_index()] == [0, 1, 2]
    b = KeySchedule(ROOT)
    assert b.next_index() == 0
