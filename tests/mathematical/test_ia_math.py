"""Mathematical tests for the IA family (FORMULA_LEDGER IA-EDNN/GATE/ATTN).

Two layers of tests:

* *Invariant* tests on the bare helpers (``difference_representation``,
  ``gate_invariant``, ``whitening_invariant``) that pin the FORMULA_LEDGER
  identities and their counterexamples.
* *Real-transform* tests that build AloePri's actual covariant obfuscation
  (wider embedding key matrix with the ``W*_embed = W_e + alpha_e E`` noise
  structure for Gate-IA; per-RoPE-block invertible maps with optional
  head/block permutation for Attn-IA) and drive the attack classes end to end,
  so the invariants are checked against the transforms the paper really uses -
  including the ``alpha_e = 0`` precondition that AloePri breaks with noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.ia.attn import (
    AttnIA,
    rope_block_columns,
    stacked_invariant,
    whitening_invariant,
)
from obfuscation_attacks.attacks.ia.ednn import difference_representation, lshift
from obfuscation_attacks.attacks.ia.gate import GateIA, gate_invariant
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability

pytestmark = pytest.mark.mathematical


# ===========================================================================
# helpers
# ===========================================================================
def _invertible(rng: np.random.Generator, k: int, min_det: float = 0.3) -> np.ndarray:
    """A well-conditioned random k x k matrix."""
    while True:
        A = rng.standard_normal((k, k))
        if abs(float(np.linalg.det(A))) > min_det:
            return A


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    k = sum(b.shape[0] for b in blocks)
    M = np.zeros((k, k))
    o = 0
    for b in blocks:
        bs = b.shape[0]
        M[o : o + bs, o : o + bs] = b
        o += bs
    return M


def _wb_ctx(weights: dict) -> AttackContext:
    return AttackContext(
        granted=frozenset({Capability.WHITE_BOX_WEIGHTS}),
        descriptor={},
        _known_weights=weights,
    )


# ===========================================================================
# EDNN: coordinate-difference invariant of the glide reflection
# ===========================================================================
def _glide(e, a, b):
    ones = np.ones_like(e)
    l = ones * a
    t = ones * b
    return e - 2.0 * (e @ l) / (l @ l) * l + t


def test_glide_preserves_coordinate_differences():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(8)
    ep = _glide(e, a=rng.standard_normal() + 1.0, b=rng.standard_normal())
    # e'[k1]-e'[k2] == e[k1]-e[k2] for all pairs
    for k1 in range(8):
        for k2 in range(8):
            assert (ep[k1] - ep[k2]) == pytest.approx(e[k1] - e[k2], abs=1e-9)


def test_lshift_is_cyclic_keeps_last_coordinate():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(lshift(x), np.array([2.0, 3.0, 4.0, 1.0]))  # wrap, not drop


def test_difference_representation_invariant_under_glide():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(10)
    ep = _glide(e, a=2.3, b=-0.7)
    assert np.allclose(difference_representation(ep), difference_representation(e), atol=1e-9)


def test_difference_representation_wraps_first_and_last():
    # IA-EDNN-3: lshift is cyclic, so the fingerprint keeps d coordinates and the
    # last difference wraps (e[-1] - e[0]); it is never dropped or zero-padded.
    e = np.array([1.0, 2.0, 4.0, 7.0])
    diff = difference_representation(e)
    assert diff.shape == e.shape  # no coordinate dropped
    assert diff[0] == pytest.approx(e[0] - e[1])  # interior: e[k] - e[k+1]
    assert diff[-1] == pytest.approx(e[-1] - e[0])  # wrap: last - first
    assert diff[-1] == pytest.approx(7.0 - 1.0)


def test_non_all_equal_normal_breaks_the_invariant():
    # The invariant is specific to l = 1*a (all-equal coordinates). A general
    # reflection normal must NOT be claimed to preserve coordinate differences
    # (the required "arbitrary normal vector" counterexample).
    rng = np.random.default_rng(2)
    e = rng.standard_normal(8)
    normal = rng.standard_normal(8)  # arbitrary direction, not all-equal
    ep = e - 2.0 * (e @ normal) / (normal @ normal) * normal
    assert not np.allclose(difference_representation(ep), difference_representation(e), atol=1e-6)


def test_non_all_equal_normal_breaks_even_with_translation():
    # Even a full glide (reflection + translation) with an arbitrary normal fails:
    # translation by 1*b cancels in differences, so only the all-equal normal can
    # preserve them.
    rng = np.random.default_rng(22)
    e = rng.standard_normal(9)
    normal = rng.standard_normal(9)
    t = np.ones_like(e) * 0.5
    ep = e - 2.0 * (e @ normal) / (normal @ normal) * normal + t
    assert not np.allclose(difference_representation(ep), difference_representation(e), atol=1e-6)


# ===========================================================================
# Gate-IA: gate-weighted average invariant
# ===========================================================================
def test_gate_average_invariant_under_column_permutation():
    rng = np.random.default_rng(3)
    vocab, d, hf = 5, 4, 7
    E = rng.standard_normal((vocab, d))
    W = rng.standard_normal((d, hf))
    zffn = rng.permutation(hf)
    W_obf = W[:, zffn]  # FFN gate column permutation
    inv_plain = gate_invariant(E, W)
    inv_obf = gate_invariant(E, W_obf)
    # Avg over the FFN dimension is invariant to the column permutation
    assert np.allclose(inv_plain, inv_obf, atol=1e-9)


def _build_gate_transform(rng, *, vocab, d, hf, h_expand, layers):
    """Build AloePri's real Gate obfuscation pieces (App D.1 / 5.2.4 / Table 9).

    Returns (W_e, W_gate, W_gate_obf, P_hat, Q_gate, perm, noise) such that the
    observed product is Pi (W_e + alpha_e*noise) P_hat . Q_gate W_gate Zffn with
    P_hat Q_gate = I_d, matching the Table 9 row  We Wgate -> Pi W*embed Wgate Zffn.
    """
    W_e = rng.standard_normal((vocab, d))
    dw = d + 2 * h_expand  # P_hat is the d x (d+2h) wider key matrix
    P_hat = rng.standard_normal((d, dw))
    Q_gate = P_hat.T @ np.linalg.inv(P_hat @ P_hat.T)  # right inverse: P_hat Q_gate = I_d
    assert np.allclose(P_hat @ Q_gate, np.eye(d), atol=1e-8)
    perm = rng.permutation(vocab)  # obf token i corresponds to plain token perm[i]
    noise = rng.standard_normal((vocab, d))
    W_gate = rng.standard_normal((layers, d, hf))
    W_gate_obf = np.empty((layers, dw, hf))
    for l in range(layers):
        zffn = rng.permutation(hf)  # FFN column permutation Zffn
        W_gate_obf[l] = (Q_gate @ W_gate[l])[:, zffn]
    return W_e, W_gate, W_gate_obf, P_hat, Q_gate, perm, noise


def _obf_embeddings(W_e, noise, P_hat, perm, alpha_e):
    E_star = W_e + alpha_e * noise  # W*_embed = W_e + alpha_e * E_embed
    return (E_star @ P_hat)[perm]  # Pi W*_embed P_hat


def test_gate_real_transform_exact_at_alpha0_and_broken_by_noise():
    # Requirement C2: with the REAL covariant transform the gate-average invariant
    # is exact ONLY at alpha_e = 0; any embedding noise (alpha_e > 0) breaks it.
    rng = np.random.default_rng(30)
    vocab, d, hf, h_expand, layers = 12, 6, 10, 4, 4
    W_e, W_gate, W_gate_obf, P_hat, _Q, perm, noise = _build_gate_transform(
        rng, vocab=vocab, d=d, hf=hf, h_expand=h_expand, layers=layers
    )
    plain_inv = gate_invariant(W_e, W_gate)

    E_obf0 = _obf_embeddings(W_e, noise, P_hat, perm, alpha_e=0.0)
    obf_inv0 = gate_invariant(E_obf0, W_gate_obf)
    # obf token i matches plaintext token perm[i] EXACTLY at alpha_e = 0
    assert np.allclose(obf_inv0, plain_inv[perm], atol=1e-9)

    E_obf1 = _obf_embeddings(W_e, noise, P_hat, perm, alpha_e=1.0)
    obf_inv1 = gate_invariant(E_obf1, W_gate_obf)
    assert not np.allclose(obf_inv1, plain_inv[perm], atol=1e-6)  # noise disrupts it


def test_gate_ia_attack_recovers_mapping_at_alpha0():
    rng = np.random.default_rng(31)
    vocab, d, hf, h_expand, layers = 12, 6, 10, 4, 4
    W_e, W_gate, W_gate_obf, P_hat, _Q, perm, noise = _build_gate_transform(
        rng, vocab=vocab, d=d, hf=hf, h_expand=h_expand, layers=layers
    )
    E_obf0 = _obf_embeddings(W_e, noise, P_hat, perm, alpha_e=0.0)
    ctx = _wb_ctx({"E": W_e, "E_obf": E_obf0, "W_gate": W_gate, "W_gate_obf": W_gate_obf})
    res = GateIA(top_k=5).run(ctx)
    assert res.status is AttackStatus.OK
    # recovered_mapping: obf token -> plaintext token == perm
    assert all(res.recovered_mapping[i] == int(perm[i]) for i in range(vocab))


def test_gate_ia_attack_degrades_under_embedding_noise():
    rng = np.random.default_rng(32)
    vocab, d, hf, h_expand, layers = 16, 6, 10, 4, 5
    W_e, W_gate, W_gate_obf, P_hat, _Q, perm, noise = _build_gate_transform(
        rng, vocab=vocab, d=d, hf=hf, h_expand=h_expand, layers=layers
    )
    E_obf1 = _obf_embeddings(W_e, noise, P_hat, perm, alpha_e=1.0)
    ctx = _wb_ctx({"E": W_e, "E_obf": E_obf1, "W_gate": W_gate, "W_gate_obf": W_gate_obf})
    res = GateIA(top_k=5).run(ctx)
    assert res.status is AttackStatus.OK
    acc = sum(res.recovered_mapping[i] == int(perm[i]) for i in range(vocab)) / vocab
    assert acc < 1.0  # the alpha_e = 1.0 defense prevents exact recovery


# ===========================================================================
# Attn-IA: whitening quadratic-form invariant
# ===========================================================================
def test_whitening_invariant_under_invertible_block_transform():
    rng = np.random.default_rng(4)
    n, block, nb = 12, 2, 3
    Q = rng.standard_normal((n, block * nb))
    # block-diagonal invertible transform
    M = np.zeros((block * nb, block * nb))
    for j in range(nb):
        A = rng.standard_normal((block, block)) + 2 * np.eye(block)
        M[j * block : (j + 1) * block, j * block : (j + 1) * block] = A
    Qt = Q @ M
    inv_a, blocks_a = whitening_invariant(Q, block)
    inv_b, blocks_b = whitening_invariant(Qt, block)
    assert blocks_a == blocks_b
    assert np.allclose(inv_a, inv_b, atol=1e-6)  # Mahalanobis form is invariant


def test_whitening_skips_singular_block_without_pseudo_inverse():
    rng = np.random.default_rng(5)
    n, block, nb = 8, 2, 3
    Q = rng.standard_normal((n, block * nb))
    Q[:, 0:block] = 0.0  # make block 0's Gram singular
    inv, used_blocks = whitening_invariant(Q, block)
    assert 0 not in used_blocks  # singular block skipped, never pseudo-inverted
    assert inv.shape[1] == len(used_blocks) == nb - 1


def test_whitening_exact_under_real_rope_block_transform():
    # Real AloePri query obfuscation: each RoPE 2x2 block is rotated (R_qk) and
    # scaled (H_qk), i.e. an invertible 2x2 map. block_size=2 is exactly invariant.
    rng = np.random.default_rng(40)
    n, nb = 14, 3
    dq = 2 * nb
    Q = rng.standard_normal((n, dq))
    Qt = Q.copy()
    for j in range(nb):
        theta = rng.uniform(0.0, 2 * np.pi)
        s = rng.uniform(0.5, 2.0)
        c, si = np.cos(theta), np.sin(theta)
        A = s * np.array([[c, -si], [si, c]])  # scaling * rotation, invertible 2x2
        Qt[:, 2 * j : 2 * j + 2] = Q[:, 2 * j : 2 * j + 2] @ A
    inv_a, _ = whitening_invariant(Q, 2)
    inv_b, _ = whitening_invariant(Qt, 2)
    assert np.allclose(inv_a, inv_b, atol=1e-8)


def test_whitening_single_column_breaks_under_rope_rotation():
    # block_size=1 treats each column alone; a RoPE rotation mixes the two columns
    # of a block, so the single-column form is NOT invariant (block_size=2 is).
    rng = np.random.default_rng(41)
    n = 12
    Q = rng.standard_normal((n, 2))
    theta = 0.7
    c, si = np.cos(theta), np.sin(theta)
    Qt = Q @ np.array([[c, -si], [si, c]])
    inv2a, _ = whitening_invariant(Q, 2)
    inv2b, _ = whitening_invariant(Qt, 2)
    assert np.allclose(inv2a, inv2b, atol=1e-9)  # 2-column block: invariant
    inv1a, blocks1 = whitening_invariant(Q, 1)
    inv1b, _ = whitening_invariant(Qt, 1)
    assert blocks1 == [0, 1]
    assert not np.allclose(inv1a, inv1b, atol=1e-6)  # single column: broken


def test_whitening_half_split_pairing_invariant():
    # rotate_half layout: the RoPE block pairs column i with column i + d/2.
    rng = np.random.default_rng(42)
    n, d = 10, 8
    half = d // 2
    Q = rng.standard_normal((n, d))
    Qt = Q.copy()
    for i in range(half):
        A = _invertible(rng, 2)
        Qt[:, [i, i + half]] = Q[:, [i, i + half]] @ A
    inv_a, ka = whitening_invariant(Q, 2, pairing="half_split")
    inv_b, kb = whitening_invariant(Qt, 2, pairing="half_split")
    assert ka == kb == list(range(half))
    assert np.allclose(inv_a, inv_b, atol=1e-8)
    # The same transform does NOT look invariant under the wrong (adjacent) pairing.
    adj_a, _ = whitening_invariant(Q, 2, pairing="adjacent")
    adj_b, _ = whitening_invariant(Qt, 2, pairing="adjacent")
    assert not np.allclose(adj_a, adj_b, atol=1e-6)


def test_rope_block_columns_layouts_and_partial_block_rejected():
    assert rope_block_columns(6, 2, "adjacent") == [(0, 1), (2, 3), (4, 5)]
    assert rope_block_columns(8, 2, "half_split") == [(0, 4), (1, 5), (2, 6), (3, 7)]
    # partial last block must be rejected, not silently used
    with pytest.raises(ValueError):
        rope_block_columns(5, 2, "adjacent")  # 5 not a multiple of 2
    with pytest.raises(ValueError):
        rope_block_columns(7, 2, "half_split")  # odd width
    with pytest.raises(ValueError):
        whitening_invariant(np.zeros((4, 5)), 2)  # width 5, partial block


# --- Attn-IA attack class (multi-head / key side / layouts) ----------------
def _head_obf(rng, Qh, perm, block=2):
    nb = Qh.shape[1] // block
    M = _block_diag([_invertible(rng, block) for _ in range(nb)])
    return (Qh @ M)[perm]


def test_attn_ia_attack_accepts_stacked_multihead_and_recovers_mapping():
    # Defect #1: a (heads, tokens, d_head) stack must NOT crash and must use
    # every head's blocks. Each head gets its own invertible RoPE-block map.
    rng = np.random.default_rng(50)
    H, n, nb, block = 3, 16, 3, 2
    d_head = block * nb
    Q = rng.standard_normal((H, n, d_head))
    perm = rng.permutation(n)
    Q_obf = np.stack([_head_obf(rng, Q[h], perm, block) for h in range(H)], axis=0)
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q_obf})
    res = AttnIA(block_size=block).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["num_heads_q"] == H
    assert res.diagnostics["num_invariant_dims"] == H * nb
    assert all(res.recovered_mapping[i] == int(perm[i]) for i in range(n))


def test_attn_ia_attack_single_head_2d_still_works():
    rng = np.random.default_rng(51)
    n, nb, block = 18, 4, 2
    Q = rng.standard_normal((n, block * nb))
    perm = rng.permutation(n)
    Q_obf = _head_obf(rng, Q, perm, block)
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q_obf})
    res = AttnIA(block_size=block).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["num_heads_q"] == 1
    assert all(res.recovered_mapping[i] == int(perm[i]) for i in range(n))


def test_attn_ia_attack_uses_key_side_when_exposed():
    # Defect #1: the attack can also exploit W_key blocks when they are exposed.
    rng = np.random.default_rng(52)
    n, nb, block = 14, 3, 2
    perm = rng.permutation(n)
    Q = rng.standard_normal((n, block * nb))
    K = rng.standard_normal((n, block * nb))
    Q_obf = _head_obf(rng, Q, perm, block)
    K_obf = _head_obf(rng, K, perm, block)
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q_obf, "K": K, "K_obf": K_obf})
    res = AttnIA(block_size=block).run(ctx)
    assert res.status is AttackStatus.OK
    assert res.diagnostics["used_key_side"] is True
    assert res.diagnostics["num_invariant_dims"] == 2 * nb  # Q blocks + K blocks
    assert all(res.recovered_mapping[i] == int(perm[i]) for i in range(n))


def test_attn_ia_attack_not_applicable_when_no_invertible_block():
    Q = np.zeros((6, 4))
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q.copy()})
    res = AttnIA(block_size=2).run(ctx)
    assert res.status is AttackStatus.NOT_APPLICABLE


def test_attn_ia_attack_fails_cleanly_on_partial_block():
    rng = np.random.default_rng(53)
    Q = rng.standard_normal((6, 5))  # width 5: a partial last 2-block
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q.copy()})
    res = AttnIA(block_size=2).run(ctx)
    assert res.status is AttackStatus.FAILED
    assert "multiple" in res.status_reason or "partial" in res.status_reason


def test_attn_ia_attack_head_permutation_defense_degrades_recovery():
    # AloePri's defense: permuting heads breaks the positional block alignment the
    # invariant relies on, so exact recovery is prevented.
    rng = np.random.default_rng(54)
    H, n, nb, block = 4, 16, 3, 2
    d_head = block * nb
    Q = rng.standard_normal((H, n, d_head))
    perm = rng.permutation(n)
    obf_heads = [_head_obf(rng, Q[h], perm, block) for h in range(H)]
    head_perm = rng.permutation(H)  # inter-head permutation (Algorithm 2)
    Q_obf = np.stack([obf_heads[head_perm[h]] for h in range(H)], axis=0)
    ctx = _wb_ctx({"Q": Q, "Q_obf": Q_obf})
    res = AttnIA(block_size=block).run(ctx)
    assert res.status is AttackStatus.OK
    acc = sum(res.recovered_mapping[i] == int(perm[i]) for i in range(n)) / n
    assert acc < 1.0


def test_stacked_invariant_keys_identify_group_head_block():
    rng = np.random.default_rng(55)
    n, nb, block = 8, 2, 2
    Q = rng.standard_normal((2, n, block * nb))  # 2 heads
    K = rng.standard_normal((n, block * nb))  # single head
    inv, keys = stacked_invariant({"Q": Q, "K": K}, block_size=block)
    assert inv.shape == (n, 2 * nb + nb)  # 2 heads of Q + 1 head of K
    assert ("Q", 0, 0) in keys and ("Q", 1, 2) in keys and ("K", 0, 0) in keys
