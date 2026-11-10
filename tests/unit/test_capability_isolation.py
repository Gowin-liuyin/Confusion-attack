"""Capability-gating / information-isolation tests (Goal section 9).

An attack may only read what the scheme exposes. A scheme that withholds a
capability must cause the dependent attack to report *not applicable*, and the
gated accessor must raise rather than leaking data; the attacker never receives
ground-truth fields through the context.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.nn_knn.euclidean_knn import EuclideanKNN
from obfuscation_attacks.attacks.vma.weight_rowsort import WeightRowSortVMA
from obfuscation_attacks.data.contracts import (
    AttackContext,
    AttackStatus,
    Capability,
    CapabilityError,
)

pytestmark = pytest.mark.unit


def _embeddings_only_ctx():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((10, 4))
    H = E[np.array([1, 2, 3])] + 0.01 * rng.standard_normal((3, 4))
    return AttackContext(
        granted=frozenset({Capability.EMBEDDINGS}),
        descriptor={"vocab_size": 10},
        _obfuscated_embeddings=H,
        reference_embeddings=E,
        # withheld even though present on a hypothetical sample:
        _known_weights={"X": np.eye(3), "Y": np.eye(3)},
    )


def test_attack_runs_when_capability_present():
    res = EuclideanKNN(top_k=3).run(_embeddings_only_ctx())
    assert res.status == AttackStatus.OK


def test_attack_not_applicable_when_capability_missing():
    # weight VMA needs WHITE_BOX_WEIGHTS, which this scheme does not expose
    res = WeightRowSortVMA().run(_embeddings_only_ctx())
    assert res.status == AttackStatus.NOT_APPLICABLE
    assert "WHITE_BOX_WEIGHTS" in res.status_reason


def test_gated_accessor_raises_without_capability():
    ctx = _embeddings_only_ctx()
    assert ctx.has(Capability.EMBEDDINGS)
    assert not ctx.has(Capability.WHITE_BOX_WEIGHTS)
    with pytest.raises(CapabilityError):
        ctx.weight("X")  # present in payload but NOT exposed -> must raise
    with pytest.raises(CapabilityError):
        ctx.require(Capability.WHITE_BOX_WEIGHTS)


def test_context_has_no_ground_truth_fields():
    ctx = _embeddings_only_ctx()
    for leaky in ("plaintext_token_ids", "secret_permutation", "secret_matrices"):
        assert not hasattr(ctx, leaky)


def test_build_context_descriptor_excludes_content_mask():
    """content_mask is derived from the true tokens -> it must not reach the attack.

    The Clean-ASR split still works because the scorer reads the mask from the
    sample directly. special_token_mask (structural: BOS/EOS/pad) stays exposed.
    """
    import numpy as np

    from obfuscation_attacks.data.contracts import AttackSample
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    sample = AttackSample(
        tokenizer_id="toy", model_id="toy", seq_len=3, vocab_size=5,
        obfuscated_embeddings=np.zeros((3, 4)),
        reference_embeddings=np.zeros((5, 4)),
        plaintext_token_ids=np.array([0, 1, 2]),
        special_token_mask=np.array([True, False, False]),
        content_mask=np.array([False, True, True]),
    )
    scheme = GenericScheme("t", frozenset({Capability.EMBEDDINGS}))
    ctx = build_context(scheme, sample)
    assert "content_mask" not in ctx.descriptor          # ground-truth-derived: withheld
    assert "special_token_mask" in ctx.descriptor        # structural: exposed
