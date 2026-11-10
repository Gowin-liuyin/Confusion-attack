import numpy as np

from obfuscation_attacks.attacks.ima.cross_instance import CrossInstanceInverter
from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability


def test_cross_instance_requires_heldout_document():
    ctx = AttackContext(
        granted=frozenset({Capability.EMBEDDINGS}),
        descriptor={"vocab_size": 4},
        _obfuscated_embeddings=np.ones((2, 3)),
        attacker_obfuscator=lambda ids: np.ones((len(ids), 3)),
        reference_corpus=[[0, 1]],
    )
    result = CrossInstanceInverter().run(ctx)
    assert result.status == AttackStatus.FAILED
    assert "insufficient_data" in result.status_reason
