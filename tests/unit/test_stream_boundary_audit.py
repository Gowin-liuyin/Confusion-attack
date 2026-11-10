import numpy as np
import pytest

from obfuscation_attacks.data.contracts import Capability, CapabilityError
from obfuscation_attacks.data.streams import RequestRecord, RequestStream, build_stream_context


class Scheme:
    def __init__(self, *caps):
        self.caps = frozenset(caps)

    def exposed_capabilities(self):
        return self.caps


def test_context_filters_and_copies_ungranted_fields():
    record = RequestRecord(99, np.array([4, 5]), vectors=np.ones((2, 3)),
                           session_id="private", epoch=4, key_id="group", arrival=9)
    stream = RequestStream([record], [np.array([1, 2])], vocab_size=8)
    view = build_stream_context(Scheme(Capability.TOKEN_CIPHERTEXT), stream)
    exposed = view.stream()[0]
    assert exposed.vectors is None and exposed.session_id is None and exposed.epoch is None
    assert exposed.key_id is None and exposed.request_id == 0
    assert "has_metadata" not in view.descriptor
    record.symbols[0] = 7
    assert exposed.symbols.tolist() == [4, 5]
    with pytest.raises(ValueError):
        exposed.symbols[0] = 0


def test_vector_only_context_does_not_expose_original_symbols():
    record = RequestRecord(1, np.array([4, 5]), vectors=np.ones((2, 3)))
    stream = RequestStream([record], [np.array([1, 2])], vocab_size=8)
    view = build_stream_context(Scheme(Capability.EMBEDDINGS), stream)
    assert view.stream(require_vectors=True)[0].symbols.tolist() == [0, 0]
    assert view.stream(require_vectors=True)[0].vectors.shape == (2, 3)
    with pytest.raises(CapabilityError):
        view.stream()
