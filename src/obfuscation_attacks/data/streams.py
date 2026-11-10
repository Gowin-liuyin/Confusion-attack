"""Multi-request stream contract (plan scenario S-C / P0-9 linkage).

Every earlier contract in this package models *one* request: one ``AttackSample``
becomes one capability-gated ``AttackContext``. The plan's cross-request attacks
(TFMA over a request log, exact/approximate linkage, rotation scan) observe a
*stream*, so they need a separate, equally gated contract:

* what the server stores per request: the symbol stream it received (token
  ciphertext, or the clustered symbols / raw vectors of a continuous wire),
  the plaintext ids *only for scoring*, and
* request metadata: session/tenant ids, arrival order and the rotation epoch the
  request belongs to.

The metadata gate encodes the plan's D13 decision directly:

* ``REQUEST_METADATA`` granted  -> **SH-P** (protocol-aware, assignment-blind).
  The attacker knows K/N and that rotation exists, so it can group requests by
  session/epoch and reason about key families by symbol equality - without ever
  seeing a key value, an assignment, or the current key.
* ``REQUEST_METADATA`` absent   -> **SH-B** (rotation-blind). The attacker can
  only pool all requests globally; the plan requires this to be reported as the
  weaker sensitivity variant, never as the main table.

Ground truth (``plaintext_ids``) stays on the stream object, exactly like
``AttackSample.plaintext_token_ids``; :func:`build_stream_context` never copies
it into the attacker view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .contracts import Capability, CapabilityError


@dataclass
class RequestRecord:
    """One request as the server logged it."""

    request_id: int
    symbols: np.ndarray  # (l,) int - cipher ids, or encoder-agnostic symbol clusters
    vectors: Optional[np.ndarray] = None  # (l, d) float - wire vectors (approx. linkage)
    session_id: Optional[str] = None
    epoch: Optional[int] = None
    key_id: Optional[str] = None
    arrival: int = 0


@dataclass(frozen=True)
class PublicRequestRecord:
    """A copied, capability-filtered request visible to an attacker."""
    request_id: int
    symbols: np.ndarray
    vectors: Optional[np.ndarray] = None
    session_id: Optional[str] = None
    epoch: Optional[int] = None
    key_id: Optional[str] = None
    arrival: int = 0


@dataclass
class RequestStream:
    """A logged stream of requests plus the ground truth used only for scoring."""

    requests: list[RequestRecord]
    plaintext_ids: list[np.ndarray]  # (l,) int per request - scoring only
    vocab_size: int
    wire: str = "token_ciphertext"  # "token_ciphertext" | "embedding"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.requests) != len(self.plaintext_ids):
            raise ValueError(
                f"{len(self.requests)} requests but {len(self.plaintext_ids)} plaintext rows"
            )
        for rec, truth in zip(self.requests, self.plaintext_ids):
            rec.symbols = np.asarray(rec.symbols, dtype=int).ravel()
            if rec.vectors is not None:
                rec.vectors = np.asarray(rec.vectors, dtype=float)
                if rec.vectors.ndim != 2 or rec.vectors.shape[0] != rec.symbols.size:
                    raise ValueError(
                        f"request {rec.request_id}: vectors {rec.vectors.shape} do not align "
                        f"with {rec.symbols.size} symbols"
                    )
            truth = np.asarray(truth, dtype=int).ravel()
            if truth.size != rec.symbols.size:
                raise ValueError(
                    f"request {rec.request_id}: {truth.size} plaintext ids vs "
                    f"{rec.symbols.size} symbols"
                )

    # -- derived, attacker-visible summaries --------------------------------
    def num_requests(self) -> int:
        return len(self.requests)

    def num_positions(self) -> int:
        return int(sum(r.symbols.size for r in self.requests))

    def describe(self) -> dict:
        return {
            "wire": self.wire,
            "num_requests": self.num_requests(),
            "num_positions": self.num_positions(),
            "vocab_size": self.vocab_size,
            "has_metadata": all(
                r.session_id is not None or r.epoch is not None for r in self.requests
            ),
        }


@dataclass
class StreamContext:
    """Capability-gated view of a :class:`RequestStream` handed to stream attacks."""

    granted: frozenset[Capability]
    descriptor: Mapping[str, Any]

    _records: Optional[tuple[PublicRequestRecord, ...]] = None
    reference_token_counts: Optional[Mapping[int, int]] = None
    reference_corpus: Optional[list] = None
    attacker_model: Optional[Any] = None
    reference_embeddings: Optional[np.ndarray] = None

    def has(self, cap: Capability) -> bool:
        return cap in self.granted

    def require(self, *caps: Capability) -> None:
        missing = [c for c in caps if c not in self.granted]
        if missing:
            raise CapabilityError(
                "scheme does not expose required capability: " + ", ".join(c.value for c in missing)
            )

    # -- gated accessors -----------------------------------------------------
    def stream(self, *, require_vectors: bool = False) -> tuple[PublicRequestRecord, ...]:
        """The logged requests.

        Symbol access requires ``TOKEN_CIPHERTEXT``. Vector access requires
        ``EMBEDDINGS``; when symbols are unavailable the returned zero array is
        only a length placeholder and must not be treated as observed symbols.
        """
        if not (
            self.has(Capability.TOKEN_CIPHERTEXT) or self.has(Capability.EMBEDDINGS)
        ):
            self.require(Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS)
        if require_vectors:
            self.require(Capability.EMBEDDINGS)
        else:
            self.require(Capability.TOKEN_CIPHERTEXT)
        if self._records is None:
            raise CapabilityError("symbol stream granted but no requests present")
        return self._records

    def request_metadata(self) -> list[dict]:
        """Session / epoch / arrival metadata (SH-P); raises without the capability."""
        self.require(Capability.REQUEST_METADATA)
        if self._records is None:
            raise CapabilityError("REQUEST_METADATA granted but no requests present")
        return [
            {
                "request_id": r.request_id,
                "session_id": r.session_id,
                "epoch": r.epoch,
                "arrival": r.arrival,
            }
            for r in self._records
        ]

    def knows_rotation(self) -> bool:
        """True in the SH-P setting, False for the rotation-blind SH-B variant."""
        return self.has(Capability.REQUEST_METADATA)


def build_stream_context(
    scheme,
    stream: RequestStream,
    *,
    attacker_model: Optional[Any] = None,
    reference_token_counts: Optional[Mapping[int, int]] = None,
    reference_corpus: Optional[list] = None,
    reference_embeddings: Optional[np.ndarray] = None,
) -> StreamContext:
    """Build the gated stream view.

    Only payloads whose capability the *scheme* exposes are attached - the same
    choke-point rule as :func:`obfuscation_attacks.schemes.base.build_context`.
    Ground truth (``plaintext_ids``) is never copied.
    """
    granted = scheme.exposed_capabilities()
    ctx = StreamContext(
        granted=granted,
        descriptor=stream.describe(),
        reference_token_counts=reference_token_counts,
        reference_corpus=reference_corpus,
        attacker_model=attacker_model,
        reference_embeddings=reference_embeddings,
    )
    if Capability.TOKEN_CIPHERTEXT in granted or Capability.EMBEDDINGS in granted:
        records = []
        metadata = Capability.REQUEST_METADATA in granted
        for index, record in enumerate(stream.requests):
            symbols = (np.array(record.symbols, copy=True) if Capability.TOKEN_CIPHERTEXT in granted
                       else np.zeros(len(record.symbols), dtype=np.int64))
            symbols.flags.writeable = False
            vectors = None
            if Capability.EMBEDDINGS in granted and record.vectors is not None:
                vectors = np.array(record.vectors, copy=True)
                vectors.flags.writeable = False
            records.append(PublicRequestRecord(
                request_id=record.request_id if metadata else index,
                symbols=symbols, vectors=vectors,
                session_id=record.session_id if metadata else None,
                epoch=record.epoch if metadata else None,
                key_id=record.key_id if metadata else None,
                arrival=record.arrival if metadata else index,
            ))
        ctx._records = tuple(records)
    if Capability.REQUEST_METADATA not in granted:
        ctx.descriptor = {key: value for key, value in ctx.descriptor.items() if key != "has_metadata"}
    return ctx


def realigned(records: Sequence[RequestRecord], epochs: Sequence[int]) -> list[RequestRecord]:
    """Copy ``records`` with the given rotation epochs (helper for scenarios/tests)."""
    out: list[RequestRecord] = []
    for rec, ep in zip(records, epochs):
        out.append(
            RequestRecord(
                request_id=rec.request_id,
                symbols=rec.symbols,
                vectors=rec.vectors,
                session_id=rec.session_id,
                epoch=int(ep),
                key_id=rec.key_id,
                arrival=rec.arrival,
            )
        )
    return out
