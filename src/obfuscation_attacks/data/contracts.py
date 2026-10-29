"""Core data contracts shared by every attack and the custom-scheme interface.

The design enforces the information-isolation rule of the Goal (section 9): an
attack may only read information that the obfuscation scheme actually *exposes*
through its declared capabilities. Ground-truth fields used for scoring
(``plaintext_token_ids``, ``secret_permutation``) live on :class:`AttackSample`,
which the *runner* holds; they are never handed to the attack through the
capability-gated :class:`AttackContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


class Capability(str, Enum):
    """What the defended/obfuscated computation exposes to the attacker.

    These are the capabilities an :class:`AttackContext` may grant. An attack
    declares the subset it *requires*; the runner marks the attack
    ``not_applicable`` when the scheme does not expose all of them.
    """

    TOKEN_CIPHERTEXT = "TOKEN_CIPHERTEXT"
    EMBEDDINGS = "EMBEDDINGS"
    HIDDEN_STATES = "HIDDEN_STATES"
    MODEL_REPLAY = "MODEL_REPLAY"
    WHITE_BOX_WEIGHTS = "WHITE_BOX_WEIGHTS"
    KNOWN_OBFUSCATION_ALGORITHM = "KNOWN_OBFUSCATION_ALGORITHM"
    DETERMINISTIC_PERMUTATION = "DETERMINISTIC_PERMUTATION"


class CapabilityError(RuntimeError):
    """Raised when an attack accesses information the scheme did not expose."""


class AttackStatus(str, Enum):
    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass
class AttackSample:
    """A single evaluation unit produced by an obfuscation scheme.

    Fields are grouped into three tiers:

    * *descriptive* - safe metadata (ids, seq length, special-token mask);
    * *exposed* - information a scheme may reveal to an attacker, each guarded
      by a :class:`Capability`;
    * *ground truth* - used only by the runner for scoring and never passed to
      an attack through :class:`AttackContext`.
    """

    # --- descriptive metadata -------------------------------------------------
    tokenizer_id: str
    model_id: str
    seq_len: int
    vocab_size: int
    layer: Optional[int] = None
    key_id: Optional[str] = None
    seed: Optional[int] = None
    text_source: Optional[str] = None
    data_split: Optional[str] = None
    special_token_mask: Optional[np.ndarray] = None  # 1D bool, True where special
    content_mask: Optional[np.ndarray] = None  # 1D bool, True for content (non stop-word)
    plaintext: Optional[str] = None  # ground-truth text (scoring only)

    # --- exposed information (gated by capability) ---------------------------
    ciphertext_token_ids: Optional[np.ndarray] = None  # TOKEN_CIPHERTEXT
    obfuscated_embeddings: Optional[np.ndarray] = None  # EMBEDDINGS
    obfuscated_hidden_states: Optional[np.ndarray] = None  # HIDDEN_STATES
    known_weights: Optional[Mapping[str, np.ndarray]] = None  # WHITE_BOX_WEIGHTS

    # --- attacker-owned resources (always available to the attacker) ---------
    reference_embeddings: Optional[np.ndarray] = None  # clean vocab x d matrix
    reference_token_counts: Optional[Mapping[int, int]] = None  # prior corpus counts
    reference_corpus: Optional[list] = None  # list of plaintext token-id sequences

    # --- ground truth (runner-only, never exposed to attacks) ----------------
    plaintext_token_ids: Optional[np.ndarray] = None
    secret_permutation: Optional[np.ndarray] = None  # plaintext_id -> cipher_id
    secret_matrices: Optional[Mapping[str, np.ndarray]] = None  # e.g. Z1, Z2

    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("special_token_mask", "content_mask"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr, dtype=bool))
        for name in ("ciphertext_token_ids", "plaintext_token_ids", "secret_permutation"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr))


@dataclass
class AttackContext:
    """Capability-gated view handed to an attack's ``run`` method.

    Accessors raise :class:`CapabilityError` when the required capability was
    not granted. Attacker-owned resources (reference embeddings, attacker model,
    reference corpus counts) are always available and are *not* gated, because
    they do not originate from the defended computation.
    """

    granted: frozenset[Capability]
    descriptor: Mapping[str, Any]

    # gated payloads
    _ciphertext_token_ids: Optional[np.ndarray] = None
    _obfuscated_embeddings: Optional[np.ndarray] = None
    _obfuscated_hidden_states: Optional[np.ndarray] = None
    _known_weights: Optional[Mapping[str, np.ndarray]] = None
    _replay_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None

    # attacker-owned (ungated)
    reference_embeddings: Optional[np.ndarray] = None
    reference_token_counts: Optional[Mapping[int, int]] = None
    reference_corpus: Optional[list] = None
    attacker_model: Optional[Any] = None
    obfuscation_fn: Optional[Callable[..., Any]] = None  # only if KNOWN_OBFUSCATION_ALGORITHM

    def has(self, cap: Capability) -> bool:
        return cap in self.granted

    def require(self, *caps: Capability) -> None:
        missing = [c for c in caps if c not in self.granted]
        if missing:
            names = ", ".join(c.value for c in missing)
            raise CapabilityError(f"scheme does not expose required capability: {names}")

    # --- gated accessors -----------------------------------------------------
    def token_ciphertext(self) -> np.ndarray:
        self.require(Capability.TOKEN_CIPHERTEXT)
        if self._ciphertext_token_ids is None:
            raise CapabilityError("TOKEN_CIPHERTEXT granted but no ciphertext present")
        return self._ciphertext_token_ids

    def obfuscated_embeddings(self) -> np.ndarray:
        self.require(Capability.EMBEDDINGS)
        if self._obfuscated_embeddings is None:
            raise CapabilityError("EMBEDDINGS granted but no embeddings present")
        return self._obfuscated_embeddings

    def obfuscated_hidden_states(self) -> np.ndarray:
        self.require(Capability.HIDDEN_STATES)
        if self._obfuscated_hidden_states is None:
            raise CapabilityError("HIDDEN_STATES granted but no hidden states present")
        return self._obfuscated_hidden_states

    def weight(self, name: str) -> np.ndarray:
        self.require(Capability.WHITE_BOX_WEIGHTS)
        if self._known_weights is None or name not in self._known_weights:
            available = sorted(self._known_weights) if self._known_weights else []
            raise CapabilityError(f"weight {name!r} not available; have {available}")
        return self._known_weights[name]

    def weight_names(self) -> list[str]:
        self.require(Capability.WHITE_BOX_WEIGHTS)
        return sorted(self._known_weights) if self._known_weights else []

    def replay(self, token_ids: Sequence[int]) -> np.ndarray:
        self.require(Capability.MODEL_REPLAY)
        if self._replay_fn is None:
            raise CapabilityError("MODEL_REPLAY granted but no replay function present")
        return self._replay_fn(token_ids)

    def knows_algorithm(self) -> bool:
        return self.has(Capability.KNOWN_OBFUSCATION_ALGORITHM)

    def is_deterministic_permutation(self) -> bool:
        return self.has(Capability.DETERMINISTIC_PERMUTATION)


@dataclass
class AttackResult:
    """Uniform result container across all attack families.

    Only the fields relevant to a given attack are filled. Scoring code reads
    whichever are present. ``status`` distinguishes a genuine run from an
    honest "not applicable" (insufficient information) or a failure.
    """

    status: AttackStatus = AttackStatus.OK
    status_reason: str = ""

    # per-position token recovery (NN, VMA-autoregressive, ISA, IMA)
    recovered_token_ids: Optional[np.ndarray] = None
    candidates: Optional[list[list[int]]] = None  # ranked per position

    # mapping recovery (TFMA, SDA, VMA-weight)
    recovered_mapping: Optional[dict[int, int]] = None  # cipher_id -> plain_id
    mapping_candidates: Optional[dict[int, list[int]]] = None  # cipher_id -> ranked plain_ids

    # permutation / matrix recovery (VMA-weight, IA)
    recovered_permutations: Optional[dict[str, np.ndarray]] = None

    # free text (SDA / IMA seq2seq)
    recovered_text: Optional[str] = None

    diagnostics: dict = field(default_factory=dict)

    @classmethod
    def not_applicable(cls, reason: str) -> "AttackResult":
        return cls(status=AttackStatus.NOT_APPLICABLE, status_reason=reason)

    @classmethod
    def failed(cls, reason: str) -> "AttackResult":
        return cls(status=AttackStatus.FAILED, status_reason=reason)
