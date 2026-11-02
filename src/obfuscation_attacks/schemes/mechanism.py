"""Obfuscation mechanisms with an explicit secret key (faithful threat model).

An :class:`ObfuscationMechanism` separates the *public algorithm* (which the
attacker is assumed to know) from the *per-client secret key*. This distinction
is essential for the learned inversion attack (IMA): a faithful attacker builds
training pairs with the known mechanism and keys it samples itself, but never
sees the client's key. AloePri defeats IMA precisely because the token-level
permutation is a secret key (paper Table 2 reports IMA TTRSR = 0.0); a faithful
harness therefore trains the inverter on attacker-sampled keys and evaluates on
the held-out client key, and the expected recovery is near chance.

This is deliberately distinct from the ``KNOWN_OBFUSCATION_ALGORITHM``
capability, which grants knowledge of the *algorithm* - not the key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ObfuscationMechanism(ABC):
    """A keyed, attacker-known obfuscation algorithm.

    ``sample_key`` draws a fresh secret key; ``obfuscate_embeddings`` applies the
    algorithm under a given key. The attacker may call both with keys it samples
    itself, but the client's key is held out by the scheme.
    """

    name: str = ""

    @abstractmethod
    def sample_key(self, rng: np.random.Generator) -> Any:
        """Sample a fresh secret key."""

    @abstractmethod
    def obfuscate_embeddings(self, token_ids: np.ndarray, key: Any) -> np.ndarray:
        """Return the (len, d) obfuscated embeddings of ``token_ids`` under ``key``."""


class AloePriToyMechanism(ObfuscationMechanism):
    """Toy AloePri-style embedding obfuscation: permutation + covariant map + noise.

    Mirrors AloePri's embedding-layer obfuscation structure (paper Sec. 5.2.2):

        obf(t) = (E[Pi(t)] + alpha_e * sigma * noise) @ A

    * ``E``        - public clean embedding matrix (vocab x d);
    * ``A``        - public shared invertible covariant map (the key-cancelling
                     transform is shared structure, not the secret);
    * ``Pi``       - the SECRET per-client token permutation = the key;
    * ``alpha_e``  - embedding-noise scale (paper's alpha_e; noise is drawn
                     deterministically from the key so one key gives one table).

    With a fresh ``Pi`` per training example, a learned inverter cannot map an
    obfuscated row back to the client's token id, matching AloePri's result that
    IMA fails under a secret key. The permutation is the only secret; ``E`` and
    ``A`` are attacker-known (``KNOWN_OBFUSCATION_ALGORITHM``).
    """

    name = "aloepri_toy_embedding"

    def __init__(
        self,
        clean_embeddings: np.ndarray,
        *,
        alpha_e: float = 0.0,
        A: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.E = np.asarray(clean_embeddings, dtype=float)
        self.vocab, self.d = self.E.shape
        self.alpha_e = float(alpha_e)
        rng = np.random.default_rng(seed)
        self.A = np.asarray(A, dtype=float) if A is not None else rng.standard_normal((self.d, self.d))
        self._sigma = float(self.E.std()) if self.E.size else 1.0

    def sample_key(self, rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(self.vocab)

    def obfuscate_embeddings(self, token_ids: np.ndarray, key: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=int).ravel()
        key = np.asarray(key, dtype=int)
        rows = self.E[key[ids]]
        if self.alpha_e:
            # Noise is a deterministic function of the key so a given key yields a
            # single, self-consistent obfuscated table (re-querying is stable).
            noise_rng = np.random.default_rng(int(np.abs(key).sum()) + 1)
            rows = rows + self.alpha_e * self._sigma * noise_rng.standard_normal(rows.shape)
        return rows @ self.A


def attacker_oracle(
    mechanism: ObfuscationMechanism,
    *,
    client_key: Any = None,
    attacker_known_key: bool = False,
    seed: int = 0,
):
    """Build the ``obfuscation_fn`` an attacker uses to synthesise training pairs.

    When ``attacker_known_key`` is False (the faithful default) each call applies
    a FRESH attacker-sampled key, so the oracle never reveals the client key.
    When True (a white-box sanity setting) it uses ``client_key`` - useful only
    to show the inverter architecture can learn at all.
    """
    rng = np.random.default_rng(seed + 1)

    def oracle(token_ids: np.ndarray) -> np.ndarray:
        key = client_key if (attacker_known_key and client_key is not None) else mechanism.sample_key(rng)
        return mechanism.obfuscate_embeddings(token_ids, key)

    return oracle
