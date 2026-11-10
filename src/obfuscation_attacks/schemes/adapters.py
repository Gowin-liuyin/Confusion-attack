"""Universal defense-arm adapter layer.

Every scheme — the three local arms, OSNIP-only, and the four external
baselines — is adapted through one interface so the same privacy tests can be
run against any of them without inventing a fake "vector wire":

* :class:`Upload` is what the client actually sends (ids, vectors or text) plus
  the metadata the server really sees.
* :class:`ServerObservation` is what the server computes (hidden states,
  logits, class probabilities, generated text) and which
  :class:`~obfuscation_attacks.data.contracts.Capability` set it exposes.
* :class:`SchemeAdapter` is the per-arm implementation: client upload, server
  observation, candidate table, attack queries, scoring truth, optional
  ``classify`` / ``generate``.

An arm whose author code, license, checkpoint or backend is missing still has a
complete contract: ``ready()`` returns the exact shortages and every execution
method raises :class:`AdapterNotReady`. It is never silently mapped to a vector
surface and never scored as zero.

The local adapters wrap the real :class:`ABDeployment`; the external adapters
delegate to small backend protocols (``ClassificationBackend``,
``ObfuscationBackend``, ``TextPerturbationBackend``, ``DPForwardBackend``) so the
day the author code/license arrives, only the backend has to be supplied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..data.contracts import Capability
from .ab_deployment import ABDeployment
from .arms import ARM_DECLARATIONS, ArmDeclaration, PAPER_ARMS


class UploadSurface(str, Enum):
    IDS = "ids"
    VECTORS = "vectors"
    TEXT = "text"


class AdapterNotReady(RuntimeError):
    """The scheme cannot be executed because a real dependency is absent."""

    def __init__(self, arm: str, shortages: Sequence[str]):
        self.arm = str(arm)
        self.shortages = [str(item) for item in shortages]
        super().__init__(f"{self.arm}: " + "; ".join(self.shortages))


@dataclass
class Upload:
    """One client request exactly as the server receives it."""

    surface: UploadSurface
    payload: Any                       # np.ndarray of ids / vectors, or str text
    request_id: int
    seq_len: int
    group_id: str | None = None
    epoch: int = 0
    truncated: bool = False
    full_len: int = 0
    metadata: dict = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        if self.surface == UploadSurface.TEXT:
            raise TypeError("text uploads are not arrays")
        return np.asarray(self.payload)


@dataclass
class ServerObservation:
    """What the honest-but-curious server can record for one request."""

    upload: Upload
    hidden_states: dict[int, np.ndarray] = field(default_factory=dict)
    logits: np.ndarray | None = None
    class_probabilities: np.ndarray | None = None
    generated_text: str | None = None
    generated_ids: np.ndarray | None = None
    capabilities: frozenset[Capability] = frozenset()
    extra: dict = field(default_factory=dict)

    def hidden(self, layer: int) -> np.ndarray:
        if layer not in self.hidden_states:
            raise KeyError(f"layer {layer} not observed; have {sorted(self.hidden_states)}")
        return self.hidden_states[layer]


# ---------------------------------------------------------------------------
# backend protocols for the external arms
# ---------------------------------------------------------------------------
class ClassificationBackend(ABC):
    """A real classification scheme's client transform + server classifier."""

    @abstractmethod
    def transform(self, text: str) -> np.ndarray:
        """Client-side transform of the token ids (already permutation/encryption)."""

    @abstractmethod
    def classify_ids(self, ids: np.ndarray) -> np.ndarray:
        """Server-side class probabilities for a transformed id sequence."""

    def prepare(self, text: str) -> np.ndarray:
        return self.transform(text)

    def embedding_table(self) -> np.ndarray:
        raise NotImplementedError("classifier does not expose its input embedding table")

    def hidden_states(self, ids: np.ndarray, layer: int = 1) -> np.ndarray:
        raise NotImplementedError("classifier does not expose hidden states")


class ObfuscationBackend(ABC):
    """ObfusLM/ObfusLM+ style model/vocabulary obfuscation."""

    @abstractmethod
    def obfuscate_ids(self, ids: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def server_logits(self, obfuscated_ids: np.ndarray) -> np.ndarray:
        ...


class TextPerturbationBackend(ABC):
    """InferDPT-style client perturbation plus local extraction."""

    @abstractmethod
    def perturb(self, text: str) -> str:
        ...

    @abstractmethod
    def extract(self, perturbed_text: str, server_completion: str) -> str:
        ...

    def generate(self, perturbed_text: str, max_new_tokens: int) -> str:
        raise NotImplementedError("the server-side black-box LLM must supply generation")


class DPForwardBackend(ABC):
    """DP-Forward style noisy representation + classifier."""

    @abstractmethod
    def noisy_representations(self, ids: np.ndarray, *, epsilon: float) -> np.ndarray:
        ...

    @abstractmethod
    def classify_representations(self, vectors: np.ndarray) -> np.ndarray:
        ...

    def embedding_table(self) -> np.ndarray:
        raise NotImplementedError("classifier does not expose its input embedding table")


# ---------------------------------------------------------------------------
# base adapter
# ---------------------------------------------------------------------------
class SchemeAdapter(ABC):
    """One executable scheme. Subclasses declare the real message path."""

    name: str
    declaration: ArmDeclaration
    tasks: tuple[str, ...] = ()
    upload_surface: UploadSurface = UploadSurface.IDS
    server_capabilities: frozenset[Capability] = frozenset()

    def ready(self, manifest: Mapping | None = None) -> list[str]:
        """Return the exact missing dependencies; empty means executable."""
        shortages: list[str] = []
        if self.declaration.status != "implemented":
            shortages.extend(self.declaration.missing or
                             (f"declaration status {self.declaration.status}",))
        manifest = manifest if isinstance(manifest, Mapping) else {}
        arms = manifest.get("arms") if isinstance(manifest.get("arms"), Mapping) else {}
        info = arms.get(self.name) if isinstance(arms.get(self.name), Mapping) else {}
        if self.declaration.status != "implemented":
            for field_name in ("license", "commit", "source_verified"):
                if not info.get(field_name):
                    shortages.append(f"manifest arms.{self.name}.{field_name} missing")
        return shortages

    def ensure_ready(self, manifest: Mapping | None = None) -> None:
        shortages = self.ready(manifest)
        if shortages:
            raise AdapterNotReady(self.name, shortages)

    @property
    def surfaces(self) -> tuple[UploadSurface, ...]:
        return (self.upload_surface,)

    # -- client side ---------------------------------------------------------
    @abstractmethod
    def client_upload(
        self,
        texts: Sequence[str],
        *,
        request_ids: Sequence[int] | None = None,
        key_indices: Sequence[int] | None = None,
        keys_per_text: int = 1,
    ) -> list[Upload]:
        """Produce the real upload records (this side may hold client secrets)."""

    # -- server side ---------------------------------------------------------
    def server_observe(
        self, uploads: Sequence[Upload], *, forward: bool = True
    ) -> list[ServerObservation]:
        raise NotImplementedError(f"{self.name} does not expose server observations")

    def classify(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support classification")

    def generate(self, prompts: Sequence[str], *, max_new_tokens: int = 64) -> list[str]:
        raise NotImplementedError(f"{self.name} does not support generation")

    # -- attacker-visible resources -----------------------------------------
    def candidate_table(self) -> np.ndarray:
        """The server's public candidate matrix for retrieval attacks."""
        raise NotImplementedError(f"{self.name} has no public candidate table")

    def attack_queries(
        self, uploads: Sequence[Upload], *, forward: bool = False,
        layer: int = 1,
    ) -> np.ndarray:
        """The representation a KNN/EDNN attack actually sees."""
        raise NotImplementedError(f"{self.name} has no vector query surface")

    def scoring_truth(self, uploads: Sequence[Upload]) -> np.ndarray:
        """Stage-1 truth in the server's own id space (scoring side only)."""
        raise NotImplementedError(f"{self.name} has no id truth")

    def describe(self) -> dict:
        return {
            "arm": self.name,
            "status": self.declaration.status,
            "tasks": list(self.tasks or self.declaration.tasks),
            "upload_surface": self.upload_surface.value,
            "capabilities": sorted(cap.value for cap in self.server_capabilities),
            "requires_training": self.declaration.requires_training,
            "randomization": self.declaration.randomization,
            "key_mechanism": self.declaration.key_mechanism,
            "source": self.declaration.source,
            "commit": self.declaration.commit,
            "license": self.declaration.license,
            "missing": list(self.declaration.missing),
        }


# ---------------------------------------------------------------------------
# local adapters backed by the real ABDeployment
# ---------------------------------------------------------------------------
class _DeploymentAdapter(SchemeAdapter):
    def __init__(self, deployment: ABDeployment) -> None:
        self.deployment = deployment
        self.declaration = ARM_DECLARATIONS[self.name]
        #: request_id -> {"plaintext": ids, "stage1": ids in the server's id space}
        self._truth: dict[int, dict] = {}
        self._next_request_id = 0

    def _request_ids(self, count: int, requested: Sequence[int] | None) -> list[int]:
        if requested is not None:
            if len(requested) != count or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in requested):
                raise ValueError("request_ids must contain one nonnegative integer per upload")
            ids = [int(value) for value in requested]
        else:
            ids = []
            candidate = self._next_request_id
            while len(ids) < count:
                if candidate not in self._truth:
                    ids.append(candidate)
                candidate += 1
        if len(set(ids)) != len(ids) or any(value in self._truth for value in ids):
            raise ValueError("request_ids must be unique and cannot replace stored truth")
        if ids:
            self._next_request_id = max(self._next_request_id, max(ids) + 1)
        return ids

    def _ids(self, text: str) -> np.ndarray:
        return self.deployment.encode(text)

    def _record_truth(self, request_id: int, plaintext: np.ndarray,
                      stage1: np.ndarray) -> None:
        if int(request_id) in self._truth:
            raise ValueError("request_id already has scoring truth")
        self._truth[int(request_id)] = {
            "plaintext": np.array(plaintext, dtype=np.int64, copy=True),
            "stage1": np.array(stage1, dtype=np.int64, copy=True),
        }

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([
            self._truth[int(upload.request_id)]["stage1"] for upload in uploads
        ])

    def plaintext_truth(self, uploads) -> np.ndarray:
        return np.concatenate([
            self._truth[int(upload.request_id)]["plaintext"] for upload in uploads
        ])

    def candidate_table(self) -> np.ndarray:
        return self.deployment.table(self._table_arm)

    def describe(self) -> dict:
        info = super().describe()
        info["deployment"] = self.deployment.describe()
        return info


class PlaintextAdapter(_DeploymentAdapter):
    name = "plaintext"
    tasks = ("classification", "generation")
    upload_surface = UploadSurface.IDS
    server_capabilities = frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS,
        Capability.HIDDEN_STATES, Capability.MODEL_REPLAY,
        Capability.WHITE_BOX_WEIGHTS,
    })
    _table_arm = "plaintext"

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        repeats = int(keys_per_text)
        if repeats < 1:
            raise ValueError("keys_per_text must be positive")
        request_uids = self._request_ids(len(texts) * repeats, request_ids)
        uploads = []
        index = 0
        for text_index, text in enumerate(texts):
            ids = self._ids(text)
            for _ in range(repeats):
                request_id = request_uids[index]
                uploads.append(Upload(
                    surface=UploadSurface.IDS, payload=ids.copy(),
                    request_id=request_id, seq_len=len(ids), full_len=len(ids),
                ))
                self._record_truth(request_id, ids, ids)
                index += 1
        return uploads

    def server_observe(self, uploads, *, forward=True) -> list[ServerObservation]:
        observations = []
        for upload in uploads:
            hidden, logits = {}, None
            if forward:
                if self.deployment.clean is None:
                    raise AdapterNotReady(self.name, ["clean model not loaded"])
                import torch

                with torch.no_grad():
                    out = self.deployment.clean(
                        input_ids=torch.from_numpy(upload.as_array()).long().unsqueeze(0),
                        output_hidden_states=True,
                    )
                hidden = {
                    layer: out.hidden_states[layer][0].float().numpy()
                    for layer in self.deployment.layers
                }
                logits = out.logits[0].float().numpy()
            observations.append(ServerObservation(
                upload=upload, hidden_states=hidden, logits=logits,
                capabilities=self.server_capabilities,
            ))
        return observations

    def attack_queries(self, uploads, *, forward=False, layer=1) -> np.ndarray:
        return np.concatenate([
            self.deployment.table("plaintext")[upload.as_array()] for upload in uploads
        ], axis=0)

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([upload.as_array() for upload in uploads])


class AloePriAdapter(_DeploymentAdapter):
    name = "aloepri"
    tasks = ("classification", "generation")
    upload_surface = UploadSurface.IDS
    server_capabilities = frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS,
        Capability.HIDDEN_STATES, Capability.WHITE_BOX_WEIGHTS,
        Capability.DETERMINISTIC_PERMUTATION,
    })
    _table_arm = "aloepri"

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        repeats = int(keys_per_text)
        if repeats < 1:
            raise ValueError("keys_per_text must be positive")
        request_uids = self._request_ids(len(texts) * repeats, request_ids)
        uploads = []
        index = 0
        for text in texts:
            ids = self._ids(text)
            permuted = self.deployment.perm[ids]
            for _ in range(repeats):
                request_id = request_uids[index]
                uploads.append(Upload(
                    surface=UploadSurface.IDS, payload=permuted.copy(),
                    request_id=request_id, seq_len=len(ids), full_len=len(ids),
                ))
                self._record_truth(request_id, ids, permuted)
                index += 1
        return uploads

    def server_observe(self, uploads, *, forward=True) -> list[ServerObservation]:
        observations = []
        for upload in uploads:
            hidden, logits = {}, None
            if forward:
                if self.deployment.model is None:
                    raise AdapterNotReady(self.name, ["deployment model not loaded"])
                import torch

                with torch.no_grad():
                    out = self.deployment.model(
                        input_ids=torch.from_numpy(upload.as_array()).long().unsqueeze(0),
                        output_hidden_states=True,
                    )
                hidden = {
                    layer: out.hidden_states[layer][0].float().numpy()
                    for layer in self.deployment.layers
                }
                logits = out.logits[0].float().numpy()
            observations.append(ServerObservation(
                upload=upload, hidden_states=hidden, logits=logits,
                capabilities=self.server_capabilities,
            ))
        return observations

    def attack_queries(self, uploads, *, forward=False, layer=1) -> np.ndarray:
        if not forward:
            # The server embeds the received (permuted) ids; without a forward pass
            # this is the trivial id/embedding identity row.
            return np.concatenate([
                self.deployment.table("aloepri")[upload.as_array()]
                for upload in uploads
            ], axis=0)
        hidden = []
        for observation in self.server_observe(uploads, forward=True):
            hidden.append(observation.hidden(layer))
        return np.concatenate(hidden, axis=0)

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([upload.as_array() for upload in uploads])


class ABAdapter(_DeploymentAdapter):
    name = "ab"
    tasks = ("classification", "generation")
    upload_surface = UploadSurface.VECTORS
    server_capabilities = frozenset({
        Capability.EMBEDDINGS, Capability.HIDDEN_STATES,
        Capability.WHITE_BOX_WEIGHTS, Capability.MODEL_REPLAY,
    })
    _table_arm = "ab"

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        repeats = int(keys_per_text)
        if repeats < 1:
            raise ValueError("keys_per_text must be positive")
        ids_for_uploads = self._request_ids(len(texts) * repeats, request_ids)
        if key_indices is not None and (len(key_indices) != len(ids_for_uploads) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in key_indices
        )):
            raise ValueError("key_indices must contain one nonnegative integer per upload")
        uploads = []
        index = 0
        for text in texts:
            ids = self._ids(text)
            for _ in range(repeats):
                request_id = ids_for_uploads[index]
                key_index = request_id if key_indices is None else int(key_indices[index])
                wire = self.deployment.observe(
                    text, arm="ab", request_id=request_id, key_index=key_index, forward=False
                )
                observed = np.asarray(wire.truth.plaintext_ids, dtype=np.int64)
                stage1 = np.asarray(wire.truth.permuted_ids, dtype=np.int64)
                vectors = np.asarray(wire.public.wire_vectors, dtype=np.float32)
                if (int(wire.public.request_id) != request_id or int(wire.public.seq_len) != len(observed)
                        or len(stage1) != len(observed) or len(vectors) != len(observed)
                        or not np.array_equal(observed, ids[:len(observed)])):
                    raise ValueError("upload and scoring truth are not aligned")
                full_len = int(wire.public.full_len or len(ids))
                if full_len != len(ids) or bool(wire.public.truncated) != (len(observed) < len(ids)):
                    raise ValueError("upload truncation metadata does not match encoded input")
                uploads.append(Upload(
                    surface=UploadSurface.VECTORS,
                    payload=vectors,
                    request_id=request_id, seq_len=len(observed), full_len=full_len,
                    group_id=wire.public.group_id, epoch=int(wire.public.epoch or 0),
                    truncated=bool(wire.public.truncated),
                    metadata={"position_map": list(range(len(observed)))},
                ))
                self._record_truth(request_id, observed, stage1)
                self._truth[request_id]["full_plaintext"] = np.array(ids, dtype=np.int64, copy=True)
                index += 1
        return uploads

    def server_observe(self, uploads, *, forward=True) -> list[ServerObservation]:
        observations = []
        for upload in uploads:
            hidden, logits = {}, None
            if forward:
                if self.deployment.model is None:
                    raise AdapterNotReady(self.name, ["deployment model not loaded"])
                import torch

                with torch.no_grad():
                    out = self.deployment.model(
                        inputs_embeds=torch.from_numpy(upload.as_array()).float().unsqueeze(0),
                        output_hidden_states=True,
                    )
                hidden = {
                    layer: out.hidden_states[layer][0].float().numpy()
                    for layer in self.deployment.layers
                }
                logits = out.logits[0].float().numpy()
            observations.append(ServerObservation(
                upload=upload, hidden_states=hidden, logits=logits,
                capabilities=self.server_capabilities,
            ))
        return observations

    def attack_queries(self, uploads, *, forward=False, layer=1) -> np.ndarray:
        if not forward:
            return np.concatenate([upload.as_array() for upload in uploads], axis=0)
        return np.concatenate(
            [obs.hidden(layer) for obs in self.server_observe(uploads, forward=True)],
            axis=0,
        )

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([
            self._truth[int(upload.request_id)]["stage1"] for upload in uploads
        ])


# ---------------------------------------------------------------------------
# OSNIP-only: vectors from a clean-backbone key-conditioned encoder
# ---------------------------------------------------------------------------
class OSNIPOnlyAdapter(SchemeAdapter):
    """Clean model + independently trained OSNIP encoder (protocol §8.3).

    The packaged encoder is A-conditioned; reusing it on the clean table is an
    out-of-distribution substitute. This adapter therefore needs its own
    ``encoder`` (a ``torch.nn.Module`` with the ``mlp_residual_v1`` interface)
    plus a clean :class:`ABDeployment`-like server model. Until those exist it
    fails closed with the exact missing item.
    """

    name = "osnip"
    tasks = ("classification", "generation")
    upload_surface = UploadSurface.VECTORS
    server_capabilities = frozenset({
        Capability.EMBEDDINGS, Capability.HIDDEN_STATES, Capability.MODEL_REPLAY,
    })

    def __init__(
        self,
        deployment: ABDeployment | None = None,
        *,
        encoder: Any | None = None,
        encoder_sha256: str | None = None,
        training_recipe: str | None = None,
    ) -> None:
        self.deployment = deployment
        self.encoder = encoder
        self.encoder_sha256 = encoder_sha256
        self.training_recipe = training_recipe
        self.declaration = ARM_DECLARATIONS[self.name]
        self._truth: dict[int, np.ndarray] = {}

    def ready(self, manifest: Mapping | None = None) -> list[str]:
        shortages: list[str] = []
        if self.deployment is None:
            shortages.append("clean server model not supplied")
        if self.encoder is None:
            shortages.append("clean-backbone OSNIP encoder not supplied")
        if not self.encoder_sha256:
            shortages.append("encoder sha256 not supplied")
        if not self.training_recipe:
            shortages.append("encoder training recipe/artifact id not supplied")
        return shortages

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        self.ensure_ready()
        import torch

        uploads = []
        index = 0
        for text in texts:
            ids = self.deployment.encode(text)
            h = torch.from_numpy(self.deployment.E_clean[ids]).float()
            for offset in range(int(keys_per_text)):
                key_index = index if key_indices is None else int(key_indices[index])
                key = self.deployment._key_for(key_index)
                z = self.deployment.iso_norm_project(
                    self.encoder(h, torch.from_numpy(key)), h
                )
                uploads.append(Upload(
                    surface=UploadSurface.VECTORS,
                    payload=z.detach().float().numpy(),
                    request_id=index, seq_len=len(ids), full_len=len(ids),
                ))
                self._truth[index] = np.asarray(ids, dtype=np.int64)
                index += 1
        return uploads

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([self._truth[int(upload.request_id)] for upload in uploads])

    def attack_queries(self, uploads, *, forward=False, layer=1) -> np.ndarray:
        return np.concatenate([upload.as_array() for upload in uploads], axis=0)

    def candidate_table(self) -> np.ndarray:
        self.ensure_ready()
        return self.deployment.E_clean

    def server_observe(self, uploads, *, forward=True) -> list[ServerObservation]:
        self.ensure_ready()
        observations = []
        for upload in uploads:
            hidden, logits = {}, None
            if forward:
                if self.deployment.clean is None:
                    raise AdapterNotReady(self.name, ["clean model not loaded"])
                import torch

                with torch.no_grad():
                    out = self.deployment.clean(
                        inputs_embeds=torch.from_numpy(upload.as_array()).float().unsqueeze(0),
                        output_hidden_states=True,
                    )
                hidden = {
                    layer: out.hidden_states[layer][0].float().numpy()
                    for layer in self.deployment.layers
                }
                logits = out.logits[0].float().numpy()
            observations.append(ServerObservation(
                upload=upload, hidden_states=hidden, logits=logits,
                capabilities=self.server_capabilities,
            ))
        return observations


# ---------------------------------------------------------------------------
# external arms: complete contracts, fail closed until backend + provenance
# ---------------------------------------------------------------------------
class _ExternalAdapter(SchemeAdapter):
    backend: Any | None = None
    backend_name = "backend"

    def __init__(self, backend: Any | None = None, *,
                 provenance: Mapping | None = None) -> None:
        self.backend = backend
        self.provenance = dict(provenance or {})
        self.declaration = ARM_DECLARATIONS[self.name]
        self._truth: dict[int, np.ndarray] = {}

    def _record_truth(self, request_id: int, ids: np.ndarray) -> None:
        self._truth[int(request_id)] = np.asarray(ids, dtype=np.int64)

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([
            self._truth[int(upload.request_id)] for upload in uploads
        ])

    def candidate_table(self) -> np.ndarray:
        self.ensure_ready()
        try:
            return np.asarray(self.backend.embedding_table(), dtype=np.float32)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{self.name} backend does not expose an embedding candidate table"
            ) from exc

    def _provenance_shortages(self, manifest: Mapping | None) -> list[str]:
        manifest = manifest if isinstance(manifest, Mapping) else {}
        arms = manifest.get("arms") if isinstance(manifest.get("arms"), Mapping) else {}
        info = arms.get(self.name) if isinstance(arms.get(self.name), Mapping) else {}
        merged = {**self.provenance, **dict(info)}
        shortages = []
        for field_name in ("adapter", "license", "commit", "source_verified"):
            if not merged.get(field_name):
                shortages.append(f"arms.{self.name}.{field_name} missing")
        if self.backend is None:
            shortages.append(f"{self.backend_name} not supplied")
        return shortages

    def ready(self, manifest: Mapping | None = None) -> list[str]:
        shortages: list[str] = []
        if self.backend is None:
            shortages.extend(self.declaration.missing)
        shortages.extend(self._provenance_shortages(manifest))
        return shortages


class SentinelLMsAdapter(_ExternalAdapter):
    """Classification-only, encrypted token ids (glide-reflection + fine-tuning)."""

    name = "sentinellms"
    backend_name = "author implementation + fine-tuned checkpoint"
    tasks = ("classification",)
    upload_surface = UploadSurface.IDS
    server_capabilities = frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.WHITE_BOX_WEIGHTS,
    })

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        self.ensure_ready()
        uploads = []
        for index, text in enumerate(texts):
            ids = np.asarray(self.backend.transform(text), dtype=np.int64)
            uploads.append(Upload(
                surface=UploadSurface.IDS, payload=ids, request_id=index,
                seq_len=len(ids), full_len=len(ids),
            ))
            self._record_truth(index, ids)
        return uploads

    def attack_queries(self, uploads, *, forward=True, layer=1) -> np.ndarray:
        self.ensure_ready()
        if forward:
            return np.concatenate([
                np.asarray(self.backend.hidden_states(upload.as_array(), layer=layer),
                           dtype=np.float32)
                for upload in uploads
            ], axis=0)
        table = self.candidate_table()
        return np.concatenate([table[upload.as_array()] for upload in uploads], axis=0)

    def classify(self, texts) -> np.ndarray:
        self.ensure_ready()
        return np.stack([
            np.asarray(self.backend.classify_ids(
                np.asarray(self.backend.transform(text), dtype=np.int64)
            ))
            for text in texts
        ])


class ObfusLMPlusAdapter(_ExternalAdapter):
    """Vocabulary/model obfuscation with the '+' layer-wise variant."""

    name = "obfuslm_plus"
    backend_name = "author implementation + fine-tuned model"
    tasks = ("classification", "generation")
    upload_surface = UploadSurface.IDS
    server_capabilities = frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.WHITE_BOX_WEIGHTS,
    })

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        self.ensure_ready()
        uploads = []
        for index, text in enumerate(texts):
            ids = np.asarray(self.backend.obfuscate_ids(
                np.asarray([ord(c) for c in text], dtype=np.int64)
            ), dtype=np.int64)
            uploads.append(Upload(
                surface=UploadSurface.IDS, payload=ids, request_id=index,
                seq_len=len(ids), full_len=len(ids),
            ))
            self._record_truth(index, ids)
        return uploads

    def attack_queries(self, uploads, *, forward=True, layer=1) -> np.ndarray:
        self.ensure_ready()
        if forward:
            return np.concatenate([
                np.asarray(self.backend.hidden_states(upload.as_array(), layer=layer),
                           dtype=np.float32)
                for upload in uploads
            ], axis=0)
        table = self.candidate_table()
        return np.concatenate([table[upload.as_array()] for upload in uploads], axis=0)


class InferDPTAdapter(_ExternalAdapter):
    """Perturbed text upload + local extraction (generation only)."""

    name = "inferdpt"
    backend_name = "author implementation (perturbation + extraction)"
    tasks = ("generation",)
    upload_surface = UploadSurface.TEXT
    server_capabilities = frozenset({Capability.TOKEN_CIPHERTEXT})

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        self.ensure_ready()
        return [
            Upload(
                surface=UploadSurface.TEXT,
                payload=self.backend.perturb(text),
                request_id=index, seq_len=len(text), full_len=len(text),
            )
            for index, text in enumerate(texts)
        ]

    def server_observe(self, uploads, *, forward=True) -> list[ServerObservation]:
        self.ensure_ready()
        observations = []
        for upload in uploads:
            completion = ""
            if forward:
                completion = self.backend.generate(str(upload.payload), 64)
            observations.append(ServerObservation(
                upload=upload, generated_text=completion,
                capabilities=self.server_capabilities,
            ))
        return observations

    def generate(self, prompts, *, max_new_tokens=64) -> list[str]:
        self.ensure_ready()
        outputs = []
        for prompt in prompts:
            perturbed = self.backend.perturb(prompt)
            completion = self.backend.generate(perturbed, max_new_tokens)
            outputs.append(self.backend.extract(perturbed, completion))
        return outputs


class DPForwardAdapter(_ExternalAdapter):
    """Classification on noisy forwarded representations."""

    name = "dp_forward"
    backend_name = "author implementation + noisy fine-tuned model"
    tasks = ("classification",)
    upload_surface = UploadSurface.VECTORS
    server_capabilities = frozenset({
        Capability.EMBEDDINGS, Capability.WHITE_BOX_WEIGHTS,
    })

    def __init__(self, backend: Any | None = None, *, epsilon: float = 1.0,
                 provenance: Mapping | None = None,
                 tokenizer: Callable[[str], np.ndarray] | None = None) -> None:
        super().__init__(backend, provenance=provenance)
        self.epsilon = float(epsilon)
        self.tokenizer = tokenizer or (
            lambda text: np.asarray([ord(char) for char in text], dtype=np.int64)
        )

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        self.ensure_ready()
        uploads = []
        for index, text in enumerate(texts):
            ids = np.asarray(self.tokenizer(text), dtype=np.int64)
            vectors = np.asarray(
                self.backend.noisy_representations(ids, epsilon=self.epsilon),
                dtype=np.float32,
            )
            uploads.append(Upload(
                surface=UploadSurface.VECTORS, payload=vectors,
                request_id=index, seq_len=len(ids), full_len=len(ids),
            ))
            self._record_truth(index, ids)
        return uploads

    def attack_queries(self, uploads, *, forward=False, layer=1) -> np.ndarray:
        return np.concatenate([upload.as_array() for upload in uploads], axis=0)

    def classify(self, texts) -> np.ndarray:
        self.ensure_ready()
        uploads = self.client_upload(texts)
        return np.stack([
            np.asarray(self.backend.classify_representations(upload.as_array()))
            for upload in uploads
        ])


# ---------------------------------------------------------------------------
# plaintext classification reference (not one of the eight paper arms)
# ---------------------------------------------------------------------------
class ClassificationReferenceAdapter(SchemeAdapter):
    """Same-backbone plaintext reference for a classification table.

    Protocol §8.3 requires a same-BERT plaintext reference before an external
    classification arm can be compared. This adapter gives that reference the
    same client/server contract as the defended arms: it uploads the clean token
    ids and runs the clean classifier, and it exposes the embedding table and
    hidden states so EDNN/KNN can score both the exact-solvable embedding
    control and the hidden-state attack.
    """

    name = "plaintext_classifier"
    tasks = ("classification",)
    upload_surface = UploadSurface.IDS
    server_capabilities = frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS,
        Capability.HIDDEN_STATES, Capability.WHITE_BOX_WEIGHTS,
    })

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.declaration = ARM_DECLARATIONS["plaintext"]
        self._truth: dict[int, np.ndarray] = {}

    def client_upload(self, texts, *, request_ids=None, key_indices=None,
                      keys_per_text=1) -> list[Upload]:
        uploads = []
        for index, text in enumerate(texts):
            ids = np.asarray(self.backend.transform(text), dtype=np.int64)
            uploads.append(Upload(
                surface=UploadSurface.IDS, payload=ids, request_id=index,
                seq_len=len(ids), full_len=len(ids),
            ))
            self._truth[index] = ids
        return uploads

    def classify(self, texts) -> np.ndarray:
        return np.stack([
            np.asarray(self.backend.classify_ids(
                np.asarray(self.backend.transform(text), dtype=np.int64)
            ))
            for text in texts
        ])

    def candidate_table(self) -> np.ndarray:
        return np.asarray(self.backend.embedding_table(), dtype=np.float32)

    def attack_queries(self, uploads, *, forward=True, layer=1) -> np.ndarray:
        if forward:
            return np.concatenate([
                np.asarray(self.backend.hidden_states(upload.as_array(), layer=layer),
                           dtype=np.float32)
                for upload in uploads
            ], axis=0)
        table = self.candidate_table()
        return np.concatenate([table[upload.as_array()] for upload in uploads], axis=0)

    def scoring_truth(self, uploads) -> np.ndarray:
        return np.concatenate([
            self._truth[int(upload.request_id)] for upload in uploads
        ])


ADAPTER_CLASSES: dict[str, type[SchemeAdapter]] = {
    "plaintext": PlaintextAdapter,
    "aloepri": AloePriAdapter,
    "ab": ABAdapter,
    "osnip": OSNIPOnlyAdapter,
    "sentinellms": SentinelLMsAdapter,
    "obfuslm_plus": ObfusLMPlusAdapter,
    "inferdpt": InferDPTAdapter,
    "dp_forward": DPForwardAdapter,
}


def build_adapter(arm: str, *, provenance: Mapping | None = None,
                  **kwargs: Any) -> SchemeAdapter:
    """Instantiate an arm adapter; blocked arms still return a contract object."""
    try:
        cls = ADAPTER_CLASSES[str(arm)]
    except KeyError as exc:
        raise KeyError(f"unknown defense arm {arm!r}; have {sorted(ADAPTER_CLASSES)}") from exc
    if str(arm) in {"plaintext", "aloepri", "ab"} and not kwargs.get("deployment"):
        raise AdapterNotReady(str(arm), ["verified ABDeployment not supplied"])
    if str(arm) in {"plaintext", "aloepri", "ab", "osnip"}:
        return cls(**kwargs)
    return cls(provenance=provenance, **kwargs)


def adapter_status_table(
    manifest: Mapping | None = None,
    deployments: Mapping[str, Any] | None = None,
    backends: Mapping[str, Any] | None = None,
) -> dict:
    """Readiness of every arm with its exact shortages and capabilities.

    ``deployments`` maps the local arms / OSNIP-only server to a constructed
    :class:`ABDeployment`; ``backends`` maps external arms to their author
    backend. Without them the arms are reported as blocked with the exact
    missing resource rather than assumed usable.
    """
    deployments = deployments if isinstance(deployments, Mapping) else {}
    backends = backends if isinstance(backends, Mapping) else {}
    table = {}
    for arm in PAPER_ARMS:
        try:
            if arm in {"plaintext", "aloepri", "ab"}:
                deployment = deployments.get(arm)
                if deployment is None:
                    declaration = ARM_DECLARATIONS[arm]
                    table[arm] = {
                        "arm": arm,
                        "ready": False,
                        "shortages": ["verified ABDeployment not supplied"],
                        "status": declaration.status,
                        "tasks": list(declaration.tasks),
                        "upload_surface": declaration.upload,
                        "capabilities": [],
                        "requires_training": declaration.requires_training,
                        "source": declaration.source,
                        "commit": declaration.commit,
                        "license": declaration.license,
                        "missing": list(declaration.missing),
                    }
                    continue
                adapter = ADAPTER_CLASSES[arm](deployment)
            elif arm == "osnip":
                adapter = OSNIPOnlyAdapter(deployments.get("osnip"))
            else:
                manifest = manifest if isinstance(manifest, Mapping) else {}
                arms = manifest.get("arms") if isinstance(manifest.get("arms"), Mapping) else {}
                adapter = ADAPTER_CLASSES[arm](
                    backends.get(arm), provenance=arms.get(arm)
                )
            shortages = adapter.ready(manifest)
            description = adapter.describe()
        except Exception as exc:  # pragma: no cover - defensive
            shortages = [f"adapter construction failed: {exc}"]
            description = {"arm": arm}
        table[arm] = {
            "ready": not shortages,
            "shortages": list(shortages),
            **description,
        }
    return table
