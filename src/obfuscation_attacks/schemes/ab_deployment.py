"""The packaged A+B deployment as an obfuscation scheme with real wire.

This module turns the runnable artifact ``osnip_ab_runnable_20261106`` (AloePri
``square_monomial`` server weights + OSNIP client encoder + client secret) into
the harness the plan's attack cards assume, so that P0 attacks can be measured on
the *deployment* rather than on a synthetic toy.

The plan's isolation rule (§1.7-9: "the evaluation harness may use the encoder /
key / client secret to generate wire, but the attack algorithm itself must not
read them") is encoded in the types:

* :meth:`ABDeployment.observe` is the only method that touches the client secret
  and the OSNIP encoder; it returns a :class:`Wire` whose ``public`` half is what
  the server logged (continuous ``z``, hidden states, permuted ids, logits) and
  whose ``truth`` half (plaintext ids, the secret permutation, the key) is used by
  the scorer only.
* every ``attack_*`` helper below takes ``wire.public`` / the candidate table and
  returns recovered ids, so a scoring bug cannot hide behind a secret read.

Arms (plan §1.5):

===========  ==========================================================  ==============
arm          what the server receives / sees                            ``S1`` object
===========  ==========================================================  ==============
``plaintext``  clean token ids, clean model                              clean embedding
``aloepri``    permuted token ids (AloePri, no OSNIP), obfuscated model   permuted id
``ab``         continuous ``z`` (AloePri + OSNIP), obfuscated model      ``z``
===========  ==========================================================  ==============

``osnip`` (clean model + OSNIP encoder) is **not** instantiable from this
artifact: the packaged encoder was trained against the obfuscated embedding table
(``osnip/config.json``: ``model_path`` points at the AloePri ``server/``), so
reusing it on the clean table would be an out-of-distribution substitute. The plan
requires that case to be reported as ``not_applicable`` with a reason, never as a
0% success rate, which is what :meth:`ABDeployment.arm_status` does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = ROOT.parent / "OSNIP_package_20261101" / "ab_runnable"
#: The protocol's primary model is the 1B **base**, never the Instruct variant.
#: Callers that do not pass ``clean_model`` must not silently load Instruct.
DEFAULT_CLEAN = Path(r"F:\Models\Llama-3.2-1B")
ARMS = ("plaintext", "aloepri", "ab")


@dataclass
class WirePublic:
    """What the honest-but-curious server logged for one request."""

    arm: str
    request_id: int
    seq_len: int
    #: continuous wire the client sent (``ab`` arm only)
    wire_vectors: Optional[np.ndarray] = None
    #: token ids the server received (``aloepri`` arm only; already permuted)
    wire_token_ids: Optional[np.ndarray] = None
    #: hidden states the server computed, ``[n_layers_reported, seq_len, d]``
    hidden_states: Optional[np.ndarray] = None
    layers: tuple[int, ...] = ()
    #: logits in the server's (permuted) output space
    logits_perm: Optional[np.ndarray] = None
    #: opaque rotation-group id (never the seed/root/key)
    group_id: Optional[str] = None
    #: backwards-compatible alias for ``group_id``; still opaque
    key_id: Optional[str] = None
    epoch: Optional[int] = None
    session_id: Optional[str] = None
    #: True when the request exceeded ``max_len``; ``full_len`` records the source length
    truncated: bool = False
    full_len: int = 0

    def __post_init__(self) -> None:
        if self.key_id is None and self.group_id is not None:
            self.key_id = self.group_id
        elif self.group_id is None and self.key_id is not None:
            self.group_id = self.key_id


@dataclass
class WireTruth:
    """Ground truth used by the scorer only - never handed to an attack."""

    plaintext_ids: np.ndarray
    permuted_ids: Optional[np.ndarray] = None
    key: Optional[np.ndarray] = None
    text: str = ""


@dataclass
class Wire:
    public: WirePublic
    truth: WireTruth


class ABDeployment:
    """Loads the packaged artifact once and produces per-arm wire observations."""

    def __init__(
        self,
        artifact: Path = DEFAULT_ARTIFACT,
        clean_model: Path = DEFAULT_CLEAN,
        *,
        dtype: str = "float32",
        layers: Sequence[int] = (1, 5, 10, 15),
        max_len: int | None = 64,
        load_models: bool = True,
        root_key: bytes | None = None,
        expected_variant: str | None = None,
        strict_length: bool = False,
    ) -> None:
        import torch

        from .key_schedule import KeySchedule

        self.artifact = Path(artifact)
        self.clean_model_path = Path(clean_model)
        self.public_artifact = self.artifact / "model_server"
        self.dtype = getattr(torch, dtype)
        self.layers = tuple(int(l) for l in layers)
        self.max_len = None if max_len is None else int(max_len)
        self.strict_length = bool(strict_length)
        self.expected_variant = expected_variant
        self.clean_model_variant_evidence = inspect_model_variant(self.clean_model_path)
        if expected_variant == "base" and self.clean_model_variant_evidence["looks_instruct"]:
            raise ValueError(
                f"clean model {self.clean_model_path} carries Instruct evidence "
                f"({self.clean_model_variant_evidence['evidence']}), but expected_variant='base'; "
                "Instruct must not substitute for the 1B base"
            )
        #: 256-bit secret root; per-request keys and opaque group ids derive from it.
        self.key_schedule = KeySchedule(root_key)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ``load_models=False`` is for the weight-only attacks (C06-W, C08): they need the
        # weight tensors and the embedding tables, never a forward pass. Loading both 1B
        # models costs ~8 GiB and, on this box, pushes MKL into swap so a 1.5 s GEMM becomes
        # minutes. Wire-generation arms must keep the default.
        # ``load_models="deployment"`` loads only the obfuscated server model: C01's
        # AloePri-only cell needs its hidden state, but not the clean model.
        self.models_loaded = load_models is True
        self.deployment_loaded = load_models is True or load_models == "deployment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifact / "model_server", local_files_only=True)
        if self.deployment_loaded:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.artifact / "model_server", local_files_only=True, dtype=self.dtype
            )
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None
        if self.models_loaded:
            self.clean = AutoModelForCausalLM.from_pretrained(
                self.clean_model_path, local_files_only=True, dtype=self.dtype
            )
            self.clean.eval()
            for p in self.clean.parameters():
                p.requires_grad_(False)
        else:
            self.clean = None

        secret = torch.load(self.artifact / "client" / "client_secret.pt", map_location="cpu", weights_only=False)
        self.perm = secret["perm_vocab"].long().numpy()
        self.inv_perm = secret["inv_perm_vocab"].long().numpy()
        self.secret_metadata = dict(secret.get("metadata", {}))
        if not np.array_equal(self.inv_perm[self.perm], np.arange(len(self.perm))):
            raise ValueError("client secret permutation is not invertible")

        cfg = json.loads((self.artifact / "osnip" / "config.json").read_text())
        self.key_dim = int(cfg.get("key_dim", 32))
        self._encoder = self._build_encoder(cfg)

        if self.deployment_loaded:
            self.vocab_size = int(self.model.config.vocab_size)
            self.hidden = int(self.model.config.hidden_size)
            self.n_layers = int(self.model.config.num_hidden_layers)
        else:
            cfg = json.loads((self.artifact / "model_server" / "config.json").read_text())
            self.vocab_size = int(cfg["vocab_size"])
            self.hidden = int(cfg["hidden_size"])
            self.n_layers = int(cfg["num_hidden_layers"])
        max_pos = int(self.model.config.max_position_embeddings) if self.deployment_loaded else 131072
        if self.max_len is not None:
            self.max_len = min(self.max_len, max_pos)

        # Candidate tables are *public weights* of the respective server (plan A2):
        # the obfuscated table for the AloePri-bearing arms, the clean table for the
        # plaintext arm. §7.5 forbids scoring an obfuscated wire against the clean
        # table, so the table is chosen by arm, never by convenience.
        self.E_dep = (
            self._embedding_table(self.model) if self.deployment_loaded
            else self.weight_tensor("model.embed_tokens.weight", "deployment")
        )
        self.E_clean = (
            self._embedding_table(self.clean) if self.models_loaded
            else self.weight_tensor("model.embed_tokens.weight", "clean")
        )
        self._E_dep_norm: Optional[np.ndarray] = None
        self._E_clean_norm: Optional[np.ndarray] = None

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _embedding_table(model) -> np.ndarray:
        return model.get_input_embeddings().weight.detach().float().cpu().numpy().copy()

    def _build_encoder(self, cfg: dict):
        import torch
        from torch import nn

        class KeyConditionedEncryptor(nn.Module):
            """Mirror of ``ab_runtime.py``'s encoder (mlp_residual_v1)."""

            def __init__(self, width: int, key_dim: int = 32, hidden_dim: int = 256) -> None:
                super().__init__()
                self.input_projection = nn.Linear(width, hidden_dim)
                self.key_projection = nn.Linear(key_dim, hidden_dim, bias=False)
                self.output_projection = nn.Linear(hidden_dim, width)

            def forward(self, h, key):
                if key.ndim == 1:
                    key = key.expand(*h.shape[:-1], -1)
                hidden = torch.tanh(self.input_projection(h) + self.key_projection(key))
                return h + self.output_projection(hidden)

        enc = KeyConditionedEncryptor(
            int(cfg["width"]), key_dim=self.key_dim, hidden_dim=int(cfg.get("hidden_dim", 256))
        )
        state = torch.load(self.artifact / "osnip" / "encryptor.pt", map_location="cpu", weights_only=True)
        enc.load_state_dict(state)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc

    @staticmethod
    def iso_norm_project(z, h, eps: float = 1e-12):
        z_norm = z.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        h_norm = h.float().norm(dim=-1, keepdim=True)
        return (z.float() * (h_norm / z_norm)).to(dtype=z.dtype)

    # -- arm availability ----------------------------------------------------
    def arm_status(self, arm: str) -> dict:
        if arm in ("plaintext", "aloepri", "ab"):
            return {"arm": arm, "status": "available"}
        if arm == "osnip":
            return {
                "arm": arm,
                "status": "resource_blocked",
                "reason": (
                    "the packaged OSNIP encoder was trained against the AloePri-obfuscated "
                    "embedding table (osnip/config.json model_path -> server/); applying it to "
                    "the clean table would be an out-of-distribution substitute, so this arm "
                    "needs its own clean-model encoder instance (plan G5, ~6.3h GPU) and must "
                    "not be reported as 0%"
                ),
            }
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS + ('osnip',)}")

    # -- encoding ------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """Full tokenisation. Truncation is explicit in ``observe``, never silent here."""
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="np")["input_ids"][0]
        return np.asarray(ids, dtype=np.int64)

    def _key_for(self, key_index: int, epoch: int = 0) -> np.ndarray:
        return self.key_schedule.derive(int(key_index), int(epoch))[: self.key_dim]

    # -- the single method that touches the secrets --------------------------
    def observe(
        self,
        text: str,
        *,
        arm: str,
        request_id: int = 0,
        key_index: int | None = None,
        key_seed: int | None = None,
        epoch: int = 0,
        session_id: Optional[str] = None,
        forward: bool = True,
    ) -> Wire:
        return self.observe_ids(
            self.encode(text), arm=arm, request_id=request_id, key_index=key_index,
            key_seed=key_seed, epoch=epoch, session_id=session_id, text=text,
            forward=forward,
        )

    def observe_ids(
        self,
        ids,
        *,
        arm: str,
        request_id: int = 0,
        key_index: int | None = None,
        key_seed: int | None = None,
        epoch: int = 0,
        session_id: Optional[str] = None,
        text: str = "",
        forward: bool = True,
    ) -> Wire:
        """Exact-prefix observation used by the stepwise generation protocol.

        ``forward=False`` returns the observation surface a KNN cell actually consumes
        (the uploaded ``z`` or the received token ids) without running the server
        model. The AloePri-only cell is defined on the layer-1 hidden state, so it
        always requires ``forward=True``.
        """
        import torch

        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} is not instantiable here: {self.arm_status(arm)}")
        if forward and arm == "plaintext" and self.clean is None:
            raise RuntimeError("plaintext forward needs the clean model (load_models=True)")
        if forward and arm == "aloepri" and self.model is None:
            raise RuntimeError(
                "the AloePri KNN cell is defined on the server hidden state; "
                "construct ABDeployment with load_models='deployment'"
            )
        if not forward and arm == "aloepri":
            raise ValueError(
                "AloePri C01 observation is the layer-1 hidden state; forward=True is required"
            )
        ids = np.asarray(ids, dtype=np.int64)
        full_len = int(len(ids))
        truncated = False
        if self.max_len is not None and full_len > self.max_len:
            if self.strict_length:
                raise ValueError(
                    f"request has {full_len} tokens > max_len {self.max_len}; "
                    "strict_length refuses to truncate"
                )
            ids = ids[: self.max_len]
            truncated = True
        if key_index is None:
            key_index = key_seed
        if key_index is None:
            key_index = self.key_schedule.next_index()
        key_index = int(key_index)
        ids_t = torch.from_numpy(ids).long()
        permuted = torch.from_numpy(self.perm[ids]).long()
        key = self._key_for(key_index, epoch)
        meta = self.key_schedule.public_metadata(key_index, epoch)
        common = dict(
            group_id=meta["group_id"],
            key_id=meta["group_id"],
            epoch=int(epoch),
            session_id=session_id,
            truncated=truncated,
            full_len=full_len,
        )

        with torch.no_grad():
            if arm == "plaintext":
                if forward:
                    out = self.clean(input_ids=ids_t.unsqueeze(0), output_hidden_states=True)
                    hiddens = np.stack([out.hidden_states[l][0].float().numpy() for l in self.layers])
                    logits = out.logits[0].float().numpy()
                else:
                    hiddens = None
                    logits = None
                public = WirePublic(
                    arm=arm, request_id=request_id, seq_len=len(ids), layers=self.layers,
                    wire_token_ids=ids.copy(),
                    hidden_states=hiddens, logits_perm=logits,
                    epoch=int(epoch), session_id=session_id,
                    truncated=truncated, full_len=full_len,
                )
                truth = WireTruth(plaintext_ids=ids, text=text)
                return Wire(public, truth)

            if arm == "aloepri":
                out = self.model(input_ids=permuted.unsqueeze(0), output_hidden_states=True)
                hiddens = np.stack([out.hidden_states[l][0].float().numpy() for l in self.layers])
                public = WirePublic(
                    arm=arm, request_id=request_id, seq_len=len(ids),
                    wire_token_ids=permuted.numpy().copy(), hidden_states=hiddens,
                    layers=self.layers, logits_perm=out.logits[0].float().numpy(),
                    **common,
                )
                truth = WireTruth(plaintext_ids=ids, permuted_ids=permuted.numpy().copy(), key=key, text=text)
                return Wire(public, truth)

            # arm == "ab": client permutes, looks up the *deployment* table, then OSNIP-encodes
            h = torch.from_numpy(self.E_dep[permuted.numpy()]).to(torch.float32)
            z = self.iso_norm_project(self._encoder(h, torch.from_numpy(key)), h)
            if forward:
                out = self.model(inputs_embeds=z.unsqueeze(0).to(self.dtype), output_hidden_states=True)
                hiddens = np.stack([out.hidden_states[l][0].float().numpy() for l in self.layers])
                logits = out.logits[0].float().numpy()
            else:
                hiddens = None
                logits = None
            public = WirePublic(
                arm=arm, request_id=request_id, seq_len=len(ids),
                wire_vectors=z.float().numpy().copy(), hidden_states=hiddens,
                layers=self.layers, logits_perm=logits,
                **common,
            )
            truth = WireTruth(
                plaintext_ids=ids, permuted_ids=permuted.numpy().copy(), key=key, text=text
            )
            return Wire(public, truth)

    # -- attacker-visible candidate tables ----------------------------------
    def table(self, arm: str, *, normalize: bool = False) -> np.ndarray:
        if arm == "plaintext":
            if normalize:
                if self._E_clean_norm is None:
                    self._E_clean_norm = _normalize(self.E_clean)
                return self._E_clean_norm
            return self.E_clean
        if arm in ("aloepri", "ab"):
            if normalize:
                if self._E_dep_norm is None:
                    self._E_dep_norm = _normalize(self.E_dep)
                return self._E_dep_norm
            return self.E_dep
        raise ValueError(f"no candidate table for arm {arm!r}: {self.arm_status(arm)}")

    # -- the attack-side public view ----------------------------------------
    def public_view(self, arm: str, scenario: str = "S-public"):
        """Return a :class:`PublicDeployment` holding only attacker-legal resources."""
        from .public_view import SafetensorsWeightSource, build_public_view

        source = SafetensorsWeightSource(self.artifact / "model_server", self.clean_model_path)
        return build_public_view(self, arm=arm, scenario=scenario, weight_source=source)

    # -- public weights for the S3 attacks ----------------------------------
    def weights(self, side: str) -> dict[str, np.ndarray]:
        """``side='clean'`` (attacker's reference) or ``'deployment'`` (observed)."""
        model = self.clean if side == "clean" else self.model
        out: dict[str, np.ndarray] = {}
        for name, tensor in model.state_dict().items():
            if tensor.dtype.is_floating_point:
                out[name] = tensor.detach().float().cpu().numpy()
        return out

    def weight_tensor(self, name: str, side: str) -> np.ndarray:
        """Read a single weight tensor from the safetensors shards (low memory).

        ``weights()`` below materialises both full state dicts (~8 GiB for a 1B model),
        which is enough to push this box into swap when a weight attack also holds the
        embedding tables. Weight attacks that need a handful of matrices should come through
        here instead: the index maps the tensor to its shard, so only that shard is decoded.
        """
        from safetensors import safe_open

        root = self.artifact / "model_server" if side == "deployment" else self.clean_model_path
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())["weight_map"]
            if name not in index:
                raise KeyError(f"{name!r} not in {index_path}")
            shard = root / index[name]
        else:
            shard = root / "model.safetensors"
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            tensor = fh.get_tensor(name)
        return tensor.detach().float().numpy().astype("float32", copy=False)

    def embedding_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """``(clean, deployment)`` embedding tables for the weight attacks."""
        return self.E_clean, self.E_dep

    def describe(self) -> dict:
        return {
            "artifact": str(self.artifact),
            "clean_model": str(self.clean_model_path),
            "expected_variant": self.expected_variant,
            "clean_model_variant_evidence": self.clean_model_variant_evidence,
            "public_artifact": str(self.public_artifact),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden,
            "num_layers": self.n_layers,
            "layers_reported": list(self.layers),
            "max_len": self.max_len,
            "strict_length": self.strict_length,
            "arms_available": (
                list(ARMS) if self.models_loaded
                else ["ab", "aloepri"] if self.deployment_loaded else "weight-only mode"
            ),
            "models_loaded": self.models_loaded,
            "arm_status": {a: self.arm_status(a) for a in ARMS + ("osnip",)},
            "obfuscation_metadata": self.secret_metadata,
            "osnip_encoder_params": int(sum(p.numel() for p in self._encoder.parameters())),
        }


def _normalize(table: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    return table / np.maximum(norms, 1e-12)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_model_variant(path: Path | str) -> dict:
    """Read public config/tokenizer evidence for base-vs-Instruct, without loading weights."""
    root = Path(path)
    evidence: list[str] = []
    name_instruct = "instruct" in root.name.lower()
    if name_instruct:
        evidence.append(f"directory name {root.name!r} contains 'instruct'")
    tokenizer_config = root / "tokenizer_config.json"
    chat_template = False
    if tokenizer_config.is_file():
        try:
            doc = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
        chat_template = doc.get("chat_template") is not None
        if chat_template:
            evidence.append("tokenizer_config.json contains a chat_template")
        if doc.get("name_or_path"):
            evidence.append(f"tokenizer name_or_path={doc.get('name_or_path')!r}")
    config_path = root / "config.json"
    architectures = None
    if config_path.is_file():
        try:
            architectures = json.loads(config_path.read_text(encoding="utf-8")).get("architectures")
        except json.JSONDecodeError:
            architectures = None
    return {
        "path": str(root),
        "name": root.name,
        "looks_instruct": bool(name_instruct or chat_template),
        "chat_template": bool(chat_template),
        "architectures": architectures,
        "evidence": evidence,
    }


def local_model_revision(model: dict) -> str:
    """Fingerprint the exact local model identity, without claiming a remote commit."""
    identity = {name: model[name] for name in
                ("config_sha256", "tokenizer_sha256", "weights_sha256", "files")}
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "local-sha256:" + hashlib.sha256(blob).hexdigest()


def verify_model_identity(path: Path | str, *, expected_variant: str | None = None) -> dict:
    """Hash the real config/tokenizer/weights and refuse an Instruct-as-base claim.

    Streams file bytes (never loads the weights into memory). A missing file is an
    explicit error; no hash is invented.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"model directory missing: {root}")
    variant_evidence = inspect_model_variant(root)
    if expected_variant == "base" and variant_evidence["looks_instruct"]:
        raise ValueError(
            f"{root.name} carries Instruct evidence {variant_evidence['evidence']}, "
            "but expected_variant='base'; Instruct must not substitute for the 1B base"
        )
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing under {root}")
    tokenizer_files = sorted(
        candidate
        for candidate in (
            root / "tokenizer.json",
            root / "tokenizer.model",
            root / "tokenizer_config.json",
            root / "special_tokens_map.json",
            root / "added_tokens.json",
            root / "chat_template.jinja",
            root / "vocab.json",
            root / "vocab.txt",
            root / "merges.txt",
            root / "spiece.model",
            root / "sentencepiece.bpe.model",
        )
        if candidate.is_file()
    )
    if not tokenizer_files:
        raise FileNotFoundError(f"no tokenizer files under {root}")
    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors weights under {root}")

    combined = hashlib.sha256()
    files: dict[str, str] = {}
    for file in tokenizer_files:
        digest = _file_sha256(file)
        files[str(file.relative_to(root))] = digest
        combined.update(digest.encode("ascii"))
    weights = hashlib.sha256()
    for file in weight_files:
        digest = _file_sha256(file)
        files[str(file.relative_to(root))] = digest
        weights.update(digest.encode("ascii"))

    files["config.json"] = _file_sha256(config)
    for filename in ("model.safetensors.index.json", "generation_config.json"):
        file = root / filename
        if file.is_file():
            files[filename] = _file_sha256(file)

    revision_file = root / "REFERENCE_REVISION"
    revision = revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else None
    return {
        "path": str(root),
        "source": "local directory",
        "name": root.name,
        "variant": expected_variant,
        "variant_evidence": variant_evidence,
        "revision": revision,
        "config_sha256": files["config.json"],
        "tokenizer_sha256": combined.hexdigest(),
        "weights_sha256": weights.hexdigest(),
        "files": files,
    }


# ---------------------------------------------------------------------------
# attack primitives over the public half only (no secret, no ground truth)
# ---------------------------------------------------------------------------
def knn_topk(
    queries: np.ndarray,
    table: np.ndarray,
    *,
    k: int = 100,
    metric: str = "cosine",
    query_chunk: int = 2048,
    table_norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``[n_queries, k]`` candidate token ids, nearest first (plan P0-1).

    ``table`` is the arm's candidate matrix (the deployment table for the
    AloePri-bearing arms - never the clean table, per plan §7.5). Full-vocabulary
    by construction: no proposal / ANN approximation is used (plan §1.7-4).
    """
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError(f"queries must be [n, d], got {queries.shape}")
    table = np.asarray(table, dtype=np.float32)
    if queries.shape[1] != table.shape[1]:
        raise ValueError(f"dimension mismatch: queries {queries.shape} vs table {table.shape}")
    n = queries.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    if metric == "cosine":
        tn = table_norm if table_norm is not None else _normalize(table)
        tn_t = tn.T
        for start in range(0, n, query_chunk):
            q = queries[start:start + query_chunk]
            qn = _normalize(q)
            sims = qn @ tn_t                      # [chunk, V]
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
            out[start:start + chunk_len(q)] = np.take_along_axis(idx, order, axis=1)
        return out
    if metric == "l2":
        table_sq = (table * table).sum(axis=1)[None, :]
        for start in range(0, n, query_chunk):
            q = queries[start:start + query_chunk]
            d2 = (q * q).sum(axis=1)[:, None] - 2.0 * (q @ table.T) + table_sq
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
            out[start:start + chunk_len(q)] = np.take_along_axis(idx, order, axis=1)
        return out
    raise ValueError(f"unknown metric {metric!r}")


def chunk_len(x: np.ndarray) -> int:
    return int(x.shape[0])


def flatten_hidden(wire: WirePublic, layer_index: int, *, skip_first: bool = True) -> np.ndarray:
    """``[seq_len, d]`` hidden states for one reported layer."""
    if wire.hidden_states is None:
        raise ValueError("wire has no hidden states")
    h = wire.hidden_states[layer_index]
    return h[1:] if skip_first else h
