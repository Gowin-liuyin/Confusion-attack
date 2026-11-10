"""The attack-side public view of the deployment (protocol §1, §8.1).

An attack receives only what an honest-but-curious server really holds: the
server model and its public weights, the observed messages, the public
tokenizer, and (per scenario) the published base model or target encoder
weights. The client root key, the token permutation, the encoder and all
ground truth stay on the generation/scoring side.

The view never holds a reference to the secret-bearing deployment object; it is
built from explicit public pieces. ``assert_public`` makes the boundary
executable in tests.
"""

from __future__ import annotations

import functools
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..data.contracts import CapabilityError

SCENARIOS = ("S-public", "S-deploy", "S-encoder", "S-pair")

FORBIDDEN_NAMES = (
    "perm",
    "inv_perm",
    "key",
    "root_key",
    "secret",
    "secret_metadata",
    "encoder",
    "_encoder",
    "observe",
    "truth",
    "plaintext_ids",
    "permuted_ids",
)


def _normalize(table: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    return table / np.maximum(norms, 1e-12)


def _encode_with_tokenizer(tokenizer, text: str) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="np")["input_ids"][0]
    return np.asarray(ids, dtype=np.int64)


def _hf_forward(model, inputs_embeds):
    """Call a HuggingFace model with the embeddings keyword, never positionally.

    The adapter serves two callers with different contracts:

    * NumPy input (VMA / BeamClean): the array is converted to a float32 tensor
      on the model's device and embedding dtype, run under ``torch.no_grad()``
      with ``output_hidden_states=True``/``use_cache=False``, and returned as an
      object whose ``hidden_states`` are detached CPU float32 NumPy arrays (and
      ``logits`` likewise, when the model produced them).
    * Torch input (ISA): the tensor is kept attached to its autograd graph (no
      detach, no copy; only ``.to`` when device/dtype differ) and the native HF
      output is returned unchanged, so the attack's backward pass reaches the
      input embeddings. Gradients are enabled for exactly this call.

    The server model is put in eval mode with frozen parameters once per model
    instance; the whole forward is never wrapped in ``no_grad``.
    """
    import torch

    if isinstance(inputs_embeds, torch.Tensor):
        return _hf_forward_torch(model, inputs_embeds)
    if isinstance(inputs_embeds, np.ndarray):
        return _hf_forward_numpy(model, inputs_embeds)
    # Fixture inputs that are neither arrays nor tensors keep the old pass-through
    # behaviour (the model receives them untouched and its output is returned).
    return _call_forward(_forward_target(model), inputs_embeds, {})


def _forward_target(model):
    """The callable to invoke: ``forward`` when present, else the model itself."""
    forward = getattr(model, "forward", None)
    return forward if forward is not None else model


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (AttributeError, StopIteration, TypeError):
            pass
    return "cpu"


def _embedding_dtype(model):
    """The dtype the model's input embeddings expect (float32 when unknown)."""
    import torch

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            weight = get_embeddings().weight
            if weight.is_floating_point():
                return weight.dtype
        except (AttributeError, TypeError):
            pass
    return torch.float32


_PREPARED_ATTR = "_public_view_prepared"
_KWARGS_ATTR = "_public_view_forward_kwargs"
_FORWARD_FLAGS = {"output_hidden_states": True, "use_cache": False}


def _prepare_model(model) -> None:
    """Put the server model in eval mode and freeze it, once per instance."""
    try:
        if getattr(model, _PREPARED_ATTR, False):
            return
    except AttributeError:
        pass
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if getattr(parameter, "requires_grad", False):
                parameter.requires_grad_(False)
    try:
        setattr(model, _PREPARED_ATTR, True)
    except (AttributeError, TypeError):
        pass


def _forward_kwargs(model, target) -> dict:
    """``output_hidden_states``/``use_cache`` flags the forward signature accepts.

    The accepted set is inspected once and cached on the model, so a model whose
    signature lacks the flags is not called with them (and a model without hidden
    states raises in :func:`_require_hidden_states` rather than silently returning
    an output without them).
    """
    supported = None
    try:
        supported = getattr(model, _KWARGS_ATTR, None)
    except AttributeError:
        supported = None
    if supported is None:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            supported = tuple(_FORWARD_FLAGS)
        else:
            parameters = signature.parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                supported = tuple(_FORWARD_FLAGS)
            else:
                supported = tuple(name for name in _FORWARD_FLAGS if name in parameters)
        try:
            setattr(model, _KWARGS_ATTR, supported)
        except (AttributeError, TypeError):
            pass
    return {name: _FORWARD_FLAGS[name] for name in supported}


def _call_forward(target, inputs_embeds, kwargs):
    if kwargs:
        return target(inputs_embeds=inputs_embeds, **kwargs)
    return target(inputs_embeds=inputs_embeds)


def _require_hidden_states(output, model, kwargs):
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        detail = (
            "its forward signature does not accept output_hidden_states"
            if "output_hidden_states" not in kwargs
            else "output_hidden_states=True did not produce hidden states"
        )
        raise RuntimeError(
            f"{type(model).__name__} cannot expose hidden states: {detail}; "
            "the public attack forward requires them"
        )
    return hidden_states


def _detached_cpu(array) -> np.ndarray:
    import torch

    if isinstance(array, np.ndarray):
        return np.asarray(array, dtype=np.float32)
    return array.detach().to(device="cpu", dtype=torch.float32).numpy()


@dataclass
class _NumpyForwardOutput:
    """Detached CPU view of an HF output for the NumPy (attack) path."""

    hidden_states: list[np.ndarray]
    logits: Optional[np.ndarray] = None


def _hf_forward_numpy(model, inputs_embeds) -> "_NumpyForwardOutput":
    import torch

    _prepare_model(model)
    target = _forward_target(model)
    kwargs = _forward_kwargs(model, target)
    tensor = torch.as_tensor(np.asarray(inputs_embeds, dtype=np.float32))
    device = torch.device(_model_device(model))
    dtype = _embedding_dtype(model)
    if tensor.device != device or tensor.dtype != dtype:
        tensor = tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        output = _call_forward(target, tensor, kwargs)
    hidden = [_detached_cpu(state) for state in _require_hidden_states(output, model, kwargs)]
    logits = getattr(output, "logits", None)
    return _NumpyForwardOutput(
        hidden_states=hidden,
        logits=None if logits is None else _detached_cpu(logits),
    )


def _hf_forward_torch(model, inputs_embeds):
    import torch

    _prepare_model(model)
    target = _forward_target(model)
    kwargs = _forward_kwargs(model, target)
    device = torch.device(_model_device(model))
    dtype = _embedding_dtype(model)
    # Never no_grad here: ISA differentiates through the input embeddings. The
    # device/dtype cast is inside the block too, so a conversion still records its
    # autograd edge even when the caller wrapped the call in ``torch.no_grad()``.
    with torch.enable_grad():
        tensor = inputs_embeds
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype)
        output = _call_forward(target, tensor, kwargs)
    _require_hidden_states(output, model, kwargs)
    return output


class SafetensorsWeightSource:
    """Read one public weight tensor without materialising a state dict.

    Holds only filesystem paths to public artifacts; it never holds the client
    secret, the permutation or the encoder.
    """
    def __init__(
        self,
        deployment_root: Path | str | None,
        clean_root: Path | str | None = None,
    ) -> None:
        self.deployment_root = Path(deployment_root) if deployment_root else None
        self.clean_root = Path(clean_root) if clean_root else None

    def get(self, name: str, side: str) -> np.ndarray:
        root = self.clean_root if side == "clean" else self.deployment_root
        if root is None:
            raise CapabilityError(f"no public {side} weights in this scenario")
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            if name not in index:
                raise KeyError(f"{name!r} not in {index_path}")
            shard = root / index[name]
        else:
            shard = root / "model.safetensors"
        from safetensors import safe_open

        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        return tensor.detach().float().cpu().numpy().astype("float32", copy=False)

    def for_scenario(self, allow_clean: bool) -> "SafetensorsWeightSource":
        """Return a source that physically cannot read the clean side when disallowed."""
        return SafetensorsWeightSource(
            self.deployment_root, self.clean_root if allow_clean else None
        )

    def __repr__(self) -> str:
        return (
            f"SafetensorsWeightSource(deployment={self.deployment_root}, "
            f"clean={self.clean_root})"
        )


class ScenarioWeightSource:
    """A weight source that refuses the clean side outside the allowed scenarios."""

    def __init__(self, inner: Any, *, allow_clean: bool) -> None:
        self._inner = inner
        self._allow_clean = bool(allow_clean)

    def get(self, name: str, side: str) -> np.ndarray:
        if side == "clean" and not self._allow_clean:
            raise CapabilityError("clean weights are not public in this scenario")
        return self._inner.get(name, side)

    def __repr__(self) -> str:
        return f"ScenarioWeightSource(allow_clean={self._allow_clean})"


@dataclass
class PublicDeployment:
    """Attacker-visible deployment surface for exactly one arm and scenario."""

    arm: str
    scenario: str
    artifact: str
    vocab_size: int
    hidden_size: int
    n_layers: int
    layers: tuple[int, ...]
    max_len: Optional[int]
    tokenizer_id: str
    n_heads: int = 1
    n_kv_heads: int = 1
    E_server: Optional[np.ndarray] = None
    E_clean: Optional[np.ndarray] = None
    weight_source: Optional[Any] = None
    forward_fn: Optional[Callable[..., Any]] = None
    clean_forward_fn: Optional[Callable[..., Any]] = None
    encode_fn: Optional[Callable[[str], np.ndarray]] = None
    decode_fn: Optional[Callable[[Any], str]] = None
    arm_status_info: dict = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    allow_clean_weights: bool = False
    _norm_cache: dict = field(default_factory=dict, repr=False)

    # -- public resources ----------------------------------------------------
    def table(self, arm: Optional[str] = None, *, normalize: bool = False) -> np.ndarray:
        selected = arm or self.arm
        if selected not in {self.arm, "plaintext"}:
            raise CapabilityError(
                f"no public table for arm {selected!r}; this view is for {self.arm!r}"
            )
        if selected == "plaintext":
            if not self.allow_clean_weights:
                raise CapabilityError(
                    f"clean table is not public in scenario {self.scenario!r}"
                )
            table = self.E_clean
            if table is None:
                raise CapabilityError(
                    f"clean table is not public in scenario {self.scenario!r}"
                )
        else:
            table = self.E_server
        if table is None:
            raise CapabilityError(
                f"no public table for arm {selected!r} in scenario {self.scenario!r}"
            )
        if not normalize:
            return table
        cached = self._norm_cache.get(id(table))
        if cached is None:
            cached = _normalize(np.asarray(table, dtype=np.float32))
            self._norm_cache[id(table)] = cached
        return cached

    def weight_tensor(self, name: str, side: str) -> np.ndarray:
        if side not in {"deployment", "clean"}:
            raise ValueError(f"side must be 'deployment' or 'clean'; got {side!r}")
        if side == "clean" and not self.allow_clean_weights:
            raise CapabilityError(
                f"clean weights are not public in scenario {self.scenario!r}"
            )
        if self.weight_source is None:
            raise CapabilityError(
                f"no public weight source for side {side!r} in this artifact"
            )
        return np.asarray(self.weight_source.get(name, side), dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        if self.encode_fn is None:
            raise CapabilityError("no public tokenizer in this view")
        return np.asarray(self.encode_fn(text), dtype=np.int64)

    def decode(self, token_ids) -> str:
        if self.decode_fn is None:
            raise CapabilityError("no public tokenizer decoder in this view")
        return self.decode_fn(token_ids)

    def forward(self, inputs_embeds, *, clean: bool = False):
        if clean and not self.allow_clean_weights:
            raise CapabilityError(
                f"clean forward is not public in scenario {self.scenario!r}"
            )
        fn = self.clean_forward_fn if clean else self.forward_fn
        if fn is None:
            raise CapabilityError(
                f"no public {'clean ' if clean else ''}forward function in this view"
            )
        return fn(inputs_embeds)

    def arm_status(self) -> dict:
        return dict(self.arm_status_info)

    # -- serialisable description (never a secret) ---------------------------
    def describe(self) -> dict:
        return {
            "arm": self.arm,
            "scenario": self.scenario,
            "artifact": self.artifact,
            "vocab_size": int(self.vocab_size),
            "hidden_size": int(self.hidden_size),
            "n_layers": int(self.n_layers),
            "layers": list(self.layers),
            "max_len": self.max_len,
            "tokenizer_id": self.tokenizer_id,
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "has_server_table": self.E_server is not None,
            "has_clean_table": self.E_clean is not None,
            "allow_clean_weights": bool(self.allow_clean_weights),
            "has_weight_source": self.weight_source is not None,
            "has_forward": self.forward_fn is not None,
            "has_clean_forward": self.clean_forward_fn is not None,
            "has_encode": self.encode_fn is not None,
            "has_decode": self.decode_fn is not None,
            "arm_status": dict(self.arm_status_info),
            "missing": list(self.missing),
        }

    def __repr__(self) -> str:
        return (
            f"PublicDeployment(arm={self.arm!r}, scenario={self.scenario!r}, "
            f"vocab={self.vocab_size}, clean_weights={self.allow_clean_weights})"
        )


def build_public_view(
    deployment,
    *,
    arm: str,
    scenario: str,
    weight_source: Optional[Any] = None,
) -> PublicDeployment:
    """Build the public view from explicit public pieces only.

    The plaintext arm's server *is* the clean model, so that arm is incompatible
    with ``S-deploy`` (deployment weights only): the combination is refused here
    rather than silently exposing the clean table, weights and forward.

    The clean side is public in ``S-public``, ``S-encoder`` and ``S-pair``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if arm == "plaintext" and scenario == "S-deploy":
        raise ValueError(
            "plaintext arm is incompatible with scenario 'S-deploy': the plaintext "
            "server is the clean model, and S-deploy exposes deployment weights only"
        )

    E_dep = getattr(deployment, "E_dep", None)
    E_clean = getattr(deployment, "E_clean", None)
    # The grant covers
    # the published clean *weights* too, so it does not depend on the clean
    # embedding table being present; ``table("plaintext")`` still fails closed when
    # ``E_clean`` itself is None.
    allow_clean = scenario in {"S-public", "S-encoder", "S-pair"}
    if arm == "plaintext":
        # The plaintext arm's server is the clean model; in a scenario that does not
        # grant the clean side it has no server table at all (fail closed).
        E_server = E_clean if allow_clean else None
    else:
        E_server = E_dep

    missing: list[str] = []
    if E_server is None:
        missing.append(f"server table unavailable for arm {arm!r}")
    if not allow_clean:
        missing.append(f"clean weights are not public in scenario {scenario!r}")

    if weight_source is None:
        weight_source = getattr(deployment, "weight_source", None)
    if weight_source is None:
        artifact = getattr(deployment, "artifact", None)
        clean = getattr(deployment, "clean_model_path", None)
        if artifact is not None:
            weight_source = SafetensorsWeightSource(Path(artifact) / "model_server", clean)
    if weight_source is not None:
        if hasattr(weight_source, "for_scenario"):
            weight_source = weight_source.for_scenario(allow_clean)
        elif not allow_clean:
            raise CapabilityError(
                "weight source cannot be restricted to the deployment side; "
                "refusing to build a clean-accessible view"
            )

    forward_fn = getattr(deployment, "forward_fn", None)
    clean_forward_fn = getattr(deployment, "clean_forward_fn", None)
    if forward_fn is None:
        model = getattr(deployment, "model", None)
        if model is not None:
            forward_fn = functools.partial(_hf_forward, model)
    if clean_forward_fn is None and allow_clean:
        clean = getattr(deployment, "clean", None)
        if clean is not None:
            clean_forward_fn = functools.partial(_hf_forward, clean)
    if not allow_clean:
        clean_forward_fn = None
    if arm == "plaintext":
        # The plaintext arm's server is the clean model, never the deployment one.
        # Without a clean forward the view stays table-only and ``forward()`` fails
        # closed (CapabilityError) instead of silently serving the wrong server.
        forward_fn = clean_forward_fn

    encode_fn = getattr(deployment, "encode_fn", None)
    decode_fn = getattr(deployment, "decode_fn", None)
    tokenizer = getattr(deployment, "tokenizer", None)
    if encode_fn is None and tokenizer is not None:
        encode_fn = functools.partial(_encode_with_tokenizer, tokenizer)
    if decode_fn is None and tokenizer is not None:
        decode_fn = tokenizer.decode

    tokenizer_id = getattr(deployment, "tokenizer_id", None) or (
        type(tokenizer).__name__ if tokenizer is not None else "unknown"
    )
    config = getattr(deployment, "config", None)
    if config is None:
        model = getattr(deployment, "model", None)
        config = getattr(model, "config", None) if model is not None else None
    if config is None:
        artifact = getattr(deployment, "artifact", None)
        config_path = Path(artifact) / "model_server" / "config.json" if artifact else None
        if config_path is not None and config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
    def _cfg(name: str, default: int) -> int:
        if isinstance(config, dict):
            return int(config.get(name, default))
        return int(getattr(config, name, default)) if config is not None else default

    n_heads = _cfg("num_attention_heads", 1)
    n_kv_heads = _cfg("num_key_value_heads", n_heads)
    status_fn = getattr(deployment, "arm_status", None)
    try:
        status = status_fn(arm) if callable(status_fn) else {}
    except ValueError:
        status = {}
    if not isinstance(status, dict):
        status = {}

    public_artifact = getattr(deployment, "public_artifact", None)
    if public_artifact is None:
        artifact = getattr(deployment, "artifact", "")
        public_artifact = str(Path(artifact) / "model_server") if artifact else ""

    return PublicDeployment(
        arm=arm,
        scenario=scenario,
        artifact=str(public_artifact),
        vocab_size=int(getattr(deployment, "vocab_size", 0)),
        hidden_size=int(getattr(deployment, "hidden", 0) or getattr(deployment, "hidden_size", 0)),
        n_layers=int(getattr(deployment, "n_layers", 0)),
        layers=tuple(int(layer) for layer in getattr(deployment, "layers", ()) or ()),
        max_len=getattr(deployment, "max_len", None),
        tokenizer_id=str(tokenizer_id),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        E_server=E_server,
        E_clean=E_clean if allow_clean else None,
        weight_source=weight_source,
        forward_fn=forward_fn,
        clean_forward_fn=clean_forward_fn,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        arm_status_info=status,
        missing=tuple(missing),
        allow_clean_weights=allow_clean,
    )


def assert_public(obj) -> bool:
    """Raise if a view carries any forbidden secret-adjacent attribute name."""
    for name in FORBIDDEN_NAMES:
        if hasattr(obj, name):
            raise AssertionError(
                f"public view exposes forbidden attribute {name!r}; "
                "attacks must not receive secrets or ground truth"
            )
    return True
