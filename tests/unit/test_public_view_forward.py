"""Acceptance tests for S01/S10: the public HuggingFace forward adapter.

S01: ``_hf_forward`` must feed a real Llama a tensor of the model's device/dtype,
return hidden states (``output_hidden_states`` was never enabled) for the NumPy
callers, and keep the ISA autograd path differentiable. S10: ``build_public_view``
must hand the tokenizer decoder to the public view.

The fixture is a tiny randomly initialised real ``LlamaForCausalLM``, so this
checks the actual HuggingFace calling contract, not a stub.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from obfuscation_attacks.data.contracts import CapabilityError
from obfuscation_attacks.schemes.ab_deployment import ABDeployment
from obfuscation_attacks.schemes.public_view import PublicDeployment, build_public_view

pytestmark = pytest.mark.unit

HIDDEN = 32
VOCAB = 64
N_LAYERS = 2


class StubTokenizer:
    """Minimal public tokenizer surface: callable encode plus decode."""

    def __init__(self) -> None:
        self.decoded: list[list[int]] = []

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str = "np"):
        ids = np.arange(len(str(text)), dtype=np.int64)
        return {"input_ids": ids[None, :]}

    def decode(self, token_ids):
        self.decoded.append([int(t) for t in token_ids])
        return "decoded:" + ",".join(str(int(t)) for t in token_ids)


class FakeDeployment:
    """Duck-typed deployment exposing only public pieces plus the server model."""

    def __init__(self, model) -> None:
        self.model = model
        self.artifact = ""
        self.vocab_size = VOCAB
        self.hidden = HIDDEN
        self.n_layers = N_LAYERS
        self.layers = (1,)
        self.max_len = None
        self.tokenizer = StubTokenizer()
        self.tokenizer_id = "stub"
        self.E_dep = np.zeros((VOCAB, HIDDEN), dtype=np.float32)
        self.E_clean = None

    def arm_status(self, arm: str) -> dict:
        return {"arm": arm, "status": "available"}


@pytest.fixture(scope="module")
def llama():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB,
    )
    model = transformers.LlamaForCausalLM(config)
    model.eval()
    return model


class _CleanWeightSource:
    """Public weight source with distinct deployment and clean tensors."""

    def __init__(self) -> None:
        self.deployment = -np.arange(16, dtype=np.float32).reshape(4, 4)
        self.clean = np.arange(16, dtype=np.float32).reshape(4, 4)

    def get(self, name: str, side: str) -> np.ndarray:
        return self.clean if side == "clean" else self.deployment

    def for_scenario(self, allow_clean: bool):
        return self


def _deployment_with_weights(weight_source, *, clean_table=None):
    return SimpleNamespace(
        artifact="",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=clean_table,
        weight_source=weight_source,
        forward_fn=lambda embeds: "deployment",
        clean_forward_fn=lambda embeds: "clean",
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )


def _public(model):
    return build_public_view(FakeDeployment(model), arm="ab", scenario="S-public")


def _deployment_with_clean():
    """A deployment that has both a deployment and a clean model surface."""
    return SimpleNamespace(
        artifact="",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=np.ones((8, 4), dtype=np.float32),
        forward_fn=lambda embeds: "deployment",
        clean_forward_fn=lambda embeds: "clean",
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )


def test_numpy_forward_matches_the_native_hidden_states(llama):
    public = _public(llama)
    embeds = np.random.default_rng(1).normal(size=(2, 5, HIDDEN)).astype(np.float32)

    out = public.forward(embeds)

    assert isinstance(out.hidden_states, list)
    assert len(out.hidden_states) == N_LAYERS + 1
    with torch.no_grad():
        native = llama(
            inputs_embeds=torch.as_tensor(embeds),
            output_hidden_states=True,
            use_cache=False,
        )
    for got, want in zip(out.hidden_states, native.hidden_states):
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.float32
        assert got.shape == tuple(want.shape)
        np.testing.assert_allclose(got, want.detach().numpy(), atol=1e-5, rtol=0.0)

    assert isinstance(out.logits, np.ndarray)
    assert out.logits.dtype == np.float32
    assert out.logits.shape == (2, 5, VOCAB)

    # Model hygiene: eval mode and frozen parameters after the call.
    assert not llama.training
    assert all(not parameter.requires_grad for parameter in llama.parameters())


def test_torch_forward_keeps_the_autograd_graph_on_the_input(llama):
    public = _public(llama)
    embeds = torch.randn(1, 4, HIDDEN, requires_grad=True)

    out = public.forward(embeds)

    hidden = out.hidden_states[-1]
    assert torch.is_tensor(hidden)
    assert hidden.dtype == torch.float32
    assert hidden.grad_fn is not None
    hidden.sum().backward()
    assert embeds.grad is not None
    assert torch.isfinite(embeds.grad).all()
    assert all(parameter.grad is None for parameter in llama.parameters())


def test_torch_forward_cast_keeps_the_graph_under_outer_no_grad(llama):
    public = _public(llama)
    embeds = torch.randn(1, 4, HIDDEN, dtype=torch.float64, requires_grad=True)

    with torch.no_grad():
        out = public.forward(embeds)

    hidden = out.hidden_states[-1]
    assert torch.is_tensor(hidden)
    assert hidden.requires_grad
    assert hidden.grad_fn is not None
    hidden.sum().backward()
    assert embeds.grad is not None
    assert embeds.grad.dtype == torch.float64
    assert torch.isfinite(embeds.grad).all()
    assert all(parameter.grad is None for parameter in llama.parameters())


def test_build_public_view_exposes_the_tokenizer_decode(llama):
    deployment = FakeDeployment(llama)
    public = build_public_view(deployment, arm="ab", scenario="S-public")

    assert public.decode([1, 2, 3]) == "decoded:1,2,3"
    assert deployment.tokenizer.decoded[-1] == [1, 2, 3]
    assert public.describe()["has_decode"] is True


def test_plaintext_arm_is_refused_in_s_deploy_and_ab_stays_deployment_only():
    deployment = SimpleNamespace(
        artifact="",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=np.ones((8, 4), dtype=np.float32),
        forward_fn=lambda embeds: "deployment",
        clean_forward_fn=lambda embeds: "clean",
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )

    # The plaintext server is the clean model; S-deploy exposes deployment weights
    # only, so the combination must be refused before anything is constructed.
    with pytest.raises(ValueError, match="plaintext arm is incompatible"):
        build_public_view(deployment, arm="plaintext", scenario="S-deploy")

    # The deployment-only invariant is unchanged.
    public = build_public_view(deployment, arm="ab", scenario="S-deploy")
    assert public.E_clean is None
    assert public.clean_forward_fn is None
    with pytest.raises(CapabilityError):
        public.table("clean")
    with pytest.raises(CapabilityError):
        public.table("plaintext")
    with pytest.raises(CapabilityError):
        public.weight_tensor("w", "clean")
    with pytest.raises(CapabilityError):
        public.forward("embeds", clean=True)
    assert public.forward("embeds") == "deployment"


@pytest.mark.parametrize("scenario", ["S-public", "S-pair"])
def test_clean_access_is_public_in_s_public_and_s_pair(scenario):
    deployment = _deployment_with_clean()
    public = build_public_view(deployment, arm="ab", scenario=scenario)

    assert public.allow_clean_weights is True
    np.testing.assert_array_equal(public.table("plaintext"), deployment.E_clean)
    assert public.forward("embeds", clean=True) == "clean"


def test_s_encoder_view_inherits_the_clean_side():
    deployment = _deployment_with_clean()
    public = build_public_view(deployment, arm="ab", scenario="S-encoder")

    assert public.allow_clean_weights is True
    np.testing.assert_array_equal(public.table("plaintext"), deployment.E_clean)
    assert public.forward("embeds", clean=True) == "clean"

    # The deployment side still works.
    np.testing.assert_array_equal(public.table("ab"), deployment.E_dep)
    assert public.forward("embeds") == "deployment"


def _ab_deployment_stub():
    """ABDeployment instance without artifact/model loading, for view wiring."""
    deployment = object.__new__(ABDeployment)
    deployment.artifact = Path("no-such-artifact")
    deployment.public_artifact = "no-such-artifact/model_server"
    deployment.clean_model_path = None
    deployment.E_dep = np.zeros((8, 4), dtype=np.float32)
    deployment.E_clean = np.ones((8, 4), dtype=np.float32)
    deployment.model = None
    deployment.clean = None
    deployment.tokenizer = None
    deployment.encode_fn = None
    deployment.decode_fn = None
    deployment.vocab_size = 8
    deployment.hidden = 4
    deployment.n_layers = 2
    deployment.layers = (1,)
    deployment.max_len = None
    return deployment


def test_ab_deployment_public_view_grants_s_encoder_clean_access():
    deployment = _ab_deployment_stub()

    granted = deployment.public_view("ab", "S-public")
    assert granted.allow_clean_weights is True
    np.testing.assert_array_equal(granted.table("plaintext"), deployment.E_clean)

    encoder = deployment.public_view("ab", "S-encoder")
    assert encoder.allow_clean_weights is True
    np.testing.assert_array_equal(encoder.table("plaintext"), deployment.E_clean)

    # The plaintext arm follows the same scenario gate for its clean-model server.
    plaintext = deployment.public_view("plaintext", "S-encoder")
    assert plaintext.allow_clean_weights is True
    np.testing.assert_array_equal(plaintext.table(), deployment.E_clean)
    with pytest.raises(CapabilityError):
        plaintext.forward("embeds")


def test_s_public_clean_weights_are_public_without_a_clean_table():
    source = _CleanWeightSource()
    public = build_public_view(
        _deployment_with_weights(source), arm="ab", scenario="S-public"
    )

    # Published base weights are protocol-public even without an E_clean table.
    assert public.allow_clean_weights is True
    assert public.E_clean is None
    np.testing.assert_array_equal(public.weight_tensor("w", "clean"), source.clean)
    assert public.forward("embeds", clean=True) == "clean"
    with pytest.raises(CapabilityError):
        public.table("plaintext")


def test_s_public_with_a_clean_table_allows_table_weights_and_forward():
    source = _CleanWeightSource()
    clean_table = np.full((8, 4), 0.5, dtype=np.float32)
    public = build_public_view(
        _deployment_with_weights(source, clean_table=clean_table),
        arm="ab",
        scenario="S-public",
    )

    np.testing.assert_array_equal(public.table("plaintext"), clean_table)
    np.testing.assert_array_equal(public.weight_tensor("w", "clean"), source.clean)
    assert public.forward("embeds", clean=True) == "clean"


def test_plaintext_arm_without_a_clean_model_fails_closed_on_forward():
    deployment = SimpleNamespace(
        artifact="",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=np.ones((8, 4), dtype=np.float32),
        forward_fn=lambda embeds: "deployment",
        tokenizer=StubTokenizer(),
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )
    public = build_public_view(deployment, arm="plaintext", scenario="S-public")

    # No clean model: the deployment forward must never stand in for it.
    with pytest.raises(CapabilityError):
        public.forward("embeds")
    # The table-only surface stays public.
    assert np.array_equal(public.table("plaintext"), deployment.E_clean)
    assert public.encode("abc").tolist() == [0, 1, 2]
    assert public.decode([1, 2]) == "decoded:1,2"


def test_plaintext_arm_with_a_clean_model_forwards_to_it(llama):
    class _DeploymentModel:
        def forward(self, inputs_embeds=None):
            raise AssertionError("the deployment model must not serve the plaintext arm")

    deployment = SimpleNamespace(
        artifact="",
        vocab_size=VOCAB,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        layers=(1,),
        E_dep=np.zeros((VOCAB, HIDDEN), dtype=np.float32),
        E_clean=np.ones((VOCAB, HIDDEN), dtype=np.float32),
        model=_DeploymentModel(),
        clean=llama,
        tokenizer=StubTokenizer(),
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )
    public = build_public_view(deployment, arm="plaintext", scenario="S-public")

    embeds = np.random.default_rng(2).normal(size=(1, 3, HIDDEN)).astype(np.float32)
    out = public.forward(embeds)
    assert len(out.hidden_states) == N_LAYERS + 1
    with torch.no_grad():
        native = llama(
            inputs_embeds=torch.as_tensor(embeds),
            output_hidden_states=True,
            use_cache=False,
        )
    np.testing.assert_allclose(
        out.hidden_states[-1], native.hidden_states[-1].detach().numpy(), atol=1e-5, rtol=0.0
    )


def test_hand_built_view_gates_clean_table_and_forward_on_allow_clean_weights():
    e_dep = np.arange(32, dtype=np.float32).reshape(8, 4)
    public = PublicDeployment(
        arm="ab",
        scenario="S-public",
        artifact="fixture",
        vocab_size=8,
        hidden_size=4,
        n_layers=2,
        layers=(1,),
        max_len=None,
        tokenizer_id="fixture",
        E_server=e_dep,
        E_clean=-e_dep,
        forward_fn=lambda embeds: "deployment",
        clean_forward_fn=lambda embeds: "clean",
        allow_clean_weights=False,
    )

    # Field presence alone must never unlock the clean surface.
    with pytest.raises(CapabilityError):
        public.table("plaintext")
    with pytest.raises(CapabilityError):
        public.forward("embeds", clean=True)

    # The deployment surface of the same hand-built view still works.
    np.testing.assert_array_equal(public.table("ab"), e_dep)
    assert public.forward("embeds") == "deployment"


def test_custom_forward_fn_still_passes_through_unchanged():
    sentinel = SimpleNamespace(
        hidden_states=[np.zeros((1, 1, 4), dtype=np.float32)], logits=None
    )
    deployment = SimpleNamespace(
        artifact="fixture",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=None,
        forward_fn=lambda embeds: sentinel,
        arm_status=lambda arm: {"arm": arm, "status": "available"},
    )
    public = build_public_view(deployment, arm="ab", scenario="S-public")
    assert public.forward("sentinel-input") is sentinel


def test_model_without_hidden_states_raises_a_clear_error():
    class NoHiddenStates:
        def forward(self, inputs_embeds=None):
            return SimpleNamespace(logits=torch.zeros(1))

    deployment = SimpleNamespace(
        artifact="fixture",
        vocab_size=8,
        hidden=4,
        n_layers=2,
        layers=(1,),
        E_dep=np.zeros((8, 4), dtype=np.float32),
        E_clean=None,
        model=NoHiddenStates(),
        arm_status=lambda arm: {},
    )
    public = build_public_view(deployment, arm="ab", scenario="S-public")
    with pytest.raises(RuntimeError, match="hidden states"):
        public.forward(np.zeros((1, 2, 4), dtype=np.float32))
