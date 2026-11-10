"""Acceptance tests for the W1 public attack view (protocol §1, §8.1).

An attack receives only what the honest-but-curious server really holds. These
tests use a fake deployment with a sentinel secret so that any leaked reference
to the permutation, key or encoder fails loudly.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from obfuscation_attacks.data.contracts import CapabilityError
from obfuscation_attacks.schemes.ab_deployment import WirePublic, knn_topk
from obfuscation_attacks.schemes.public_view import (
    PublicDeployment,
    assert_public,
    build_public_view,
)

pytestmark = pytest.mark.unit

FORBIDDEN = (
    "perm", "inv_perm", "key", "root_key", "secret", "secret_metadata",
    "encoder", "_encoder", "observe", "truth", "plaintext_ids", "permuted_ids",
)


class FakeWeightSource:
    def __init__(self) -> None:
        self.weights = {
            ("deployment", "w"): np.arange(16, dtype=np.float32).reshape(4, 4),
            ("clean", "w"): -np.arange(16, dtype=np.float32).reshape(4, 4),
        }

    def get(self, name: str, side: str) -> np.ndarray:
        return self.weights[(side, name)]

    def for_scenario(self, allow_clean: bool):
        if allow_clean:
            return self
        return _DeploymentOnlySource(self.weights)


class _DeploymentOnlySource:
    def __init__(self, weights) -> None:
        self._deployment = dict(weights)

    def get(self, name: str, side: str) -> np.ndarray:
        if side != "deployment":
            raise CapabilityError(f"{side} weights are not public in S-deploy")
        return self._deployment[(side, name)]


class FakeDeployment:
    """Duck-typed stand-in with a sentinel permutation and secret metadata."""

    def __init__(self) -> None:
        self.artifact = "sentinel-artifact"
        self.vocab_size = 8
        self.hidden = 4
        self.n_layers = 2
        self.layers = (1,)
        self.max_len = 8
        self.tokenizer_id = "fake-tokenizer"
        self.E_dep = np.arange(32, dtype=np.float32).reshape(8, 4)
        self.E_clean = -self.E_dep
        self.perm = np.array([3, 1, 0, 2, 7, 6, 5, 4], dtype=np.int64)
        self.inv_perm = np.argsort(self.perm)
        self.secret_metadata = {"sentinel": "SENTINEL-SECRET-VALUE"}
        self._encoder = object()
        self.weight_source = FakeWeightSource()
        self.arm_status_info = {"ab": {"arm": "ab", "status": "available"}}

    def encode(self, text: str) -> np.ndarray:
        return np.arange(len(text), dtype=np.int64)

    def table(self, arm: str, normalize: bool = False) -> np.ndarray:
        return self.E_dep


def _view(scenario: str = "S-public", arm: str = "ab") -> PublicDeployment:
    return build_public_view(FakeDeployment(), arm=arm, scenario=scenario)


def test_public_view_exposes_no_secret_surface():
    fake = FakeDeployment()
    view = build_public_view(fake, arm="ab", scenario="S-public")
    for name in FORBIDDEN:
        assert not hasattr(view, name), name
    assert_public(view)
    described = view.describe()
    blob = json.dumps(described) + repr(view)
    assert "SENTINEL-SECRET-VALUE" not in blob
    assert fake.perm.tobytes() not in pickle.dumps(view)
    assert fake.inv_perm.tobytes() not in pickle.dumps(view)


def test_public_prediction_is_invariant_to_truth_mutation():
    fake = FakeDeployment()
    view = build_public_view(fake, arm="ab", scenario="S-public")
    queries = np.asarray(fake.E_dep[:2], dtype=np.float32)
    first = knn_topk(queries, view.table("ab"), k=3, metric="l2")
    fake.perm = np.arange(8, dtype=np.int64)[::-1].copy()
    fake.inv_perm = np.argsort(fake.perm)
    fake.secret_metadata["sentinel"] = "CHANGED"
    second = knn_topk(queries, view.table("ab"), k=3, metric="l2")
    assert np.array_equal(first, second)


def test_clean_weights_follow_the_scenario():
    view = _view("S-deploy")
    with pytest.raises(CapabilityError):
        view.weight_tensor("w", "clean")
    ok = _view("S-public")
    assert np.array_equal(ok.weight_tensor("w", "clean"), -np.arange(16).reshape(4, 4))
    assert np.array_equal(ok.weight_tensor("w", "deployment"), np.arange(16).reshape(4, 4))
    for scenario in ("S-encoder", "S-pair"):
        stronger = _view(scenario)
        assert np.array_equal(stronger.weight_tensor("w", "clean"), ok.weight_tensor("w", "clean"))
        assert_public(stronger)


def test_wire_public_group_id_is_opaque_and_truncation_is_explicit():
    wire = WirePublic(
        arm="ab", request_id=0, seq_len=2, group_id="g-0123456789abcdef",
        epoch=0, truncated=True, full_len=9,
    )
    assert wire.group_id == "g-0123456789abcdef"
    assert wire.key_id == wire.group_id
    assert not hasattr(wire, "key_index")
    assert wire.truncated is True and wire.full_len == 9


def test_missing_scenario_resource_is_recorded_not_invented():
    fake = FakeDeployment()
    fake.E_clean = None
    view = build_public_view(fake, arm="ab", scenario="S-deploy")
    assert view.E_clean is None
    assert any("clean" in reason for reason in view.missing)
    with pytest.raises(CapabilityError):
        view.table("clean")
    with pytest.raises(CapabilityError):
        view.table("plaintext")


def test_s_deploy_cannot_reach_clean_weights_or_clean_forward():
    fake = FakeDeployment()
    fake.forward_fn = lambda embeds: ("deployment", embeds)
    fake.clean_forward_fn = lambda embeds: ("clean", embeds)
    view = build_public_view(fake, arm="ab", scenario="S-deploy")
    with pytest.raises(CapabilityError):
        view.weight_tensor("w", "clean")
    with pytest.raises(CapabilityError):
        view.weight_source.get("w", "clean")
    assert view.clean_forward_fn is None
    with pytest.raises(CapabilityError):
        view.forward("x", clean=True)
    assert view.forward("x") == ("deployment", "x")


def test_public_artifact_is_the_server_subtree_not_the_secret_bundle():
    view = _view("S-public")
    assert "client" not in view.artifact
    assert view.artifact.endswith("model_server") or view.artifact == "fixture"


def test_plaintext_arm_server_is_the_clean_model():
    fake = FakeDeployment()
    fake.forward_fn = lambda embeds: ("deployment", embeds)
    fake.clean_forward_fn = lambda embeds: ("clean", embeds)
    view = build_public_view(fake, arm="plaintext", scenario="S-public")
    assert view.forward("x") == ("clean", "x")


def test_hf_model_is_called_with_the_inputs_embeds_keyword():
    class HFModel:
        def forward(self, input_ids=None, inputs_embeds=None):
            if inputs_embeds is None:
                raise RuntimeError("float embeddings were passed as input_ids")
            return {"hidden": inputs_embeds}

    fake = FakeDeployment()
    fake.model = HFModel()
    fake.clean = HFModel()
    view = build_public_view(fake, arm="ab", scenario="S-public")
    assert view.forward("embeds") == {"hidden": "embeds"}


def test_variant_evidence_uses_content_not_only_the_directory_name(tmp_path):
    from obfuscation_attacks.schemes.ab_deployment import (
        inspect_model_variant,
        verify_model_identity,
    )

    instruct = tmp_path / "Llama-3.2-1B"
    instruct.mkdir()
    (instruct / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}"}), encoding="utf-8"
    )
    evidence = inspect_model_variant(instruct)
    assert evidence["looks_instruct"] is True
    with pytest.raises(ValueError):
        verify_model_identity(instruct, expected_variant="base")

    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert inspect_model_variant(base)["looks_instruct"] is False
