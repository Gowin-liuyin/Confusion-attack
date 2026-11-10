"""Universal arm-adapter tests: every paper arm has an executable contract.

Local arms wrap a (fake) deployment; external arms must fail closed until the
author backend plus license/commit are supplied, and must then delegate real
upload/classification/generation calls without inventing a surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.evaluation.generic_runner import (
    adapter_attack_matrix,
    run_classification,
    run_text_surface,
    run_vector_knn,
)
from obfuscation_attacks.schemes import adapters
from obfuscation_attacks.schemes.adapters import (
    ABAdapter,
    AdapterNotReady,
    AloePriAdapter,
    DPForwardAdapter,
    InferDPTAdapter,
    ObfusLMPlusAdapter,
    OSNIPOnlyAdapter,
    PlaintextAdapter,
    SentinelLMsAdapter,
    adapter_status_table,
    build_adapter,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth
from obfuscation_attacks.schemes.arms import PAPER_ARMS

pytestmark = pytest.mark.unit

V, D = 64, 8
RNG = np.random.default_rng(0)
E = RNG.normal(size=(V, D)).astype(np.float32)
PERM = RNG.permutation(V).astype(np.int64)
INV = np.empty_like(PERM)
INV[PERM] = np.arange(V)


class FakeDeployment:
    def __init__(self) -> None:
        self.E = E
        self.E_clean = E
        self.E_dep = E
        self.perm = PERM
        self.inv_perm = INV
        self.layers = (1,)
        self.vocab_size = V
        self.clean = None
        self.model = None
        self.artifact = "fixture"

    def encode(self, text):
        return np.asarray([ord(char) % V for char in text], dtype=np.int64)

    def table(self, arm):
        return self.E_clean if arm == "plaintext" else self.E_dep

    def _key_for(self, index, epoch=0):
        return np.zeros(4, dtype=np.float32)

    @staticmethod
    def iso_norm_project(z, h, eps=1e-12):
        return z

    def observe(self, text, *, arm, request_id=0, key_index=None, key_seed=None,
                forward=True, **kwargs):
        ids = self.encode(text)
        permuted = PERM[ids]
        if arm == "ab":
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_vectors=E[permuted].copy(), group_id="g")
        else:
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_token_ids=ids.copy())
        return Wire(public, WireTruth(plaintext_ids=ids, permuted_ids=permuted.copy()))

    def describe(self):
        return {"vocab_size": V, "hidden_size": D, "num_layers": 1}


def test_default_clean_model_is_the_base_not_instruct():
    from obfuscation_attacks.schemes.ab_deployment import DEFAULT_CLEAN

    assert "instruct" not in DEFAULT_CLEAN.name.lower()


def test_registry_matches_the_eight_paper_arms():
    assert set(adapters.ADAPTER_CLASSES) == set(PAPER_ARMS)
    for arm in ("plaintext", "aloepri", "ab"):
        with pytest.raises(AdapterNotReady):
            build_adapter(arm)


def test_all_eight_arms_have_contracts_and_fail_closed():
    table = adapter_status_table()
    assert set(table) == set(PAPER_ARMS) and len(table) == 8
    for arm in ("sentinellms", "obfuslm_plus", "inferdpt", "dp_forward"):
        assert table[arm]["ready"] is False
        assert table[arm]["shortages"]
    for arm in ("sentinellms", "inferdpt"):
        adapter = build_adapter(arm)
        with pytest.raises(AdapterNotReady):
            adapter.ensure_ready()
    # an unsupported arm is a hard error, never a silent vector surface
    with pytest.raises(KeyError):
        build_adapter("not_a_defense")


def test_local_adapters_produce_real_uploads_and_truth():
    dep = FakeDeployment()
    texts = ["abcdefgh", "ijklmnop"]
    plain = PlaintextAdapter(dep)
    uploads = plain.client_upload(texts)
    assert uploads[0].surface.value == "ids"
    assert np.array_equal(plain.attack_queries(uploads), E[plain.scoring_truth(uploads)])
    ab = ABAdapter(dep)
    uploads = ab.client_upload(texts)
    assert uploads[0].surface.value == "vectors"
    assert np.array_equal(ab.attack_queries(uploads),
                          np.concatenate([u.as_array() for u in uploads]))
    assert np.array_equal(ab.scoring_truth(uploads),
                          np.concatenate([PERM[dep.encode(t)] for t in texts]))
    aloe = AloePriAdapter(dep)
    ups = aloe.client_upload(texts)
    assert np.array_equal(ups[0].as_array(), PERM[dep.encode(texts[0])])


def test_vector_knn_runs_on_plaintext_ab_and_osnip():
    dep = FakeDeployment()
    texts = ["abcdefgh", "ijklmnop"]
    for adapter in (PlaintextAdapter(dep), ABAdapter(dep)):
        result = run_vector_knn(adapter, texts, metric="l2", top_k=16)
        assert result["status"] == "ok"
        assert result["metrics"]["stage1_permuted_id"]["top1"]["value"] == 1.0
        assert result["n_tokens"] == sum(len(dep.encode(text)) for text in texts)
        assert result["ci"]["top1_micro"]["n_clusters"] == 2
    osnip = OSNIPOnlyAdapter(
        dep, encoder=lambda h, key: h, encoder_sha256="a" * 64,
        training_recipe="fixture-recipe",
    )
    assert osnip.ready() == []
    result = run_vector_knn(osnip, texts, metric="l2", top_k=16)
    assert result["status"] == "ok"
    assert result["metrics"]["stage1_permuted_id"]["top1"]["value"] == 1.0


def test_key_robustness_uses_text_clusters():
    dep = FakeDeployment()
    adapter = ABAdapter(dep)
    texts = ["abcdefgh", "ijklmnop"]
    result = run_vector_knn(adapter, texts, metric="l2", top_k=16, keys_per_text=5)
    assert result["n_keys"] == 5
    assert result["n_prompts"] == 2
    assert result["ci"]["top1_micro"]["n_clusters"] == 2
    assert len(result["per_unit_predictions"]) == 10


def test_ab_respects_request_and_key_ids_across_batches_and_truncation():
    class Truncated(FakeDeployment):
        def __init__(self):
            super().__init__()
            self.calls = []

        def observe(self, text, *, arm, request_id=0, key_index=None, forward=True, **kwargs):
            self.calls.append((request_id, key_index))
            full = self.encode(text)
            ids = full[:64]
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_vectors=E[PERM[ids]].copy(), group_id=f"g{key_index}",
                                truncated=len(ids) < len(full), full_len=len(full))
            return Wire(public, WireTruth(plaintext_ids=ids, permuted_ids=PERM[ids]))

    dep = Truncated()
    adapter = ABAdapter(dep)
    first = adapter.client_upload(["x" * 80], request_ids=[8, 19],
                                  key_indices=[3, 7], keys_per_text=2)
    second = adapter.client_upload(["abc"], request_ids=[42], key_indices=[3])
    assert dep.calls == [(8, 3), (19, 7), (42, 3)]
    assert [u.request_id for u in first + second] == [8, 19, 42]
    assert first[0].truncated and first[0].full_len == 80 and first[0].seq_len == 64
    assert len(adapter.scoring_truth(first + second)) == 131
    assert len(adapter._truth[8]["full_plaintext"]) == 80
    with pytest.raises(ValueError, match="unique"):
        adapter.client_upload(["different"], request_ids=[8])
    assert len(adapter.plaintext_truth(first)) == 128
    assert adapter.client_upload(["more"])[0].request_id == 43


def test_local_adapters_do_not_overwrite_truth_across_batches():
    for adapter in (PlaintextAdapter(FakeDeployment()), AloePriAdapter(FakeDeployment())):
        first = adapter.client_upload(["a"])
        second = adapter.client_upload(["b"])
        assert [first[0].request_id, second[0].request_id] == [0, 1]
        assert adapter.plaintext_truth(first).tolist() == adapter.deployment.encode("a").tolist()


class FakeClassifier:
    def transform(self, text):
        return np.asarray([ord(char) % 4 for char in text], dtype=np.int64)

    def classify_ids(self, ids):
        probabilities = np.zeros(3, dtype=np.float32)
        probabilities[int(ids.sum()) % 3] = 1.0
        return probabilities

    def embedding_table(self):
        return E

    def hidden_states(self, ids, layer=1):
        return E[np.asarray(ids)]


class FakeObfuscator(FakeClassifier):
    def obfuscate_ids(self, ids):
        return (ids + 1) % V

    def server_logits(self, ids):
        return np.zeros((len(ids), V), dtype=np.float32)


class FakeInferDPT:
    def perturb(self, text):
        return " ".join(reversed(text.split()))

    def extract(self, perturbed_text, server_completion):
        return server_completion

    def generate(self, perturbed_text, max_new_tokens):
        return "completion"


class FakeDPForward:
    def noisy_representations(self, ids, *, epsilon):
        return E[np.asarray(ids) % V]

    def classify_representations(self, vectors):
        probabilities = np.zeros(3, dtype=np.float32)
        probabilities[int(np.asarray(vectors).sum()) % 3] = 1.0
        return probabilities

    def embedding_table(self):
        return E


def _provenance(arm: str) -> dict:
    return {
        "arms": {
            arm: {"adapter": "fixture", "license": "fixture", "commit": "abc",
                  "source_verified": True}
        }
    }


_PROVENANCE = {"adapter": "fixture", "license": "fixture", "commit": "abc",
               "source_verified": True}


def test_external_backends_make_adapters_ready_and_executable():
    sentinel = SentinelLMsAdapter(FakeClassifier(), provenance=_PROVENANCE)
    assert sentinel.ready(_provenance("sentinellms")) == []
    uploads = sentinel.client_upload(["abc", "de"])
    assert uploads[0].surface.value == "ids"
    probabilities = sentinel.classify(["abc", "de"])
    assert probabilities.shape == (2, 3)

    obfus = ObfusLMPlusAdapter(FakeObfuscator(), provenance=_PROVENANCE)
    assert obfus.ready(_provenance("obfuslm_plus")) == []
    assert obfus.client_upload(["abc"])[0].surface.value == "ids"

    infer = InferDPTAdapter(FakeInferDPT(), provenance=_PROVENANCE)
    assert infer.ready(_provenance("inferdpt")) == []
    trace = run_text_surface(infer, ["a b c"], manifest=_provenance("inferdpt"))
    assert trace["status"] == "ok"
    assert trace["traces"][0]["uploaded_text"] == "c b a"
    assert trace["traces"][0]["server_completion"] == "completion"

    dp = DPForwardAdapter(
        FakeDPForward(), epsilon=1.0, provenance=_PROVENANCE,
        tokenizer=lambda t: np.asarray([ord(c) % V for c in t], dtype=np.int64),
    )
    assert dp.ready(_provenance("dp_forward")) == []
    ups = dp.client_upload(["ab"])
    assert ups[0].surface.value == "vectors"
    assert dp.classify(["ab"]).shape == (1, 3)


def test_classification_runner_reports_accuracy():
    adapter = SentinelLMsAdapter(FakeClassifier(), provenance=_PROVENANCE)
    examples = [{"text": "abc", "label": int(np.asarray([97 % 4, 98 % 4, 99 % 4]).sum() % 3)}]
    result = run_classification(adapter, examples, manifest=_provenance("sentinellms"))
    assert result["status"] == "ok"
    assert result["accuracy"] == 1.0


def test_classification_ednn_and_dp_forward_knn():
    from obfuscation_attacks.evaluation.classification import run_ednn_classifier

    sentinel = SentinelLMsAdapter(FakeClassifier(), provenance=_PROVENANCE)
    texts = ["abc", "de"]
    knn = run_vector_knn(sentinel, texts, top_k=16, use_hidden=True)
    assert knn["status"] == "ok"
    assert knn["metrics"]["stage1_permuted_id"]["top1"]["value"] == 1.0
    ednn = run_ednn_classifier(sentinel.backend, texts, layer=1, top_k=16)
    assert ednn["status"] == "ok"
    assert ednn["n_tokens"] == sum(len(sentinel.backend.transform(t)) for t in texts)
    assert ednn["metrics"]["stage1_permuted_id"]["top1"]["value"] == 1.0

    dp = DPForwardAdapter(
        FakeDPForward(), provenance=_PROVENANCE,
        tokenizer=lambda t: np.asarray([ord(c) % V for c in t], dtype=np.int64),
    )
    dp_knn = run_vector_knn(dp, texts, top_k=16)
    assert dp_knn["status"] == "ok"
    assert dp_knn["metrics"]["stage1_permuted_id"]["top1"]["value"] == 1.0


def test_matrix_covers_every_arm_and_attack():
    from obfuscation_attacks.evaluation.paper_profile import PAPER_ATTACKS

    dep = FakeDeployment()
    manifest = {"arms": {arm: dict(_PROVENANCE) for arm in PAPER_ARMS}}
    matrix = adapter_attack_matrix(
        manifest=manifest,
        deployments={"plaintext": dep, "aloepri": dep, "ab": dep},
        backends={
            "sentinellms": FakeClassifier(),
            "obfuslm_plus": FakeObfuscator(),
            "inferdpt": FakeInferDPT(),
            "dp_forward": FakeDPForward(),
        },
    )
    assert set(matrix["matrix"]) == set(PAPER_ARMS)
    for arm in PAPER_ARMS:
        assert set(matrix["matrix"][arm]) == set(PAPER_ATTACKS)
        for cell in matrix["matrix"][arm].values():
            assert cell["status"] in {"adapter_ready", "blocked", "not_applicable"}
    assert matrix["matrix"]["plaintext"]["knn_l2"]["status"] == "adapter_ready"
    assert matrix["matrix"]["plaintext"]["knn_l2"]["executor_wired"] is True
    assert matrix["matrix"]["ab"]["knn_l2_key_robustness"]["executor_wired"] is True
    assert matrix["matrix"]["plaintext"]["knn_l2_key_robustness"]["executor_wired"] is False
    # external backs are ready at the adapter level, but not executor-wired
    assert matrix["matrix"]["sentinellms"]["utility_classification"]["status"] == "adapter_ready"
    assert matrix["matrix"]["sentinellms"]["utility_classification"]["executor_wired"] is False
