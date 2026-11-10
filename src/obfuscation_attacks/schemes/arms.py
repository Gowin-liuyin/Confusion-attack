"""Eight-arm declaration matrix (protocol §0.1, §8.3; W5).

Each defense arm declares its real message type, server-visible resources,
task support, training adaptation, randomization/key mechanism and provenance.
The matrix is used by the preflight manifest and the job planner; it is not a
second experiment protocol.

External arms are marked ``unverified`` or ``resource_blocked`` with the exact
missing source/commit/license/artifact. They are never silently marked
``not_applicable`` and never given a fabricated vector upload.
"""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARMS = (
    "plaintext", "aloepri", "osnip", "ab",
    "sentinellms", "obfuslm_plus", "inferdpt", "dp_forward",
)


@dataclass(frozen=True)
class ArmDeclaration:
    name: str
    upload: str                      # "text" | "ids" | "vectors" | "none"
    download: str
    tasks: tuple[str, ...]           # "classification" | "generation"
    server_visible: tuple[str, ...]
    public_resources: tuple[str, ...]
    secret_resources: tuple[str, ...]
    requires_training: bool
    randomization: str
    key_mechanism: str
    source: str | None
    commit: str | None
    license: str | None
    status: str                      # implemented | resource_blocked | unverified
    missing: tuple[str, ...] = ()
    client_cost: str = "not measured"
    server_cost: str = "not measured"
    communication_cost: str = "not measured"
    notes: str = ""

    def manifest_entry(self) -> dict:
        return {
            "adapter": self.name if self.status == "implemented" else None,
            "license": self.license,
            "commit": self.commit,
            "source_verified": self.status == "implemented",
        }

    def capability(self) -> dict:
        return {
            "upload": self.upload,
            "download": self.download,
            "tasks": list(self.tasks),
            "server_visible": list(self.server_visible),
            "public_resources": list(self.public_resources),
            "secret_resources": list(self.secret_resources),
            "requires_training": self.requires_training,
            "randomization": self.randomization,
            "key_mechanism": self.key_mechanism,
            "status": self.status,
            "missing": list(self.missing),
            "source": self.source,
            "commit": self.commit,
            "license": self.license,
            "client_cost": self.client_cost,
            "server_cost": self.server_cost,
            "communication_cost": self.communication_cost,
        }


ARM_DECLARATIONS: dict[str, ArmDeclaration] = {
    "plaintext": ArmDeclaration(
        name="plaintext",
        upload="ids",
        download="logits/ids",
        tasks=("classification", "generation"),
        server_visible=("token ids", "hidden states", "logits", "server weights"),
        public_resources=("clean base model", "public tokenizer"),
        secret_resources=(),
        requires_training=False,
        randomization="none",
        key_mechanism="none",
        source="local baseline",
        commit="repository",
        license="internal",
        status="implemented",
        notes="native path positive control; no perturbation",
    ),
    "aloepri": ArmDeclaration(
        name="aloepri",
        upload="ids",
        download="permuted logits/ids",
        tasks=("classification", "generation"),
        server_visible=("permuted ids", "hidden states", "permuted logits", "deployment weights"),
        public_resources=("deployment weights", "obfuscation algorithm"),
        secret_resources=("token permutation", "client key/seed", "unuploaded plaintext"),
        requires_training=False,
        randomization="secret token/A matrix permutation",
        key_mechanism="secret permutation",
        source="AloePri technical report (arXiv:2603.01499v2)",
        commit=None,
        license=None,
        status="implemented",
        notes=(
            "engineering A artifact only: the packaged single-side dense re-basing is not a "
            "complete AloePri Algorithm-1 reproduction and must not be labelled as one"
        ),
    ),
    "osnip": ArmDeclaration(
        name="osnip",
        upload="vectors",
        download="logits/ids",
        tasks=("classification", "generation"),
        server_visible=("continuous wire vectors", "hidden states", "logits"),
        public_resources=("clean base model", "encoder structure/training recipe"),
        secret_resources=("client key", "encoder instance key schedule"),
        requires_training=True,
        randomization="key-conditioned client encoder",
        key_mechanism="per-request client key",
        source="OSNIP, ICML 2026 (PMLR v306)",
        commit=None,
        license=None,
        status="resource_blocked",
        missing=("a clean-base encoder trained independently of the A artifact",),
        notes="reusing the A+B encoder on the clean table is an out-of-distribution substitute",
    ),
    "ab": ArmDeclaration(
        name="ab",
        upload="vectors",
        download="logits/ids",
        tasks=("classification", "generation"),
        server_visible=("continuous wire vectors", "hidden states", "logits", "deployment weights"),
        public_resources=("deployment weights", "clean base weights", "public tokenizer"),
        secret_resources=("token permutation", "client root key", "encoder", "unuploaded plaintext"),
        requires_training=False,
        randomization="key-conditioned OSNIP encoder over permuted embeddings",
        key_mechanism="256-bit root -> per-request HKDF key, opaque group id",
        source="this project's A+B artifact",
        commit="repository",
        license="internal",
        status="implemented",
        notes="full-prefix re-encode/re-upload path",
    ),
    "sentinellms": ArmDeclaration(
        name="sentinellms",
        upload="ids",
        download="class logits",
        tasks=("classification",),
        server_visible=("transformed token ids", "model weights", "logits"),
        public_resources=("transformation algorithm", "public model"),
        secret_resources=("glide-reflection parameters", "fine-tuned encoder"),
        requires_training=True,
        randomization="fixed geometric transform + fine-tuning",
        key_mechanism="defense-side transformation (not a per-user key)",
        source="SentinelLMs, AAAI 2024",
        commit=None,
        license=None,
        status="unverified",
        missing=("author code commit", "license", "task adaptation verified", "training recipe verified"),
        notes="classification-only; migration to a decoder needs separate numerical validation",
    ),
    "obfuslm_plus": ArmDeclaration(
        name="obfuslm_plus",
        upload="ids",
        download="logits/ids",
        tasks=("classification", "generation"),
        server_visible=("obfuscated text/ids", "model weights", "logits"),
        public_resources=("+ version algorithm", "public model"),
        secret_resources=("obfuscation vocabulary mapping", "training adaptation"),
        requires_training=True,
        randomization="model- and vocabulary-level obfuscation",
        key_mechanism="obfuscation mapping + fine-tuning",
        source="ObfusLM, ACL 2025",
        commit=None,
        license=None,
        status="unverified",
        missing=("author code commit", "license", "model/vocabulary obfuscation reproduction", "fine-tuning recipe"),
        notes="the weaker base version must not be reported as ObfusLM+",
    ),
    "inferdpt": ArmDeclaration(
        name="inferdpt",
        upload="text",
        download="perturbed text",
        tasks=("generation",),
        server_visible=("perturbed text", "server logits", "generated continuation"),
        public_resources=("perturbation algorithm", "local extraction guidance"),
        secret_resources=("private text", "local extraction model"),
        requires_training=False,
        randomization="token-level perturbation",
        key_mechanism="perturbation sampling",
        source="InferDPT, TDSC 2025",
        commit=None,
        license=None,
        status="unverified",
        missing=("author code commit", "license", "both perturbation and local extraction modules"),
        notes="a RANTEXT-only run compares uploaded-text privacy, not full-system utility",
    ),
    "dp_forward": ArmDeclaration(
        name="dp_forward",
        upload="vectors",
        download="logits",
        tasks=("classification",),
        server_visible=("noisy forwarded representations", "server logits"),
        public_resources=("DP mechanism definition", "public model"),
        secret_resources=("privacy budget randomization",),
        requires_training=True,
        randomization="forward-pass differential privacy noise",
        key_mechanism="DP noise, not a user key",
        source="DP-Forward, CCS 2023",
        commit=None,
        license=None,
        status="unverified",
        missing=("author code commit", "license", "noise layer/mechanism verification", "task adaptation"),
        notes="DP-SGD is not a substitute for forward-pass DP",
    ),
}


def arm_matrix() -> dict:
    return {
        arm: {
            "declaration": declaration.capability(),
            "manifest_entry": declaration.manifest_entry(),
        }
        for arm, declaration in ARM_DECLARATIONS.items()
    }


def manifest_arms() -> dict:
    """Manifest block for the paper-v1 preflight."""
    return {arm: declaration.manifest_entry() for arm, declaration in ARM_DECLARATIONS.items()}


def text_upload_arms() -> list[str]:
    return [arm for arm, decl in ARM_DECLARATIONS.items() if decl.upload == "text"]


def vector_upload_arms() -> list[str]:
    return [arm for arm, decl in ARM_DECLARATIONS.items() if decl.upload == "vectors"]
