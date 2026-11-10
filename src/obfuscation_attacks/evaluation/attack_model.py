"""Attack-model alignment layer (plan ``privacy_test_plan_20261106`` §1.4 / App. A).

The plan's first discipline is that **every number carries its attack model**:
which surface it observed, what the attacker knew, whether a client key, a
deployment-key pairing or rotation awareness was involved, and whether the
candidate set was the full vocabulary. The plan forbids reporting a bare ASR
without those fields (§8.5, App. A, checklist §10.1).

This module makes that machine-checkable instead of prose:

* :class:`AttackModelSpec` pins one attack class (C01..C14) to a surface, a
  knowledge level and the scenario the plan allows it in (§1.11).
* :data:`ATTACK_MODELS` maps every registered attack name to its spec. A test
  fails when a registered attack has no spec, or when a spec names an unknown
  attack - so a new attack cannot be merged without declaring its model.
* :func:`result_payload` renders the plan's App.-A JSON (plus the extra
  ``attack_uses_encoder/key/client_secret=false`` flags required by §1.7-9) and
  :func:`validate_payload` rejects a payload that is missing a required field.

Knowledge levels (plan §1.4): ``SH`` is the honest-but-curious server; ``K0``
(no-key observation), ``KW`` (weight-visible), ``KI`` (self-built instance) and
``SH-P`` (protocol-aware, assignment-blind) are *sub-capabilities* of SH and may
enter the main table; ``KP`` (known deployment-key pairs), ``KA`` (adaptive
white-box) and ``EXT-K`` (key leakage) are extensions and may only appear in the
appendix / ablation. The JSON records the level actually used, so an appendix
number can never be silently promoted to the main table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Plan §1.3 observation surfaces.
SURFACES = {"S1_wire", "S2_state", "S3_weights", "S4_stream", "S5_output"}
#: Plan §1.4 knowledge levels.
KNOWLEDGE = {"SH", "SH-P", "SH-B", "KP", "KA", "EXT-K"}
#: Plan §1.4 *sub-capabilities* of the honest-but-curious server. ``attacker_knowledge``
#: is the coarse plan code (SH / SH-P / ...); this is the fine one the plan's catalogue
#: (§3 "知识" column) uses, e.g. KNN is ``K0`` and VMA-W is ``KW``. Both are recorded so
#: a reader can check the row against §3 without re-deriving it from prose.
SUB_CAPABILITIES = {"K0", "KW", "KI", "KP", "KA", "KP+EXT-K", "KA+KP", "n/a"}
#: Plan §1.11 attack scenarios.
SCENARIOS = {
    "S-A", "S-B", "S-C", "S-D", "E-1", "E-2", "E-3",
    # protocol §1 information scenarios (paper-v1); historical labels stay readable
    "S-public", "S-deploy", "S-encoder", "S-pair",
}
#: Knowledge levels that may appear in the main table (plan D1).
MAIN_TABLE_KNOWLEDGE = {"SH", "SH-P"}

#: ``attack name -> plan §3 catalogue sub-capability``.
#: Transcribed one-to-one from the plan's "知识" column so the mapping is auditable:
#:   A1 K0, A2 KI/KP, A3 K0, A4 KA, A5 KI, B1-B3 K0, B4 KI, C1-C3 KW,
#:   D1/D2 KP, D3 KI, D4 KP+EXT-K, D5 KP, D6 KA+KP, E1/E2/E5/E6/E7 K0, F1 K0/KP,
#:   F2 K0/KP, F3 K0.
PLAN_SUBCAPABILITY: dict[str, str] = {
    "nn_knn.euclidean": "K0",
    "nn_knn.cosine": "K0",
    "nn_knn.adaptive_wk": "KA",
    "nn_knn.adaptive_oracle": "KA",
    "ima.cross_instance": "KI",
    "ima.cross_checkpoint": "KI",
    "ima.cross_key": "KP+EXT-K",
    "ima.mixed_key": "KP",
    "ima.adaptive": "KA+KP",
    "isa.er": "K0",
    "isa.tbs": "K0",
    "isa.aloepri_state": "K0",
    "vma.autoregressive": "K0",
    "vma.osnip_projection": "K0",
    "vma.weight_rowsort": "KW",
    "ia.ednn": "K0",
    "ia.gate": "KW",
    "ia.attn": "KW",
    "ima.classify": "KP",
    "ima.seq2seq": "KP",
    "ima.qwen2_aloepri": "KP",
    "tfma.independent": "K0",
    "tfma.global_assignment": "K0",
    "tfma.per_key": "K0",
    "tfma.output": "K0",
    "sda.supervised_substitution_diagnostic": "KP",
    "sda.sda2021": "K0",
    "sda.sda2023": "K0",
    "linkage.exact": "K0",
    "linkage.approximate": "K0",
    "beamclean.gaussian": "KI",
    "beamclean.laplacian": "KI",
    "pii.extract": "K0",
    "lm_recon.ngram": "K0",
    "lm_recon.local_lm": "K0",
    "vec2text.iterative": "K0",
}


REQUIRED_FIELDS = (
    "attack",
    "class_id",
    "target",
    "observation_surface",
    "attacker_knowledge",
    "knowledge_subcapability",
    "scenario",
    "uses_client_key",
    "uses_known_pairs",
    "rotation_aware",
    "candidate_mode",
    "full_vocabulary",
    "positive_control",
    "n_prompts",
    "n_tokens",
    "n_keys",
    "n_types",
    "reference_tokens",
    "metrics",
    "ci",
    "source_sha256",
    "data_sha256",
    "attack_uses_encoder",
    "attack_uses_key",
    "attack_uses_client_secret",
)


@dataclass(frozen=True)
class AttackModelSpec:
    """The attack model one registered attack is allowed to assume."""

    class_id: str  # plan class C01..C14 (orchestration doc)
    family: str
    observation_surface: str
    attacker_knowledge: str
    scenario: str
    paper: str
    uses_client_key: bool = False
    uses_known_pairs: bool = False
    rotation_aware: bool = False
    candidate_mode: str = "full"
    main_table: bool = True
    implemented: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "family": self.family,
            "observation_surface": self.observation_surface,
            "attacker_knowledge": self.attacker_knowledge,
            "scenario": self.scenario,
            "uses_client_key": self.uses_client_key,
            "uses_known_pairs": self.uses_known_pairs,
            "rotation_aware": self.rotation_aware,
            "candidate_mode": self.candidate_mode,
            "main_table": self.main_table,
            "paper": self.paper,
            "note": self.note,
        }


def _spec(**kw) -> AttackModelSpec:
    return AttackModelSpec(**kw)


#: One entry per registered attack. Key = registered attack name.
ATTACK_MODELS: dict[str, AttackModelSpec] = {
    # --- C01 KNN / NN (S1, no key) -----------------------------------------
    "nn_knn.euclidean": _spec(
        class_id="C01",
        family="nn_knn",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Song & Raghunathan, CCS 2020 (arXiv:2004.00053); OSNIP App. C.1 / Table 6",
        note="K0 sub-capability of SH: no key, no pairing, public embedding table.",
    ),
    "nn_knn.cosine": _spec(
        class_id="C01",
        family="nn_knn",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Song & Raghunathan, CCS 2020 (arXiv:2004.00053); OSNIP App. C.1 / Table 6",
        note="K0; cosine variant of the same retrieval attack.",
    ),
    # --- C02 learned embedding inversion (S1, self-built instance) ----------
    "ima.cross_instance": _spec(
        class_id="C02",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-D",
        paper="Kugler et al. 2021 (arXiv:2109.10104) InvBERT; DeepInvert 2026 (arXiv:2608.04477)",
        main_table=False,
        note=(
            "KI sub-capability: the inverter is trained on pairs minted by the attacker's "
            "*own* instance of the mechanism. Never reads the deployment key - the KI/KP "
            "boundary the plan draws in D8."
        ),
    ),
    # --- C04 cross-checkpoint / transfer attack (S1, self-built instance) ---
    "ima.cross_checkpoint": _spec(
        class_id="C04",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-D",
        paper=(
            "protocol is this project's extension: the plan's cited arXiv:2610.27817 was "
            "verified to be a learned-transform *defense*, not a transfer protocol"
        ),
        main_table=False,
        note=(
            "KI transfer: train the inverter on a different checkpoint of the same "
            "architecture, evaluate on the deployment checkpoint. Positive control is the "
            "same-checkpoint in-instance accuracy (>=95%)."
        ),
    ),
    # --- C05 ISA (S2, no key) ----------------------------------------------
    "isa.er": _spec(
        class_id="C05",
        family="isa",
        observation_surface="S2_state",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Dong et al., USENIX Security 2025 (arXiv:2507.16372)",
        note="K0; emb-reconstruction variant optimising input embeddings against a state.",
    ),
    "isa.tbs": _spec(
        class_id="C05",
        family="isa",
        observation_surface="S2_state",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Dong et al., USENIX Security 2025 (arXiv:2507.16372)",
        note="K0; token-by-token / SVD re-parameterised variant.",
    ),
    "isa.aloepri_state": _spec(
        class_id="C05",
        family="isa",
        observation_surface="S2_state",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="AloePri App. D.1 / [24]",
        note="K0; AloePri-adapted State1/State2 objective.",
    ),
    # --- C06 VMA (S2 hidden-state matching + S3 weight permutation) ---------
    "vma.autoregressive": _spec(
        class_id="C06",
        family="vma",
        observation_surface="S2_state",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Thomas et al., ICML 2025 (arXiv:2505.18332), Alg. 1/4/6",
        note="VMA-H: the autogressive vocabulary-matching variant; K0 + white-box replay.",
    ),
    "vma.osnip_projection": _spec(
        class_id="C06",
        family="vma",
        observation_surface="S2_state",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="OSNIP App. C.2 / Table 7",
        note="VMA-H (OSNIP projection form); K0.",
    ),
    "vma.weight_rowsort": _spec(
        class_id="C06",
        family="vma",
        observation_surface="S3_weights",
        attacker_knowledge="SH",
        scenario="S-B",
        paper="AloePri App. D.1 / Table 9; Thomas et al. 2025",
        note=(
            "VMA-W: recovers the token *permutation* from deployment weights (KW). "
            "Content recovery needs the composed chain (stage-1 + stage-2)."
        ),
    ),
    # --- C08 IA (S3 weights) ------------------------------------------------
    "ia.ednn": _spec(
        class_id="C08",
        family="ia",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Lin et al., EMNLP 2024 (2024.emnlp-main.126)",
        note=(
            "Original EDNN: a wire-level invariant attack on the obfuscated embedding "
            "matrix (glide-reflection difference invariant). Reported under C08 because "
            "the plan's IA family lists EDNN as its original member."
        ),
    ),
    "ia.gate": _spec(
        class_id="C08",
        family="ia",
        observation_surface="S3_weights",
        attacker_knowledge="SH",
        scenario="S-B",
        paper="AloePri App. D.1",
        note="Gate-IA (W_gate statistical invariant); KW.",
    ),
    "ia.attn": _spec(
        class_id="C08",
        family="ia",
        observation_surface="S3_weights",
        attacker_knowledge="SH",
        scenario="S-B",
        paper="AloePri App. D.1",
        note="Attn-IA (Q/K block invariant, whitening); KW. Not merged with Gate-IA.",
    ),
    # --- C09 IMA (paired inversion) ----------------------------------------
    "ima.classify": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KP",
        scenario="E-1",
        paper="Kugler et al. 2021 (arXiv:2109.10104) InvBERT Classify; AloePri App. D.1",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        note=(
            "KB/KP: training pairs are minted by the *deployment* obfuscator (in this "
            "harness ``obfuscation_fn`` closes over the deployment secret), i.e. the "
            "attacker holds deployment-key pairs. Appendix only (plan D8)."
        ),
    ),
    "ima.seq2seq": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KP",
        scenario="E-1",
        paper="Kugler et al. 2021 (arXiv:2109.10104) InvBERT Seq2Seq",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        note="BB/KP variant of the same paired-inversion family; appendix only.",
    ),
    "ima.qwen2_aloepri": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KP",
        scenario="E-1",
        paper="AloePri Table 2 / App. D.1 (2 layers, 8 heads Qwen2)",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        note="Paper-shaped inverter, same KP pair provenance; appendix only.",
    ),
    # --- C10 TFMA (S4, S-C) -------------------------------------------------
    "tfma.independent": _spec(
        class_id="C10",
        family="tfma",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="AloePri Sec. 7.6 / Table 5 / [27] (Aldarrab & May 2021)",
        rotation_aware=True,
        note="id-level wire; the embedding-level variant must set attack_variant.",
    ),
    "tfma.global_assignment": _spec(
        class_id="C10",
        family="tfma",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="AloePri Sec. 7.6 (Hungarian one-to-one completion; named extension)",
        rotation_aware=True,
        main_table=False,
        note="Hungarian assignment is an extension, reported apart from the paper method.",
    ),
    # --- C11 SDA (S4, S-C) --------------------------------------------------
    "sda.supervised_substitution_diagnostic": _spec(
        class_id="C11", family="sda", observation_surface="S4_stream",
        attacker_knowledge="KP", scenario="S-pair", paper="supervised synthetic diagnostic",
        uses_known_pairs=True, main_table=False, implemented=False,
        note="Known synthetic cipher/plain training pairs; not recurrence CLM SDA.",
    ),
    "sda.sda2021": _spec(
        class_id="C11",
        family="sda",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="Aldarrab & May, ACL 2021 (2021.acl-long.561); AloePri [27]",
        rotation_aware=True,
        main_table=False,
    ),
    "sda.sda2023": _spec(
        class_id="C11",
        family="sda",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="Kambhatla et al., Findings EACL 2023 (2023.findings-eacl.160); AloePri [28]",
        rotation_aware=True,
        main_table=False,
    ),
    # --- C12 linkage (S4, S-C) ---------------------------------------------
    "linkage.exact": _spec(
        class_id="C12",
        family="linkage",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="AloePri frequency leakage; protocol extension of this project (no source paper)",
        rotation_aware=True,
        note="Exact symbol-equality linking; SH-B is the same attack without metadata.",
    ),
    "linkage.approximate": _spec(
        class_id="C12",
        family="linkage",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="AloePri frequency leakage; protocol extension of this project (no source paper)",
        rotation_aware=True,
        note="Cosine-threshold linking with TPR/FPR reported per threshold.",
    ),
    # --- C07 BeamClean (S1, LM-prior decoding) -----------------------------
    "beamclean.gaussian": _spec(
        class_id="C07",
        family="beamclean",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Kale et al. 2025, BeamClean: Language Aware Embedding Reconstruction (arXiv:2505.13758)",
        note=(
            "No LM weights and no obfuscation mechanism are assumed: only the obfuscated "
            "embeddings, the model's public embedding table and a language-model prior."
        ),
    ),
    "beamclean.laplacian": _spec(
        class_id="C07",
        family="beamclean",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Kale et al. 2025 (arXiv:2505.13758)",
        note="Laplacian noise-model variant of the same decoder.",
    ),
    # --- C13 PII extraction (S5, Limitations + appendix) -------------------
    "pii.extract": _spec(
        class_id="C13",
        family="pii",
        observation_surface="S5_output",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="AloePri [14] (Papillon/PUPA dataset, NAACL 2025); AloePri entity scorer",
        main_table=False,
        note="Entity-level PIIRSR; never a defense claim for the input-side scheme (D11).",
    ),
    # --- C14 output-side LM / n-gram reconstruction (S5) -------------------
    "lm_recon.ngram": _spec(
        class_id="C14",
        family="lm_recon",
        observation_surface="S5_output",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="InferDPT (TDSC 2025); RANTEXT n-gram / CosSim metrics",
        main_table=False,
        note="n-gram / CosSim / BLEU-4 / exact-match metrics over the recovered output.",
    ),
    "lm_recon.local_lm": _spec(
        class_id="C14",
        family="lm_recon",
        observation_surface="S5_output",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="InferDPT (TDSC 2025) GPT-4 reconstruction, local-LM substitute",
        main_table=False,
        note=(
            "Substitute for GPT-4 reconstruction: requires a local generative LM handle. "
            "Without one the attack returns not_applicable (never a fabricated score)."
        ),
    ),
    # --- C03 Vec2Text: protocol recorded, not implemented this round --------
    "vec2text.iterative": _spec(
        class_id="C03",
        family="vec2text",
        observation_surface="S1_wire",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="Morris et al., EMNLP 2023 (arXiv:2310.06816)",
        main_table=False,
        implemented=False,
        note="Cite-only in this round (plan C03: P2, Phase 3). No code, no numbers.",
    ),
    # --- C01 ablation A4: adaptive KNN (plan §4.14 / OSNIP Table 8) ---------
    "nn_knn.adaptive_wk": _spec(
        class_id="C01",
        family="nn_knn",
        observation_surface="S1_wire",
        attacker_knowledge="KA",
        scenario="E-2",
        paper="OSNIP Table 8 (ADAPTIVE-WK); plan §4.14",
        main_table=False,
        implemented=False,
        note=(
            "KA ablation: the attacker holds the encryptor weights but uses the *wrong* key. "
            "Outside the semi-honest model (A3/A4), so it may only appear in the ablation table."
        ),
    ),
    "nn_knn.adaptive_oracle": _spec(
        class_id="C01",
        family="nn_knn",
        observation_surface="S1_wire",
        attacker_knowledge="KA",
        scenario="E-2",
        paper="OSNIP Table 8 (ADAPTIVE-ORACLE); plan §4.14",
        main_table=False,
        implemented=False,
        note=(
            "KA ablation upper bound: encryptor weights *and* the correct key. Directly violates "
            "A3, so it is an upper bound that explains why the key is the only line of defense."
        ),
    ),
    # --- C09 appendix variants D4/D5/D6 (plan §4.13 / §4.14) ----------------
    "ima.cross_key": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KP",
        scenario="E-1",
        paper="this project's extension (plan §4.13), no source paper",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        implemented=False,
        note="KP+EXT-K: pairs from key A, test on key B. Appendix only; rotation shrinks but does not zero it.",
    ),
    "ima.mixed_key": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KP",
        scenario="E-1",
        paper="this project's extension (plan §4.13), no source paper",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        implemented=False,
        note="Rotation-blind training on a mixed-key pool; tests whether pooled pairs re-identify a key.",
    ),
    "ima.adaptive": _spec(
        class_id="C09",
        family="ima",
        observation_surface="S1_wire",
        attacker_knowledge="KA",
        scenario="E-2",
        paper="this project's extension (plan §4.14), no source paper",
        uses_client_key=True,
        uses_known_pairs=True,
        main_table=False,
        implemented=False,
        note="KA+KP: inverter adapts with encryptor weights plus pairs; ablation only.",
    ),
    # --- C10 per-key tables (E5) and output-side TFMA (F1) ------------------
    "tfma.per_key": _spec(
        class_id="C10",
        family="tfma",
        observation_surface="S4_stream",
        attacker_knowledge="SH-P",
        scenario="S-C",
        paper="AloePri Table 5 protocol + plan E5 (per-key tables); this project's K-scan extension",
        rotation_aware=True,
        implemented=False,
        note=(
            "Per-key grouping is executed by scripts/run_full_battery.py::run_c10 (SH-P groups "
            "requests by epoch); there is no standalone registered class yet, so this entry is a "
            "protocol record rather than a wired attack. "
            "Per-key frequency table (one table per key family) with a K/N scan. This is the "
            "strongest legitimate SH-P variant and the one the main table must use (D13)."
        ),
    ),
    "tfma.output": _spec(
        class_id="C10",
        family="tfma",
        observation_surface="S5_output",
        attacker_knowledge="SH",
        scenario="S-A",
        paper="AloePri Table 5 output-side application; plan F1",
        main_table=False,
        implemented=False,
        note="Output-side TFMA: the server's own generated ids, not the input wire.",
    ),
}


def spec_for(attack_name: str, family: Optional[str] = None) -> AttackModelSpec:
    """Return the spec for ``attack_name`` (or the family's first spec)."""
    if attack_name in ATTACK_MODELS:
        return ATTACK_MODELS[attack_name]
    if family:
        for name, spec in ATTACK_MODELS.items():
            if spec.family == family:
                return spec
    raise KeyError(
        f"no attack-model spec for {attack_name!r}; declare one in "
        f"obfuscation_attacks/evaluation/attack_model.py"
    )


def _sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def result_payload(
    *,
    attack_name: str,
    scheme_name: str,
    metrics: Mapping[str, Any],
    diagnostics: Optional[Mapping[str, Any]] = None,
    target: Optional[str] = None,
    positive_control: Optional[Mapping[str, Any]] = None,
    n_prompts: int = 0,
    n_tokens: int = 0,
    n_keys: int = 1,
    n_types: int = 0,
    reference_tokens: int = 0,
    ci: Optional[Mapping[str, Any]] = None,
    source: Any = None,
    data: Any = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Render the plan's App.-A result JSON for one attack run.

    ``overrides`` lets a run narrow its declared model (e.g. a TFMA run without
    request metadata must record ``attacker_knowledge="SH-B"``); it can never
    widen it, because the spec is the ceiling.
    """
    spec = spec_for(attack_name)
    ov = dict(overrides or {})
    knowledge = ov.get("attacker_knowledge", spec.attacker_knowledge)
    if knowledge not in KNOWLEDGE:
        raise ValueError(f"unknown attacker_knowledge {knowledge!r}")
    rotation_aware = bool(ov.get("rotation_aware", spec.rotation_aware))
    payload = {
        "attack": attack_name,
        "class_id": spec.class_id,
        "family": spec.family,
        "target": target or scheme_name,
        "observation_surface": spec.observation_surface,
        "attacker_knowledge": knowledge,
        "knowledge_subcapability": ov.get(
            "knowledge_subcapability", PLAN_SUBCAPABILITY.get(attack_name, "n/a")
        ),
        "scenario": spec.scenario,
        "uses_client_key": bool(ov.get("uses_client_key", spec.uses_client_key)),
        "uses_known_pairs": bool(ov.get("uses_known_pairs", spec.uses_known_pairs)),
        "rotation_aware": rotation_aware,
        "candidate_mode": ov.get("candidate_mode", spec.candidate_mode),
        "full_vocabulary": ov.get("candidate_mode", spec.candidate_mode) == "full",
        "positive_control": dict(positive_control or {}),
        "n_prompts": int(n_prompts),
        "n_tokens": int(n_tokens),
        "n_keys": int(n_keys),
        "n_types": int(n_types),
        "reference_tokens": int(reference_tokens),
        "metrics": dict(metrics),
        "ci": dict(ci or {}),
        "source_label_sha256": _sha256(source if source is not None else attack_name),
        "data_label_sha256": _sha256(data if data is not None else scheme_name),
        "source_sha256": None,
        "data_sha256": None,
        # Plan §1.7-9: the attack algorithm must not read the encoder, the key or
        # the client secret. A KP attack (appendix) flips these honestly.
        "attack_uses_encoder": bool(ov.get("attack_uses_encoder", spec.uses_client_key)),
        "attack_uses_key": bool(ov.get("attack_uses_key", spec.uses_client_key)),
        "attack_uses_client_secret": bool(
            ov.get("attack_uses_client_secret", spec.uses_client_key)
        ),
        "main_table_eligible": False,
        "paper": spec.paper,
        "paper_parameter_undisclosed": bool(ov.get("paper_parameter_undisclosed", False)),
        "diagnostics": dict(diagnostics or {}),
    }
    return payload


def _fraction_problems(obj: Any, path: str = "payload") -> list[str]:
    """Check every explicit numerator/denominator pair is non-negative and consistent."""
    problems: list[str] = []
    if isinstance(obj, Mapping):
        if "numerator" in obj and "denominator" in obj:
            num, den = obj["numerator"], obj["denominator"]
            if isinstance(num, bool) or not isinstance(num, (int, float)) or num < 0:
                problems.append(f"{path}.numerator is not a non-negative number: {num!r}")
            if isinstance(den, bool) or not isinstance(den, (int, float)) or den < 0:
                problems.append(f"{path}.denominator is not a non-negative number: {den!r}")
            if (
                isinstance(num, (int, float)) and not isinstance(num, bool)
                and isinstance(den, (int, float)) and not isinstance(den, bool)
                and num > den
            ):
                problems.append(f"{path}: numerator {num} exceeds denominator {den}")
            value = obj.get("value")
            if (
                value is not None and isinstance(value, (int, float)) and not isinstance(value, bool)
                and isinstance(den, (int, float)) and not isinstance(den, bool) and den > 0
                and isinstance(num, (int, float)) and not isinstance(num, bool)
            ):
                if abs(float(value) - float(num) / float(den)) > 1e-6:
                    problems.append(
                        f"{path}.value {value} does not equal {num}/{den}"
                    )
        for key, item in obj.items():
            problems.extend(_fraction_problems(item, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            problems.extend(_fraction_problems(item, f"{path}[{index}]"))
    return problems


COUNT_FIELDS = ("n_prompts", "n_tokens", "n_keys", "n_types", "reference_tokens")


def validate_payload(payload: Mapping[str, Any]) -> list[str]:
    """Return the list of missing/invalid fields (empty list == valid).

    This is a fail-closed schema check on experiment facts, not a protocol audit:
    a run can never declare itself paper-scale. ``main_table_eligible`` must be
    False in a result payload; eligibility is established by the separate,
    read-only audit document produced from the run's evidence, never by a block
    the payload (or anyone editing it) writes about itself.
    """
    problems: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in payload:
            problems.append(f"missing field {field_name!r}")
    if payload.get("observation_surface") not in SURFACES:
        problems.append(f"bad observation_surface {payload.get('observation_surface')!r}")
    if payload.get("attacker_knowledge") not in KNOWLEDGE:
        problems.append(f"bad attacker_knowledge {payload.get('attacker_knowledge')!r}")
    if payload.get("knowledge_subcapability") not in SUB_CAPABILITIES:
        problems.append(
            f"bad knowledge_subcapability {payload.get('knowledge_subcapability')!r} "
            f"(plan §1.4: {sorted(SUB_CAPABILITIES)})"
        )
    if payload.get("scenario") not in SCENARIOS:
        problems.append(f"bad scenario {payload.get('scenario')!r}")
    if payload.get("candidate_mode") not in {"full", "proposal"}:
        problems.append(f"bad candidate_mode {payload.get('candidate_mode')!r}")
    if payload.get("candidate_mode") == "proposal" and payload.get("main_table_eligible"):
        problems.append("a proposal-mode result must not be main-table eligible")
    for field_name in COUNT_FIELDS:
        value = payload.get(field_name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"{field_name} is not a non-negative integer: {value!r}")
    problems.extend(_fraction_problems(payload.get("metrics", {}), "metrics"))
    if payload.get("main_table_eligible") is True:
        problems.append(
            "main_table_eligible: a result payload cannot self-certify eligibility; "
            "the independent audit document is the only source"
        )
        control = (payload.get("diagnostics") or {}).get("positive_control_passed")
        if control is not True:
            problems.append(
                f"main_table_eligible requires a passed positive control, got {control!r}"
            )
    return problems
