"""paper-v1 execution carrier.

Numbers are the section 8 floors from ``docs/PAPER_EVALUATION_PROTOCOL.md``.
``configs/paper/paper-v1.json`` is an execution carrier, not a second policy.
Importing this module does not register a faithful entry and does not import
torch or transformers. Current ``run_c08`` / ``real_battery`` entries are not
paper-faithful and are not auto-registered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


class ProfileError(Exception):
    """The named profile cannot be bound to the protocol."""


class PreflightError(Exception):
    """A paper-v1 start or budget missed one or more floors."""

    def __init__(self, shortages: list[str] | str):
        if isinstance(shortages, str):
            shortages = [shortages]
        self.shortages = [str(item) for item in shortages]
        super().__init__("; ".join(self.shortages))


PAPER_PROFILE_NAME = "paper-v1"
PAPER_ARMS = (
    "plaintext",
    "aloepri",
    "osnip",
    "ab",
    "sentinellms",
    "obfuslm_plus",
    "inferdpt",
    "dp_forward",
)
AUTHORITY = "docs/PAPER_EVALUATION_PROTOCOL.md"

MODEL_IDS = (
    "llama-3.2-1b-base",
    "llama-3.2-3b-instruct",
    "qwen2.5-7b-instruct",
)
PRIMARY_MODEL = "llama-3.2-1b-base"
SCENARIOS = ("S-public", "S-deploy", "S-encoder", "S-pair")
PAPER_ATTACKS = (
    "knn_l2",
    "knn_cosine",
    "knn_l2_key_robustness",
    "ednn",
    "vma_h_recovered_prefix",
    "vma_h_observed_prefix",
    "isa_er",
    "isa_tbs",
    "vma_w",
    "gate_ia",
    "attn_ia",
    "least_squares",
    "ima_ridge",
    "ima_neural",
    "linkage_exact",
    "linkage_cosine",
    "linkage_knn_aligned",
    "linkage_recovered_text",
    "aggregation",
    "tfma",
    "tfma_after_inversion",
    "sda",
    "text_bayes",
    "text_lm_rerank",
    "output_pii",
    "utility_generation",
    "utility_classification",
    "efficiency",
)
KNOWN_ATTACKS = frozenset(PAPER_ATTACKS)
KNOWN_CLASSES = frozenset(
    {f"C{i:02d}" for i in range(1, 15)} | {"C06W", "C06H", "C08A"}
)
LAYERS_1B = [1, 5, 10, 15]
TRAINING_SEEDS = {17, 29, 43}
EXTERNAL_ARMS = frozenset({"sentinellms", "obfuslm_plus", "inferdpt", "dp_forward"})
TEXT_ARMS = frozenset({"inferdpt"})
CLASSIFICATION_ARMS = frozenset({"sentinellms", "dp_forward"})
#: arms whose server observes a token/character symbol stream directly (ids or text)
SYMBOL_SURFACE_ARMS = frozenset({
    "plaintext", "aloepri", "sentinellms", "obfuslm_plus", "inferdpt",
})
#: arms whose server observes an embedding vector, so symbols need inversion first
VECTOR_SURFACE_ARMS = frozenset({"ab", "osnip", "dp_forward"})
VECTOR_SURFACE = frozenset({
    "knn_l2",
    "knn_cosine",
    "knn_l2_key_robustness",
    "ednn",
    "ima_ridge",
    "ima_neural",
})
DECODER_STATE = frozenset({
    "vma_h_recovered_prefix",
    "vma_h_observed_prefix",
    "isa_er",
    "isa_tbs",
})
WEIGHT_SURFACE = frozenset({"gate_ia", "attn_ia", "vma_w", "least_squares"})
LAYER_ATTACKS = frozenset({
    "vma_h_recovered_prefix",
    "vma_h_observed_prefix",
    "isa_er",
    "isa_tbs",
    "gate_ia",
    "attn_ia",
    "vma_w",
})
GENERATION_ONLY = frozenset({"output_pii", "utility_generation"})
CELL_FILTER_KEYS = frozenset({"arms"})

SOURCE_ANCHORS = {
    "arms": "section 8.1",
    "models": "section 8.1",
    "training_seeds": "section 8.1",
    "split_seed": "section 8.1",
    "root_key_bits": "section 8.1",
    "bootstrap_replicates": "section 8.1",
    "knn": "section 8.1",
    "vma_h": "section 8.2",
    "isa": "section 8.2",
    "weight": "section 8.2",
    "ima": "section 8.2",
    "stream": "section 8.2",
    "linkage": "section 8.2",
    "aggregation_m": "section 8.2",
    "tfma": "section 8.2",
    "sda": "section 8.2",
    "text": "section 8.2",
    "output": "section 8.2",
    "utility": "section 8.3",
    "efficiency": "section 8.3",
    "baseline_grids": "section 8.3",
}

CANONICAL_FLOORS = {
    "training_seeds": [17, 29, 43],
    "split_seed": 20261107,
    "root_key_bits": 256,
    "bootstrap_replicates": 2000,
    "arms": list(PAPER_ARMS),
    "models": {
        "primary": {
            "name": "Llama-3.2-1B",
            "variant": "base",
            "id": "llama-3.2-1b-base",
        },
        "scale": {
            "name": "Llama-3.2-3B-Instruct",
            "variant": "instruct",
            "id": "llama-3.2-3b-instruct",
        },
        "cross_family": {
            "name": "Qwen2.5-7B-Instruct",
            "variant": "instruct",
            "id": "qwen2.5-7b-instruct",
        },
    },
    "knn": {
        "main_texts": 1000,
        "main_keys": 1,
        "robust_texts": 200,
        "robust_keys": 5,
        "char_min": 20,
        "char_max": 512,
        "top_k": 100,
        "metrics": ["l2", "cosine"],
    },
    "vma_h": {
        "prompts": 200,
        "keys": 5,
        "layers_1b": list(LAYERS_1B),
        "prefix": "recovered",
    },
    "isa": {
        "prompts": 200,
        "keys": 5,
        "steps": 50000,
        "checkpoints": [1000, 10000, 25000, 50000],
        "extension_steps": [100000, 200000],
        "restarts": 3,
        "lr_grid": [1e-4, 5e-4, 1e-3, 1e-2],
        "lambda_dm": [0, 1e-3],
        "val_prompts": 40,
        "length_strata": {"32-64": 50, "65-128": 50, "129-256": 50, "257-512": 50},
        "optimizer": "AdamW",
        "weight_decay": 0,
        "loss": "MSE",
        "tbs_alpha": 5 / math.pi,
        "tbs_basis": "full",
    },
    "weight": {
        "min_types": 1000,
        "noise_tiers": [[0, 0], [0.2, 0.01], [1.0, 0.2]],
    },
    "ima": {
        "train_docs": 8192,
        "min_train_tokens": 1_000_000,
        "val_docs": 1024,
        "test_docs": 1000,
        "expansion_docs": [16384, 32768],
        "ridge_lambda": [0.01, 0.1, 1, 10],
        "epochs": [10, 20, 40],
        "epoch_extension": 80,
        "lr": [2e-5, 1e-4, 3e-4],
        "batch": 64,
    },
    "stream": {
        "n_streams": 3,
        "requests": 5000,
        "min_tokens": 100_000,
        "min_types": 1000,
        "key_pool": [4, 16, 64],
        "rotate_every": [1, 16, 64, 256],
    },
    "linkage": {
        "min_positives": 10000,
        "min_negatives": 100000,
        "fpr": [0.01, 0.001],
    },
    "aggregation_m": [1, 2, 4, 8, 16],
    "tfma": {
        "reference_tokens": 10_000_000,
        "min_types": 50000,
        "curve": [4_450_000, 10_000_000, 20_000_000],
        "top_k_distinct": 100,
        "control_top1": 0.90,
    },
    "sda": {
        "train_sequences": 60000,
        "val_sequences": 2000,
        "length": 128,
        "beam": [10, 50, 100],
        "epochs": [6, 12, 24],
    },
    "text": {
        "val": 200,
        "test": 1000,
        "bayes_epochs": [3, 10],
        "bayes_beam": [10, 50, 100],
        "lm_max_new_tokens": 512,
        "lm_model": {"id": "qwen2.5-7b-instruct"},
        "decoding": "greedy",
    },
    "output": {
        "prompts": 1000,
        "max_new_tokens": 256,
        "decoding": "greedy",
        "entities_per_type": 100,
        "entity_types": ["name", "email", "phone", "address", "id"],
    },
    "utility": {
        "classification_tasks": ["mmlu", "arc_easy", "hellaswag", "piqa", "mnli", "sst2",
                                 "anli_r1", "anli_r2", "anli_r3", "wic", "qnli"],
        "cross_family_classification_tasks": ["mmlu", "arc_easy", "hellaswag", "piqa", "sst2", "qnli"],
        "generation_tasks": ["wikitext2_ppl", "cnn_dailymail_summary"],
        "ppl_window": 512,
        "ppl_stride": 256,
        "summary_docs": 1000,
        "summary_max_input_tokens": 1024,
        "summary_max_new_tokens": 256,
        "shots": 0,
    },
    "efficiency": {
        "input_lengths": [128, 512, 1024],
        "output_length": 128,
        "concurrency": [1, 4, 16],
        "warmup": 20,
        "scored_requests": 200,
        "rounds": 3,
    },
    "baseline_grids": {
        "epsilon": [1, 3, 8],
        "obfuslm_k": [5, 10, 20],
        "obfuslm_epsilon": [0.1, 1, 3],
        "sentinel_glide": [1, 5, 10],
    },
}

# attack_id -> {"arms": set[str], "models": set[str] | None}
_FAITHFUL: dict[str, dict] = {}


class Profile:
    def __init__(self, name: str, protocol_sha256: str, floors: dict, source_anchors: dict):
        self.name = name
        self.protocol_sha256 = protocol_sha256
        self.floors = floors
        self.source_anchors = source_anchors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_root(root: Path | str | None) -> Path:
    return _repo_root() if root is None else Path(root)


def _protocol_path(root: Path | str | None) -> Path:
    return _as_root(root) / "docs" / "PAPER_EVALUATION_PROTOCOL.md"


def _carrier_path(root: Path | str | None) -> Path:
    return _as_root(root) / "configs" / "paper" / "paper-v1.json"


def protocol_sha256(root: Path | str | None = None) -> str:
    path = _protocol_path(root)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ProfileError(f"protocol file missing at {path}") from exc
    return hashlib.sha256(data).hexdigest()


def execution_carrier_document(protocol_bytes: bytes) -> dict:
    """Build the non-normative JSON carrier from the section 8 floors."""
    return {
        "role": "execution_carrier",
        "normative": False,
        "authority": AUTHORITY,
        "profile": PAPER_PROFILE_NAME,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "floors": CANONICAL_FLOORS,
    }


def _floors_mismatch(got, expected, path: str = "floors") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return [f"{path} got {got!r} floor {expected!r}"]
        mismatches = []
        for key, value in expected.items():
            if key not in got:
                mismatches.append(f"{path}.{key} missing floor {value!r}")
            else:
                mismatches.extend(_floors_mismatch(got[key], value, f"{path}.{key}"))
        return mismatches
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return [f"{path} got {got!r} floor {expected!r}"]
        mismatches = []
        for index, (item, want) in enumerate(zip(got, expected)):
            mismatches.extend(_floors_mismatch(item, want, f"{path}[{index}]"))
        return mismatches
    if isinstance(expected, float):
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return [f"{path} got {got!r} floor {expected!r}"]
        if not math.isclose(float(got), expected, rel_tol=1e-9, abs_tol=1e-12):
            return [f"{path} got {got!r} floor {expected!r}"]
        return []
    if got != expected:
        return [f"{path} got {got!r} floor {expected!r}"]
    return []


def bind_profile(profile_doc: dict, protocol_bytes: bytes) -> Profile:
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    stored = profile_doc.get("protocol_sha256") if isinstance(profile_doc, dict) else None
    if stored != digest:
        raise ProfileError(
            f"protocol hash drift: stored {stored!r} does not match protocol file hash {digest}"
        )
    if profile_doc.get("role") != "execution_carrier" or profile_doc.get("normative") is not False:
        raise ProfileError("paper-v1 carrier must be a non-normative execution_carrier")
    if profile_doc.get("authority") != AUTHORITY:
        raise ProfileError(f"paper-v1 carrier authority drifted from {AUTHORITY}")
    if "floors" not in profile_doc:
        raise ProfileError("paper-v1 carrier missing floors")
    mismatches = _floors_mismatch(profile_doc.get("floors"), CANONICAL_FLOORS)
    if mismatches:
        raise ProfileError(
            "paper-v1 floor drift: " + "; ".join(mismatches[:12])
        )
    return Profile(
        name=PAPER_PROFILE_NAME,
        protocol_sha256=digest,
        floors=copy.deepcopy(profile_doc["floors"]),
        source_anchors=dict(SOURCE_ANCHORS),
    )


def require_profile(name: str | None, *, root: Path | str | None = None) -> Profile:
    if name is None or (isinstance(name, str) and name.strip() == ""):
        raise ProfileError("profile is required")
    if not isinstance(name, str):
        raise ProfileError(f"profile must be a string, got {type(name).__name__}")
    base = _as_root(root)
    if name == "probe":
        digest = ""
        proto = _protocol_path(base)
        if proto.is_file():
            digest = hashlib.sha256(proto.read_bytes()).hexdigest()
        return Profile(
            name="probe",
            protocol_sha256=digest,
            floors={},
            source_anchors={"probe": "not a section 8 formal profile"},
        )
    if name != PAPER_PROFILE_NAME:
        raise ProfileError(f"unknown profile {name!r}")
    proto = _protocol_path(base)
    if not proto.is_file():
        raise ProfileError(
            f"protocol file missing at {proto}; refusing paper-v1 without a protocol hash"
        )
    carrier = _carrier_path(base)
    if not carrier.is_file():
        raise ProfileError(f"paper-v1 execution carrier missing at {carrier}")
    try:
        doc = json.loads(carrier.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"paper-v1 carrier is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ProfileError("paper-v1 carrier must be a JSON object")
    return bind_profile(doc, proto.read_bytes())


def layers_for_depth(n_layers: int) -> list[int]:
    if n_layers < 2:
        raise PreflightError([f"layers depth got {n_layers} floor 2"])
    values = {
        1,
        math.ceil(n_layers / 3),
        math.ceil(2 * n_layers / 3),
        n_layers - 1,
    }
    return sorted(values)


def layers_for_model(model_id: str) -> list[int]:
    if model_id == PRIMARY_MODEL:
        return list(LAYERS_1B)
    raise PreflightError(
        [f"model depth unverified for {model_id}; refusing to copy 1B layers {LAYERS_1B}"]
    )


def register_faithful_entry(
    attack_id: str,
    *,
    arms: set[str],
    models: set[str] | None = None,
    scenarios: set[str] | None = None,
) -> None:
    _FAITHFUL[str(attack_id)] = {
        "arms": set(arms),
        "models": None if models is None else set(models),
        "scenarios": None if scenarios is None else set(scenarios),
    }


def unregister_faithful_entry(attack_id: str) -> None:
    _FAITHFUL.pop(str(attack_id), None)


def clear_faithful_entries() -> None:
    _FAITHFUL.clear()


def _faithful_for(attack_id: str, arm: str, model: str | None = None,
                  scenario: str | None = None) -> bool:
    entry = _FAITHFUL.get(attack_id)
    if not entry or arm not in entry["arms"]:
        return False
    if model is not None and entry["models"] is not None and model not in entry["models"]:
        return False
    if (
        scenario is not None
        and entry.get("scenarios") is not None
        and scenario not in entry["scenarios"]
    ):
        return False
    return True


def _id_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _negative_shortages(value, field: str, shortages: list[str]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and value < 0:
        shortages.append(f"{field} got {value} floor 0")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _negative_shortages(item, str(key), shortages)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _negative_shortages(item, field, shortages)


def _max_of(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            parts = [int(part) for part in value.replace(" ", "").split(",") if part]
        except ValueError:
            return None
        return max(parts) if parts else 0
    if isinstance(value, (list, tuple)):
        try:
            nums = [int(part) for part in value]
        except (TypeError, ValueError):
            return None
        return max(nums) if nums else 0
    return None


def _as_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _budget_shortage(shortages: list[str], field: str, value, floor) -> None:
    got = _as_number(value)
    if got is None:
        shortages.append(f"{field} got {value!r} floor {floor} (not numeric)")
    elif got < floor:
        shortages.append(f"{field} got {value} floor {floor}")


def check_requested_budget(profile: Profile, requested: dict | None) -> None:
    """Refuse a paper-v1 request that undercuts section 8. Probe is not enforced."""
    if profile is None or getattr(profile, "name", None) != PAPER_PROFILE_NAME:
        return
    requested = requested or {}
    floors = profile.floors or CANONICAL_FLOORS
    shortages: list[str] = []
    _negative_shortages(requested, "count", shortages)

    if "knn_texts" in requested:
        _budget_shortage(shortages, "knn_texts", requested["knn_texts"], floors["knn"]["main_texts"])
    if "isa_steps" in requested:
        maximum = _max_of(requested["isa_steps"])
        floor = floors["isa"]["steps"]
        if maximum is None:
            shortages.append(
                f"isa_steps got {requested['isa_steps']!r} maximum None floor {floor}"
            )
        elif maximum < floor:
            shortages.append(
                f"isa_steps got {requested['isa_steps']} maximum {maximum} floor {floor}"
            )
    if "isa_prompts" in requested:
        _budget_shortage(shortages, "isa_prompts", requested["isa_prompts"], floors["isa"]["prompts"])
    if "vma_prompts" in requested:
        _budget_shortage(shortages, "vma_prompts", requested["vma_prompts"], floors["vma_h"]["prompts"])
    if "max_len" in requested:
        _budget_shortage(shortages, "max_len", requested["max_len"], floors["knn"]["char_max"])
    if "top_k" in requested:
        _budget_shortage(shortages, "top_k", requested["top_k"], floors["knn"]["top_k"])
    if "tfma_top100_distinct" in requested:
        _budget_shortage(
            shortages,
            "tfma_top100_distinct",
            requested["tfma_top100_distinct"],
            floors["tfma"]["top_k_distinct"],
        )
    if "n_prompts" in requested:
        got = _as_number(requested["n_prompts"])
        if got is None:
            shortages.append(f"n_prompts got {requested['n_prompts']!r} floor 0 (not numeric)")
        elif got < 0:
            shortages.append(f"n_prompts got {requested['n_prompts']} floor 0")
    if "layers" in requested:
        layers = list(requested["layers"])
        if layers != list(floors["vma_h"]["layers_1b"]):
            shortages.append(
                f"layers got {layers} floor {list(floors['vma_h']['layers_1b'])}"
            )
    if "training_seeds" in requested:
        seeds = list(requested["training_seeds"])
        rejected = [seed for seed in seeds if seed not in TRAINING_SEEDS]
        if rejected:
            shortages.append(
                f"training_seeds got {seeds} floor {list(floors['training_seeds'])}; "
                f"rejected {rejected}"
            )
    for key, known in (("classes", KNOWN_CLASSES), ("attacks", KNOWN_ATTACKS)):
        if key not in requested:
            continue
        for item in _id_list(requested[key]):
            token = str(item)
            lookup = token.upper() if key == "classes" else token
            if lookup not in known and token not in known:
                shortages.append(f"{key} unknown id {token}")
    if "arms" in requested:
        present = set(_id_list(requested["arms"]))
        missing = [arm for arm in PAPER_ARMS if arm not in present]
        if missing:
            shortages.append(
                f"arms missing {', '.join(missing)} got {list(requested['arms'])} "
                f"floor {list(PAPER_ARMS)}"
            )
    if shortages:
        raise PreflightError(shortages)


def _looks_like_instruct_base(info: dict) -> bool:
    blob = " ".join(
        str(info.get(key) or "")
        for key in ("variant", "name", "path", "model_path", "weights_path")
    )
    return "instruct" in blob.lower()


#: attack -> data fields this cell actually consumes (protocol §2/§8 floors)
def data_requirements(attack: str, floors: dict) -> list[tuple[str, int | float]]:
    """Per-attack data floors. A KNN cell needs the KNN set, not the IMA corpus."""
    knn = floors["knn"]
    if attack in {"knn_l2", "knn_cosine", "ednn"}:
        return [("knn_texts", knn["main_texts"])]
    if attack == "knn_l2_key_robustness":
        # the robust slice is defined after the main 1000; without the main set it is
        # just the first 200 filtered rows and the independence claim is unverifiable
        return [("knn_texts", knn["main_texts"]),
                ("robust_texts", knn["robust_texts"])]
    if attack in {"vma_h_recovered_prefix", "vma_h_observed_prefix"}:
        return [("vma_prompts", floors["vma_h"]["prompts"])]
    if attack in {"isa_er", "isa_tbs"}:
        return [("isa_prompts", floors["isa"]["prompts"])]
    if attack in {"ima_ridge", "ima_neural"}:
        return [
            ("train_docs", floors["ima"]["train_docs"]),
            ("train_tokens", floors["ima"]["min_train_tokens"]),
            ("val_docs", floors["ima"]["val_docs"]),
            ("test_docs", floors["ima"]["test_docs"]),
        ]
    if attack in {"tfma", "tfma_after_inversion"}:
        return [
            ("reference_tokens", floors["tfma"]["reference_tokens"]),
            ("reference_types", floors["tfma"]["min_types"]),
        ]
    if attack in {
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned",
        "linkage_recovered_text", "aggregation",
    }:
        return [
            ("stream_requests", floors["stream"]["requests"]),
            ("stream_tokens", floors["stream"]["min_tokens"]),
            ("stream_types", floors["stream"]["min_types"]),
        ]
    if attack in {"text_bayes", "text_lm_rerank"}:
        return [("text_test", floors["text"]["test"])]
    if attack == "output_pii":
        return [("output_prompts", floors["output"]["prompts"])]
    return []


def _data_shortages(manifest: dict | None, floors: dict, attacks: set[str] | None) -> list[str]:
    """Check only the data floors of the attacks in the requested cells."""
    manifest = manifest if isinstance(manifest, dict) else {}
    data = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}
    selected = sorted(attacks) if attacks is not None else list(PAPER_ATTACKS)
    requirements: list[tuple[str, int | float]] = []
    for attack in selected:
        for field, floor in data_requirements(attack, floors):
            if (field, floor) not in requirements:
                requirements.append((field, floor))
    shortages: list[str] = []
    for field, floor in requirements:
        got = data.get(field)
        if got is None:
            shortages.append(f"data {field} missing floor {floor}")
        else:
            try:
                if float(got) < float(floor):
                    shortages.append(f"data {field} got {got} floor {floor}")
            except (TypeError, ValueError):
                shortages.append(f"data {field} got {got!r} is not numeric (floor {floor})")
    if "knn_l2_key_robustness" in selected:
        got_keys = data.get("robust_keys")
        floor_keys = floors["knn"]["robust_keys"]
        if got_keys != floor_keys:
            shortages.append(f"data robust_keys got {got_keys} floor {floor_keys}")
    if not requirements:
        return shortages
    if data.get("split_seed") != floors["split_seed"]:
        shortages.append(
            f"data split_seed got {data.get('split_seed')} floor {floors['split_seed']}"
        )
    if data.get("hashes_disjoint") is not True:
        shortages.append("split hashes are not disjoint")
    if set(selected) & {"knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn"}:
        from ..data.corpora import CLUSTERED_SPLIT

        if data.get("split_strategy") != CLUSTERED_SPLIT:
            shortages.append("data clustered split strategy missing or unverified")
        digest = data.get("split_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            shortages.append("data split_sha256 missing or invalid")
    split_hashes = data.get("split_hashes")
    if isinstance(split_hashes, dict) and split_hashes:
        lists = {name: list(split_hashes.get(name) or []) for name in ("train", "val", "test")}
        if any(not values for values in lists.values()):
            shortages.append("split_hashes missing train/val/test content hashes")
        elif (
            set(lists["train"]) & set(lists["val"])
            or set(lists["train"]) & set(lists["test"])
            or set(lists["val"]) & set(lists["test"])
        ):
            shortages.append("split_hashes content overlap across train/val/test")
    return shortages


def _requested_scope(requested: dict | None) -> tuple[set[str] | None, set[str] | None]:
    """Return (attacks, arms) scoping a per-cell preflight; None means every cell."""
    if not isinstance(requested, dict):
        return None, None
    attacks = None if requested.get("attacks") is None else {str(a) for a in _id_list(requested["attacks"])}
    arms = None if requested.get("arms") is None else {str(a) for a in _id_list(requested["arms"])}
    return attacks, arms


def _identity_shortages(
    manifest: dict | None,
    floors: dict,
    *,
    attacks: set[str] | None = None,
    arms: set[str] | None = None,
    models: set[str] | None = None,
) -> list[str]:
    manifest = manifest if isinstance(manifest, dict) else {}
    shortages: list[str] = []
    models_doc = manifest.get("models") if isinstance(manifest.get("models"), dict) else {}
    if not models_doc:
        shortages.append(
            "missing model identity: revision and config/tokenizer/weights sha256"
        )
    expected = {
        "llama-3.2-1b-base": "base",
        "llama-3.2-3b-instruct": "instruct",
        "qwen2.5-7b-instruct": "instruct",
    }
    required_models = models if models is not None else set(expected)
    for model_id, variant in expected.items():
        if model_id not in required_models:
            continue
        info = models_doc.get(model_id) if isinstance(models_doc.get(model_id), dict) else {}
        if not info:
            shortages.append(
                f"missing model identity for {model_id}: revision and sha256"
            )
            continue
        if not info.get("revision"):
            shortages.append(f"{model_id} missing revision")
        if not info.get("path"):
            shortages.append(f"{model_id} missing path")
        for field in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
            if not info.get(field):
                shortages.append(f"{model_id} missing {field}")
        if model_id == PRIMARY_MODEL and _looks_like_instruct_base(info):
            shown = info.get("variant") or info.get("name") or info.get("path")
            shortages.append(
                f"{model_id} primary variant got {shown} floor base; "
                "Instruct is not the 1B base"
            )
        elif info.get("variant") != variant:
            shortages.append(f"{model_id} variant got {info.get('variant')!r} floor {variant}")
    shortages.extend(_data_shortages(manifest, floors, attacks))
    if attacks is not None:
        data = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}
        content_bound = {"knn_l2", "knn_cosine", "ednn", "ima_ridge", "ima_neural"}
        if attacks & content_bound and not isinstance(data.get("knn_sha256"), str):
            shortages.append("data knn_sha256 missing for content binding")
        if "knn_l2_key_robustness" in attacks and not isinstance(
            data.get("robust_sha256"), str
        ):
            shortages.append("data robust_sha256 missing for content binding")
    arms_doc = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    required_arms = arms if arms is not None else set(PAPER_ARMS)
    for arm in PAPER_ARMS:
        if arm not in required_arms:
            continue
        info = arms_doc.get(arm) if isinstance(arms_doc.get(arm), dict) else {}
        if not info.get("adapter"):
            shortages.append(f"{arm} missing adapter")
        if arm in EXTERNAL_ARMS:
            for field in ("license", "commit", "source_verified"):
                if not info.get(field):
                    shortages.append(f"{arm} missing {field}")
    if arms is not None and arms & {"ab", "aloepri"}:
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
        ab = artifacts.get("ab") if isinstance(artifacts.get("ab"), dict) else {}
        if not ab.get("path"):
            shortages.append("ab artifact missing path")
        files = ab.get("files_sha256") if isinstance(ab.get("files_sha256"), dict) else {}
        if not files:
            shortages.append("ab artifact missing file content hashes")
    return shortages


def _applicability(arm: str, attack: str) -> tuple[bool, str]:
    if arm in TEXT_ARMS and attack in VECTOR_SURFACE:
        return False, (
            "InferDPT uploads perturbed text, not an embedding vector; "
            "inventing an uploaded vector is not allowed"
        )
    if arm in TEXT_ARMS and attack in DECODER_STATE:
        return False, "InferDPT uploads text and does not expose decoder hidden states"
    if arm in CLASSIFICATION_ARMS and attack in DECODER_STATE:
        return False, (
            "classification defense has no validated decoder; "
            "hidden-state recovery does not apply"
        )
    if arm == "osnip" and attack in WEIGHT_SURFACE:
        return False, (
            "OSNIP does not expose a weight permutation this attack can align"
        )
    if attack == "tfma" and arm in VECTOR_SURFACE_ARMS:
        return False, (
            "embedding-vector upload has no directly observed symbol stream; "
            "use tfma_after_inversion"
        )
    if attack == "tfma_after_inversion" and arm not in VECTOR_SURFACE_ARMS:
        return False, (
            "this arm already exposes symbols; there is no inversion step to compose"
        )
    if attack in GENERATION_ONLY and arm in CLASSIFICATION_ARMS:
        return False, (
            "classification defense has no generation path; "
            "output PII does not apply"
        )
    if attack == "least_squares" and arm not in {"aloepri", "ab"}:
        return False, (
            "least_squares is the one-sided dense weight variant, not vma_w, "
            "and this arm has no such weight alignment"
        )
    return True, ""


def applicable(arm: str, attack: str) -> bool:
    """Public wrapper for the §8 applicability table (used by the auditor)."""
    return _applicability(str(arm), str(attack))[0]


def _composed(arm: str, attack: str) -> bool:
    if not _applicability(arm, attack)[0]:
        return False
    if attack in {"tfma_after_inversion", "linkage_recovered_text"}:
        return True
    if attack in {"text_bayes", "text_lm_rerank"} and arm not in TEXT_ARMS:
        return True
    return False


def _requested_pairs(requested: dict | None) -> list[tuple[str, str]] | None:
    """None means every applicable cell. A list is the requested subset."""
    if not requested:
        return None
    attacks = requested.get("attacks")
    arms = requested.get("arms")
    classes = requested.get("classes")
    if attacks is None and arms is None and classes is None:
        return None
    attack_ids = _id_list(attacks) if attacks is not None else list(PAPER_ATTACKS)
    arm_ids = _id_list(arms) if arms is not None else list(PAPER_ARMS)
    return [(str(attack), str(arm)) for attack in attack_ids for arm in arm_ids]


def _faithful_shortages(requested: dict | None) -> list[str]:
    pairs = _requested_pairs(requested)
    if pairs is None:
        pairs = [
            (attack, arm)
            for attack in PAPER_ATTACKS
            for arm in PAPER_ARMS
            if _applicability(arm, attack)[0]
        ]
    missing = []
    for attack, arm in pairs:
        if attack not in KNOWN_ATTACKS or arm not in PAPER_ARMS:
            continue
        if not _applicability(arm, attack)[0]:
            continue
        if not _faithful_for(attack, arm):
            missing.append(
                f"not_run: no configured faithful entry for {attack}/{arm}"
            )
    if not missing:
        return []
    if len(missing) > 8:
        return missing[:8] + [
            f"not_run: {len(missing)} cells have no configured faithful entry"
        ]
    return missing


def preflight_formal_start(
    profile: Profile,
    manifest: dict | None,
    requested: dict | None = None,
    *,
    models: set[str] | None = None,
) -> None:
    if getattr(profile, "name", None) != PAPER_PROFILE_NAME:
        raise PreflightError(["formal start requires profile paper-v1"])
    floors = profile.floors or CANONICAL_FLOORS
    req = requested if isinstance(requested, dict) else {}
    # ``arms`` here is a cell filter. A subset is not a reduced eight-arm budget.
    budget_view = {key: value for key, value in req.items() if key not in CELL_FILTER_KEYS}
    shortages: list[str] = []
    try:
        check_requested_budget(profile, budget_view)
    except PreflightError as exc:
        shortages.extend(exc.shortages)
    attacks, arms = _requested_scope(req if req else None)
    if models is None:
        if attacks is None and arms is None:
            models = set(MODEL_IDS)
        else:
            present = manifest.get("models") if isinstance(manifest, dict) else None
            present = present if isinstance(present, dict) else {}
            scoped = {model for model in MODEL_IDS if model in present}
            models = scoped or set(MODEL_IDS)
    shortages.extend(
        _identity_shortages(manifest, floors, attacks=attacks, arms=arms, models=models)
    )
    shortages.extend(_faithful_shortages(req if req else None))
    if shortages:
        raise PreflightError(shortages)


def _job_budget(floors: dict, attack: str, model: str) -> dict:
    knn = floors["knn"]
    if attack in {"knn_l2", "knn_cosine", "ednn"}:
        budget = {
            "n_texts": knn["main_texts"],
            "n_keys": knn["main_keys"],
            "top_k": knn["top_k"],
            "char_min": knn["char_min"],
            "char_max": knn["char_max"],
        }
    elif attack == "knn_l2_key_robustness":
        budget = {
            "n_texts": knn["robust_texts"],
            "n_keys": knn["robust_keys"],
            "top_k": knn["top_k"],
        }
    elif attack in DECODER_STATE:
        family = "vma_h" if attack.startswith("vma_h") else "isa"
        budget = {
            "prompts": floors[family]["prompts"],
            "keys": floors[family]["keys"],
        }
        if attack == "vma_h_recovered_prefix":
            budget["prefix"] = "recovered"
        elif attack == "vma_h_observed_prefix":
            budget["prefix"] = "observed"
    elif attack in {"utility_generation", "utility_classification"}:
        budget = {
            "shots": floors["utility"]["shots"],
            "summary_docs": floors["utility"]["summary_docs"],
        }
    else:
        budget = {}
    families = {
        "vma_h_recovered_prefix": ("vma_h",), "vma_h_observed_prefix": ("vma_h",),
        "isa_er": ("isa",), "isa_tbs": ("isa",),
        "vma_w": ("weight",), "gate_ia": ("weight",), "attn_ia": ("weight",),
        "least_squares": ("weight",), "ima_ridge": ("ima",), "ima_neural": ("ima",),
        "linkage_exact": ("stream", "linkage"), "linkage_cosine": ("stream", "linkage"),
        "linkage_knn_aligned": ("stream", "linkage"),
        "linkage_recovered_text": ("stream", "linkage"),
        "aggregation": ("stream", "aggregation_m"),
        "tfma": ("stream", "tfma"), "tfma_after_inversion": ("stream", "tfma"),
        "sda": ("stream", "sda"), "text_bayes": ("text",), "text_lm_rerank": ("text",),
        "output_pii": ("output",), "utility_generation": ("utility",),
        "utility_classification": ("utility",), "efficiency": ("efficiency",),
    }
    if attack in families:
        budget["requirements"] = {name: copy.deepcopy(floors[name]) for name in families[attack]}
        if attack in {"ima_neural", "sda"}:
            budget["requirements"]["training_seeds"] = list(floors["training_seeds"])
    if attack in LAYER_ATTACKS and model == PRIMARY_MODEL:
        budget["layers"] = list(LAYERS_1B)
    return budget


def _job_blockers(manifest: dict | None, floors: dict, arm: str, model: str, attack: str) -> list[str]:
    blockers: list[str] = []
    manifest = manifest if isinstance(manifest, dict) else {}
    if attack in LAYER_ATTACKS and model != PRIMARY_MODEL:
        blockers.append(
            f"model depth unverified for {model}; refusing to invent layers "
            f"or copy 1B layers {LAYERS_1B}"
        )
    models = manifest.get("models") if isinstance(manifest.get("models"), dict) else {}
    info = models.get(model) if isinstance(models.get(model), dict) else {}
    if not info.get("revision"):
        blockers.append(f"{model} missing revision")
    if not info.get("path"):
        blockers.append(f"{model} missing path")
    for field in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
        if not info.get(field):
            blockers.append(f"{model} missing {field}")
    if model == PRIMARY_MODEL and (not info or _looks_like_instruct_base(info)):
        if not info or info.get("variant") != "base":
            blockers.append("primary variant unverified; Instruct must not be substituted for the 1B base")
    expected_variants = {
        "llama-3.2-1b-base": "base",
        "llama-3.2-3b-instruct": "instruct",
        "qwen2.5-7b-instruct": "instruct",
    }
    if model in expected_variants and info.get("variant") != expected_variants[model]:
        blockers.append(
            f"{model} variant got {info.get('variant')!r} floor {expected_variants[model]}"
        )
    data = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}
    if attack in {"knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn"}:
        from ..data.corpora import CLUSTERED_SPLIT

        if data.get("split_strategy") != CLUSTERED_SPLIT:
            blockers.append("data clustered split strategy missing or unverified")
        digest = data.get("split_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            blockers.append("data split_sha256 missing or invalid")
    requirements = data_requirements(attack, floors)
    for field, floor in requirements:
        value = data.get(field)
        if value is None:
            blockers.append(f"data {field} got {value} floor {floor}")
        else:
            try:
                if float(value) < float(floor):
                    blockers.append(f"data {field} got {value} floor {floor}")
            except (TypeError, ValueError):
                blockers.append(
                    f"data {field} got {value!r} is not numeric (floor {floor})"
                )
    if requirements:
        if data.get("split_seed") != floors["split_seed"]:
            blockers.append(
                f"data split_seed got {data.get('split_seed')} floor {floors['split_seed']}"
            )
        if data.get("hashes_disjoint") is not True:
            blockers.append("split hashes are not disjoint")
    if attack in {"knn_l2", "knn_cosine", "ednn", "ima_ridge", "ima_neural"} and not isinstance(
        data.get("knn_sha256"), str
    ):
        blockers.append("data knn_sha256 missing for content binding")
    if attack == "knn_l2_key_robustness":
        if not isinstance(data.get("robust_sha256"), str):
            blockers.append("data robust_sha256 missing for content binding")
        if data.get("robust_keys") != floors["knn"]["robust_keys"]:
            blockers.append(
                f"data robust_keys got {data.get('robust_keys')} "
                f"floor {floors['knn']['robust_keys']}"
            )
    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    arm_info = arms.get(arm) if isinstance(arms.get(arm), dict) else {}
    if not arm_info.get("adapter"):
        blockers.append(f"{arm} missing adapter")
    if arm in EXTERNAL_ARMS:
        for field in ("license", "commit", "source_verified"):
            if not arm_info.get(field):
                blockers.append(f"{arm} missing {field}")
    if arm in {"ab", "aloepri"}:
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
        ab = artifacts.get("ab") if isinstance(artifacts.get("ab"), dict) else {}
        if not ab.get("path"):
            blockers.append("ab artifact missing path")
        if not isinstance(ab.get("files_sha256"), dict) or not ab.get("files_sha256"):
            blockers.append("ab artifact missing file content hashes")
    return blockers


def _provenance_missing(blockers: list[str]) -> bool:
    return any(
        "missing license" in item or "missing commit" in item or "missing source_verified" in item
        for item in blockers
    )


def plan_jobs(profile: Profile, manifest: dict | None = None, *, root: Path | str | None = None) -> dict:
    del root  # plans do not read weights or construct a deployment
    floors = profile.floors if isinstance(profile.floors, dict) and "knn" in profile.floors else CANONICAL_FLOORS
    jobs = []
    for arm in PAPER_ARMS:
        for model in MODEL_IDS:
            for scenario in SCENARIOS:
                for attack in PAPER_ATTACKS:
                    applicable, na_reason = _applicability(arm, attack)
                    blockers = _job_blockers(manifest, floors, arm, model, attack)
                    faithful = _faithful_for(attack, arm, model, scenario)
                    composed = _composed(arm, attack)
                    if not applicable:
                        status = "not_applicable"
                        reason = na_reason
                    elif faithful and not blockers:
                        status = "configured"
                        reason = "configured faithful entry"
                    elif composed:
                        status = "depends_on"
                        reason = (
                            "depends_on a prior inversion; not configured until a faithful entry exists"
                        )
                    elif _provenance_missing(blockers):
                        status = "unverified"
                        reason = "; ".join(blockers)
                    elif blockers:
                        status = "resource_blocked"
                        reason = "; ".join(blockers)
                    elif _faithful_for(attack, arm, model):
                        status = "not_run"
                        reason = f"scenario {scenario} has no configured faithful entry"
                    else:
                        status = "not_run"
                        reason = "not_run: no configured faithful entry"
                    jobs.append({
                        "job_id": f"{arm}|{model}|{scenario}|{attack}",
                        "arm": arm,
                        "model": model,
                        "scenario": scenario,
                        "attack": attack,
                        "status": status,
                        "applicable": applicable,
                        "reason": reason,
                        "blockers": blockers,
                        "faithful_entry": faithful,
                        "budget": _job_budget(floors, attack, model),
                    })
    return {
        "profile": profile.name,
        "protocol_sha256": profile.protocol_sha256,
        "authority": AUTHORITY,
        "normative": False,
        "granularity": "attack_arm_model_scenario",
        "axis_expansion": "not_expanded; budget grids are requirements, not individual scheduled runs",
        "expanded_matrix_tool": "scripts/plan_paper_matrix.py",
        "jobs": jobs,
    }


def formal_aggregate_allowed(record: dict) -> bool:
    if not isinstance(record, dict) or record.get("profile") != PAPER_PROFILE_NAME:
        return False
    try:
        digest = protocol_sha256()
    except ProfileError:
        return False
    if record.get("protocol_sha256") != digest:
        return False
    audit = record.get("audit")
    if not isinstance(audit, dict):
        return False
    return (
        audit.get("verdict") == "audited_paper_scale"
        and audit.get("auditor") == "protocol_auditor"
        and audit.get("protocol_sha256") == digest
    )


def discover_local_resources(search_roots: list[Path]) -> dict:
    """Report local model directories without inventing a weights hash."""
    models = {
        "llama-3.2-1b-base": {
            "expected_name": "Llama-3.2-1B",
            "expected_variant": "base",
            "weights_sha256": None,
            "config_sha256": None,
            "tokenizer_sha256": None,
            "matched_path": None,
            "rejected": [],
        },
        "llama-3.2-3b-instruct": {
            "expected_name": "Llama-3.2-3B-Instruct",
            "expected_variant": "instruct",
            "weights_sha256": None,
            "matched_path": None,
            "rejected": [],
        },
        "qwen2.5-7b-instruct": {
            "expected_name": "Qwen2.5-7B-Instruct",
            "expected_variant": "instruct",
            "weights_sha256": None,
            "matched_path": None,
            "rejected": [],
        },
    }
    notes: list[str] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            notes.append(f"missing search root {root}")
            continue
        candidates = [root] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_dir())
        for path in candidates:
            low = path.name.lower().replace("_", "-")
            if "llama-3.2-1b" in low and "instruct" in low:
                models["llama-3.2-1b-base"]["rejected"].append({
                    "path": str(path),
                    "name": path.name,
                    "variant": "instruct",
                    "is_base": False,
                    "reason": f"{path.name} is Instruct, not the Llama-3.2-1B base",
                    "weights_sha256": None,
                })
                notes.append(f"{path.name} is Instruct, not the base")
            elif low in {"llama-3.2-1b", "llama-3.2-1b-base"}:
                models["llama-3.2-1b-base"]["matched_path"] = str(path)
                notes.append(f"{path.name} may be the base; weights hash was not computed")
    return {"models": models, "notes": notes}


def gate_battery_args(
    profile,
    *,
    plan_only: bool,
    requested: dict,
    manifest: dict | None = None,
    root: Path | str | None = None,
) -> dict:
    if profile is None or profile == "":
        message = "profile is required; refusing to start without an explicit profile"
        return {"action": "refuse", "code": 2, "message": message, "shortages": [message]}
    try:
        bound = profile if isinstance(profile, Profile) else require_profile(profile, root=root)
    except ProfileError as exc:
        return {"action": "refuse", "code": 2, "message": str(exc), "shortages": [str(exc)]}
    if bound.name != PAPER_PROFILE_NAME:
        if plan_only:
            return {"action": "plan", "code": 0, "plan": plan_jobs(bound, manifest, root=root)}
        return {"action": "run_probe", "code": 0, "aggregate": False}
    if requested:
        budget_view = {key: value for key, value in requested.items() if key != "arms"}
        try:
            check_requested_budget(bound, budget_view)
        except PreflightError as exc:
            return {
                "action": "refuse",
                "code": 2,
                "message": str(exc),
                "shortages": list(exc.shortages),
            }
    if plan_only:
        return {"action": "plan", "code": 0, "plan": plan_jobs(bound, manifest, root=root)}
    try:
        preflight_formal_start(bound, manifest, requested or None)
    except PreflightError as exc:
        return {
            "action": "refuse",
            "code": 2,
            "message": str(exc),
            "shortages": list(exc.shortages),
        }
    return {"action": "run_paper", "code": 0, "budget": copy.deepcopy(bound.floors)}
