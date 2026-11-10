"""paper-v1 real executor (W7): verified artifacts -> real wire -> KNN -> immutable evidence.

The executor is the only bridge from the CLI to a real attack. Before any model,
encoder or table is touched it re-verifies the manifest against the files on disk
(content hashes of the model, the artifact, the corpus and the protocol) and runs
the per-cell preflight again. A missing item, a hash mismatch, an unsupported arm
or a budget cut raises :class:`ExecutorRefusal`; nothing is executed.

Only cells registered as faithful by :mod:`paper_runner` are dispatched here. The
first unit is C01 (KNN/EDNN) on the packaged A+B artifact; other attack ids are
rejected with a reason instead of producing an empty success.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..schemes.ab_deployment import ABDeployment, local_model_revision, verify_model_identity
from . import real_battery
from .export_randomness import verify_export_randomness
from .square_lineage import verify_square_lineage
from .square_endpoints import verify_square_endpoints
from .encoder_lineage import verify_encoder_binding
from .public_guard import assert_public_payload
from .paper_profile import (
    PAPER_PROFILE_NAME,
    PreflightError,
    preflight_formal_start,
    protocol_sha256,
)
from .run_store import content_sha256

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = ROOT / "reports" / "privacy_rebuild_20261107" / "runs"
SUPPORTED_ATTACKS = {"knn_l2", "knn_cosine", "knn_l2_key_robustness"}
SUPPORTED_ARMS = {"plaintext", "aloepri", "ab"}
SUPPORTED_SCENARIOS = {"S-public"}


class ExecutorRefusal(RuntimeError):
    """The cell did not reach the attack because an input failed verification."""

    def __init__(self, shortages: Sequence[str] | str):
        if isinstance(shortages, str):
            shortages = [shortages]
        self.shortages = [str(item) for item in shortages]
        super().__init__("; ".join(self.shortages))


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def texts_sha256(texts: Sequence[str]) -> str:
    """Content digest of the exact frozen text set (order-sensitive)."""
    digest = hashlib.sha256()
    for text in texts:
        blob = str(text).encode("utf-8")
        digest.update(len(blob).to_bytes(8, "little"))
        digest.update(blob)
    return digest.hexdigest()


def _git_identity() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip().splitlines()
    except Exception as exc:  # pragma: no cover - git is present in this repo
        return {"commit": None, "error": str(exc)}
    files = {}
    # The run identity must separate algorithm revisions too, not only the runner
    # files: a change in an attack/core module can change every number in the run.
    source_root = ROOT / "src" / "obfuscation_attacks"
    for path in sorted(source_root.rglob("*.py")):
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    for relative in (
        "scripts/run_full_battery.py",
        "scripts/audit_paper_run.py",
    ):
        path = ROOT / relative
        files[relative] = sha256_file(path) if path.is_file() else None
    return {
        "commit": commit or None,
        "n_uncommitted": len(status),
        "uncommitted_sample": status[:12],
        "files_sha256": files,
    }


class PaperExecutor:
    """Execute one configured C01 cell against verified real artifacts."""

    def __init__(
        self,
        profile,
        manifest: Mapping | None,
        *,
        root_key: bytes | None = None,
        result_root: Path | str | None = None,
        evidence_root: Path | str | None = None,
        deployment_factory: Callable[[Path, Path, bytes | None, bool], ABDeployment] | None = None,
        data_loader: Callable[..., dict] | None = None,
        code_identity: dict | None = None,
    ) -> None:
        if getattr(profile, "name", None) != PAPER_PROFILE_NAME:
            raise ExecutorRefusal("paper executor requires the paper-v1 profile")
        if not isinstance(manifest, Mapping):
            raise ExecutorRefusal("paper executor requires a manifest")
        self.profile = profile
        self.manifest = dict(manifest)
        self.root_key = root_key
        self.result_root = Path(result_root or DEFAULT_RESULT_ROOT)
        self.evidence_root = Path(evidence_root or (self.result_root / "evidence"))
        self.deployment_factory = deployment_factory or self._default_deployment
        self.data_loader = data_loader or self._default_data_loader
        self.code_identity = code_identity if code_identity is not None else _git_identity()
        self._base_identity: dict | None = None
        self._identity_by_role: dict[str, dict] = {}
        self._data_actual: dict | None = None
        self._texts: list[str] | None = None
        self._corpus: dict | None = None
        self._reference: tuple[dict, int] | None = None
        self._deployments: dict[bool, ABDeployment] = {}
        self._controls: dict[str, dict] = {}

    # -- manifest/file verification -----------------------------------------
    def _default_data_loader(
        self, path: Path, *, seed: int, n_knn: int, n_vma: int = 200, n_robust: int = 0
    ) -> dict:
        from ..data.corpora import CLUSTERED_SPLIT, load_alpaca

        return load_alpaca(path, seed=seed, n_knn=n_knn, n_vma=n_vma, n_robust=n_robust,
                           split_strategy=CLUSTERED_SPLIT)

    def _default_deployment(
        self, artifact: Path, model_path: Path, root_key: bytes | None, needs_forward: bool
    ) -> ABDeployment:
        return ABDeployment(
            artifact=artifact,
            clean_model=model_path,
            dtype="float32",
            layers=tuple(self.profile.floors["vma_h"]["layers_1b"]),
            max_len=None,
            load_models="deployment" if needs_forward else False,
            strict_length=True,
            expected_variant="base",
            root_key=root_key,
        )

    def _model_info(self, model_id: str) -> dict:
        models = self.manifest.get("models") if isinstance(self.manifest.get("models"), dict) else {}
        info = models.get(model_id)
        if not isinstance(info, dict) or not info.get("path"):
            raise ExecutorRefusal(f"{model_id} has no verified path in the manifest")
        return info

    def _artifact_info(self) -> dict:
        artifacts = self.manifest.get("artifacts") if isinstance(self.manifest.get("artifacts"), dict) else {}
        info = artifacts.get("ab")
        if not isinstance(info, dict) or not info.get("path"):
            raise ExecutorRefusal("ab artifact has no verified path in the manifest")
        files = info.get("files_sha256")
        if not isinstance(files, dict) or not files:
            raise ExecutorRefusal("ab artifact has no file content hashes in the manifest")
        return info

    def _verify_model(self, model_id: str, info: dict, shortages: list[str]) -> dict:
        path = Path(str(info["path"]))
        if not path.is_dir():
            shortages.append(f"{model_id} path missing on disk: {path}")
            return {}
        try:
            actual = verify_model_identity(path, expected_variant=info.get("variant"))
        except (FileNotFoundError, ValueError, OSError) as exc:
            shortages.append(f"{model_id} identity unverifiable: {exc}")
            return {}
        if info.get("revision") != local_model_revision(actual):
            shortages.append(f"{model_id} revision is not the verified local content identity")
        for field in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
            declared = info.get(field)
            if declared != actual.get(field):
                shortages.append(
                    f"{model_id} {field} drift: manifest {declared!r} != file {actual.get(field)!r}"
                )
        if actual.get("variant_evidence", {}).get("looks_instruct") and info.get("variant") == "base":
            shortages.append(f"{model_id} carries Instruct evidence but the manifest says base")
        return actual

    def _verify_artifact(self, info: dict, model_path: str, shortages: list[str], *, require_encoder: bool = False) -> dict:
        root = Path(str(info["path"]))
        if not root.is_dir():
            shortages.append(f"ab artifact path missing on disk: {root}")
            return {}
        declared = dict(info.get("files_sha256") or {})
        # Completeness: every real file must be declared, and the declared set must
        # contain the deployment, the client secret and the encoder. A manifest that
        # lists only one file cannot silently leave the rest unverified.
        on_disk = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        required = {
            "manifest.json",
            "model_server/config.json",
            "model_server/tokenizer.json",
            "model_server/tokenizer_config.json",
            "client/client_secret.pt",
            "osnip/encryptor.pt",
            "osnip/config.json",
        }
        index_path = root / "model_server" / "model.safetensors.index.json"
        if index_path.is_file():
            required.add("model_server/model.safetensors.index.json")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for shard in sorted(set((index.get("weight_map") or {}).values())):
                required.add(f"model_server/{shard}")
        else:
            required.add("model_server/model.safetensors")
        missing_required = sorted(required - set(declared))
        if missing_required:
            shortages.append(
                "ab artifact manifest does not cover required files: "
                + ", ".join(missing_required[:6])
            )
        undeclared = sorted(set(on_disk) - set(declared))
        if undeclared:
            shortages.append(
                "ab artifact has files with no declared hash: " + ", ".join(undeclared[:6])
            )
        actual: dict[str, str] = {}
        for relative, expected in sorted(declared.items()):
            path = root / relative
            if not path.is_file():
                shortages.append(f"ab artifact file missing: {relative}")
                continue
            digest = sha256_file(path)
            actual[relative] = digest
            if digest != expected:
                shortages.append(
                    f"ab artifact {relative} sha256 drift: manifest {expected} != file {digest}"
                )
        weights_sha = info.get("weights_sha256")
        combined = _combined_weight_sha(actual)
        if not weights_sha:
            shortages.append("ab artifact manifest missing weights_sha256")
        elif combined and weights_sha != combined:
            shortages.append(
                f"ab artifact weights_sha256 drift: manifest {weights_sha} != file {combined}"
            )
        lineage = info.get("lineage") if isinstance(info.get("lineage"), dict) else {}
        evidence = lineage.get(str(model_path)) if isinstance(lineage, dict) else None
        if not isinstance(evidence, dict):
            shortages.append(
                f"ab artifact has no lineage evidence for model {model_path}"
            )
        elif evidence.get("verdict") != "matches":
            shortages.append(
                f"ab artifact lineage verdict {evidence.get('verdict')!r} != 'matches' "
                f"for {model_path}"
            )
        try:
            randomness = verify_export_randomness(root)
        except Exception as exc:
            shortages.append(f"ab artifact export randomness verification failed: {type(exc).__name__}: {exc}")
            randomness = {"verified": False}
        decoder_lineage = {"verified": False}
        if randomness.get("verified") is True:
            try:
                decoder_lineage = verify_square_lineage(root, model_path)
            except Exception as exc:
                shortages.append(f"ab artifact decoder lineage verification failed: {type(exc).__name__}: {exc}")
        endpoint_lineage = {"verified": False}
        if decoder_lineage.get("verified") is True:
            try:
                endpoint_lineage = verify_square_endpoints(root, model_path)
            except Exception as exc:
                shortages.append(f"ab artifact endpoint lineage verification failed: {type(exc).__name__}: {exc}")
        if require_encoder:
            try:
                verify_encoder_binding(root)
            except Exception as exc:
                shortages.append(f"ab artifact encoder binding verification failed: {type(exc).__name__}: {exc}")
        return {"path": str(root), "files_sha256": actual,
                "weights_sha256": combined or weights_sha, "lineage": evidence,
                "export_randomness": randomness, "decoder_lineage": decoder_lineage,
                "endpoint_lineage": endpoint_lineage}

    def _verify_data(self, shortages: list[str]) -> dict:
        from ..data.corpora import CLUSTERED_SPLIT, audit_corpus_isolation

        data = self.manifest.get("data") if isinstance(self.manifest.get("data"), dict) else {}
        path_value = data.get("path")
        if not path_value:
            shortages.append("data path missing in the manifest")
            return {}
        path = Path(str(path_value))
        if not path.is_file():
            shortages.append(f"data file missing on disk: {path}")
            return {}
        digest = sha256_file(path)
        if data.get("source_sha256") and data["source_sha256"] != digest:
            shortages.append(
                f"data source_sha256 drift: manifest {data['source_sha256']} != file {digest}"
            )
        seed = data.get("split_seed")
        main_floor = int(self.profile.floors["knn"]["main_texts"])
        if int(data.get("knn_texts") or 0) < main_floor:
            shortages.append(
                f"data knn_texts got {data.get('knn_texts')} floor {main_floor}"
            )
        robust_n = int(data.get("robust_texts") or 0)
        try:
            corpus = self.data_loader(
                path, seed=int(seed), n_knn=int(data.get("knn_texts") or 0),
                n_robust=robust_n,
            )
        except Exception as exc:
            shortages.append(f"data split failed: {exc}")
            return {}
        if data.get("split_strategy") != CLUSTERED_SPLIT:
            shortages.append("data split_strategy lacks the required clustered split")
        try:
            isolation = audit_corpus_isolation(corpus)
        except Exception as exc:
            shortages.append(f"data isolation verification failed: {exc}")
            return {}
        if isolation["cross_role_clusters"]:
            shortages.append("data near-duplicate clusters cross role boundaries")
        if isolation["within_role_duplicate_clusters"]:
            shortages.append("data near-duplicate documents occur within a role")
        if data.get("split_sha256") != isolation["split_sha256"]:
            shortages.append("data split_sha256 drift or missing")
        for role, count in isolation["role_counts"].items():
            if isolation["role_distinct_hashes"][role] != count:
                shortages.append(f"data {role} contains repeated documents")
        knn = corpus["knn"]
        texts = knn.texts()
        if len(texts) != int(data.get("knn_texts") or 0):
            shortages.append(
                f"data knn_texts drift: split {len(texts)} != manifest {data.get('knn_texts')}"
            )
        for text in texts:
            if not (int(data.get("char_min", 20)) <= len(text) <= int(data.get("char_max", 512))):
                shortages.append("data text outside the frozen character window")
                break
        digest_texts = texts_sha256(texts)
        if data.get("knn_sha256") and data["knn_sha256"] != digest_texts:
            shortages.append(
                f"data knn_sha256 drift: manifest {data['knn_sha256']} != split {digest_texts}"
            )
        result = {
            "path": str(path),
            "source_sha256": digest,
            "knn_sha256": digest_texts,
            "n_texts": len(texts),
            "main_knn_texts": len(texts),
            "split_seed": seed,
            "char_min": data.get("char_min"),
            "char_max": data.get("char_max"),
            "manifest": knn.manifest,
            "split_strategy": CLUSTERED_SPLIT,
            "split_sha256": isolation["split_sha256"],
            "isolation": isolation,
        }
        if robust_n:
            expected_keys = int(self.profile.floors["knn"]["robust_keys"])
            if int(data.get("robust_keys") or 0) != expected_keys:
                shortages.append(
                    f"data robust_keys got {data.get('robust_keys')} floor {expected_keys}"
                )
            robust = corpus["robust"].texts()
            if len(robust) != robust_n:
                shortages.append(
                    f"data robust_texts drift: split {len(robust)} != manifest {robust_n}"
                )
            for text in robust:
                if not (int(data.get("robust_char_min") or 20)
                        <= len(text) <= int(data.get("robust_char_max") or 512)):
                    shortages.append("robust text outside the frozen character window")
                    break
            robust_digest = texts_sha256(robust)
            if data.get("robust_sha256") and data["robust_sha256"] != robust_digest:
                shortages.append(
                    f"data robust_sha256 drift: manifest {data['robust_sha256']} != split {robust_digest}"
                )
            result.update({
                "robust_texts": len(robust),
                "robust_sha256": robust_digest,
                "robust_char_min": data.get("robust_char_min"),
                "robust_char_max": data.get("robust_char_max"),
            })
        self._corpus = corpus
        return result

    def verify(self, job: Mapping) -> dict:
        """Run this cell's preflight and return its role-specific identity.

        File contents are rechecked for each job, even when model and corpus
        objects are reused in memory.
        """
        shortages: list[str] = []
        if str(job.get("arm")) in {"ab", "aloepri"} and self.root_key is None:
            shortages.append(
                "root key: ab/aloepri execution requires --root-key-file; a deployment "
                "that generates its own random key produces results nobody can audit"
            )
        protocol = self.manifest.get("protocol_sha256")
        if not protocol:
            shortages.append("manifest missing protocol_sha256")
        elif protocol != getattr(self.profile, "protocol_sha256", None):
            shortages.append(
                f"protocol_sha256 drift: manifest {protocol} != profile {self.profile.protocol_sha256}"
            )
        if protocol != protocol_sha256():
            shortages.append("protocol_sha256 drift: protocol file changed after profile loading")
        recorded_code = self.code_identity.get("files_sha256")
        if recorded_code and _git_identity().get("files_sha256") != recorded_code:
            shortages.append("code drift: source files changed after executor initialization")
        try:
            preflight_formal_start(
                self.profile,
                self.manifest,
                {
                    "attacks": [job["attack"]],
                    "arms": [job["arm"]],
                    "scenarios": [job["scenario"]],
                },
                models={job["model"]},
            )
        except PreflightError as exc:
            shortages.extend(exc.shortages)
        if shortages:
            raise ExecutorRefusal(shortages)

        model_id = str(job["model"])
        model_info = self._model_info(model_id)
        artifact_info = self._artifact_info()
        model_actual = self._verify_model(model_id, model_info, shortages)
        artifact_actual = self._verify_artifact(
            artifact_info, str(model_info.get("path")), shortages, require_encoder=str(job.get("arm")) == "ab"
        )
        data_actual = self._verify_data(shortages)
        if self._base_identity is not None:
            previous_model = self._base_identity["model"]
            for field in ("path", "source", "revision", "variant"):
                if model_info.get(field) != previous_model.get(field):
                    shortages.append(f"model {field} changed after the executor loaded it")
            for field in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
                if model_actual.get(field) != previous_model.get(field):
                    shortages.append(f"model {field} changed after the executor loaded it")
            if artifact_actual != self._base_identity["artifact"]:
                shortages.append("artifact identity changed after the executor loaded it")
            if data_actual != self._data_actual:
                shortages.append("corpus identity changed after the executor loaded it")
            current_root = hashlib.sha256(self.root_key).hexdigest() if self.root_key else None
            if current_root != self._base_identity["key_schedule"]["root_commitment"]:
                shortages.append("key schedule changed after the executor loaded it")
        if shortages:
            raise ExecutorRefusal(shortages)
        if self._base_identity is None:
            self._texts = self._corpus["knn"].texts()
            self._robust_texts = (
                self._corpus["robust"].texts() if "robust" in self._corpus else []
            )
            self._data_actual = data_actual
            self._base_identity = {
                "profile": PAPER_PROFILE_NAME,
                "protocol_sha256": self.profile.protocol_sha256,
                "model": {
                    "id": model_id,
                    "path": model_info.get("path"),
                    "source": model_info.get("source"),
                    "revision": model_info.get("revision"),
                    "variant": model_info.get("variant"),
                    "config_sha256": model_actual.get("config_sha256", model_info.get("config_sha256")),
                    "tokenizer_sha256": model_actual.get("tokenizer_sha256", model_info.get("tokenizer_sha256")),
                    "weights_sha256": model_actual.get("weights_sha256", model_info.get("weights_sha256")),
                    "variant_evidence": model_actual.get("variant_evidence"),
                },
                "artifact": artifact_actual,
                "code": self.code_identity,
                "key_schedule": {
                    "kdf": "HKDF-SHA256",
                    "root_commitment": (
                        hashlib.sha256(self.root_key).hexdigest() if self.root_key else None
                    ),
                    "per_request": "fresh derived key per request index",
                    "root_key_available": self.root_key is not None,
                },
            }

        robust_job = str(job.get("attack")) == "knn_l2_key_robustness"
        role = "robust" if robust_job else "knn"
        if robust_job and not self._robust_texts:
            raise ExecutorRefusal("key-robustness cell needs the independent robust set")
        if role not in self._identity_by_role:
            data_actual = dict(self._data_actual or {})
            if robust_job:
                data_actual = {
                    **data_actual,
                    "role": "robust",
                    "n_texts": data_actual.get("robust_texts"),
                    "knn_sha256": data_actual.get("robust_sha256"),
                    "char_min": data_actual.get("robust_char_min"),
                    "char_max": data_actual.get("robust_char_max"),
                }
            else:
                data_actual = {**data_actual, "role": "knn",
                               "main_knn_texts": data_actual.get("n_texts")}
            self._identity_by_role[role] = {**self._base_identity, "data": data_actual}
        return self._identity_by_role[role]

    # -- execution -----------------------------------------------------------
    def _deployment(self, needs_forward: bool) -> ABDeployment:
        if needs_forward not in self._deployments:
            artifact = Path(str(self.manifest["artifacts"]["ab"]["path"]))
            model_path = Path(str(self.manifest["models"][PRIMARY_MODEL_ID]["path"]))
            self._deployments[needs_forward] = self.deployment_factory(
                artifact, model_path, self.root_key, needs_forward
            )
        return self._deployments[needs_forward]

    def _reference_counts(self, dep: ABDeployment) -> tuple[dict, int]:
        if self._reference is None:
            from ..data.corpora import reference_counts

            texts = self._corpus["reference"].texts() if self._corpus else []
            counts, tokens, _ = reference_counts(
                texts, dep.encode, vocab_size=dep.vocab_size
            )
            self._reference = (counts, tokens)
        return self._reference

    def _source_paths(self) -> list[Path]:
        artifact = Path(str(self.manifest["artifacts"]["ab"]["path"]))
        candidates = [
            artifact / "manifest.json",
            artifact / "client" / "client_secret.pt",
            artifact / "osnip" / "encryptor.pt",
            artifact / "osnip" / "config.json",
            artifact / "model_server" / "config.json",
        ]
        candidates.extend(sorted((ROOT / "src" / "obfuscation_attacks").rglob("*.py")))
        candidates.extend([ROOT / "scripts" / "run_full_battery.py", ROOT / "scripts" / "audit_paper_run.py"])
        return [path for path in candidates if path.is_file()]

    def _run_arm(self, job: Mapping, identity: dict, *, arm: str, metric: str,
                 control: dict | None, texts: list[str] | None = None,
                 keys_per_text: int = 1, attack_id: str | None = None,
                 workpoint: str | None = None) -> dict:
        needs_forward = arm == "aloepri"
        dep = self._deployment(needs_forward)
        ref_counts, _ = self._reference_counts(dep)
        data_path = Path(str(self.manifest["data"]["path"]))
        scenario = str(job["scenario"])
        seed = int(self.manifest["data"]["split_seed"])
        evidence_dir = self.evidence_root / str(job["job_id"]).replace("|", "__")
        if workpoint is None and arm == "aloepri":
            workpoint = "full_hidden_layer1"
        out = real_battery.run_c01(
            dep,
            list(texts if texts is not None else (self._texts or [])),
            ref_counts,
            verbose=False,
            profile=PAPER_PROFILE_NAME,
            protocol_sha256=self.profile.protocol_sha256,
            metric=metric,
            arms=(arm,),
            forward={arm: needs_forward},
            model_id=str(job["model"]),
            scenario=scenario,
            seed=seed,
            data_paths=[data_path],
            source_paths=self._source_paths(),
            control=control,
            identity=identity,
            oracle_diagnostics=True,
            emit_not_applicable=False,
            result_root=self.result_root,
            evidence_dir=evidence_dir,
            write_queries=True,
            workpoint=workpoint,
            keys_per_text=int(keys_per_text),
            key_base=0,
            attack_id=attack_id,
        )
        return out[arm]

    def _control(self, metric: str, identity: dict, *, robust: bool = False) -> dict:
        key = (metric, bool(robust))
        if key in self._controls:
            return self._controls[key]
        attack_name = "nn_knn.euclidean" if metric == "l2" else "nn_knn.cosine"
        control_attack_id = "knn_l2" if metric == "l2" else "knn_cosine"
        workpoint = "full_robust_1key" if robust else "full"
        texts = self._robust_texts if robust else self._texts
        role = "robust" if robust else "knn"
        from .run_store import load_completed, make_run_id

        data_sha = content_sha256([Path(self.manifest["data"]["path"])])
        run_id = make_run_id(
            profile=PAPER_PROFILE_NAME,
            model=f"{PRIMARY_MODEL_ID}/plaintext",
            arm="plaintext",
            scenario="S-public",
            attack=attack_name,
            workpoint=workpoint,
            seed=int(self.manifest["data"]["split_seed"]),
            data_sha256=data_sha,
            binding=real_battery.identity_binding(
                identity, self.profile.protocol_sha256
            ),
        )
        existing = load_completed(
            self.result_root / "C01" / attack_name / "plaintext", run_id
        )
        if existing:
            dep = self._deployment(False)
            expected_total = sum(len(dep.encode(text)) for text in (texts or []))
            problems: list[str] = []
            if existing.get("protocol_sha256") != self.profile.protocol_sha256:
                problems.append("protocol hash differs")
            if existing.get("data_sha256") != data_sha:
                problems.append("data hash differs")
            ex_model = (existing.get("identity") or {}).get("model") or {}
            for field in ("weights_sha256", "tokenizer_sha256", "config_sha256"):
                if ex_model.get(field) != identity.get("model", {}).get(field):
                    problems.append(f"model {field} differs")
            ex_role = ((existing.get("identity") or {}).get("data") or {}).get("role", "knn")
            if ex_role != role:
                problems.append(f"data role {ex_role!r} != {role!r}")
            rows = (existing.get("diagnostics") or {}).get("per_unit_predictions") or []
            total = sum(int(row.get("count", 0)) for row in rows)
            hits = sum(
                1 for row in rows for value in (row.get("rank") or []) if value <= 1
            )
            if not rows or total != expected_total:
                problems.append(
                    f"per-unit counts sum {total} != frozen corpus tokens {expected_total}"
                )
            if problems:
                raise ExecutorRefusal(
                    "stored positive control failed verification (new identity required): "
                    + "; ".join(problems)
                )
            top1 = (hits / total) if total else None
            control_path = existing.get("result_path") or str(
                self.result_root / "C01" / attack_name / "plaintext" / run_id / "result.json"
            )
            self._controls[key] = {
                "name": "clean_embedding_retrieval",
                "top1": top1,
                "required_top1": 0.99,
                "n_texts": existing.get("n_prompts"),
                "top_k": existing.get("top_k"),
                "metric": metric,
                "run_id": existing.get("run_id"),
                "result_path": control_path,
                "positive_control_passed": top1 is not None and top1 >= 0.99,
                "reused": True,
                "recomputed_top1": top1,
            }
            return self._controls[key]
        result = self._run_arm(
            {
                "job_id": f"control|{metric}|{role}",
                "attack": control_attack_id,
                "arm": "plaintext",
                "scenario": "S-public",
                "model": PRIMARY_MODEL_ID,
            },
            identity,
            arm="plaintext",
            metric=metric,
            control=None,
            texts=texts,
            keys_per_text=1,
            attack_id=control_attack_id,
            workpoint=workpoint,
        )
        self._controls[key] = {
            "name": "clean_embedding_retrieval",
            "top1": result.get("stage1_top1"),
            "required_top1": 0.99,
            "n_texts": result.get("n_texts"),
            "top_k": 100,
            "metric": metric,
            "run_id": result.get("run_id"),
            "result_path": result.get("result_path"),
            "positive_control_passed": result.get("positive_control_passed"),
        }
        return self._controls[key]

    def __call__(self, job: Mapping) -> dict:
        """Dispatcher entry: the CLI passes the executor itself into dispatch_configured."""
        return self.run(job)

    def run(self, job: Mapping) -> dict:
        """Execute one configured job; always returns an explicit outcome record."""
        job = dict(job)
        if "plan_spec_id" in job or "attack_axes" in job or "plan_identity" in job:
            return self._failed(job, "expanded matrix axes are not yet wired to this executor")
        attack = str(job.get("attack"))
        arm = str(job.get("arm"))
        scenario = str(job.get("scenario"))
        model_id = str(job.get("model"))
        if attack not in SUPPORTED_ATTACKS:
            return self._failed(job, f"attack {attack!r} has no paper executor entry")
        if arm not in SUPPORTED_ARMS:
            return self._failed(job, f"arm {arm!r} has no paper executor entry")
        if scenario not in SUPPORTED_SCENARIOS:
            return self._failed(job, f"scenario {scenario!r} is not configured for this executor")
        if model_id != PRIMARY_MODEL_ID:
            return self._failed(job, f"model {model_id!r} is not the verified primary model")
        try:
            identity = self.verify(job)
        except ExecutorRefusal as exc:
            return self._failed(job, "verification refused: " + "; ".join(exc.shortages))
        except Exception as exc:  # pragma: no cover - defensive
            return self._failed(job, f"verification error: {exc}", traceback.format_exc())
        metric = "l2" if attack in {"knn_l2", "knn_l2_key_robustness"} else "cosine"
        robust = attack == "knn_l2_key_robustness"
        if robust and arm != "ab":
            return self._failed(
                job, "key-robustness is only wired for the key-conditioned ab arm"
            )
        texts = self._robust_texts if robust else self._texts
        try:
            control = (
                self._control(metric, identity, robust=robust)
                if arm != "plaintext" else None
            )
            result = self._run_arm(
                job, identity, arm=arm, metric=metric, control=control,
                texts=list(texts or []),
                keys_per_text=5 if robust else 1,
                attack_id=attack if robust else None,
                workpoint="full_robust_5key" if robust else None,
            )
        except Exception as exc:
            return self._failed(job, f"execution error: {exc}", traceback.format_exc())
        passed = result.get("positive_control_passed")
        outcome = "executed" if passed is True else "control_failed"
        return {
            "outcome": outcome,
            "job_id": job.get("job_id"),
            "attack": attack,
            "arm": arm,
            "scenario": scenario,
            "model": model_id,
            "run_id": result.get("run_id"),
            "result_path": result.get("result_path"),
            "metrics": {
                "stage1_top1": result.get("stage1_top1"),
                "stage1_top5": result.get("stage1_top5"),
                "stage1_top10": result.get("stage1_top10"),
                "stage1_top100": result.get("stage1_top100"),
                "composed_top1_oracle": result.get("composed_top1_oracle"),
            },
            "positive_control": result.get("positive_control", control),
            "positive_control_passed": passed,
            "evidence": result.get("evidence"),
        }

    def _failed(self, job: Mapping, reason: str, detail: str | None = None) -> dict:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe = str(job.get("job_id", "job")).replace("|", "__").replace("/", "_")
        path = self.result_root / "failures" / f"{safe}.{stamp}.json"
        record = {
            "job": dict(job),
            "reason": reason,
            "traceback": detail,
            "timestamp": stamp,
        }
        # Failure records are public evidence too: fail closed before writing.
        assert_public_payload(record, label=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "outcome": "failed",
            "job_id": job.get("job_id"),
            "reason": reason,
            "error_path": str(path),
        }


PRIMARY_MODEL_ID = "llama-3.2-1b-base"


def _combined_weight_sha(files: Mapping[str, str]) -> str | None:
    """Hash of the model_server safetensors set, matching build_paper_manifest."""
    shards = sorted(
        (name, digest)
        for name, digest in files.items()
        if name.startswith("model_server/") and name.endswith(".safetensors")
    )
    if not shards:
        return None
    combined = hashlib.sha256()
    for name, digest in shards:
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()
