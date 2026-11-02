"""Run I/O helpers: JSON/YAML persistence, run directories, environment snapshot."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays for metrics output."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return super().default(obj)


def save_json(path: str | Path, data: Any) -> Path:  # noqa: ANN401
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder, sort_keys=True)
    return p


def load_json(path: str | Path) -> Any:  # noqa: ANN401
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_yaml(path: str | Path, data: Any) -> Path:  # noqa: ANN401
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=True)
    return p


def load_yaml(path: str | Path) -> Any:  # noqa: ANN401
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def environment_snapshot() -> dict:
    """Capture interpreter / platform / library versions for ``environment.json``."""
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for mod in ("numpy", "scipy", "yaml", "torch", "transformers", "nltk"):
        try:
            imported = __import__(mod)
            snapshot[mod] = getattr(imported, "__version__", "unknown")
        except Exception:
            snapshot[mod] = None
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        snapshot["cuda_available"] = False
    return snapshot


@dataclass
class RunPaths:
    """Standard artifact layout for one experiment run (see docs/ section 11)."""

    root: Path
    name: str
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        ensure_dir(self.dir)

    @property
    def dir(self) -> Path:
        return self.root / self.name

    @property
    def resolved_config(self) -> Path:
        return self.dir / "config.resolved.yaml"

    @property
    def environment(self) -> Path:
        return self.dir / "environment.json"

    @property
    def metrics(self) -> Path:
        return self.dir / "metrics.json"

    @property
    def predictions(self) -> Path:
        return self.dir / "predictions.json"

    @property
    def run_log(self) -> Path:
        return self.dir / "run.log"

    @property
    def fidelity(self) -> Path:
        return self.dir / "fidelity.json"
