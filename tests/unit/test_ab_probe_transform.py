import importlib.util
from pathlib import Path

import numpy as np


def test_column_fit_handles_permutations_and_negative_scales():
    path = Path(__file__).resolve().parents[2] / "scripts" / "ab_probe_transform.py"
    spec = importlib.util.spec_from_file_location("ab_probe_transform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(21)
    clean = rng.normal(size=(100, 4))
    deployed = clean[:, [2, 0, 3, 1]] * np.array([-2.0, 1.5, -0.5, 3.0])
    _, mapping, residual = module.column_fit(clean, deployed)
    assert mapping.tolist() == [1, 3, 0, 2]
    assert residual.max() < 1e-10
    damaged = deployed.copy()
    damaged[:, 0] += rng.normal(size=100)
    assert module.column_fit(clean, damaged)[2].mean() > 0.01
