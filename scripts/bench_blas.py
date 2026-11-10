import time

import numpy as np


def t(label, fn, repeats=1):
    fn()  # warm-up
    t0 = time.time()
    for _ in range(repeats):
        fn()
    dt = (time.time() - t0) / repeats
    print(f"{label:52s} {dt:8.2f}s")
    return dt


rng = np.random.default_rng(0)
M = rng.normal(size=(2048, 2048)).astype(np.float64)
E = rng.normal(size=(128256, 2048)).astype(np.float32)
Q = rng.normal(size=(1000, 2048)).astype(np.float32)

t("inv(2048 float64)", lambda: np.linalg.inv(M))
M32 = np.linalg.inv(M).astype(np.float32)
t("[1000,2048] @ [2048,2048] float32", lambda: Q @ M32)
t("[1000,2048] @ [2048,128256] float32", lambda: Q @ E.T)
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
t("normalise 128256x2048 (copy+norm)", lambda: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12))
Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
t("[1000,2048] @ [2048,128256] normalised", lambda: Qn @ En.T)
print("threads:", np.__config__.show(mode="dicts").get("Build Dependencies", {}))
import os

print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
try:
    import threadpoolctl

    print(threadpoolctl.threadpool_info())
except Exception as exc:  # pragma: no cover
    print("threadpoolctl unavailable:", exc)
