"""Probe VMA-H (context-aware) — plaintext control first, then the A+B arm."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(min(24, os.cpu_count() or 8)))

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.real_vma import score_vma, vma_h_recovered  # noqa: E402
from obfuscation_attacks.schemes.ab_deployment import ABDeployment  # noqa: E402

PROMPTS = [
    "The capital of France is Paris and the river that runs through it is the Seine.",
    "Machine learning models learn patterns from data and generalize to unseen examples.",
]


def main() -> None:
    dep = ABDeployment(layers=(1, 5, 10, 15), max_len=32)
    n_propose = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for arm in ("plaintext", "ab"):
        wires = [
            dep.observe(text, arm=arm, request_id=i, key_index=i)
            for i, text in enumerate(PROMPTS)
        ]
        public = dep.public_view(arm, "S-public")
        out = vma_h_recovered(
            public, [wire.public for wire in wires], layers=(5,), full_vocab=False,
            n_propose=n_propose, max_positions=6, verbose=True,
        )
        truth = [
            (wire.truth.plaintext_ids if arm == "plaintext" else wire.truth.permuted_ids)[:6]
            for wire in wires
        ]
        scored = score_vma(out, truth, layers=(5,))
        for layer, metrics in scored.items():
            print(
                f"{arm:9s} {layer}: total_top1={metrics['total_top1']['value']:.3f} "
                f"coverage={metrics['total_top1']['denominator']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
