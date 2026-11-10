#!/usr/bin/env python3
"""Real classification-backend probe for the universal arm adapter.

Runs a public DistilBERT SST-2 classifier through the classification adapter:
server accuracy, the exact-solvable embedding control, and the EDNN/KNN attack
on the server-visible hidden state. This validates the classification surface of
the adapter layer with a real transformer; it is explicitly a probe, not a
paper-v1 defense result (no external defense is reproduced here).

Usage::

    python scripts/run_classification_probe.py --model F:/Models/distilbert-sst2 \
        --out reports/privacy_rebuild_20261107/round4/classification_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.classification import HFClassificationBackend  # noqa: E402
from obfuscation_attacks.evaluation.generic_runner import (  # noqa: E402
    run_classification,
    run_vector_knn,
)
from obfuscation_attacks.schemes.ab_deployment import verify_model_identity  # noqa: E402
from obfuscation_attacks.schemes.adapters import ClassificationReferenceAdapter  # noqa: E402

#: small hand-written SST-2-style fixture (1 positive, 0 negative); probe only
EXAMPLES = [
    ("a stirring and beautifully acted drama", 1),
    ("the movie is a joy from start to finish", 1),
    ("warm, funny and genuinely moving", 1),
    ("a smart script with terrific performances", 1),
    ("one of the best films of the year", 1),
    ("i loved every minute of it", 1),
    ("a delightful and charming comedy", 1),
    ("the cast is excellent and the story is engaging", 1),
    ("an intelligent, emotionally satisfying film", 1),
    ("it works beautifully as both comedy and drama", 1),
    ("a dull, lifeless mess", 0),
    ("the plot makes no sense and the acting is wooden", 0),
    ("a boring and predictable thriller", 0),
    ("i hated this movie", 0),
    ("the film is a complete waste of time", 0),
    ("clumsy direction and a terrible script", 0),
    ("an incoherent and deeply unfunny comedy", 0),
    ("the performances are flat and the story drags", 0),
    ("a joyless, poorly made sequel", 0),
    ("it fails on almost every level", 0),
    ("a surprisingly tender and well-observed film", 1),
    ("the ending is clumsy and the pacing is awful", 0),
    ("the director brings real warmth to the story", 1),
    ("a tedious and forgettable experience", 0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "Models" / "distilbert-sst2")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "privacy_rebuild_20261107" / "round4"
                        / "classification_probe.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.float32
    )
    model.eval()
    load_seconds = round(time.time() - t0, 1)
    backend = HFClassificationBackend(model, tokenizer)
    adapter = ClassificationReferenceAdapter(backend)

    texts = [text for text, _ in EXAMPLES]
    examples = [{"text": text, "label": label} for text, label in EXAMPLES]
    t0 = time.time()
    utility = run_classification(adapter, examples)
    utility_seconds = round(time.time() - t0, 2)

    control = run_vector_knn(adapter, texts[:8], use_hidden=False, top_k=100, metric="l2")
    hidden = run_vector_knn(adapter, texts[:8], use_hidden=True, layer=1, top_k=100, metric="l2")
    identity = verify_model_identity(args.model)
    result = {
        "kind": "probe_only_not_a_paper_result",
        "note": (
            "Public DistilBERT SST-2 classifier used to validate the classification "
            "adapter path (utility + exact embedding control + hidden-state EDNN). "
            "No external defense is reproduced; not a paper-v1 comparison."
        ),
        "model": {
            "path": str(args.model),
            "config_sha256": identity["config_sha256"],
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "weights_sha256": identity["weights_sha256"],
            "variant_evidence": identity["variant_evidence"],
        },
        "fixture": {
            "n": len(EXAMPLES),
            "sha256": hashlib.sha256(
                json.dumps(EXAMPLES, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "source": "hand-written SST-2-style probe fixture",
        },
        "load_seconds": load_seconds,
        "utility": utility,
        "utility_seconds": utility_seconds,
        "ednn": {
            "embedding_control": {
                "top1": control["metrics"]["stage1_permuted_id"]["top1"]["value"],
                "n_tokens": control["n_tokens"],
            },
            "hidden_layer1": {
                "top1": hidden["metrics"]["stage1_permuted_id"]["top1"]["value"],
                "top5": hidden["metrics"]["stage1_permuted_id"]["top5"]["value"],
                "top10": hidden["metrics"]["stage1_permuted_id"]["top10"]["value"],
                "top100": hidden["metrics"]["stage1_permuted_id"]["top100"]["value"],
                "n_tokens": hidden["n_tokens"],
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
