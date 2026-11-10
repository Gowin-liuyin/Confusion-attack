"""Fidelity verifier - exits 0 only if the reproduction's faithfulness holds.

Checks (static + a few fast runtime invariants):

1. All 7 families register, with the expected (name, variant) separation of the
   original / AloePri / OSNIP lineages.
2. The AloePri Qwen2 inverter is fixed at 2 decoder layers / 8 attention heads.
3. configs/original, configs/aloepri, configs/osnip exist, are non-empty, and
   every *full* config declares no silent approximation (sampling, truncation,
   quantization, low precision, pruning, early stopping, candidate truncation,
   approximate nearest neighbour).
4. Every formula row in docs/FORMULA_LEDGER.md points at code and test files
   that exist (paper -> code -> test traceability).
5. Runtime invariants that would break if an algorithm were swapped for a
   non-faithful one (L1->L2, RowSort->NN, glide difference invariant, gate
   average invariant, sorted-L1 permutation invariance).
6. SDA encodings at runtime: 2021 frequency-rank is injective per symbol and
   differs from 2023 recurrence encoding (first-appearance order).
7. Full SDA configs carry paper scale (beam width, training-cipher count).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


EXPECTED = {
    "nn_knn.euclidean": "original",
    "nn_knn.cosine": "original",
    "vma.autoregressive": "original",
    "vma.weight_rowsort": "aloepri",
    "vma.osnip_projection": "osnip",
    "ia.ednn": "original",
    "ia.gate": "aloepri",
    "ia.attn": "aloepri",
    "isa.er": "original",
    "isa.tbs": "original",
    "isa.aloepri_state": "aloepri",
    "ima.classify": "original",
    "ima.seq2seq": "original",
    "ima.qwen2_aloepri": "aloepri",
    "tfma.independent": "aloepri",
    "tfma.global_assignment": "experimental_completion",
    "sda.sda2021": "frequency_rank_lookup_variant",
    "sda.sda2023": "recurrence_lookup_variant",
}


def check_registry() -> None:
    from obfuscation_attacks.attacks import load_all, REGISTRY

    load_all()
    for name, variant in EXPECTED.items():
        check(name in REGISTRY, f"attack not registered: {name}")
        if name in REGISTRY:
            got = REGISTRY[name].variant
            check(got == variant, f"{name} variant={got!r}, expected {variant!r}")
    families = {cls.family for cls in REGISTRY.values()}
    for fam in ("nn_knn", "vma", "ia", "isa", "ima", "tfma", "sda"):
        check(fam in families, f"family missing from registry: {fam}")


def check_qwen2() -> None:
    try:
        from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter

        model = Qwen2AloePriInverter().build_model(obf_dim=6, vocab_size=16)
        check(model.config.num_hidden_layers == 2, "Qwen2 inverter must have 2 decoder layers")
        check(model.config.num_attention_heads == 8, "Qwen2 inverter must have 8 attention heads")
    except Exception as exc:  # pragma: no cover
        FAILURES.append(f"Qwen2 inverter construction failed: {exc}")


def _ledger_ids() -> set[str]:
    ledger = ROOT / "docs" / "FORMULA_LEDGER.md"
    if not ledger.is_file():
        return set()
    ids: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("ID", "---") or set(cells[0]) <= {"-", ":"}:
            continue
        ids.add(cells[0])
    return ids


def check_configs() -> None:
    ledger_ids = _ledger_ids()
    for sub in ("original", "aloepri", "osnip"):
        d = ROOT / "configs" / sub
        check(d.is_dir(), f"missing config dir: configs/{sub}")
        files = list(d.glob("*.yaml")) if d.is_dir() else []
        check(len(files) > 0, f"configs/{sub} is empty")
        for f in files:
            cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
            if cfg.get("mode") == "full":
                approx = (cfg.get("fidelity") or {}).get("approximations")
                check(
                    isinstance(approx, dict) and bool(approx),
                    f"{f.relative_to(ROOT)} mode: full must declare "
                    "fidelity.approximations (absence is not 'no approximation')",
                )
                for key, val in (approx or {}).items():
                    check(
                        val is False,
                        f"{f.relative_to(ROOT)} declares approximation {key}={val} (must be false)",
                    )
                missing = [
                    ref for ref in (cfg.get("formula_refs") or [])
                    if ref not in ledger_ids
                ]
                check(
                    not missing,
                    f"{f.relative_to(ROOT)} formula_refs not in FORMULA_LEDGER: {missing}",
                )


def check_formula_ledger() -> None:
    ledger = ROOT / "docs" / "FORMULA_LEDGER.md"
    check(ledger.is_file(), "docs/FORMULA_LEDGER.md missing")
    if not ledger.is_file():
        return
    text = ledger.read_text(encoding="utf-8")
    # table rows look like: | ID | formula | paper | section | variant | code | test |
    path_re = re.compile(r"`([^`]+\.py)[^`]*`")
    n_rows = 0
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("ID", "---") or set(cells[0]) <= {"-", ":"}:
            continue
        n_rows += 1
        for col in (cells[5], cells[6]):  # code, test columns
            for m in path_re.finditer(col):
                rel = m.group(1)
                check((ROOT / rel).is_file(), f"FORMULA_LEDGER path does not exist: {rel}")
    check(n_rows >= 15, f"FORMULA_LEDGER has too few rows ({n_rows}); expected the full formula set")


def check_runtime_invariants() -> None:
    # glide coordinate-difference invariant (EDNN) -- must NOT be L2/NN
    from obfuscation_attacks.attacks.ia.ednn import difference_representation

    rng = np.random.default_rng(0)
    e = rng.standard_normal(8)
    ones = np.ones(8)
    a, b = 1.7, -0.4
    ep = e - 2.0 * (e @ (ones * a)) / ((ones * a) @ (ones * a)) * (ones * a) + ones * b
    check(
        np.allclose(difference_representation(e), difference_representation(ep), atol=1e-9),
        "EDNN difference invariant broken",
    )

    # VMA weight RowSort exact recovery (must be RowSort+match, not a decoder)
    from obfuscation_attacks.attacks.vma.weight_rowsort import recover_permutations

    X = rng.standard_normal((8, 5))
    z1, z2 = rng.permutation(8), rng.permutation(5)
    Y = X[np.ix_(z1, z2)]
    res = recover_permutations(X, Y)
    check(res.get("status") == "ok" and np.array_equal(res["Z1"], z1), "VMA RowSort recovery broken")

    # Gate average invariant under column permutation
    from obfuscation_attacks.attacks.ia.gate import gate_invariant

    E = rng.standard_normal((5, 4))
    W = rng.standard_normal((4, 7))
    zffn = rng.permutation(7)
    check(np.allclose(gate_invariant(E, W), gate_invariant(E, W[:, zffn]), atol=1e-9),
          "Gate-IA average invariant broken")

    # sorted-L1 permutation invariance (VMA hidden-dim permutation)
    from obfuscation_attacks.attacks.vma.autoregressive import _sorted_l1

    v = rng.standard_normal(10)
    check(_sorted_l1(v, v[rng.permutation(10)]) < 1e-9, "sorted-L1 permutation invariance broken")


def check_sda_encodings() -> None:
    """Pin the SDA encodings at runtime (the CRITICAL 2021 fix + 2021 vs 2023).

    * 2021 frequency-rank must be INJECTIVE per symbol (distinct cipher symbols ->
      distinct codes); the old competition-ranking collapsed equal-count symbols
      to one code, destroying the substitution structure.
    * recurrence encoding (first-appearance order) must DIFFER from frequency-rank
      encoding when frequency order != first-appearance order, and must be the
      1-indexed first-appearance sequence (ABACAD -> 1 2 1 3 1 4).
    """
    from obfuscation_attacks.attacks.sda import encodings as enc

    # counts: 0->3, 1->2, 2->2 (a tie between 1 and 2)
    seq = [0, 1, 2, 0, 1, 2, 0]
    codes = list(enc.frequency_rank_encode(seq))
    sym2code: dict[int, int] = {}
    injective = True
    for s, c in zip(seq, codes):
        if s in sym2code and sym2code[s] != c:
            injective = False
        sym2code[s] = c
    check(injective, "SDA-2021 frequency-rank code is not a function of the symbol")
    check(
        len(set(sym2code.values())) == len(set(seq)),
        "SDA-2021 frequency-rank encoding is not injective per symbol "
        f"({len(set(sym2code.values()))} codes for {len(set(seq))} symbols) - "
        "equal-count symbols must still get distinct codes (paper Fig. 5)",
    )

    # recurrence vs frequency-rank: frequency order (3,5,1) != first-appearance (5,3,1)
    s2 = [5, 5, 3, 3, 3, 1]
    rec2 = list(enc.recurrence_encode(s2))
    fr2 = list(enc.frequency_rank_encode(s2))
    check(rec2 != fr2, "recurrence encoding must differ from frequency-rank encoding")

    abacad = [0, 1, 0, 2, 0, 3]  # A B A C A D
    check(
        list(enc.recurrence_encode(abacad)) == [1, 2, 1, 3, 1, 4],
        "recurrence encoding ABACAD must be the 1-indexed first-appearance sequence 1 2 1 3 1 4",
    )


def check_sda_config_scale() -> None:
    """Full SDA configs must carry paper scale, not shrunk smoke values."""
    def load(rel: str) -> dict:
        return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))

    c21 = load("configs/original/sda2021.yaml").get("attack", {}).get("params", {})
    check(int(c21.get("beam_width", 0)) >= 100, "SDA-2021 full config beam_width < 100 (paper uses 100)")
    check(int(c21.get("num_train", 0)) >= 2_000_000, "SDA-2021 full config num_train < 2e6 (paper)")
    c23 = load("configs/original/sda2023.yaml").get("attack", {}).get("params", {})
    check(int(c23.get("beam_width", 0)) >= 200, "SDA-2023 full config beam_width < 200 (paper uses 200)")
    check(int(c23.get("num_train", 0)) >= 1_200_000, "SDA-2023 full config num_train < 1.2e6 (paper)")
    # Verify the executable representation and options, not just the YAML budgets.
    from obfuscation_attacks.attacks.sda.sda2021_seq2seq import Seq2SeqConfig, Seq2SeqCipherModel
    from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import CausalCipherLM, CausalLMConfig
    cfg21 = Seq2SeqConfig(vocab_size=8, src_vocab_size=9,
                         d_model=int(c21.get("d_model", 32)), nhead=int(c21.get("nhead", 4)),
                         num_layers=int(c21.get("num_layers", 2)), dim_ff=int(c21.get("dim_ff", 64)),
                         dropout=float(c21.get("dropout", 0.0)))
    model21 = Seq2SeqCipherModel(cfg21)
    check(len(model21.encoder.layers) == 6 and len(model21.decoder.layers) == 6,
          f"SDA-2021 executable encoder/decoder layers={len(model21.encoder.layers)}/{len(model21.decoder.layers)}, paper configuration specifies 6/6")
    check(not ({"num_tune", "num_test"} & set(c21)),
          "SDA-2021 original config contains num_tune/num_test not consumed by the trainer")
    cfg23 = CausalLMConfig(vocab_size=8, max_rec=8,
                           d_model=int(c23.get("d_model", 48)), dim_ff=int(c23.get("dim_ff", 96)),
                           num_layers=int(c23.get("num_layers", 2)),
                           nhead=int(c23.get("nhead", 4)), dropout=float(c23.get("dropout", 0.0)),
                           representation=c23.get("representation", "lookup"))
    model = CausalCipherLM(cfg23)
    actual_layers = len(model.blocks.layers)
    actual_heads = model.blocks.layers[0].self_attn.num_heads if actual_layers else 0
    check(actual_layers == 12 and actual_heads == 12,
          f"SDA-2023 executable model is {actual_layers} layers/{actual_heads} heads, source config specifies 12/12")
    actual_ff = model.blocks.layers[0].linear1.out_features if actual_layers else 0
    check(actual_ff == 1536,
          f"SDA-2023 executable feed-forward dimension={actual_ff}, source config specifies 1536")
    actual_dropout = model.blocks.layers[0].dropout.p if actual_layers else None
    actual_attention_dropout = model.blocks.layers[0].self_attn.dropout if actual_layers else None
    check(actual_dropout == 0.1 and actual_attention_dropout == 0.1,
          f"SDA-2023 executable dropout={actual_dropout}, attention dropout={actual_attention_dropout}; source config specifies 0.1/0.1")
    check(model.recurrence_encoder is not None,
          "SDA-2023 uses recurrence lookup embeddings; configured character CNN/highway input is absent")
    check(cfg23.representation != "decimal_char_cnn",
          "SDA-2023 decimal character CNN is an unverified engineering variant, not author-code fidelity evidence")


def main() -> int:
    check_registry()
    check_qwen2()
    check_configs()
    check_formula_ledger()
    check_runtime_invariants()
    check_sda_encodings()
    check_sda_config_scale()

    if FAILURES:
        print("FIDELITY CHECK FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("FIDELITY CHECK PASSED: 7 families, variants separated, formulas traceable, "
          "invariants hold (incl. injective SDA frequency-rank vs recurrence), full configs "
          "declare no approximations and carry paper scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
