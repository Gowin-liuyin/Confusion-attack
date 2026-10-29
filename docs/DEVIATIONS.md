# Deviations and Completions

Choices the papers did not fully specify (`UNSPECIFIED`) and engineering completions that were
necessary to make a runnable harness (`EXPERIMENTAL_COMPLETION`). None of these change an attack's
mathematics; none are presented as paper parameters.

## EXPERIMENTAL_COMPLETION

- **SDA version mixing (AloePri).** AloePri *cites* Aldarrab & May 2021 (frequency-rank +
  encoder-decoder) but its appendix *describes* recurrence encoding + a causal Transformer LM, which
  matches Kambhatla et al. 2023. `configs/aloepri/sda.yaml` runs the 2023 recurrence+CLM method to
  match the appendix algorithm, and records the citation/appendix mismatch. Both lineages are
  implemented separately (`sda.sda2021`, `sda.sda2023`).
- **TFMA global assignment.** The paper matches each ciphertext symbol to its nearest-frequency
  plaintext token (per-symbol, `tfma.independent`). The one-to-one Hungarian assignment
  (`tfma.global_assignment`, `variant=experimental_completion`) is an added extension, named and
  reported separately, never substituted for the paper method.
- **Synthetic smoke scenarios.** `src/obfuscation_attacks/data/synthetic.py` builds small, seeded
  scenarios that contain the genuine leakage each attack exploits. They exist to exercise and verify
  the algorithms end-to-end on CPU; they are explicitly not paper reproductions.
- **Toy models for verification.** `CausalToyModel` (numpy, causal) and `TorchToyModel` (differentiable)
  stand in for real LLMs in math/sanity tests. Real-model runs are specified in the `mode: full` configs.
- **ISA L_dm default off.** ER's distribution-matching penalty `L_dm` is fully implemented (fixed
  random Gaussian feature network + batch mean/variance). The default `lambda_dm = 0` follows the
  paper's default (DM is its optional "improved setting", lambda in {0, 1e-3}); set `lambda_dm > 0`
  to enable it (its correctness is covered by the ISA math tests).
- **ISA TBS arctan scale.** We use plain `arctan` (alpha = 1) so the stated `(-pi/2, pi/2)` bound holds
  and is testable. The paper multiplies by `alpha = 5/pi` only to accelerate convergence.
- **SDA encodings 1-indexed.** Both recurrence and frequency-rank codes start at 1; relative structure
  is unchanged (the 2021 paper illustrates 0-indexed ranks).
- **SDA frequency-rank tie rule.** Competition ranking ("1224"): equal-count symbols share a rank and
  the next rank is skipped; ties are reported via `frequency_rank_info`, never broken by an invented
  order.
- **IMA WB vocabulary probing.** In the white-box setting the inverter additionally probes one
  obfuscated embedding per vocabulary id to guarantee full-class coverage; this assumes token-local
  obfuscation (exact for permutation/linear schemes). Gated by `wb_probe_vocab`.
- **IMA Seq2Seq tokenizer.** Inputs are integer token ids, so InvBERT's learned BB tokenizer step is
  unnecessary; a known plaintext-id vocabulary is used. BOS is a dedicated embedding row, never emitted.
- **SDA / smoke architectures.** Tiny Transformers (2 layers, small d_model) are used for the smoke/
  regression runs; the char-CNN+highway front end of the SDA papers is not reproduced because inputs
  are integer ids, not characters. Paper-scale sizes live in the full configs.

## UNSPECIFIED (not given by the paper; defaulted and recorded)

- NN/EDNN/OSNIP default `top_k = 10`; OSNIP VMA L1 by default, sorted-L1 optional.
- ISA default `lr = 0.1`, `steps = 300`, TBS `r = min(d, 8)` for the toy setting; full configs carry
  the paper-scale `lr`/`steps`.
- IMA Classify `hidden_dims=(128,)`, `epochs=150`, `lr=1e-2`, AdamW; Seq2Seq `d_model=64, nhead=4,
  layers=2`; Qwen2 `hidden_size=64, intermediate_size=128`. These are harness defaults for the toy
  setting, overridable via params.
- BB query budget for IMA Classify defaults to `min(len(corpus), 64)`.

## Not implemented (documented, not faked)

- **NN-3** (Song & Raghunathan white-box continuous-relaxation optimisation) is documented as the
  original paper's extended objective but not implemented; only the nearest-neighbour retrieval is.
  It is marked "not implemented" in the formula ledger rather than stubbed.

## Environment deviation

- **OpenMP duplicate runtime.** The conda MKL + pip torch combination raises
  `OMP: Error #15 (libiomp5md.dll already initialized)`. The accepted workaround
  `KMP_DUPLICATE_LIB_OK=TRUE` is set in `tests/conftest.py` and all scripts. Recorded in
  `RESOURCE_REQUIREMENTS.md`.
