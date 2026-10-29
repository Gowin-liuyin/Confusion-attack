# Fidelity Statement

This repository reproduces the **attack algorithms** of the seven families faithfully and
verifies them with mathematical and end-to-end tests. It does **not** claim to have executed
the full paper-scale experiments (see `RESOURCE_REQUIREMENTS.md`).

## Non-negotiable fidelity rules honoured

1. **No metric/algorithm substitution.** L1 stays L1 (autoregressive VMA), RowSort stays RowSort
   (weight VMA, never a decoder), ER/TBS stay optimisation over internal-state distance (never a
   frequency rank), recurrence encoding stays first-appearance order (never frequency rank).
   `scripts/verify_fidelity.py` asserts these invariants at runtime.
2. **Lineage separation.** Original, AloePri and OSNIP variants are separate modules
   (`attacks/<family>/...`), separate configs (`configs/original|aloepri|osnip`), separate ledger
   rows, and separate `variant` tags on each attack class.
3. **No silent approximation.** Every `mode: full` config declares
   `fidelity.approximations` with every flag `false` (sampling, truncation, quantization, low
   precision, pruning, early stopping, candidate truncation, approximate nearest neighbour). The
   fidelity verifier fails if any is set true.
4. **No parameter shrinkage disguised as the paper's.** Reduced sizes appear only in `mode: smoke`
   configs and in tests, clearly labelled as synthetic. Paper parameters are transcribed verbatim
   in the `mode: full` configs.
5. **Honest status.** When an attack lacks the information a scheme exposes it returns
   `not_applicable` (never a fabricated 0%). When a full experiment cannot run it is recorded as
   `resource_blocked` (never fabricated metrics).
6. **Formula traceability.** Every formula in `FORMULA_LEDGER.md` cites paper, section, code file and
   test file; the verifier checks the code/test files exist.

## What is verified here (CPU, tiny models / synthetic scenarios)

- Mathematical identities: NN L2/cosine ranking and zero-vector handling; sorted-L1 permutation
  invariance; `Y = Z1 X Z2` RowSort recovery incl. duplicate-row disambiguation and multi-layer
  voting; EDNN glide difference invariant (with a general-reflection counterexample); Gate-IA average
  invariance; Attn-IA whitening invariance incl. singular-block skipping; cross-entropy and softmax;
  the two-term SDA-2023 CLM loss vs the target-only SDA-2021 loss; recurrence vs frequency-rank
  encodings; Qwen2 inverter fixed at 2 layers / 8 heads.
- End-to-end recovery on synthetic scenarios that contain the genuine leakage: each attack recovers
  the planted secret (tokens / mapping / permutation) through the capability-gated interface.

## What is NOT claimed

- No claim that paper numeric results (e.g. AloePri TTRSR tables, OSNIP ASR tables) were reproduced.
- No claim that this is a full reproduction of the originating papers' entire contributions - only
  the seven attack routes the two target papers exercise.
