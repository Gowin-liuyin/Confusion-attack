# Algorithm Fidelity Reference

This file describes algorithm checks, not experimental policy. The only authority for threat
models, data, budgets and publication acceptance is [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md).

## What the implementation checks establish

- Original, AloePri-adapted and OSNIP-adapted attacks have separate registrations/configs.
- [FORMULA_LEDGER.md](FORMULA_LEDGER.md) connects paper formulas to code and mathematical tests.
- `scripts/verify_fidelity.py` checks registration, declared approximation flags, selected paper
  parameters and mathematical invariants. It does not inspect every real deployment experiment.
- Capability-gated contexts and attack-isolation tests check that attack functions receive the
  declared inputs; secret-bearing sample generation and scoring are separate concerns.
- `scripts/privacy_family_gates.py` exercises implementation-level controls. Its passing status
  does not prove adequate real data, converged attack training, full-vocabulary search, or the
  current protocol's paper-scale sample size.

## Mechanism distinctions to preserve

- VMA-H is hidden-state replay with recovered prefixes. VMA-W is weight/permutation matching;
  the two do not share an interchangeable score or observation surface.
- A proposal-based replay, CPU classifier, bigram prior or reduced candidate set must retain its
  actual implementation label. Such substitutes are not automatically the original paper attack.
- Self-built-instance training pairs are different from target-client pairs. Known original
  weights are different from knowing the target token permutation or client key.
- A raw mapping score, a permuted-ID score and an end-to-end plaintext score measure different
  objectives. Their distinction is preserved in the protocol and in per-run output.
- A single-sided dense transform in the bundled engineering artifact is not the complete
  two-sided AloePri Algorithm 1/2. Read [DEVIATIONS.md](DEVIATIONS.md) and the actual artifact metadata.

## Limits of historical evidence

Synthetic fixtures, plaintext closed-loop runs and small real-artifact probes validate different
parts of the implementation. None alone reproduces a full defended-model paper table or proves
security. Original source/config parameters may differ from the current project's planned
evaluation; keep both provenance and actual execution metadata rather than silently relabeling.

Historical closed-loop, blocker and alignment records are in [archive/](archive/README.md).
Current facts come from actual `reports/` artifacts, checked under the unified protocol. Detailed
implementation deviations remain in [DEVIATIONS.md](DEVIATIONS.md).
