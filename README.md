# Obfuscation / Inversion Attack Reproduction

Faithful, runnable, testable reproduction of the **seven** obfuscation / inversion attacks exercised
by two target papers - **AloePri** (`2603.01499v2`) and **OSNIP** (`2601.22752v1`) - and their
underlying original papers. Original, AloePri-adapted and OSNIP-adapted variants are kept strictly
separate in code, configs, tests and documentation.

| Family | Attacks (registered name) |
|--------|---------------------------|
| NN / KNN | `nn_knn.euclidean`, `nn_knn.cosine` |
| VMA | `vma.autoregressive` (original), `vma.weight_rowsort` (AloePri), `vma.osnip_projection` (OSNIP) |
| IA | `ia.ednn` (original EDNN), `ia.gate` (AloePri), `ia.attn` (AloePri) |
| ISA | `isa.er`, `isa.tbs` (original), `isa.aloepri_state` (AloePri) |
| IMA | `ima.classify`, `ima.seq2seq` (InvBERT), `ima.qwen2_aloepri` (AloePri, 2 layers / 8 heads) |
| TFMA | `tfma.independent` (AloePri), `tfma.global_assignment` (extension) |
| SDA | `sda.sda2021` (frequency-rank + seq2seq), `sda.sda2023` (recurrence + causal LM) |

See `docs/FORMULA_LEDGER.md` for paper -> code -> test traceability of every formula and
`docs/FIDELITY.md` for the faithfulness rules.

## Install from zero

```powershell
# Python 3.10+ (this build used 3.13). From the repo root:
python -m pip install -e .            # core: numpy, scipy, pyyaml
python -m pip install -e .[neural]    # torch + transformers (ISA/IMA/SDA, Qwen2 inverter)
# optional, for real-data full runs: python -m pip install -e .[data]
```

On a conda+pip mix you may hit `OMP: Error #15 (libiomp5md.dll)`. Export the standard workaround
(already set inside the test suite and scripts):

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Run the tests

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m pytest                      # unit + mathematical + regression + integration
python -m pytest -m mathematical      # just the formula-pinning tests
python -m pytest -m "not neural"      # skip torch-backed tests
```

## The four entry points

```powershell
python -m pytest                                  # all tests
python scripts/run_experiment.py --all-smoke      # run every synthetic smoke config
python scripts/run_experiment.py --config configs/osnip/knn.yaml   # a full config -> resource_blocked
python scripts/verify_fidelity.py                 # fidelity gate (families, variants, invariants, ledger)
python scripts/audit_placeholders.py              # no TODO / stub / unimplemented attack gate
```

Smoke runs write artifacts under `reports/runs/<name>/`:
`config.resolved.yaml`, `environment.json`, `metrics.json`, `predictions.json`, `run.log`,
`fidelity.json`.

## Full paper experiments

The `configs/original|aloepri|osnip/*.yaml` files transcribe the real paper setups
(models, datasets, hyperparameters, seeds). They are `mode: full` and are **resource-blocked** on an
8 GB laptop GPU (see `docs/RESOURCE_REQUIREMENTS.md`); the runner records a `resource_blocked` status
rather than fabricating numbers. To run them, provision adequate hardware, download the datasets, and
point a real model/data backend at the loader (see that doc). The attack algorithms themselves are
verified by the mathematical and end-to-end synthetic tests.

## Plug in a custom obfuscation scheme (Goal section 9)

A scheme is the defender. Subclass `ObfuscationScheme`, declare which capabilities you expose, and
produce `AttackSample`s; the capability-gated `AttackContext` then lets an attacker use *only* the
information you reveal.

```python
from obfuscation_attacks.data.contracts import Capability
from obfuscation_attacks.schemes.base import ObfuscationScheme, build_context

class MyScheme(ObfuscationScheme):
    name = "my_scheme"
    def exposed_capabilities(self):
        return frozenset({Capability.EMBEDDINGS})   # reveal only perturbed embeddings
    def build_sample(self, ...):
        ...  # fill AttackSample.obfuscated_embeddings, reference_embeddings, ground truth
```

Worked example: `src/obfuscation_attacks/schemes/custom_example.py` +
`configs/custom_scheme/example.yaml` (run it with
`python scripts/run_experiment.py --config configs/custom_scheme/example.yaml`). Attacks that need a
capability you withhold are reported `not_applicable` (never fed your secret, never scored 0%).
Capabilities: `TOKEN_CIPHERTEXT`, `EMBEDDINGS`, `HIDDEN_STATES`, `MODEL_REPLAY`,
`WHITE_BOX_WEIGHTS`, `KNOWN_OBFUSCATION_ALGORITHM`, `DETERMINISTIC_PERMUTATION`.

## Repository layout

```
configs/{original,aloepri,osnip,smoke,custom_scheme}/   experiment configs (full vs smoke separated)
src/obfuscation_attacks/
  data/         contracts (AttackSample / AttackContext / capabilities), synthetic scenarios
  schemes/      ObfuscationScheme base, capability-gated context builder, custom example
  models/       ModelHandle + numpy causal toy + differentiable torch toy
  attacks/      nn_knn/ vma/ ia/ isa/ ima/ tfma/ sda/   (original/AloePri/OSNIP in separate modules)
  metrics/      ASR@k, token P/R/F1, TTRSR, Total/Clean ASR, mapping ASR
  evaluation/   runner + scoring + artifacts
  utils/        seeding, IO, environment snapshot
tests/{unit,mathematical,regression,integration}/       the four test tiers
scripts/        run_experiment.py, verify_fidelity.py, audit_placeholders.py
docs/           BIBLIOGRAPHY, FORMULA_LEDGER, EXPERIMENT_MATRIX, FIDELITY, DEVIATIONS, RESOURCE_REQUIREMENTS
```

## Faithfulness and scope

This reproduces the seven attack **routes** the two target papers exercise, with every formula traced
and tested. It does **not** claim to reproduce the papers' full numeric tables (resource-blocked) nor
the entirety of the original papers' contributions. Deviations and unspecified choices are recorded in
`docs/DEVIATIONS.md`.
