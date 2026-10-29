"""Faithful reproduction of seven obfuscation / inversion attacks.

The package reproduces the attacks that appear in the experiments of two target
papers and their underlying original papers:

* NN / KNN     - nearest-neighbour embedding inversion
* VMA          - vocabulary-matching attack (autoregressive, weight RowSort, OSNIP projection)
* IA           - invariant attack (EDNN, Gate-IA, Attn-IA)
* ISA          - internal-state attack (ER, TBS, AloePri state matching)
* IMA          - inversion-model attack (InvBERT classify/seq2seq, Qwen2 AloePri)
* TFMA         - token-frequency matching attack
* SDA          - substitution-deciphering attack (2021 frequency-rank, 2023 recurrence)

Original algorithms, AloePri adaptations and OSNIP adaptations are kept in
separate modules, configs and tests. See ``docs/FORMULA_LEDGER.md`` for the
paper -> code -> test traceability of every formula.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
