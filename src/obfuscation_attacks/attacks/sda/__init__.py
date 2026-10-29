"""SDA - Substitution-Deciphering Attack family.

The two paper lineages are implemented in SEPARATE modules and are never merged:

* :mod:`.sda2021_seq2seq` - Aldarrab & May (2021): frequency-rank encoding +
  Transformer encoder-decoder, objective ``L_Seq2Seq = -log P(Y|X)``, beam
  search, 1:1 ciphers.
* :mod:`.sda2023_recurrence_clm` - Kambhatla et al. (2023): recurrence encoding +
  causal LM, objective ``L_CLM = -log P(X) - log P(Y|X)``, 1:1 and homophonic
  ciphers.

Importing this package registers both attacks (``sda.sda2021`` and
``sda.sda2023``) and pulls the torch-backed model modules.
"""

from __future__ import annotations

from . import encodings
from . import sda2021_seq2seq
from . import sda2023_recurrence_clm
from .sda2021_seq2seq import SDA2021
from .sda2023_recurrence_clm import SDA2023

__all__ = [
    "encodings",
    "sda2021_seq2seq",
    "sda2023_recurrence_clm",
    "SDA2021",
    "SDA2023",
]
