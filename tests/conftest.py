"""Shared pytest configuration.

Sets the OpenMP duplicate-library workaround required by the conda MKL + torch
combination on this machine (recorded in docs/RESOURCE_REQUIREMENTS.md), keeps
tokenizers quiet, and ensures ``src`` is importable even without an editable
install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
