"""Placeholder / unfinished-work auditor - exits 0 only if the code is complete.

Fails (exit 1) when it finds any of:

* unfinished-work markers (TODO / FIXME / XXX / HACK) anywhere in ``src`` or
  ``scripts``;
* ``NotImplementedError`` outside the single abstract base method
  ``attacks/base.py`` (i.e. a real attack left unimplemented);
* a function whose body is only ``pass`` or ``...`` (a stub), except genuine
  ``@abstractmethod`` declarations;
* a registered attack that does not override ``_run`` with its own body.

This is the "no placeholder attacks / no fake results / no TODO" gate.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "obfuscation_attacks"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

SELF = Path(__file__).resolve()
BASE_MODULE = (SRC / "attacks" / "base.py").resolve()

MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
NIE_RE = re.compile(r"NotImplementedError")

problems: list[str] = []


def _iter_py() -> list[Path]:
    files = list(SRC.rglob("*.py")) + list(SCRIPTS.glob("*.py"))
    return [f for f in files if f.resolve() != SELF]


def scan_markers(path: Path, text: str) -> None:
    for i, line in enumerate(text.splitlines(), 1):
        if MARKER_RE.search(line):
            problems.append(f"{path.relative_to(ROOT)}:{i}: unfinished-work marker -> {line.strip()}")
        if NIE_RE.search(line) and path.resolve() != BASE_MODULE:
            problems.append(
                f"{path.relative_to(ROOT)}:{i}: NotImplementedError outside the abstract base"
            )


def _has_abstract_decorator(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", []):
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
        if "abstractmethod" in str(name):
            return True
    return False


def _is_stub_body(body: list[ast.stmt]) -> bool:
    stmts = body
    # drop a leading docstring
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(
        getattr(stmts[0], "value", None), ast.Constant
    ) and isinstance(stmts[0].value.value, str):
        stmts = stmts[1:]
    if len(stmts) != 1:
        return False
    only = stmts[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) and only.value.value is ...:
        return True
    return False


def scan_stubs(path: Path, text: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover
        problems.append(f"{path.relative_to(ROOT)}: syntax error {exc}")
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _has_abstract_decorator(node):
                continue
            if _is_stub_body(node.body):
                problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: stub function {node.name!r} "
                    f"(body is only pass/...)"
                )


def check_attacks_implemented() -> None:
    from obfuscation_attacks.attacks import load_all, REGISTRY
    from obfuscation_attacks.attacks.base import Attack

    load_all()
    for name, cls in REGISTRY.items():
        own = cls._run is not Attack._run
        problems_append = own and callable(cls._run)
        if not problems_append:
            problems.append(f"attack {name} does not provide a real _run implementation")


def main() -> int:
    for path in _iter_py():
        text = path.read_text(encoding="utf-8")
        scan_markers(path, text)
        scan_stubs(path, text)
    check_attacks_implemented()

    if problems:
        print("PLACEHOLDER AUDIT FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PLACEHOLDER AUDIT PASSED: no TODO/FIXME markers, no unimplemented attacks, no stubs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
