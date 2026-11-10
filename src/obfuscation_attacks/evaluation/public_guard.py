"""Fail-closed public-artifact guard (protocol §1.7-9 / §4.4).

Every result payload, plan, status document and summary written by the battery is
public evidence. Ground-truth ids, the secret permutation, key roots and client
secrets must never appear in them. Keys are normalised (lowercase, non-alphanumerics
stripped) before matching, so ``Plaintext_IDs`` and ``inv_perm`` are caught as well
as their snake_case forms.

The only sanctioned ground-truth fields are the protocol-audit evidence rows under
``per_unit_predictions`` / ``per_unit_predictions_by_layer``: a strict row schema
(``{unit,hits,count,rank,predicted,truth,prompt,key,request,text}`` with integer
ids and integer lists) is validated and nothing else is exempted. Hash commitments
(``client_secret_sha256`` etc.) must be single 64-hex strings; lists, dicts or
non-hex values are rejected.

The guard is shared by the result writers (``real_battery.write_payload``) and the
battery's plan/status/summary writer. The serialized corpus-text scan lives in
``scripts/run_full_battery.py`` because only that caller knows the evaluated texts.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: Tokens that mark a mapping key as carrying ground truth or secret material.
FORBIDDEN_KEY_TOKENS = (
    "truth", "plaintextid", "permutedid", "invperm", "rootkey",
    "clientsecret", "secret", "encoder", "perm",
)

#: Structural prompt-identity keys and their expected value types.
ALLOWED_PROMPT_KEYS = frozenset({
    "promptindex", "promptsha256", "promptid", "nprompts", "keysperprompt",
})
STRUCTURAL_PROMPT_COUNT_KEYS = frozenset({
    "prompts", "promptcount", "promptcounts", "numprompts", "valprompts",
})

#: The per-prompt result container is allowed only as a list of row mappings;
#: its rows are still key-scanned, so a raw ``prompt`` field inside is rejected.
PER_PROMPT_CONTAINER_KEYS = frozenset({"perprompt"})

#: Strict schema for the sanctioned per-unit scoring evidence.
EVIDENCE_CONTAINERS = frozenset({
    "perunitpredictions", "perunitpredictionsbylayer",
})
EVIDENCE_ROW_KEYS = frozenset({
    "unit", "hits", "count", "rank", "predicted", "truth",
    "prompt", "key", "request", "text",
})
EVIDENCE_INT_KEYS = frozenset({"unit", "hits", "count", "prompt", "key",
                               "request", "text"})
EVIDENCE_LIST_KEYS = frozenset({"rank", "predicted", "truth"})

#: Normalised keys that merely contain a token word without carrying material:
#: boolean capability/permission flags and hash commitments.
BENIGN_KEYS = frozenset({
    "attackusesencoder",          # boolean capability flag
    "attackusesclientsecret",     # boolean capability flag (never the secret)
    "usesclientsecret",           # boolean capability flag (never the secret)
    "encodersha256",              # hash commitment, not encoder parameters
    "clientsecretsha256",         # hash commitment, not the client secret bytes
    "attackusessecretpermutation",            # boolean permission record
    "diagnosticvariantsusesecretpermutation",  # boolean permission record
    "rowsortusessecretpermutation",           # boolean permission record
    "knownsecretdiagnostic",                  # boolean premise-diagnostic flag
    "stage1permutedid",            # attack estimate container, not truth
    "composedwithoraclepermutation",  # oracle upper-bound metric, attack outputs
    "usessecretpermutation",                  # per-diagnostic boolean flag
    "usessecret",                             # boolean flag on alignment records
    "permisvalid",                            # boolean validity verdict of the
                                              # deployment permutation (artifact
                                              # metadata echo; the permutation
                                              # itself stays forbidden)
    # These names contain the "perm" token but only declare CAPABILITIES; they
    # are boolean flags and are listed in ``_BOOLEAN_BENIGN_KEYS`` so a raw
    # secret string or list under them is still rejected.
    "permission", "permissions", "permitted", "permit", "permits",
    "hyperparameter", "hyperparameters",
})

_HASH_COMMITMENT = re.compile(r"^[0-9a-fA-F]{64}$")
_FLAG_SUFFIXES = ("available", "flag", "used", "uses", "enabled", "present")
_COMMITMENT_SUFFIXES = ("sha256", "commitment", "digest")
#: A key that IS one of these forbidden names never becomes a commitment just
#: because its value looks like a hash (``root_key_sha256`` is fine, ``root_key``
#: is not).
_FORBIDDEN_EXACT_KEYS = frozenset({
    "truth", "plaintextid", "plaintextids", "permutedid", "permutedids",
    "invperm", "rootkey", "clientsecret", "secret", "encoder", "perm",
})
#: Numeric structural keys (lengths, shapes, counts) are metadata, not material.
#: Allowed when the value is an int or a list/tuple of ints.
_NUMERIC_STRUCTURAL_SUFFIXES = (
    "len", "length", "count", "size", "shape", "dim", "ndim", "numel", "nbytes",
)
_BOOLEAN_BENIGN_KEYS = frozenset({
    "attackusesencoder", "attackusesclientsecret", "usesclientsecret",
    "attackusessecretpermutation", "rowsortusessecretpermutation",
    "usessecretpermutation", "usessecret",
    "knownsecretdiagnostic",
    "permission", "permissions", "permitted", "permit", "permits",
    "hyperparameter", "hyperparameters",
    "permisvalid",
})
#: This permission record may be a boolean or the list of diagnostic variant
#: NAMES (short identifiers only, never free text).
_DIAGNOSTIC_VARIANT_FLAG = "diagnosticvariantsusesecretpermutation"
_VARIANT_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def normalise_key(key) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


#: Only these keys may carry a 64-hex commitment (and only a 64-hex string,
#: or a mapping of path -> 64-hex for a file-hash map). Keys named root_key /
#: client_secret / inv_perm / perm / secret stay forbidden regardless of value.
COMMITMENT_KEYS = frozenset({
    "clientsecretsha256", "deploymentstatesha256", "rootkeycommitment",
    "corpusha256", "referencecountssha256", "studysha256", "datasha256",
    "sourcesha256", "encodersha256", "filessha256",
})


def _commitment_gaps(normal: str, value: Any, path: str) -> list[str]:
    """Commitment values must be a single 64-hex string (or an unbound empty).

    ``None``/``""`` mean "not bound" (e.g. a probe run's empty
    ``protocol_sha256``); they carry no material and are allowed. Lists, dicts
    and non-hex strings are rejected.
    """
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        gaps: list[str] = []
        for child, digest in value.items():
            if digest is None or digest == "":
                continue
            if not (isinstance(digest, str) and _HASH_COMMITMENT.match(digest)):
                gaps.append(
                    f"{path}.{child}: commitment map value is not a 64-hex string"
                )
        return gaps
    if not (isinstance(value, str) and _HASH_COMMITMENT.match(value)):
        return [f"{path}: commitment is not a single 64-hex string"]
    return []


def _is_benign_value(normal: str, value: Any) -> bool:
    """Boolean availability flags are not material; hashes are handled above."""
    return isinstance(value, bool) and normal.endswith(_FLAG_SUFFIXES)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_int_list(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and all(_is_int(item) for item in value)
    )


def _is_numeric_structural(normal: str, value: Any) -> bool:
    """Lengths/shapes/counts are metadata: ints or int lists only, never texts."""
    if not normal.endswith(_NUMERIC_STRUCTURAL_SUFFIXES):
        return False
    if _is_int(value):
        return True
    return _is_int_list(value)


def _evidence_row_gaps(row: Any, path: str) -> list[str]:
    if not isinstance(row, Mapping):
        return [f"{path}: evidence row is not a mapping"]
    gaps: list[str] = []
    for key, value in row.items():
        normal = normalise_key(key)
        if normal not in EVIDENCE_ROW_KEYS:
            gaps.append(
                f"{path}.{key}: key outside the evidence row schema "
                f"(normalised {normal!r})"
            )
            continue
        if normal in EVIDENCE_INT_KEYS and not _is_int(value):
            gaps.append(f"{path}.{key}: evidence id is not an integer")
        if normal in EVIDENCE_LIST_KEYS and not _is_int_list(value):
            gaps.append(f"{path}.{key}: evidence list is not a list of integers")
    return gaps


def _evidence_container_gaps(normal: str, value: Any, path: str) -> list[str]:
    gaps: list[str] = []
    if normal == "perunitpredictions":
        if not isinstance(value, (list, tuple)):
            return [f"{path}: per_unit_predictions is not a list"]
        for index, row in enumerate(value):
            gaps.extend(_evidence_row_gaps(row, f"{path}[{index}]"))
        return gaps
    if not isinstance(value, Mapping):
        return [f"{path}: per_unit_predictions_by_layer is not a mapping"]
    for layer, rows in value.items():
        layer_normal = normalise_key(layer)
        if any(token in layer_normal for token in FORBIDDEN_KEY_TOKENS):
            gaps.append(f"{path}.{layer}: forbidden key inside evidence layers")
            continue
        if not isinstance(rows, (list, tuple)):
            gaps.append(f"{path}.{layer}: layer evidence is not a list")
            continue
        for index, row in enumerate(rows):
            gaps.extend(_evidence_row_gaps(row, f"{path}.{layer}[{index}]"))
    return gaps


def _prompt_value_gaps(normal: str, value: Any, path: str) -> list[str]:
    if normal == "promptsha256":
        if not (isinstance(value, str) and _HASH_COMMITMENT.match(value)):
            return [f"{path}: prompt_sha256 is not a 64-hex commitment"]
        return []
    if not _is_int(value):
        return [f"{path}: structural prompt field is not an integer"]
    return []


def forbidden_material_gaps(obj: Any, path: str = "payload") -> list[str]:
    """Paths of forbidden truth/secret keys or values anywhere in ``obj``."""
    gaps: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = str(key)
            normal = normalise_key(name)
            here = f"{path}.{name}"
            if normal in EVIDENCE_CONTAINERS:
                gaps.extend(_evidence_container_gaps(normal, value, here))
                continue
            if normal in PER_PROMPT_CONTAINER_KEYS:
                if (not isinstance(value, (list, tuple))
                        or not all(isinstance(row, Mapping) for row in value)):
                    gaps.append(
                        f"{here}: per_prompt must be a list of row mappings"
                    )
                    continue
                for index, row in enumerate(value):
                    gaps.extend(
                        forbidden_material_gaps(row, f"{here}[{index}]")
                    )
                continue
            if _is_numeric_structural(normal, value):
                continue
            if normal in ALLOWED_PROMPT_KEYS:
                gaps.extend(_prompt_value_gaps(normal, value, here))
                continue
            if normal in STRUCTURAL_PROMPT_COUNT_KEYS:
                if not _is_int(value):
                    gaps.append(f"{here}: prompt count is not an integer")
                continue
            if normal in COMMITMENT_KEYS:
                gaps.extend(_commitment_gaps(normal, value, here))
                continue
            if (normal.endswith(_COMMITMENT_SUFFIXES)
                    and normal not in _FORBIDDEN_EXACT_KEYS):
                # Any *_sha256/*_commitment/*_digest key is a commitment when its
                # value is a single 64-hex string (e.g. logits_perm_sha256).
                gaps.extend(_commitment_gaps(normal, value, here))
                continue
            if normal in BENIGN_KEYS:
                # Even an allowlisted key must carry the benign VALUE type: a
                # commitment is a single 64-hex string, a flag is a boolean.
                if normal == _DIAGNOSTIC_VARIANT_FLAG:
                    valid = isinstance(value, bool) or (
                        isinstance(value, (list, tuple))
                        and all(isinstance(item, str)
                                and _VARIANT_NAME.match(item) for item in value)
                    )
                    if not valid:
                        gaps.append(
                            f"{here}: permission record must be a boolean or "
                            "variant-name list"
                        )
                elif normal.endswith(_COMMITMENT_SUFFIXES):
                    if not (isinstance(value, str)
                            and _HASH_COMMITMENT.match(value)):
                        gaps.append(
                            f"{here}: commitment is not a single 64-hex string"
                        )
                elif normal in _BOOLEAN_BENIGN_KEYS and not isinstance(value, bool):
                    gaps.append(f"{here}: capability flag is not a boolean")
            elif normal in {"continuation", "continuationids", "chosenid", "chosenids"}:
                gaps.append(f"{here}: client-side generation truth is not public evidence")
            elif any(token in normal for token in FORBIDDEN_KEY_TOKENS):
                if not _is_benign_value(normal, value):
                    gaps.append(
                        f"{here}: forbidden truth/secret key "
                        f"(normalised {normal!r})"
                    )
            elif "prompt" in normal:
                gaps.append(
                    f"{here}: raw prompt-bearing key (normalised {normal!r})"
                )
            gaps.extend(forbidden_material_gaps(value, here))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            gaps.extend(forbidden_material_gaps(value, f"{path}[{index}]"))
    return gaps


def assert_public_payload(payload: Mapping, *, label: str = "payload") -> None:
    """Raise ``ValueError`` (writing nothing) when the payload leaks material."""
    gaps = forbidden_material_gaps(payload, path=label)
    if gaps:
        raise ValueError(
            f"refusing to write {label} with truth/secret material: "
            + ", ".join(gaps[:8])
        )
