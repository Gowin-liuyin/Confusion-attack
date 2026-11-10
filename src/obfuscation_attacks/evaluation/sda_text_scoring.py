"""Scoring-side character alignment for reconstructed text.

The attacker produces text independently; private reference text and annotations
are passed only to this scorer. No truth text or entity values are returned.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence
import numpy as np

_WORD = re.compile(r"\w+(?:['\u2019]\w+)*", re.UNICODE)


def _unambiguous_character_alignment(reference: str, predicted: str, max_cells: int):
    """Positions matched identically at one position in every optimal edit path."""
    n, m = len(reference), len(predicted)
    if (n + 1) * (m + 1) > max_cells:
        raise ValueError("character alignment exceeds max_cells; no truncation was performed")
    forward = np.zeros((n + 1, m + 1), dtype=np.int32)
    backward = np.zeros((n + 1, m + 1), dtype=np.int32)
    forward[:, 0] = np.arange(n + 1)
    forward[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            forward[i, j] = min(forward[i - 1, j] + 1, forward[i, j - 1] + 1,
                                forward[i - 1, j - 1] + (reference[i - 1] != predicted[j - 1]))
    backward[:, m] = np.arange(n, -1, -1)
    backward[n, :] = np.arange(m, -1, -1)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            backward[i, j] = min(backward[i + 1, j] + 1, backward[i, j + 1] + 1,
                                 backward[i + 1, j + 1] + (reference[i] != predicted[j]))
    distance = int(forward[n, m])
    mapping = []
    for i in range(n):
        matches = []
        ambiguous = False
        for j in range(m + 1):
            if int(forward[i, j]) + 1 + int(backward[i + 1, j]) == distance:
                ambiguous = True  # an optimal path deletes this reference character
            if j < m:
                cost = int(reference[i] != predicted[j])
                if int(forward[i, j]) + cost + int(backward[i + 1, j + 1]) == distance:
                    if cost:
                        ambiguous = True
                    else:
                        matches.append(j)
        mapping.append(matches[0] if not ambiguous and len(matches) == 1 else None)
    return mapping, distance


def _span_recovered(mapping: Sequence[int | None], start: int, end: int) -> bool:
    positions = mapping[start:end]
    return bool(positions) and all(p is not None for p in positions) and all(
        positions[i] == positions[0] + i for i in range(len(positions))
    )


def _entity_recovered(mapping, reference: str, predicted: str, start: int, end: int) -> bool:
    if not _span_recovered(mapping, start, end):
        return False
    left, right = mapping[start], mapping[end - 1] + 1
    is_word = lambda char: bool(re.fullmatch(r"\w", char, re.UNICODE))
    if (is_word(reference[start]) and (start == 0 or not is_word(reference[start - 1]))
            and left > 0 and is_word(predicted[left - 1])):
        return False
    if (is_word(reference[end - 1]) and (end == len(reference) or not is_word(reference[end]))
            and right < len(predicted) and is_word(predicted[right])):
        return False
    return True


def score_text_reconstruction(reference: str, predicted: str, *, entities: Sequence[Mapping] = (),
                              max_cells: int = 2_000_000) -> dict:
    """Count exact, unambiguous reference word/entity recovery after character alignment.

    Offsets are Python Unicode codepoint offsets, not UTF-8 bytes. Every supplied
    entity remains in the denominator. Ambiguous optimal alignments count as
    misses; no case folding or Unicode normalization is performed. Word
    boundaries use Python's Unicode ``\\w`` rule and are not a tokenizer metric.
    """
    if not isinstance(reference, str) or not isinstance(predicted, str):
        raise TypeError("reference and predicted must be strings")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")
    annotations = []
    for entity in entities:
        start, end = entity.get("start"), entity.get("end")
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, int) or not isinstance(end, int) or
                start < 0 or start >= end or end > len(reference)):
            raise ValueError("entity offsets must be non-empty codepoint spans within the reference")
        annotations.append((start, end, str(entity.get("type", "unknown"))))
    mapping, distance = _unambiguous_character_alignment(reference, predicted, max_cells)
    words = list(_WORD.finditer(reference))
    predicted_words = {(word.start(), word.end()) for word in _WORD.finditer(predicted)}
    word_hits = sum(
        _span_recovered(mapping, word.start(), word.end())
        and (mapping[word.start()], mapping[word.end() - 1] + 1) in predicted_words
        for word in words
    )
    entity_flags = [(kind, _entity_recovered(mapping, reference, predicted, start, end))
                    for start, end, kind in annotations]
    def counts(hits, total):
        return {"value": hits / total if total else None, "numerator": int(hits), "denominator": int(total)}
    by_type = {}
    for kind, flag in entity_flags:
        values = by_type.setdefault(kind, [0, 0])
        values[0] += int(flag)
        values[1] += 1
    return {
        "character_edit": {"edits": distance, "reference_characters": len(reference),
                           "value": distance / len(reference) if reference else None,
                           "metric": "unicode_codepoint_levenshtein_not_TER"},
        "sda2021_character_ter": {"edits": distance, "reference_characters": len(reference),
                                   "value": distance / len(reference) if reference else None,
                                   "metric": "sda2021_character_insert_delete_substitute_rate",
                                   "source": "https://aclanthology.org/2021.acl-long.561/",
                                   "normalization": "none", "shift_operations": False,
                                   "unit": "unicode_codepoints",
                                   "fidelity": "paper_defined_operations; implementation_not_author_verified"},
        "exact_sequence": counts(int(reference == predicted), 1) if reference else counts(0, 0),
        "word_recovery": counts(word_hits, len(words)),
        "entity_recovery": {**counts(sum(flag for _, flag in entity_flags), len(entity_flags)),
                            "by_type": {kind: counts(*values) for kind, values in by_type.items()}},
        "alignment": {"rule": "all optimal Levenshtein paths must agree on identical contiguous characters",
                      "unrecovered_or_ambiguous_reference_characters": sum(value is None for value in mapping),
                      "offset_unit": "unicode_codepoints", "normalization": "none",
                      "word_rule": "exact unicode word regex span and boundaries",
                      "entity_rule": "contiguous exact span with retained outer word boundaries"},
    }


def score_sda_text_rows(references: Mapping[int, str], predictions: Mapping[int, str], *,
                        entities: Mapping[int, Sequence[Mapping]] | None = None,
                        max_cells: int = 2_000_000) -> dict:
    """Aggregate scoring-only per-request text metrics over the entire reference set.

    Missing predictions count as empty outputs; unknown prediction/annotation
    IDs are rejected. This does not reconstruct a reference-token Top-k metric.
    """
    entities = entities or {}
    if set(predictions) - set(references) or set(entities) - set(references):
        raise ValueError("prediction or entity request ID is absent from references")
    rows = []
    for request_id, reference in references.items():
        result = score_text_reconstruction(reference, predictions.get(request_id, ""),
                                           entities=entities.get(request_id, ()), max_cells=max_cells)
        rows.append({"request_id": request_id, **result})
    def aggregate(name):
        numerator = sum(row[name]["numerator"] for row in rows)
        denominator = sum(row[name]["denominator"] for row in rows)
        return {"value": numerator / denominator if denominator else None,
                "numerator": numerator, "denominator": denominator}
    edits = sum(row["character_edit"]["edits"] for row in rows)
    characters = sum(row["character_edit"]["reference_characters"] for row in rows)
    entity_types = {}
    for row in rows:
        for kind, counts in row["entity_recovery"]["by_type"].items():
            combined = entity_types.setdefault(kind, [0, 0])
            combined[0] += counts["numerator"]
            combined[1] += counts["denominator"]
    per_type = {kind: {"value": hits / total if total else None,
                       "numerator": hits, "denominator": total}
                for kind, (hits, total) in entity_types.items()}
    return {"status": "scored" if rows else "not_evaluated", "n_requests": len(rows),
            "missing_predictions": len(set(references) - set(predictions)),
            "character_edit": {"value": edits / characters if characters else None,
                               "edits": edits, "reference_characters": characters,
                               "metric": "unicode_codepoint_levenshtein_not_TER"},
            "sda2021_character_ter": {"value": edits / characters if characters else None,
                                       "edits": edits, "reference_characters": characters,
                                       "metric": "sda2021_character_insert_delete_substitute_rate",
                                       "source": "https://aclanthology.org/2021.acl-long.561/",
                                       "normalization": "none", "shift_operations": False,
                                       "unit": "unicode_codepoints", "aggregation": "micro",
                                       "fidelity": "paper_defined_operations; implementation_not_author_verified"},
            "exact_sequence": aggregate("exact_sequence"),
            "word_recovery": aggregate("word_recovery"),
            "entity_recovery": {**aggregate("entity_recovery"), "by_type": per_type},
            "reference_token_topk": {"status": "not_evaluated",
                                      "reason": "character alignment does not establish token-level candidates"},
            "rows": rows}
