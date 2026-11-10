"""Entity/PII scoring with character-span to token-offset mapping (F07/W6).

Annotations are character spans in the real prompt/continuation text. Token
spans come from the tokenizer's offset mapping, so BPE boundaries are handled by
overlap rather than by re-tokenizing an isolated entity string. Every annotated
entity stays in the denominator: an entity absent from the generated text, an
uncovered position, a partial recovery and an early EOS all count as failures.
Input-prompt and generated-output entities are scored separately.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def char_spans_to_token_spans(offsets: Sequence[tuple[int, int]],
                              entities: Sequence[Mapping]) -> list[dict]:
    out = []
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        span = [
            index for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        covered_until = start
        for token_start, token_end in sorted(offsets[i] for i in span):
            if token_start > covered_until:
                break
            covered_until = max(covered_until, token_end)
        out.append({
            "type": str(entity.get("type", "unknown")),
            "value": entity.get("value"),
            "start": start,
            "end": end,
            "token_span": span,
            "present": bool(span),
            "fully_covered": end > start and covered_until >= end,
        })
    return out


def score_entity_recovery(
    truth_ids: Sequence[int],
    recovered_ids: Sequence[int],
    entities: Sequence[Mapping],
    offsets: Sequence[tuple[int, int]],
) -> dict:
    """Score prevalidated entity offsets over every annotated entity."""
    truth = np.asarray(truth_ids, dtype=np.int64)
    recovered = np.asarray(recovered_ids, dtype=np.int64)
    spans = char_spans_to_token_spans(offsets, entities)
    flags: list[bool] = []
    by_type: dict[str, list[bool]] = {}
    for entity in spans:
        span = entity["token_span"]
        ok = False
        if entity["fully_covered"] and span:
            if max(span) < recovered.size and max(span) < truth.size:
                ok = bool(np.all(recovered[np.asarray(span)] == truth[np.asarray(span)]))
        flags.append(ok)
        by_type.setdefault(entity["type"], []).append(ok)
    total = len(flags)
    hits = int(sum(flags))
    return {
        "value": (hits / total) if total else None,
        "numerator": hits,
        "denominator": total,
        "by_type": {
            kind: {
                "value": (sum(values) / len(values)) if values else None,
                "numerator": int(sum(values)),
                "denominator": len(values),
            }
            for kind, values in by_type.items()
        },
        "entities": spans,
        "denominator_rule": "every annotated entity, including absent/uncovered/partial",
    }


def score_verified_entity_recovery(tokenizer, text: str, truth_ids: Sequence[int],
                                   recovered_ids: Sequence[int], entities: Sequence[Mapping],
                                   offsets: Sequence[tuple[int, int]]) -> dict:
    """Score only when the annotation offsets are bound to the actual token sequence."""
    if entities:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            token_ids = [int(token) for token in encoded["input_ids"]]
            actual_offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
            supplied_offsets = [(int(start), int(end)) for start, end in offsets]
        except (KeyError, TypeError, ValueError, NotImplementedError):
            return {"status": "not_run", "reason": "tokenizer offsets unavailable or invalid"}
        if token_ids != [int(token) for token in truth_ids] or actual_offsets != supplied_offsets:
            return {"status": "not_run", "reason": "annotation offsets or token IDs do not match the actual text"}
        for entity in entities:
            try:
                start, end = int(entity["start"]), int(entity["end"])
            except (KeyError, TypeError, ValueError):
                return {"status": "not_run", "reason": "invalid entity span"}
            if start < 0 or end <= start:
                return {"status": "not_run", "reason": "invalid entity span"}
            value = entity.get("value")
            if end <= len(text) and value is not None and text[start:end] != value:
                return {"status": "not_run", "reason": "entity value does not match its text span"}
    return {"status": "scored", **score_entity_recovery(truth_ids, recovered_ids, entities, offsets)}


def score_generation_pii(
    record,
    *,
    tokenizer,
    prompt_text: str,
    prompt_entities: Sequence[Mapping],
    output_entities: Sequence[Mapping],
    prompt_offsets: Sequence[tuple[int, int]],
    output_offsets: Sequence[tuple[int, int]],
    recovered_output_ids: Sequence[int],
) -> dict:
    """Score the input prompt and the generated output in separate namespaces."""
    prompt = score_verified_entity_recovery(
        tokenizer, prompt_text, record.prompt_ids, record.prompt_ids, prompt_entities, prompt_offsets
    )
    try:
        output_text = (tokenizer.decode(record.continuation_ids, clean_up_tokenization_spaces=False)
                       if output_entities else "")
    except (TypeError, ValueError, NotImplementedError):
        output = {"status": "not_run", "reason": "generated text could not be decoded for annotation verification"}
    else:
        output = score_verified_entity_recovery(
            tokenizer, output_text, record.continuation_ids, recovered_output_ids, output_entities, output_offsets
        )
    return {
        "input_prompt": prompt,
        "generated_output": output,
        "note": "the input prompt is not an attack result; it is the exposure reference",
    }
