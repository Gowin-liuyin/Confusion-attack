"""Utility interfaces bound to the same artifact (protocol §3, §8.3; W6).

The PPL counter uses a fixed sliding window and counts every target token exactly
once. Generation tasks use the same message path as the privacy run; a scripted
answer is never accepted as utility or output evidence.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def sliding_windows(ids: Sequence[int], *, window: int = 512, stride: int = 256) -> list[tuple[list[int], list[int], int]]:
    """Return ``(window_ids, target_positions, start)`` with each token counted once."""
    tokens = [int(token) for token in ids]
    n = len(tokens)
    if window < 2 or stride < 1 or stride >= window:
        raise ValueError("window must be >= 2 and 1 <= stride < window")
    if n < 2:
        return []
    windows: list[tuple[list[int], list[int], int]] = []
    start = 0
    previous_end = 1  # position 0 is context only, never a target
    while start < n - 1:
        end = min(start + window, n)
        targets = list(range(max(start + 1, previous_end), end))
        if targets:
            windows.append((tokens[start:end], targets, start))
            previous_end = end
        if end >= n:
            break
        start += stride
    return windows


def evaluate_ppl(model, tokenizer, texts: Iterable[str], *, window: int = 512,
                 stride: int = 256) -> dict:
    """Mean negative log-likelihood per counted token (each token exactly once)."""
    import torch

    total_nll = 0.0
    total_tokens = 0
    per_text = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        windows = sliding_windows(ids, window=window, stride=stride)
        text_nll = 0.0
        text_tokens = 0
        for window_ids, targets, start in windows:
            input_ids = torch.tensor([window_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_ids).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            for position in targets:
                local = position - start
                text_nll -= float(log_probs[local - 1, window_ids[local]])
                text_tokens += 1
        if text_tokens:
            per_text.append(text_nll / text_tokens)
        total_nll += text_nll
        total_tokens += text_tokens
    return {
        "ppl": float(np.exp(total_nll / total_tokens)) if total_tokens else None,
        "mean_nll": (total_nll / total_tokens) if total_tokens else None,
        "n_tokens": int(total_tokens),
        "per_text_nll": per_text,
        "window": int(window),
        "stride": int(stride),
        "counting_rule": "each target token after BOS is counted exactly once",
    }


def _choice_loglikelihood(
    model, tokenizer, prompt: str, choice: str, *, max_length: int
) -> float:
    """Sum of ``log p(choice_t | prompt + choice_<t)`` over the choice tokens.

    This is the standard 0-shot multiple-choice score. Scoring only the first
    choice token (the previous implementation) ties whenever choices share their
    first token and is not the task metric.
    """
    import torch

    if not choice or max_length < 2:
        raise ValueError("choice must be nonempty and max_length must leave context")
    try:
        encoded = tokenizer(prompt + choice, add_special_tokens=True, truncation=False,
                            return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
    except (TypeError, NotImplementedError, KeyError, ValueError):
        encoded = tokenizer(prompt + choice, add_special_tokens=True, truncation=False)
        offsets = None
    full = list(encoded["input_ids"])
    if offsets is not None:
        if len(offsets) != len(full):
            raise ValueError("token offsets do not align with input ids")
        positions = [i for i, (begin, end) in enumerate(offsets)
                     if end > len(prompt) and end > begin]
        if not positions:
            raise ValueError("choice produced no scoreable tokens")
        start, end = positions[0], positions[-1] + 1
        if positions != list(range(start, end)):
            raise ValueError("choice token offsets are not contiguous")
        crossing = offsets[start][0] < len(prompt)
    else:
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        if full[:len(prompt_ids)] != prompt_ids:
            raise ValueError("tokenizer cannot establish choice boundary without offsets")
        start, end = len(prompt_ids), len(full)
        if start == end:
            raise ValueError("choice produced no scoreable tokens")
        crossing = False
    if crossing:
        # A boundary-spanning BPE token also contains prompt text. Score the
        # continuation as the joint/prompt likelihood ratio, sharing context.
        prompt_encoded = tokenizer(prompt, add_special_tokens=True, truncation=False,
                                   return_offsets_mapping=True)
        prompt_offsets = prompt_encoded["offset_mapping"]
        prompt_tokens = list(prompt_encoded["input_ids"])
        real_positions = [i for i, (begin, finish) in enumerate(prompt_offsets) if finish > begin]
        prompt_end = real_positions[-1] + 1 if real_positions else 0
        prompt_tokens = prompt_tokens[:prompt_end]
        common = 0
        while common < min(end, len(prompt_tokens)) and full[common] == prompt_tokens[common]:
            common += 1
        longest_suffix = max(end - common, len(prompt_tokens) - common)
        if common < 1 or longest_suffix >= max_length:
            raise ValueError("choice cannot fit with a preceding context token")
        left = max(0, common - (max_length - longest_suffix))

        def suffix_logprob(ids, boundary):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids], dtype=torch.long)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            return sum(float(log_probs[pos - 1, ids[pos]]) for pos in range(boundary, len(ids)))

        return float(suffix_logprob(full[left:end], common - left)
                     - suffix_logprob(prompt_tokens[left:], common - left))
    if start < 1 or end - start >= max_length:
        raise ValueError("choice cannot fit with a preceding context token")
    left = max(0, end - max_length)
    full = full[left:end]
    start -= left
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([full], dtype=torch.long)).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(sum(
        float(log_probs[position - 1, full[position]])
        for position in range(start, len(full))
    ))


def classification_utility(model, tokenizer, examples: Iterable[dict], *, max_length: int = 512) -> dict:
    """0-shot classification interface; task templates come from the frozen manifest."""
    import torch

    hits = 0
    total = 0
    saw_choices = False
    for example in examples:
        prompt = example["prompt"]
        label = str(example["label"])
        choices = list(example.get("choices") or [])
        saw_choices = saw_choices or bool(choices)
        if choices:
            scores = [
                _choice_loglikelihood(model, tokenizer, prompt, choice, max_length=max_length)
                for choice in choices
            ]
            prediction = choices[int(np.argmax(scores))]
        else:
            ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]
            prediction = tokenizer.decode([int(torch.argmax(logits))])
        hits += int(prediction.strip() == label)
        total += 1
    return {
        "accuracy": (hits / total) if total else None,
        "numerator": hits,
        "denominator": total,
        "shots": 0,
        "scoring": "sum_loglikelihood" if saw_choices else "free_form",
    }
