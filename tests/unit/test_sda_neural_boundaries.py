import pytest
import torch

from obfuscation_attacks.attacks.sda._neural import (
    SinusoidalPositionalEncoding, length_matched_beam_search,
)


def test_positional_capacity_error_is_explicit():
    encoding = SinusoidalPositionalEncoding(4, 256)
    assert encoding(torch.zeros(1, 256, 4)).shape == (1, 256, 4)
    with pytest.raises(ValueError, match="positional capacity"):
        encoding(torch.zeros(1, 257, 4))


def test_beam_rejects_invalid_width_and_empty_finite_candidates():
    scores = lambda batch: torch.full((len(batch), 3), float("-inf"))
    with pytest.raises(ValueError, match="beam_width"):
        length_matched_beam_search(scores, 1, 0)
    with pytest.raises(ValueError, match="no finite candidates"):
        length_matched_beam_search(scores, 1, 2)


def test_beam_does_not_return_masked_candidates():
    scores = lambda batch: torch.tensor([[0.0, -1.0, -2.0]]).repeat(len(batch), 1)
    best, _, beams = length_matched_beam_search(scores, 1, 3, allowed_ids=torch.tensor([1]))
    assert best == [1]
    assert len(beams) == 1


def test_constrained_beam_scores_final_prefix_and_enforces_injection():
    def scores(batch):
        rows = []
        for seq in batch.tolist():
            logits = torch.tensor([2.0, 1.0, -1.0]) + (0.4 if seq and seq[-1] == 1 else 0) * torch.tensor([0.0, 1.0, 0.0])
            rows.append(torch.log_softmax(logits, dim=0))
        return torch.stack(rows)

    symbols = [10, 20, 10, 20]
    tokens, score, _ = length_matched_beam_search(
        scores, 4, 3, cipher_symbols=symbols, injective=True,
    )
    assert tokens[0] == tokens[2] and tokens[1] == tokens[3]
    assert tokens[0] != tokens[1]
    expected = sum(float(scores(torch.tensor([tokens[:i]], dtype=torch.long))[0, t])
                   for i, t in enumerate(tokens))
    assert score == pytest.approx(expected)
    homophones, _, _ = length_matched_beam_search(
        scores, 4, 1, cipher_symbols=symbols, injective=False,
    )
    assert homophones == [0, 0, 0, 0]


def test_vectorized_beam_matches_reference_with_ties_and_constraints():
    import math
    def reference(step, symbols, width, injective):
        beams = [([], 0.0)]
        for pos, symbol in enumerate(symbols):
            batch = torch.tensor([tokens for tokens, _ in beams], dtype=torch.long)
            logp = step(batch)
            masked = torch.full_like(logp, float("-inf"))
            masked[:, [0, 1, 2, 3]] = logp[:, [0, 1, 2, 3]]
            for row, (tokens, _) in enumerate(beams):
                mapping = {c: t for c, t in zip(symbols[:pos], tokens)}
                if symbol in mapping:
                    value = masked[row, mapping[symbol]].clone()
                    masked[row].fill_(float("-inf"))
                    masked[row, mapping[symbol]] = value
                elif injective and mapping:
                    masked[row, list(mapping.values())] = float("-inf")
            values, indices = torch.topk(masked, min(width, masked.shape[-1]), dim=-1)
            choices = []
            for row, (tokens, score) in enumerate(beams):
                for value, index in zip(values[row].tolist(), indices[row].tolist()):
                    if math.isfinite(value):
                        choices.append((tokens + [int(index)], score + float(value)))
            choices.sort(key=lambda item: item[1], reverse=True)
            beams = choices[:width]
        return beams

    def step(batch):
        scores = []
        for prefix in batch.tolist():
            offset = (sum(prefix) + len(prefix)) % 3
            scores.append(torch.tensor([0.0, -1.0, -1.0, -2.0, -4.0]).roll(offset))
        return torch.stack(scores)
    for symbols in ([8, 4, 8, 3, 4], [8, 4, 3, 8, 4, 8, 3, 4] * 8):
        for width in (1, 2, 4):
            for injective in (False, True):
                _, _, actual = length_matched_beam_search(
                    step, len(symbols), width, allowed_ids=torch.arange(4),
                    cipher_symbols=symbols, injective=injective)
                assert actual == reference(step, symbols, width, injective)


@pytest.mark.parametrize("injective", [True, False])
def test_segmented_beam_matches_full_prefix_search(injective):
    symbols = [9, 8, 9, 8, 7, 9, 7, 8, 9]
    lengths = [3, 2, 4]
    starts = [0, 3, 5]
    seen_lengths = []
    def local_score(batch, position):
        seen_lengths.append((position, batch.shape[1]))
        rows = []
        for seq in batch.tolist():
            offset = (sum(seq) + position) % 3
            rows.append(torch.tensor([0.0, -1.0, -1.0, -2.0]).roll(offset))
        return torch.stack(rows)
    def full_score(batch):
        position = batch.shape[1]
        start = max(x for x in starts if x <= position)
        return local_score(batch[:, start:], position)
    expected = length_matched_beam_search(full_score, len(symbols), 4,
                                         cipher_symbols=symbols, injective=injective)
    seen_lengths.clear()
    actual = length_matched_beam_search(local_score, len(symbols), 4, cipher_symbols=symbols,
                                       injective=injective, sequence_lengths=lengths)
    assert actual == expected
    assert seen_lengths == [(position, position - max(x for x in starts if x <= position))
                            for position in range(len(symbols))]
    with pytest.raises(ValueError, match="sequence_lengths"):
        length_matched_beam_search(local_score, len(symbols), 4, cipher_symbols=symbols,
                                   sequence_lengths=[3, 2])


def test_segmented_beam_limits_scorer_prefix_on_many_requests():
    symbols = [9, 8, 9, 8] * 128
    maximum = 0
    def score(batch, position):
        nonlocal maximum
        maximum = max(maximum, batch.shape[1])
        return torch.tensor([[0.0, -1.0, -2.0]]).repeat(len(batch), 1)
    tokens, _, beams = length_matched_beam_search(score, len(symbols), 2, cipher_symbols=symbols,
                                                  injective=True, sequence_lengths=[4] * 128)
    assert len(tokens) == 512 and len(beams) == 2
    assert maximum == 3
    assert tokens == [tokens[0], tokens[1], tokens[0], tokens[1]] * 128
