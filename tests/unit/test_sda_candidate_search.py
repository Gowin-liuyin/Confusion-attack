import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obfuscation_attacks.attacks.sda.candidate_search import BertCandidateScorer, candidate_prefix_beam_search
from obfuscation_attacks.evaluation.sda_public import decode_candidate_observations, prepare_sda_observations
from obfuscation_attacks.schemes.adapters import Upload, UploadSurface


def test_candidate_beam_matches_exhaustive_search_when_beam_covers_all_prefixes():
    lattice = [[0, 1], [2, 3], [4, 5]]
    calls = []
    def score(batch):
        calls.append(len(batch))
        return [sum((3 ** i) * value for i, value in enumerate(row)) + 0.1 * len(row) for row in batch]
    result = candidate_prefix_beam_search(lattice, score, beam_width=8, batch_size=3)
    expected = sorted((list(tokens) for tokens in itertools.product(*lattice)),
                      key=lambda row: -score([row])[0])
    assert [row["tokens"] for row in result["beams"]] == expected
    assert result["tokens"] == expected[0]
    assert max(calls) <= 3
    assert [row["evaluated_prefixes"] for row in result["steps"]] == [2, 4, 8]


def test_candidate_search_ties_and_invalid_scorers_fail_closed():
    result = candidate_prefix_beam_search([[4, 2], [7, 5]], lambda rows: [0.0] * len(rows), beam_width=2)
    assert [row["tokens"] for row in result["beams"]] == [[4, 7], [4, 5]]
    with pytest.raises(ValueError, match="distinct"):
        candidate_prefix_beam_search([[1, 1]], lambda rows: [0.0] * len(rows), beam_width=1)
    with pytest.raises(ValueError, match="one score"):
        candidate_prefix_beam_search([[1]], lambda rows: [], beam_width=1)
    with pytest.raises(ValueError, match="finite"):
        candidate_prefix_beam_search([[1]], lambda rows: [float("nan")], beam_width=1)


def test_public_candidate_search_uses_nonfirst_candidate_without_truth():
    upload = Upload(surface=UploadSurface.VECTORS, request_id=7, seq_len=2, full_len=2,
                    payload=np.array([[0.1], [0.1]], dtype=np.float32), metadata={"truth": "not read"})
    observations = prepare_sda_observations([upload], vector_method="nearest_table",
                                             candidate_table=np.array([[0.], [2.]], dtype=np.float32),
                                             candidate_k=2)
    assert observations["rows"][0]["symbols"] == [0, 0]
    decoded = decode_candidate_observations(observations, lambda rows: [sum(row) for row in rows],
                                             beam_width=2, scorer_sha256="a" * 64,
                                             candidate_id_space="public_plaintext_token_ids")
    assert decoded["rows"][0]["decode"]["tokens"] == [1, 1]
    assert decoded["variant"] == "candidate_prefix_scoring_probe"
    assert decoded["candidate_table_semantics"] == "caller_supplied_unverified"
    assert "not read" not in str(decoded)
    with pytest.raises(ValueError, match="plaintext token IDs"):
        decode_candidate_observations(observations, lambda rows: [0] * len(rows), beam_width=2,
                                      scorer_sha256="a" * 64, candidate_id_space="deployment_rows")


def test_classifier_adapter_checks_capacity_and_scores_logit_difference():
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=3)
        def forward(self, input_ids, attention_mask):
            value = (input_ids * attention_mask).sum(dim=1).float()
            return SimpleNamespace(logits=torch.stack((-value, value), dim=1))
    def tokenizer(texts, **kwargs):
        width = max(len(text.split()) for text in texts)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, text in enumerate(texts):
            parts = [int(x) for x in text.split()]
            ids[i, :len(parts)] = torch.tensor(parts)
            mask[i, :len(parts)] = 1
        return {"input_ids": ids, "attention_mask": mask}
    scorer = BertCandidateScorer(Classifier(), tokenizer, lambda ids: " ".join(map(str, ids)))
    assert scorer([[1], [1, 2]]) == [2.0, 6.0]
    with pytest.raises(ValueError, match="no truncation"):
        scorer([[1, 2, 3, 4]])
