"""Unit tests for the six plan classes added in the 14-class pass.

Scope (plan ``privacy_test_plan_20261106`` §4 / orchestration doc §0):

* C02 ``ima.cross_instance``  - KI: pairs minted by the attacker's *own* instance;
* C04 ``ima.cross_checkpoint``- KI transfer across checkpoints, and the KI/KP
  boundary that makes the number mean anything (D8 / A5);
* C07 ``beamclean.*``         - LM-prior decoding with a naive-NN control;
* C12 ``linkage.*``           - SH-P vs SH-B on a rotating request stream;
* C13 ``pii.extract``         - PIIRSR with the "no entity -> null" denominator;
* C14 ``lm_recon.*``          - output-side n-gram metrics and the honest
  ``not_applicable`` when no generative handle exists.

These pin the *protocol* (which arm sees what, which control must pass, which
denominator is used). They deliberately do not claim paper-scale numbers: those
are blocked on hardware and recorded as such in ``docs/RESOURCE_REQUIREMENTS.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.data.synthetic_privacy import build_cross_checkpoint_scenario
from obfuscation_attacks.evaluation.runner import run_scenario

pytestmark = pytest.mark.unit


def _run(attack: str, scenario: str, **kwargs) -> dict:
    return run_scenario(attack, scenario, **kwargs)


# ---------------------------------------------------------------------------
# C13 PII extraction
# ---------------------------------------------------------------------------
def test_pii_plaintext_control_recovers_every_entity():
    """Plan §7.1: with plaintext output the extractor must recover the entities.

    The plaintext arm exposes no DETERMINISTIC_PERMUTATION capability, so the
    attack reads the stream as tokens instead of forcing a frequency decode onto
    a stream that carries no substitution.
    """
    out = _run("pii.extract", "pii", scenario_params={"identity": True})
    metrics = out["metrics"]
    assert metrics["status"] == "ok"
    assert metrics["pii"]["pii_rsr"] == pytest.approx(1.0)
    assert metrics["pii"]["n_entities_recovered"] == metrics["pii"]["n_entities"] > 0
    assert metrics["diagnostics"]["method"].startswith("plaintext wire")


def test_pii_permuted_arm_uses_the_substitution_decode():
    out = _run("pii.extract", "pii", scenario_params={"n_entities": 3})
    metrics = out["metrics"]
    assert metrics["status"] == "ok"
    assert metrics["diagnostics"]["method"].startswith("frequency_decode")
    # entity-level recovery is all-or-nothing, so it cannot silently inherit the
    # token-level rate of the decode
    assert metrics["pii"]["pii_rsr"] <= metrics["token_recovery"]["ttrsr"] + 1e-12


def test_pii_without_entities_is_null_not_zero():
    """Plan §5.4 formula (3): no annotated entity -> ``null``, never ``0%``."""
    out = _run("pii.extract", "pii", scenario_params={"n_entities": 0})
    assert out["metrics"]["pii"]["pii_rsr"] is None


def test_frequency_decode_does_not_compare_different_id_spaces():
    from obfuscation_attacks.attacks._decode import frequency_decode
    from obfuscation_attacks.data.contracts import AttackContext, Capability

    ctx = AttackContext(
        granted=frozenset({Capability.TOKEN_CIPHERTEXT}),
        descriptor={"vocab_size": 10},
        _ciphertext_token_ids=np.asarray([7, 8, 9], dtype=int),
        reference_token_counts={0: 5, 1: 3},
    )
    out = frequency_decode(ctx, top_k=4)
    assert out["coverage"] is None
    assert out["reference_supported_types"] == 2
    assert out["num_observed_symbols"] == 3
    bad = AttackContext(
        granted=frozenset({Capability.TOKEN_CIPHERTEXT}),
        descriptor={"vocab_size": 10},
        _ciphertext_token_ids=np.asarray([10], dtype=int),
        reference_token_counts={0: 5},
    )
    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        frequency_decode(bad, top_k=4)


# ---------------------------------------------------------------------------
# C14 output-side reconstruction
# ---------------------------------------------------------------------------
def test_lm_recon_plaintext_control_is_exact():
    out = _run("lm_recon.ngram", "lm_recon", scenario_params={"identity": True})
    rec = out["metrics"]["output_reconstruction"]
    assert rec["ngram_overlap"]["1-gram"] == pytest.approx(1.0)
    assert rec["exact_sequence_match"] == 1
    assert rec["bleu4"] == pytest.approx(1.0)


def test_lm_recon_permuted_arm_reports_paper_metrics():
    out = _run("lm_recon.ngram", "lm_recon")
    rec = out["metrics"]["output_reconstruction"]
    for key in ("1-gram", "2-gram", "3-gram", "4-gram"):
        assert 0.0 <= rec["ngram_overlap"][key] <= 1.0
    assert "token_embedding_cosine_proxy" in rec and "bleu4" in rec


def test_local_lm_substitute_is_not_applicable_without_generative_handle():
    """No GPT-4 / local generative backend -> no number (never a fabricated 0%)."""
    out = _run("lm_recon.local_lm", "lm_recon")
    assert out["metrics"]["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# C12 cross-request linkage
# ---------------------------------------------------------------------------
def test_linkage_sh_p_and_sh_b_use_the_same_pair_universe():
    params = {"key_size": 4, "rotation_n": 8, "n_requests": 24}
    shp = _run("linkage.exact", "linkage", scenario_params=params)
    shb = _run(
        "linkage.exact", "linkage", scenario_params={**params, "metadata_visible": False}
    )
    assert shp["result"].diagnostics["attacker_knowledge"] == "SH-P"
    assert shb["result"].diagnostics["attacker_knowledge"] == "SH-B"
    assert shp["result"].diagnostics["rotation_aware"] is True
    assert shb["result"].diagnostics["rotation_aware"] is False
    assert shp["metrics"]["linkage"]["link_rate"] == shb["metrics"]["linkage"]["link_rate"]
    assert shp["result"].stream_metrics["total_pairs"] == shb["result"].stream_metrics["total_pairs"]


def test_linkage_single_key_positive_control_is_one():
    out = _run(
        "linkage.exact", "linkage",
        scenario_params={"key_size": 1, "n_requests": 12, "rotation_n": 64},
    )
    assert out["metrics"]["linkage"]["link_rate"] == pytest.approx(1.0)


def test_linkage_approximate_reports_tpr_and_fpr_per_threshold():
    out = _run(
        "linkage.approximate", "linkage",
        attack_params={"max_queries": 20},
        scenario_params={"key_size": 4, "rotation_n": 8, "n_requests": 16},
    )
    sweep = out["metrics"]["linkage"]["threshold_sweep"]
    assert [row["threshold"] for row in sweep] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for row in sweep:
        assert row["tpr"] is not None and row["fpr"] is not None
    # TPR must be monotone non-increasing in the threshold (FPR too): a sweep
    # that violates this would mean the scores/labels are misaligned.
    tprs = [row["tpr"] for row in sweep]
    assert tprs == sorted(tprs, reverse=True)


# ---------------------------------------------------------------------------
# C07 BeamClean
# ---------------------------------------------------------------------------
def test_beamclean_naive_control_is_exact_on_clean_wire():
    out = _run("beamclean.gaussian", "beamclean", scenario_params={"noise": 0.0})
    assert out["metrics"]["token_recovery"]["asr@1"] == pytest.approx(1.0)
    assert out["result"].diagnostics["lm_prior"] == "bigram_reference"
    assert out["result"].diagnostics["naive_nn_agreement"] == pytest.approx(1.0)


def test_beamclean_estimates_noise_and_beats_chance_under_noise():
    out = _run("beamclean.gaussian", "beamclean", scenario_params={"noise": 1.2})
    diagnostics = out["result"].diagnostics
    assert diagnostics["noise_scale"] > 0.0
    assert len(diagnostics["noise_scale_history"]) >= 1  # joint (EM) estimate ran
    assert out["metrics"]["token_recovery"]["asr@1"] > 1.0 / 24  # > chance (vocab 24)


def test_beamclean_laplacian_uses_the_laplacian_likelihood():
    out = _run(
        "beamclean.laplacian", "beamclean",
        scenario_params={"noise": 1.2, "noise_model": "laplacian"},
    )
    assert out["result"].diagnostics["noise_model"] == "laplacian"
    assert out["metrics"]["status"] == "ok"


# ---------------------------------------------------------------------------
# C02 / C04 inversion, and the KI/KP boundary
# ---------------------------------------------------------------------------
def test_cross_instance_inverter_mints_its_own_pairs():
    out = _run("ima.cross_instance", "ima_cross_instance", attack_params={"epochs": 60})
    diagnostics = out["result"].diagnostics
    assert diagnostics["uses_deployment_key"] is False
    assert diagnostics["training_pair_source"] == "attacker_own_instance"
    assert diagnostics["in_instance_accuracy"] >= 0.95  # plan §6.5 control
    assert diagnostics["positive_control"]["passed"] is True


def test_cross_checkpoint_attacker_shares_no_deployment_secret():
    """C04 must vary table *and* secret; sharing the secret would be KP (D8/A5)."""
    scenario = build_cross_checkpoint_scenario(seed=0)
    instances = scenario.sample.meta["instances"]
    dep, atk = instances["deployment"], instances["attacker"]
    assert not np.allclose(dep["table"], atk["table"])  # different checkpoint
    assert not np.array_equal(dep["perm"], atk["perm"])  # own permutation
    assert not np.allclose(dep["matrix"], atk["matrix"])  # own mechanism matrix
    # and the two instances really do produce different observations
    ids = np.asarray(scenario.sample.plaintext_token_ids)
    assert not np.allclose(
        np.asarray(scenario.sample.obfuscated_embeddings),
        np.asarray(scenario.attacker_obfuscator(ids)),
    )


def test_cross_checkpoint_attack_keeps_the_same_control_floor():
    out = _run("ima.cross_checkpoint", "cross_checkpoint", attack_params={"epochs": 60})
    diagnostics = out["result"].diagnostics
    assert diagnostics["uses_deployment_key"] is False
    assert diagnostics["positive_control"]["passed"] is True
    # a transfer attack that needs the deployment key would be the KP appendix;
    # this one is trained only on its own instance
    assert diagnostics["instance_kind"] == "ki_other_checkpoint"


# ---------------------------------------------------------------------------
# C07 BeamClean: vectorized row-prior equivalence and scale behaviour
# ---------------------------------------------------------------------------
def _reference_beam_decode(Z, E, logprob, *, kind, scale, beam, lambda_lm, candidate_topk):
    """Literal transcription of the former per-candidate loop (the semantics the
    vectorized :func:`beam_decode` must reproduce exactly, ties included)."""
    from obfuscation_attacks.attacks.beamclean.core import noise_loglik

    n, V = Z.shape[0], E.shape[0]
    ll = noise_loglik(Z, E, scale, kind)
    pruned = candidate_topk is not None and int(candidate_topk) < V
    beams = [(0.0, [], [])]
    for i in range(n):
        if pruned:
            step_cand = np.argsort(-ll[i])[: int(candidate_topk)]
            step_cand.sort()
        else:
            step_cand = np.arange(V)
        expanded = []
        for score, toks, cands in beams:
            for v in step_cand.tolist():
                new_score = (score + float(ll[i, v])
                             + (float(lambda_lm) * logprob(toks, v) if lambda_lm else 0.0))
                expanded.append((new_score, toks + [int(v)], cands))
        expanded.sort(key=lambda t: t[0], reverse=True)
        beams = expanded[: max(1, int(beam))]
    best_score, best_tokens, _ = beams[0]
    candidates = []
    for i in range(n):
        prefix = best_tokens[:i]
        if pruned:
            pool = np.argsort(-ll[i])[: int(candidate_topk)]
            pool.sort()
        else:
            pool = np.arange(V)
        scores = np.array([float(ll[i, v]) + (float(lambda_lm) * logprob(prefix, v)
                                              if lambda_lm else 0.0)
                           for v in pool.tolist()])
        order = pool[np.argsort(-scores, kind="stable")]
        candidates.append(order.tolist())
    return (best_tokens, best_score), candidates


def test_beamclean_vectorized_beam_matches_the_per_candidate_reference():
    from obfuscation_attacks.attacks.beamclean.core import (
        beam_decode, bigram_logprob_row, fit_bigram_lm, ngram_logprob,
    )

    for seed in range(3):
        rng = np.random.default_rng(seed)
        V, d, n = 40, 8, 12
        E = rng.normal(size=(V, d))
        Z = E[rng.integers(0, V, size=n)] + 0.3 * rng.normal(size=(n, d))
        corpus = [rng.integers(0, V, size=30).tolist() for _ in range(6)]
        lm = fit_bigram_lm(corpus, V)
        row = lambda prefix: bigram_logprob_row(lm, prefix)
        scalar = lambda prefix, token: ngram_logprob(lm, prefix, token)
        for kind in ("gaussian", "laplacian"):
            for beam in (1, 4):
                for lambda_lm in (0.0, 0.7, 1.0):
                    for topk in (None, 10):
                        got = beam_decode(Z, E, row, kind=kind, scale=0.4, beam=beam,
                                          lambda_lm=lambda_lm, candidate_topk=topk)
                        want = _reference_beam_decode(
                            Z, E, scalar, kind=kind, scale=0.4, beam=beam,
                            lambda_lm=lambda_lm, candidate_topk=topk)
                        assert got[0].tokens == want[0][0]
                        assert got[0].score == pytest.approx(want[0][1], abs=1e-9)
                        assert got[1] == want[1]


def test_beamclean_neural_prior_runs_once_per_prefix_not_per_candidate():
    from obfuscation_attacks.attacks.beamclean.core import beam_decode

    calls = {"n": 0}

    class Handle:
        def next_token_logprobs(self, prefix):
            calls["n"] += 1
            table = np.full(50, -np.log(50))
            if prefix:
                table[int(prefix[-1])] = -0.1  # slight self-repeat preference
            return table

    rng = np.random.default_rng(0)
    E = rng.normal(size=(50, 6))
    Z = E[rng.integers(0, 50, size=6)] + 0.2 * rng.normal(size=(6, 6))
    result, candidates = beam_decode(Z, E, lambda p: np.asarray(Handle().next_token_logprobs(p)),
                                     kind="gaussian", scale=0.3, beam=4, lambda_lm=1.0)
    # One distribution per (beam, position) during the search plus one per
    # position for the winner's candidate lists - never one per candidate.
    assert calls["n"] <= 6 * 4 + 6
    assert len(result.tokens) == 6 and all(len(c) == 50 for c in candidates)


def test_beamclean_full_vocabulary_mode_is_feasible_at_real_scale():
    """The plan's main-table mode (full 128,256-entry vocabulary, no proposal
    pruning) must complete in seconds, not per-candidate Python time."""
    import time as _time

    from obfuscation_attacks.attacks.beamclean.core import (
        beam_decode, bigram_logprob_row, fit_bigram_lm,
    )

    rng = np.random.default_rng(0)
    V, d, n = 128256, 256, 24
    E = rng.normal(size=(V, d)).astype(np.float32)
    Z = (E[rng.integers(0, V, size=n)] + 0.05 * rng.normal(size=(n, d))).astype(np.float32)
    lm = fit_bigram_lm([rng.integers(0, 1000, size=64).tolist() for _ in range(8)], V)
    t0 = _time.time()
    result, candidates = beam_decode(Z, E, lambda p: bigram_logprob_row(lm, p),
                                     kind="gaussian", scale=0.05, beam=8, lambda_lm=1.0)
    elapsed = _time.time() - t0
    assert len(result.tokens) == n
    assert all(len(c) == V for c in candidates)
    assert elapsed < 60.0, f"full-vocabulary decode took {elapsed:.1f}s"
