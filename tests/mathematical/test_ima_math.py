"""Mathematical tests pinning the IMA paper formulae to code.

(a) hand-computed cross-entropy == torch CE (IMA-CE-1);
(b) the classify softmax formula == manual softmax (IMA-CLS-1), and softmax at
    the target == exp(-CE), tying IMA-CLS-1 to IMA-CE-1;
(c) Qwen2 config assertion: num_hidden_layers==2, num_attention_heads==8 (and
    num_key_value_heads==8, an UNSPECIFIED assumption), and overriding them raises;
(d) the embedding-path loss equals the written MSE formula (IMA-MSE). This MSE is
    the loss used to train the paper-experiment embedding path - an implementation
    choice - not a loss AloePri claims to use.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from obfuscation_attacks.attacks.ima.classify import (
    PerPositionClassifier,
    classify_probabilities,
    cross_entropy_objective,
)


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@pytest.mark.mathematical
@pytest.mark.neural
def test_cross_entropy_matches_hand_computed():
    """L = - sum_t log p_theta(y_t | context) computed by hand == torch CE."""
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    targets = torch.tensor([0, 2])

    npl = logits.numpy()
    per_example = [
        -np.log(_stable_softmax(npl[i])[t]) for i, t in enumerate(targets.numpy())
    ]
    manual_sum = float(np.sum(per_example))
    manual_mean = float(np.mean(per_example))

    assert np.isclose(
        float(cross_entropy_objective(logits, targets, "sum")), manual_sum, atol=1e-6
    )
    assert np.isclose(
        float(cross_entropy_objective(logits, targets, "mean")), manual_mean, atol=1e-6
    )


@pytest.mark.mathematical
@pytest.mark.neural
def test_classify_softmax_matches_manual():
    """p_theta = softmax(logits); softmax at target == exp(-CE)."""
    logits = torch.tensor([1.0, 2.0, 0.5, -0.3])
    probs = classify_probabilities(logits).numpy()
    manual = _stable_softmax(logits.numpy())

    assert np.allclose(probs, manual, atol=1e-7)
    assert np.isclose(probs.sum(), 1.0, atol=1e-7)

    target = torch.tensor([2])
    ce = float(cross_entropy_objective(logits.unsqueeze(0), target, "sum"))
    assert np.isclose(probs[2], np.exp(-ce), atol=1e-6)


@pytest.mark.mathematical
@pytest.mark.neural
def test_classify_formula_matches_model_structure():
    """classify_probabilities(model(h)) == softmax(W MLP_theta(h) + b)."""
    model = PerPositionClassifier(input_dim=5, vocab_size=4, hidden_dims=(8,), seed=0).eval()
    h = torch.randn(3, 5)
    with torch.no_grad():
        structured = torch.softmax(model.head(model.mlp(h)), dim=-1).numpy()
        via_api = classify_probabilities(model(h)).numpy()
    assert np.allclose(structured, via_api, atol=1e-7)


@pytest.mark.mathematical
@pytest.mark.neural
def test_qwen2_config_layers_heads():
    """AloePri inverter is fixed at 2 decoder layers / 8 heads; overrides raise.

    num_hidden_layers and num_attention_heads are paper values (App. D.1);
    num_key_value_heads==8 is an UNSPECIFIED assumption (= attention heads) and
    is pinned the same way so the architecture is fully determined.
    """
    from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter

    inv = Qwen2AloePriInverter()
    model = inv.build_model(obf_dim=6, vocab_size=16)
    assert model.config.num_hidden_layers == 2
    assert model.config.num_attention_heads == 8
    assert model.config.num_key_value_heads == 8  # assumption, not a paper value

    with pytest.raises(ValueError):
        Qwen2AloePriInverter(num_hidden_layers=3)
    with pytest.raises(ValueError):
        Qwen2AloePriInverter(num_attention_heads=4)
    with pytest.raises(ValueError):
        Qwen2AloePriInverter(num_key_value_heads=2)


@pytest.mark.mathematical
@pytest.mark.neural
def test_qwen2_embedding_path_loss_matches_mse_formula():
    """Embedding-path loss: L_MSE = (1/N) sum ||e_hat_t - e_t||^2 (per-row sq. L2, averaged).

    MSE is the implementation-choice loss for the paper-experiment embedding path;
    AloePri specifies the embedding *target*, not this loss.
    """
    from obfuscation_attacks.attacks.ima.qwen2_aloepri import mse_regression_objective

    e_hat = torch.tensor([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
    e = torch.tensor([[1.5, 1.0], [0.0, 0.0], [2.0, 1.0]])
    manual = float(np.mean(np.sum((e_hat.numpy() - e.numpy()) ** 2, axis=1)))
    assert np.isclose(float(mse_regression_objective(e_hat, e)), manual, atol=1e-6)
