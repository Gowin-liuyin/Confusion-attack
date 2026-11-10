import pytest

from obfuscation_attacks.evaluation.real_sda import make_substitution_pairs, train_sda_transformer


def test_training_classes_do_not_depend_on_holdout_and_oov_never_counts_as_recovered():
    first = make_substitution_pairs([[1, 2, 1], [8, 9, 8]], vocab_size=20,
                                    held_out=1, n_classes=2)
    second = make_substitution_pairs([[1, 2, 1], [14, 15, 14]], vocab_size=20,
                                     held_out=1, n_classes=2)
    assert first["classes"].tolist() == second["classes"].tolist() == [1, 2]
    out = train_sda_transformer(first, held_out=1, epochs=1, batch=1,
                                d_model=4, n_heads=1, n_layers=1, verbose=False)
    assert out["token_count"] == 3
    assert out["token_accuracy"] == 0
    assert out["exact_match"] == 0
    assert out["class_coverage"] == 0
    assert out["bleu4_mean"] is None


def test_diagnostic_rejects_empty_or_mismatched_split():
    with pytest.raises(ValueError):
        make_substitution_pairs([[1]], vocab_size=2, held_out=1)
    data = make_substitution_pairs([[1], [2]], vocab_size=3, held_out=1)
    with pytest.raises(ValueError, match="split"):
        train_sda_transformer(data, held_out=2, verbose=False)
