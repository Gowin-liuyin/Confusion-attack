"""Inversion-Model Attack (IMA) family.

A learned inverter maps obfuscated representations back to plaintext tokens. The
cost is paid up front in training; at attack time inversion is a forward pass.
The attacker knows the obfuscation *algorithm* (it synthesises training pairs by
keying the algorithm itself) but never the client's secret key - see
``schemes/mechanism.py`` for the faithful key-handling model that the tests use.

* ``classify`` - InvBERT Classify (Kugler et al., **GB** attack): a shared
  per-position MLP token classifier over the obfuscated contextual embeddings,
  trained with cross-entropy. Supports the WB and GB knowledge settings only;
  BB is InvBERT Seq2Seq and returns ``not_applicable`` here (InvBERT Sec. 3.2).
* ``seq2seq`` - InvBERT Seq2Seq (Kugler et al., **BB** attack): a tiny
  Transformer decoder that consumes the whole-sentence CTEs as an *order-aware*
  memory (positional encoding on the memory rows) and predicts token ids over an
  attacker-side vocabulary, decoding to variable length until EOS (teacher
  forcing in training, greedy decode at attack time).
* ``qwen2_aloepri`` - AloePri Qwen2 inverter (App. D.1): a HuggingFace Qwen2
  backbone fixed at 2 decoder layers and 8 attention heads. The paper-experiment
  default inverts obfuscated embeddings to plaintext token *embeddings*
  (embedding regression; MSE loss is an implementation choice, not claimed as
  AloePri's); a cross-entropy token-index head is the clearly-labelled
  alternative (App. D.1 "or token indices").
"""

from . import classify, seq2seq, qwen2_aloepri

__all__ = ["classify", "seq2seq", "qwen2_aloepri"]
