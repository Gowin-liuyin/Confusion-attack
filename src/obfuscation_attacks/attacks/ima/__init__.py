"""Inversion-Model Attack (IMA) family.

A learned inverter maps obfuscated representations back to plaintext tokens. The
cost is paid up front in training; at attack time inversion is a forward pass.

* ``classify`` - InvBERT Classify (Kugler et al., GB line): a shared per-position
  MLP token classifier over the obfuscated contextual embeddings, trained with
  cross-entropy. Exposes the WB/GB/BB knowledge settings, which differ only in
  how the training pairs are obtained.
* ``seq2seq`` - InvBERT Seq2Seq (Kugler et al., BB line): a tiny Transformer
  decoder that consumes the whole-sentence CTEs as memory and predicts token ids
  sequentially (teacher forcing in training, greedy decode at attack time).
* ``qwen2_aloepri`` - AloePri Qwen2 inverter (App. D.1): a HuggingFace Qwen2
  backbone fixed at 2 decoder layers and 8 attention heads, inverting obfuscated
  embeddings to plaintext token indices (cross-entropy), with an explicitly
  optional embedding-regression (MSE) output path.
"""

from . import classify, seq2seq, qwen2_aloepri

__all__ = ["classify", "seq2seq", "qwen2_aloepri"]
