# Bibliography

## Target papers (the two whose experiments define the seven attacks)

- **AloePri** - *Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)*.
  Yu Lin, Qizhi Zhang, Wenqiang Ruan, Daode Zhang, Jue Hong, Ye Wu. Local id `2603.01499v2`.
  Source of the AloePri-adapted attacks: weight VMA (App. D.1, Table 9), Gate-IA / Attn-IA (App. D.1),
  ISA State1/State2 (App. D.1), Qwen2 IMA inverter (App. D.1), TFMA (Sec. 7.6), SDA (Sec. 7.6 / App. D.1),
  and the NN baseline (Sec. 7.2, Table 2). Hyperparameters in App. D.2.
- **OSNIP** - *OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated
  Semantic Null Space*. Zhiyuan Cao, Zeyu Ma, Chenhao Yang, Han Zheng, Mingang Chen. Local id `2601.22752v1`.
  Source of the OSNIP-adapted attacks: KNN (App. C.1) and the projection VMA (App. C.2, Table 6).

## Original / underlying papers

- **NN / KNN** - Congzheng Song, Ananth Raghunathan. *Information Leakage in Embedding Models*. CCS 2020.
  arXiv:2004.00053.
- **VMA** - Rahul Thomas, Louai Zahran, Erica Choi, Akilesh Potti, Micah Goldblum, Arka Pal.
  *Hidden No More: Attacking and Defending Private Third-Party LLM Inference*. ICML 2025. arXiv:2505.18332.
- **IA / EDNN** - Yu Lin et al. *An Inversion Attack Against Obfuscated Embedding Matrix in Language Model
  Inference*. EMNLP 2024. ACL Anthology 2024.emnlp-main.126.
- **ISA** - Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu.
  *Depth Gives a False Sense of Privacy: LLM Internal States Inversion*. USENIX Security 2025. arXiv:2507.16372.
- **IMA** - Kai Kugler, Simon Munker, Johannes Hohmann, Achim Rettinger.
  *InvBERT: Reconstructing Text from Contextualized Word Embeddings by Inverting the BERT Pipeline*. arXiv:2109.10104.
- **SDA 2021** - Nada Aldarrab, Jonathan May. *Can Sequence-to-Sequence Models Crack Substitution Ciphers?*
  ACL 2021. 2021.acl-long.561.
- **SDA 2023** - Nishant Kambhatla, Logan Born, Anoop Sarkar. *Decipherment as Regression: Solving Historical
  Substitution Ciphers by Learning Symbol Recurrence Relations*. Findings of EACL 2023. 2023.findings-eacl.160.

## Attribution

Paper PDFs and extracted text live under `F:\Learn\Learning\混淆攻击\` (read-only, external to this
repository). TFMA has no standalone originating paper; it is treated here as AloePri's classical
frequency-analysis baseline (Sec. 7.6), not attributed to a fabricated source.
