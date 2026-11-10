# Bibliography

> Technical reference only. Experimental authority: [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md). Historical parameters/results do not define current permissions, budgets or acceptance.

## Target papers (the two whose experiments define the seven attacks)

- **AloePri** - *Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)*.
  Yu Lin, Qizhi Zhang, Wenqiang Ruan, Daode Zhang, Jue Hong, Ye Wu. Local id `2603.01499v2`.
  Source of the AloePri-adapted attacks: weight VMA (App. D.1, Table 9), Gate-IA / Attn-IA (App. D.1),
  ISA State1/State2 (App. D.1), Qwen2 IMA inverter (App. D.1), TFMA (Sec. 7.6), SDA (Sec. 7.6 / App. D.1),
  and the NN baseline (Sec. 7.2, Table 2). Hyperparameters in App. D.2.
- **OSNIP** - *OSNIP: Balancing the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated
  Semantic Null Space*. Zhiyuan Cao, Zeyu Ma, Chenhao Yang, Han Zheng, Mingang Chen.
  [ICML 2026, PMLR 306:11433–11455](https://proceedings.mlr.press/v306/cao26q.html).
  The earlier arXiv version is `2601.22752v1` (title used “Breaking”).
  Source of the OSNIP-adapted attacks: KNN (App. C.1/Table 6) and VMA-H (App. C.2/Table 7).

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

## Additional papers for the 14-class pass (plan classes C02/C04/C07/C12/C13/C14)

- **BeamClean** (C07, the attack) - Kaan Kale, Kyle Mylonakis, Jay Roberts, Sidhartha Roy.
  *BeamClean: Language Aware Embedding Reconstruction*. arXiv:2505.13758 (2025). Verified via the
  arXiv abstract page: adversary has no LM and no obfuscation mechanism, sees the obfuscated
  embeddings plus the public embedding table; jointly estimates the noise parameters and decodes
  with a language-model prior.
- **Stained Glass Transform** (C07, the *defense* being attacked) - Jay Roberts, Kyle Mylonakis,
  Sidhartha Roy, Kaan Kale. *Learning Obfuscations Of LLM Embedding Sequences: Stained Glass
  Transform*. arXiv:2506.09452 (2025). "SGT" in the plan is this defense, **not** an attack paper.
- **DeepInvert** (C02) - Zhicong Huang, Cheng Hong, Tao Wei. *DeepInvert: Semi-Supervised Embedding
  Inversion Against Obfuscated Language Models*. arXiv:2608.04477 (2026). Verified.
- **Papillon** (C13, AloePri ref. [14]) - Siyan Li, Vethavikashini Chithrra Raghuram, Omar Khattab,
  Julia Hirschberg, Zhou Yu. *Papillon: Privacy Preservation from Internet-based and Local Language
  Model Ensembles*. NAACL 2025. **PUPA is the dataset in this paper, not an attack paper.**
- **InferDPT** (C14) - *Privacy-Preserving Inference for Closed-box Large Language Model* (TDSC 2025;
  official repo `mengtong0110/InferDPT`). Source of the n-gram / CosSim / Top-K reconstruction
  metrics; the GPT-4 reconstruction is its strongest variant.
- **C12 linkage** and **C04 cross-checkpoint transfer** have **no source paper**: the plan declares
  them as this project's extensions (§4 P0-9, §4.17). The plan's `arXiv:2610.27817` attribution for C04
  was checked and is **wrong** (that paper is a learned input-embedding-transform *defense*); see
  `docs/ATTACK_PROVENANCE_14.md` F1.

## Attribution

Paper PDFs and extracted text live under `F:\Learn\Learning\混淆攻击\` (read-only, external to this
repository). The OSNIP PMLR and AloePri technical-report sources are also mirrored under
`F:\OSNIP_package_20261101\references\papers\` and `...\references\aloepri_code_20261103\`, which is
what `docs/ATTACK_PROVENANCE_14.md` §3 quotes. TFMA has no standalone originating paper; it is treated
here as AloePri's classical frequency-analysis baseline (Sec. 7.6), not attributed to a fabricated source.
