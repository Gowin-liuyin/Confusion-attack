# 14 类攻击的出处论文与实现对照（已核对）

> Technical reference only. Experimental authority: [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md). Historical parameters/results do not define current permissions, budgets or acceptance.

> 用途：回答「每类攻击到底出自哪篇论文、论文里到底写了什么、仓库里哪段代码对应它、哪条测试钉住它」。
> 本文件只记录**已核对**的出处；核对不到的一律标 `UNVERIFIED`，不编 DOI、不编作者。
> 本文件只保留出处和公式/代码映射，不定义实验口径。现行规则见 [唯一实验协议](PAPER_EVALUATION_PROTOCOL.md)；文中“导向文档”及旧章节编号指历史核对对象，不是该协议的章节。
> 核对方式见 §3（含可复跑命令与抽出的原文行号）。

## 0. 一句话结论

14 类里有 **11 类**能落到具体论文的原式（KNN、ISA、VMA-H、VMA-W、Gate-IA、Attn-IA、IMA、TFMA、SDA、BeamClean、学习式反演），
**2 类是本方案扩展**（C12 链接、C04 跨 checkpoint 迁移，后者被导向文档挂了一篇**内容不符**的 arXiv 编号，见 §2-F1），
**1 类只有数据集、没有攻击论文**（C13 PII/PUPA，见 §2-F3）。
另有 1 处口径混淆已改正：导向文档把 **SGT**（防御）当成 C07 的攻击出处，实际攻击论文是 **BeamClean**（§2-F2）。

## 1. 逐类对照表

| 类 | 攻击（注册名） | 出处论文（已核对） | 论文锚点 | 仓库实现 | 钉住它的测试 |
|---|---|---|---|---|---|
| C01 | KNN / NN（`nn_knn.euclidean`、`nn_knn.cosine`） | Song & Raghunathan, *Information Leakage in Embedding Models*, CCS 2020, arXiv:2004.00053；OSNIP App. C.1 / Table 6 | Alpaca instruction+input，20–512 字符，随机 1,000 条；L2/cosine；ASR@Top-1/5/10；Llama-3.2-1B OSNIP = 5.17/6.37/6.61% | `attacks/nn_knn/` | `tests/mathematical/test_nn_math.py`、`scripts/privacy_family_gates.py`（C01 明文臂 = 1.0） |
| C02 | 学习式嵌入反演（`ima.cross_instance`） | Kugler et al., *InvBERT: Reconstructing Text from Contextualized Word Embeddings by Inverting the BERT Pipeline*, arXiv:2109.10104；DeepInvert, arXiv:2608.04477（半监督嵌入反演，2026-08，已核标题/摘要） | 攻击者用**自己的机制实例**造配对训练反演器（KI），不使用部署密钥；域内正控 ≥95% | `attacks/ima/cross_instance.py` | `tests/unit/test_privacy_families_14.py::test_cross_instance_inverter_mints_its_own_pairs` |
| C03 | Vec2Text（`vec2text.iterative`） | Morris et al., *Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023, arXiv:2310.06816 | 迭代「生成假设 → 再嵌入 → 修正」的文本级反演 | **未实现**（导向 §4.15 定为 P2/仅引用） | `attack_model.py` 中 `implemented=False`；测试断言不得凭空出数（见 `test_attack_model` 覆盖注册表） |
| C04 | 跨 checkpoint / 迁移（`ima.cross_checkpoint`） | **本方案扩展**；导向文档原挂 arXiv:2610.27817，经核对内容不符 → 见 §2-F1 | 同架构、不同 checkpoint 训练反演器，迁移到部署 checkpoint；同 checkpoint 正控 ≥95% | `attacks/ima/cross_instance.py`（`CrossCheckpointInverter`） | `test_cross_checkpoint_attacker_shares_no_deployment_secret`、`test_cross_checkpoint_attack_keeps_the_same_control_floor` |
| C05 | ISA（`isa.er`、`isa.tbs`、`isa.aloepri_state`） | Dong et al., *Depth Gives a False Sense of Privacy: LLM Internal States Inversion*, USENIX Security 2025, arXiv:2507.16372（AloePri [24]）；AloePri App. D.1 | 记录 State₁（私有输入 X₁ 的隐态），随机初始化 X₂，前向得 State₂，用 State₁/State₂ 的损失优化 X₂ 以恢复 X₁ | `attacks/isa/` | `tests/mathematical/test_isa_math.py`、`tests/regression/test_isa_regression.py`、`tests/unit/test_isa.py` |
| C06 | VMA（`vma.autoregressive` = VMA-H；`vma.weight_rowsort` = VMA-W） | Thomas et al., *Hidden No More: Attacking and Defending Private Third-Party LLM Inference*, ICML 2025, arXiv:2505.18332（AloePri [25]）；OSNIP App. C.2 / Table 7；AloePri App. D.1 / Table 9 | ① VMA-H：隐态投影到词表空间，**自回归**用自己 top-1 当 prefix，L1；Total/Clean（NLTK 停用词）；② VMA-W：`Y = Z1 X Z2`，先按行排序消去 Z2，再用邻居匹配恢复 Z1；Table 9 六类权重组合；多层投票 | `attacks/vma/`（三个模块严格分 lineage） | `tests/mathematical/test_vma_math.py`、`scripts/privacy_family_gates.py`（C07 无关；VMA 见 `tests/regression/test_numpy_families.py`） |
| C07 | BeamClean（`beamclean.gaussian`、`beamclean.laplacian`） | Kale, Mylonakis, Roberts, Roy, *BeamClean: Language Aware Embedding Reconstruction*, arXiv:2505.13758（2025） | 攻击者**没有** LM、**不知道**混淆机制，只看到混淆嵌入 + 模型公开嵌入表；联合估计噪声参数 + 用语言模型先验做 beam 解码；对 Laplacian/Gaussian **总优于**纯距离攻击 | `attacks/beamclean/` | `test_beamclean_naive_control_is_exact_on_clean_wire`、`test_beamclean_estimates_noise_and_beats_chance_under_noise`、`test_beamclean_laplacian_uses_the_laplacian_likelihood` |
| C08 | IA（`ia.gate`、`ia.attn`；原式在 `ia.ednn`） | Lin et al., *An Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference*, EMNLP 2024, pp. 2100–2104（AloePri [34]）；AloePri App. D.1 | ① Gate-IA：α_e=0 时置换保持 `Avg(e·W_gate)` 不变 → 用该统计不变量反推映射；② Attn-IA：把 `W_query/W_key` 按 RoPE 分块，若两向量的 e 相同则 `e(Qᵀ_{:,j}Q_{:,j})⁻¹eᵀ` 不变；head/block 置换破坏该一致性 | `attacks/ia/` | `tests/mathematical/test_ia_math.py` |
| C09 | IMA（`ima.classify`、`ima.seq2seq`、`ima.qwen2_aloepri`） | Kugler et al., arXiv:2109.10104（InvBERT）；AloePri App. D.1 / Table 2 | 攻击者知道混淆机制，在**公开数据**上**用目标机制**生成混淆嵌入，训练反演器；论文具体是 **Qwen2，2 层 decoder / 8 头**，反演到明文 token 嵌入 | `attacks/ima/` | `tests/mathematical/test_ima_math.py`、`tests/regression/test_ima_regression.py` |
| C10 | TFMA（`tfma.independent`、`tfma.global_assignment`） | AloePri §7.6 / Table 5（[27]）——**无独立原始论文** | 参考语料统计 token 频率 → 与密文符号频率匹配 → 恢复 token；三种先验：CCI3 / MedDialog / Huatuo26M-Lite 子集；Top-10/100 | `attacks/tfma/` | `tests/mathematical/test_tfma_math.py` |
| C11 | SDA（`sda.sda2021`、`sda.sda2023`） | Aldarrab & May, *Can Sequence-to-Sequence Models Crack Substitution Ciphers?*, ACL 2021（AloePri [27]）；Kambhatla, Born, Sarkar, *Decipherment as Regression*, Findings EACL 2023（AloePri [28]） | 2021：频次秩编码 + seq2seq；2023：**首次出现顺序**的 recurrence 编码 + causal LM，两项损失 | `attacks/sda/` | `tests/mathematical/test_sda_math.py`、`tests/regression/test_sda_regression.py` |
| C12 | 链接（`linkage.exact`、`linkage.approximate`） | **本方案扩展**（AloePri 的频率泄漏讨论派生）；导向文档 §3/§4 P0-9 明确「本方案扩展，无独立论文」 | 精确同符号率；近似 cos≥T 的 TPR/FPR；Δ 分桶 1/2–3/4–15/16–63/64–127/128–511/≥512；SH-P 按纪元分组、SH-B 全局聚类 | `attacks/linkage/` + `data/streams.py` | `tests/unit/test_privacy_families_14.py`（SH-P/SH-B、阈值扫描单调性、单钥正控） |
| C13 | PII 提取（`pii.extract`） | AloePri [14] = Li et al., *Papillon: Privacy Preservation from Internet-based and Local Language Model Ensembles*, NAACL 2025；**PUPA 是其中的数据集，不是攻击论文** | PIIRSR = 完整正确恢复的实体 / 标注实体；部分正确算失败；无实体返回 null | `attacks/pii/` + `attacks/_decode.py` | `test_pii_plaintext_control_recovers_every_entity`、`test_pii_without_entities_is_null_not_zero` |
| C14 | 输出侧 LM 重构（`lm_recon.ngram`、`lm_recon.local_lm`） | InferDPT（TDSC 2025，官方仓库 `mengtong0110/InferDPT`）/ RANTEXT 的 n-gram、CosSim、Top-K 重构评估；GPT-4 重构为其最强变体 | 从服务器可见的输出恢复文本：1..4-gram 重叠、CosSim、BLEU-4、token 恢复 | `attacks/lm_recon/` | `test_lm_recon_plaintext_control_is_exact`、`test_local_lm_substitute_is_not_applicable_without_generative_handle` |

## 2. 核对中发现并已处理的 5 条出处问题

**F1（高）C04 的 arXiv 编号张冠李戴。**
导向文档 §4.17/附录 B 把「跨 checkpoint 迁移」挂在 `arXiv:2610.27817`。实际核对（https://arxiv.org/abs/2610.27817 ，2026-10-20 提交）：
标题 *Privacy-preserving inference on hosted models: a learned input-embedding transform for frozen Qwen3.8-27B*，作者 Mahendra Roopa, Srikanth Roopa，
内容是一个**学习式输入嵌入变换（防御）**及其 NN 解码隐私评估（30.1% content token 不再解码回原 token），**不是迁移攻击协议**。
→ 处理：代码 `attack_model.py` 已把 C04 记为「本项目扩展」，并在 `paper` 字段写明该编号是防御论文。**不得**再把它写成迁移攻击的出处。

**F2（中）SGT 是防御，不是攻击。**
导向文档把 `SGT（arXiv 2506.09452）` 列在「邻域论文攻击」一栏。实际核对：2506.09452 = Roberts et al., *Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform*（**被攻击的防御方案**）。
攻击论文是同组的 **BeamClean**（arXiv:2505.13758）。→ C07 的论文锚点改为 BeamClean；SGT 只作为「被攻击的防御」出现在相关工作。

**F3（中）PUPA 是数据集名。**
AloePri 的 [14] 原文为 *Papillon: Privacy preservation from internet-based and local language model ensembles*（Li et al., NAACL 2025）。
「PUPA」是其中的数据集；PII 打分口径来自 AloePri 官方代码 `scripts/evaluation/privacy/evaluate_entities.py`。
→ 报告里写「PUPA 数据集 + AloePri 实体打分器」，不写「PUPA 攻击论文」。

**F4（低）TFMA 没有独立论文。**
导向文档把 TFMA 与 SDA 都标成 [27]，但 [27] 是 Aldarrab & May 2021（SDA 原始论文）。TFMA 在 AloePri §7.6 中是作为经典频率分析基线引入的。
→ TFMA 的出处写「AloePri §7.6 / Table 5（无独立原始论文）」，不编造。

**F5（低）VMA 有两个「版本名」。**
同一篇 arXiv:2505.18332 在 ICML 2025 正式版叫 *Hidden No More: Attacking and Defending Private Third-Party LLM Inference*；预印本标题是 *An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs*。
→ 引用时写 ICML 2025 正式标题，并保留 arXiv 编号，避免两处标题被当成两篇论文。

## 3. 核对证据（可复跑）

本机可用 OSNIP PMLR 版全文与 AloePri 技术报告 PDF，因此**目标论文侧是原文核对**，不是二手转述：

```powershell
# OSNIP（PMLR v306）App. C.1/C.2 与 Table 6/7
#   references/papers/osnip-pmlr-v306-cao26q-2026.txt L980-L1096
#   KNN：Alpaca instruction+input、20–512 字符、随机 1,000 条
#   VMA-H：复合池 300k（WikiText cap 150k）、200 实例×5 key、NLTK 停用词、自回归 own-prefix、L1
#   Table 6 Llama-3.2-1B OSNIP = 5.17 / 6.37 / 6.61 %
#   Table 7 Llama-3.2-1B Total = 40.67 / 49.51 / 19.26 / 29.90 %，Clean = 11.27 / 18.26 / 5.54 / 8.12 %（L1/L5/L10/L15）

# AloePri 技术报告（PDF → 文本）
python -c "import fitz; d=fitz.open(r'<AloePri TR>.pdf'); open(r'%TEMP%\aloepri_tr.txt','w',encoding='utf-8').write('\n'.join(p.get_text() for p in d)); print(d.page_count)"
#   App. D.1（VMA/IA/ISA/IMA/TFMA/SDA 逐条定义 + Table 9 六类权重组合）
#   参考文献 [14] Papillon、[24] Dong et al.、[25] Thomas et al.、[26] Kugler et al.、[27] Aldarrab & May、[28] Kambhatla et al.、[34] Lin et al.
```

外部编号核对（arXiv 摘要页，逐条抓取）：

| 编号 | 核对结果 |
|---|---|
| 2505.13758 | ✅ BeamClean: Language Aware Embedding Reconstruction（Kale, Mylonakis, Roberts, Roy） |
| 2505.18332 | ✅ Hidden No More（Thomas et al.，ICML 2025） |
| 2608.04477 | ✅ DeepInvert: Semi-Supervised Embedding Inversion（Huang, Hong, Wei，2026-08） |
| 2506.09452 | ✅ Stained Glass Transform（**防御**，Roberts et al.） |
| 2610.27817 | ⚠️ 内容与「跨 checkpoint 迁移」不符 → 见 F1 |
| 2004.00053 / 2109.10104 / 2507.16372 / 2310.06816 | ✅ 与仓库既有书目一致（`docs/BIBLIOGRAPHY.md`） |

## 4. 与论文的已知差异（必须随数字一起报）

| 类 | 差异 | 影响 |
|---|---|---|
| C07 | 论文用**神经** LM 先验；本仓无生成式后端时退回 `bigram_reference` | 大词表/长句上偏**弱**，低分不得读成「防住了」；`fidelity` 标记 `lm_prior=bigram_reference` |
| C13/C14 | 论文是文本/PII 语料；本仓是 token 流 + 合成实体 | 只验证协议与打分口径，不产论文数值 |
| C02/C04 | 论文的反演器是 2 层 8 头 Qwen2；本仓 KI 模块用逐位置分类器（小规模 cpu） | 只验证 KI/KP 边界与正控，不产论文 TTRSR |
| C09 | 论文版反演器 `ima.qwen2_aloepri` 存在，但本机无 GPU（torch 为 CPU 构建） | 训练规模受限，正控用测试内小预算 |
| 全部 | 本次出处核对不认证论文数值复现 | 历史资源与状态记录在 `docs/archive/`；当前运行事实查 `reports/`，准入查统一协议 |
