> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# 出处论文核对（2026-11-06 复核）

> 上游：`privacy_test_plan_20261106.md` §2.3 / §4 的“论文依据”列；`docs/ATTACK_PROVENANCE_14.md`（历史记录，保留不改）。
> 本文件只回答一件事：**每类攻击的出处到底是不是这篇**。核对方法：直接抓 arXiv abs 页看标题/作者/摘要，不看二手引用；
> 本地有的（OSNIP PMLR v306、AloePri 技术报告、AloePri 官方代码）以本地 PDF/代码为准。
> 状态：`verified`（标题+内容与攻击机理一致）/ `corrected`（编号或归属错，已给正确锚点）/ `pending`（本轮未核，禁止在论文里写成已核）。

## 0. 结论先行

- **14 类的攻击机理来源，9 类可核到具体一手文献**；C12（链接）与 C04（跨 checkpoint）**没有**一手出处，必须写成“本方案扩展/协议”，不能编 DOI。
- 发现 **2 处出处错误**，必须改导向文档（errata），否则论文评审会抓：
  1. **C04（跨 checkpoint 迁移）**：导向 §4.17 引 `arXiv:2610.27817`。该文实际是
     *“Privacy-preserving inference on hosted models: a learned input-embedding transform for frozen Qwen3.8-27B”*（Roopa & Roopa, 2026-10-20），
     是一篇**防御**论文（学习式输入嵌入变换 + KNN 解码率 30.1%），**不是迁移攻击协议**。
  2. **C07（BeamClean）**：导向 §2.3 把 BeamClean 归在 “SGT” 名下。SGT（`arXiv:2506.09452`）是
     *Stained Glass Transform*——一种**防御**；BeamClean（`arXiv:2505.13758`）才是攻击。两者不能互相代替。
- 另：导向 §2.3 表把 “OSNIP 2026 VMA-H” 写成 “原文按层报 ASR，未显式 top-k”，与本地 OSNIP 文本一致，可保留。

## 1. 逐类核对表

| 类 | 攻击 | 导向锚点 | 核对结果 | 一手文献 / URL | 判定 |
|---|---|---|---|---|---|
| C01 | KNN / NN | Song & Raghunathan CCS 2020；OSNIP App. C.1 | 标题《Information Leakage in Embedding Models》，攻击机理（embedding 反演）一致 | arXiv:2004.00053 · https://arxiv.org/abs/2004.00053 | verified |
| C01 | KNN（OSNIP 侧参数） | OSNIP PMLR v306 Table 6 | 本地有 PMLR v306 PDF/txt（`references/papers/`），参数以本地原文为准 | 本地 `osnip-pmlr-v306-cao26q-2026.pdf` | verified(local) |
| C02 | 学习式嵌入反演（BERT inverter） | TextObfuscator ACL 2023；InvBERT | InvBERT 标题《InvBERT: Reconstructing Text from Contextualized Word Embeddings by inverting the BERT pipeline》，正是“训练反演器”路线 | arXiv:2109.10104 · https://arxiv.org/abs/2109.10104 | verified（InvBERT）；TextObfuscator 本轮 pending |
| C03 | Vec2Text | Morris et al. 2023 | 标题《Text Embeddings Reveal (Almost) As Much As Text》，迭代文本反演即 Vec2Text | arXiv:2310.06816 · https://arxiv.org/abs/2310.06816 | verified |
| C04 | 跨 checkpoint / 迁移 | 导向 §4.17：arXiv:2610.27817 | **不是迁移攻击**；该文是学习式输入嵌入变换**防御** | arXiv:2610.27817 · https://arxiv.org/abs/2610.27817（防御） | **corrected**：C04 无一手出处，按“本方案扩展”报；可引 Dong et al. 2025 的跨模型/跨层泛化观察作旁证 |
| C05 | ISA（内部状态反演） | AloePri App. D.1 + [24] | [24] 对应《Depth Gives a False Sense of Privacy: LLM Internal States Inversion》（Dong et al.），优化输入嵌入匹配隐态正是该文方法 | arXiv:2507.16372 · https://arxiv.org/abs/2507.16372 | verified |
| C06 | VMA（H 与 W 同源机理） | Thomas et al. ICML 2025；OSNIP App. C.2；AloePri Table 9 | 《An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs》，ICML 2025 正会改名《Hidden No More: Attacking and Defending Private Third-Party LLM Inference》；排序/匹配恢复置换的机理一致 | arXiv:2505.18332 · https://arxiv.org/abs/2505.18332 | verified |
| C07 | BeamClean | 导向 §2.3 写作 “SGT” | BeamClean = 《BeamClean: Language Aware Embedding Reconstruction》；SGT 是防御 | arXiv:2505.13758 · https://arxiv.org/abs/2505.13758；SGT `arXiv:2506.09452` https://arxiv.org/abs/2506.09452 | **corrected** |
| C08 | Gate-IA / Attn-IA | AloePri App. D.1 | 本地 AloePri 技术报告（`references/aloepri_code_20261103/.../docs/Lin 等 - 2026 …pdf`）含 App. D.1 攻击族；Gate/Attn 不变量公式以本地报告为准 | 本地 PDF | verified(local) |
| C09 | IMA（反演器 / 岭） | AloePri Table 2 / App. D.1；InvBERT | 反演器路线同 C02（InvBERT verified）；AloePri 的 2 层 8 头 Qwen2 规格以本地报告为准 | arXiv:2109.10104 + 本地 AloePri 报告 | verified |
| C10 | TFMA | AloePri Table 5 + [27] | 本地 AloePri 报告 §7.6 / Table 5 有 id 级替换密码频率攻击；[27] 具体条目本轮 pending | 本地 PDF | verified(local)（[27] pending） |
| C11 | SDA（2021 频次秩 / 2023 复现编码） | Aldarrab & May ACL 2021；Kambhatla et al. Findings EACL 2023 | 仓内 `docs/BIBLIOGRAPHY.md` 已登记 ACL Anthology 编号；本轮未再抓原页 | ACL Anthology（待补 URL） | pending |
| C12 | 链接（精确 + 近似） | 导向自述“本方案扩展” | 无一手出处；不得编造 | — | verified(no-source) |
| C13 | PII 提取 / PIIRSR | AloePri [14]（PUPA） | PUPA 数据集/指标本轮 pending；AloePri 报告内引用可核 | 本地 AloePri 报告 | pending |
| C14 | 输出侧 LM 重构 | InferDPT / RANTEXT（TDSC 2025）；Morris | Morris（2310.06816）verified；InferDPT/RANTEXT 本轮 pending | https://arxiv.org/abs/2310.06816 | pending（部分） |

## 2. 需要在导向文档登记的 errata

| # | 位置 | 现状 | 应改为 |
|---|---|---|---|
| E1 | §4.17 / §2.5 “C04 跨 checkpoint” | 引 arXiv:2610.27817 作迁移攻击 | 删去该锚点；C04 标 `no_source_paper=true`，写“本方案扩展（KI 迁移）”，旁证只能用 Dong et al. 2025 的跨层/跨模型观察 |
| E2 | §2.3 表 “C07 BeamClean ← SGT 2026” | 把防御当攻击 | BeamClean 锚 `arXiv:2505.13758`；SGT 作为**防御对照**单独列，不得混用 |
| E3 | §2.3 表 “SGT 2026 / NN / BeamClean” | 同一行混两个方法 | 拆两行：SGT（防御，NN/BeamClean 是打它的攻击）与 BeamClean（攻击） |

## 3. 本地一手材料清单（可直接复核，不需联网）

- OSNIP：`F:\OSNIP_package_20261101\references\papers\osnip-pmlr-v306-cao26q-2026.pdf` / `.txt`（Table 6/7/8 参数来源）。
- AloePri 技术报告：`references\aloepri_code_20261103\Aloepri_Llama-3.2-1B-Instruct_obfuscation_code_20261103\docs\Lin 等 - 2026 - Towards Privacy-Pre…pdf`。
- AloePri 官方混淆代码与攻击代码：同目录 `src/`（`obfuscate_embed_head.py`、`keymat_*.py`、`transforms.py`）。
- A+B 可运行工件：`F:\OSNIP_package_20261101\ab_runnable\`（`manifest.json` 记录 sha256 与训练配方）。

> 纪律：`pending` 的三条（TextObfuscator、SDA 两篇、PUPA、InferDPT）在补到 URL 前，论文正文只能写“相关工作报告于某年”，不能写具体编号或页码。
