> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# 14类攻击总控派工单（以 privacy_test_plan_20261106.md 为导向）

> **状态**：Phase 0 未冻结。导向文档 §0 明确「未在 §0.4 决策表签字前不得启动 Phase 1 实验」，
> 本文件是 Phase 0 派工准备，不构成 Phase 1 授权。所有数值口径以导向文档为准，本文不新增阈值。
> **作者**：主控 Worker（2026-11-06）。已核查：`verify_fidelity` PASS、`audit_placeholders` PASS、
> `pytest -m "not neural"` 107 passed（见 §7 基线）。

## 0. 14类口径（与导向文档 §3.3 对齐）

导向文档按「攻击机理去重」计 15 族；若把 Gate-IA/Attn-IA 合并为 IA（AloePri 的一个族）则为 **14 族**，
P0 主表为 10 项协议条目，总目录 36 个协议条目。本派工按用户要求的 14 类展开：

| 总控 | 攻击族 | 协议条目 | 主/附 | 优先级 |
|---|---|---|---|---|
| C01 | KNN（含 NN） | A1, A4 | 主 | P0 |
| C02 | 学习式嵌入反演 | A2 | 附 | P1 |
| C03 | Vec2Text | A3 | 引 | P2 |
| C04 | 跨 checkpoint/迁移攻击 | A5 | 主(T9)/附 | P1 |
| C05 | ISA | B1, B2 | 主 | P0 |
| C06 | VMA（VMA-H + VMA-W） | B3, C1 | 主 | P0 |
| C07 | BeamClean | B4 | 附 | P1 |
| C08 | IA（Gate-IA + Attn-IA） | C2, C3 | 主/附 | P0 |
| C09 | IMA（岭 + 反演器，同钥/跨钥/跨实例/自适应） | D1–D6 | 主(跨实例)/附(其余) | P0/P1 |
| C10 | TFMA（id/嵌入级 + 逐钥） | E1, E2, E5, F1-TFMA | 主 | P0 |
| C11 | SDA（统计 + Transformer） | E3, E4, F1-SDA | 附 | P1 |
| C12 | 链接攻击（精确 + 近似） | E6, E7 | 主 | P0 |
| C13 | PII 提取（PUPA） | F2 | Limitations+附 | P1 |
| C14 | GPT-4/LM 重构 | F3 | 主(外部臂)/附 | P1 |
| — | 非攻击条目（不计族）：E8 先验基线、E9 度量、F4 LLM-judge、G1–G5 正控/消融 | — | — | — |

**分工风险提示**：C06（VMA-H/W 观测面不同、协议不同）、C08（Gate/Attn 不变量公式不同）、
C09（六种变体威胁模型不同）各含两个/多种协议条目，总控内必须分列报告，禁止合并成一个数字（导向 §0.2-1）。

## 1. 统一威胁模型（14 个总控共享，禁止各自改口径）

基线攻击者 = 半诚实服务器 SH：严格按协议执行、不篡改/注入/重放/MITM，只记录合法可见信息离线分析。
主表只有一个攻击者；K0（无密钥观测）、KW（权重可见）、KI（自建实例）、SH-P（协议知情）是 SH 在不同观测面上的子能力；
KP（已知配对）、KA（自适应白盒）、EXT-K（密钥泄漏）、SH-B（轮换盲）一律进附录/消融（导向 §1.4）。

| 场景 | 观测 | 允许攻击 | 表格 |
|---|---|---|---|
| S-A 在线推理 | wire + 隐态 + 输出 | KNN/ISA/VMA-H/输出 TFMA-SDA（非 PII/GPT-4） | T2/T4 |
| S-B 离线权重 | 部署权重（含混淆权重） | VMA-W/Gate-IA/Attn-IA | T4 |
| S-C 多请求日志 | 跨请求符号流 + 元数据 | TFMA/链接/K/N 扫描 | T5 |
| S-D 训练反演 | 公开数据 + 自建实例 | 跨实例 IMA/学习式反演 | T2/T9 |
| E-1 客户端泄漏 | 部署密钥配对 | 同钥/跨钥/混合钥 IMA | 附录 A |
| E-2 白盒/oracle | encryptor 权重/正确钥 | ADAPTIVE-WK/ORACLE | 消融 |
| E-3 输出外露 | 明文回答 | PII/GPT-4 重构 | Limitations |

**轮换口径（最易错，§0.0/§1.9-A7/D13/G15）**：主表 = SH-P（知道 K/N 与轮换存在，可按会话推纪元边界、
按符号相等聚类密钥族；不知道钥值/分配/当前钥）；SH-B 只作敏感性附录；EXT-K 不属主表。
现有 TFMA/链接数字若按 SH-B 产生，只能当下界，主表须按 SH-P 重新分组。

**八条主表假设 A1–A8**（任一违反即移附录）：A1 半诚实；A2 白盒权重；A3 无客户端密钥；
A4 无选择明文 oracle；A5 无目标密钥配对（IMA 主表仅 KI）；A6 信道安全；A7 SH-P；A8 可跨请求关联。
**协议消息流**（§1.8）：A+B 服务器看不到 token id 流，只见连续 z + 混淆权重；id 级 TFMA 对 A+B 必须先做嵌入聚类，
标 `attack_variant=embedding_level`；AloePri/A+B 的 VMA-W 恢复的是置换不是内容，必须报 stage-1/stage-2/composed 三行。

**四臂恢复链（§1.5）**：plaintext→明文 id；only-AloePri→混淆 id→（VMA-W/IA）→明文；
only-OSNIP→明文 id；A+B→混淆 id→（VMA-W/IA）→明文；token 基线→替换词。
KNN/ISA/VMA-H 必须用部署侧实际嵌入表/输出头，禁止用干净公开表打分。

**攻击强度下限（paper-strength floors，§6）**：KNN 1,000 文本全词表；VMA-H 200×5 全词表层 1/5/10/15；
VMA-W/IA 六类组合全层 ≥1,000 types；ISA ≥1,000 步 3 restarts + 饱和曲线；IMA 2 层 8 头 Qwen2 ≥4,096 序列；
TFMA 参考 ≥4.45M tokens、观测 ≥20k 出现、三种先验、SH-P、合成正控 ≥90%；
链接 ≥1,000 请求 ≥580k 同 token 对；PII ≥1,000 prompts。
**正控失败 = 攻击无效**（§7.5 决策树），低分不得进主表、不得在结论引用；保留 `attack_invalid` 产物并披露。

**JSON 统一 schema（附录 A）**：每个结果必含 `attack/target/observation_surface/attacker_knowledge/
uses_client_key/uses_known_pairs/rotation_aware/candidate_mode/full_vocabulary/positive_control/
n_prompts/n_tokens/n_keys/n_types/reference_tokens/metrics/ci/source_sha256/data_sha256`；
另须 `attack_uses_encoder/key/client_secret=false`（主表）、`scenario`、`paper_parameter_undisclosed` 标记。
产物根 `reports/privacy_v1/<phase>/<attack>/<target>/`，每 job 含 command/env/stdout/result/status/.done；
主表从 summary 聚合生成，不手抄。

## 2. 出处论文（每类一句话锚点）

| 总控 | 攻击 | 出处（导向文档锚点 + BIBLIOGRAPHY 核对状态） |
|---|---|---|
| C01 | KNN | OSNIP App. C.1/Table 6；Song & Raghunathan CCS 2020 (arXiv:2004.00053) ✅ 本仓核对 |
| C02 | 学习式反演 | TextObfuscator ACL 2023；ObfusLM；SGT 的 NN ✅ 本仓 EDNN 属 IA 原始版，学习式反演器未实现 |
| C03 | Vec2Text | Morris et al. 2023 ✅ 仅引用 |
| C04 | 跨 checkpoint 迁移 | arXiv 2610.27817；SGT 迁移失效 ⚠️ 编号待主控核对原文 |
| C05 | ISA | AloePri App. D.1 + [24]；Dong et al. USENIX Sec 2025 ER/TBS (arXiv:2507.16372) ✅ 本仓核对 |
| C06 | VMA | OSNIP App. C.2/Table 7 (VMA-H)；Thomas et al. ICML 2025 (arXiv:2505.18332) 原始 VMA ✅；AloePri Table 9 (VMA-W) ✅ |
| C07 | BeamClean | SGT ⚠️ 待核对原题名 |
| C08 | IA | AloePri App. D.1 (Gate/Attn) ✅；原始 EDNN: Lin et al. EMNLP 2024 ✅ 本仓核对 |
| C09 | IMA | AloePri Table 2/App. D.1 (Qwen2 反演器；KI 跨实例) ✅；原始 InvBERT: Kugler et al. (arXiv:2109.10104) ✅ |
| C10 | TFMA | AloePri Table 5 (id 级替换密码) + [27] ✅ 本仓无独立论文，视为 AloePri 基线 |
| C11 | SDA | AloePri Table 5 + [27]；Aldarrab & May ACL 2021 ✅；Kambhatla et al. Findings EACL 2023 ✅ 本仓核对 |
| C12 | 链接 | AloePri 频率泄漏；本方案扩展（精确 + 近似阈值）——无独立论文，不得编造出处 |
| C13 | PII | AloePri [14] PUPA；SGT ⚠️ 待核对 |
| C14 | GPT-4 重构 | InferDPT/RANTEXT TDSC 2025；Morris et al. ⚠️ 待核对 |

## 3. 并行派工图（依赖 → 波次）

**Wave 0 — 冻结（阻塞，后续全部 blocked_by）**：
T0.1 协议 v1.2 + §0.4 D1–D13 签字 → `configs/privacy_protocol_v1.json`；
T0.2 数据 manifest + SHA-256（KNN 1,000 / VMA 200×5 / TFMA 1,000 请求 / 参考 4.45M / IMA 语料）；
T0.3 JSON schema `schemas/attack_result_v1.json`；T2/T4/T5/T9/T10 表格骨架。
签字前禁止跑 Phase 1。**当前状态：三件均未产出（已核查 configs/ 下无 privacy_protocol_v1.json）。**

**Wave 1 — 数据与入口（CPU/可并行）**：
T1.1 VMA-H 在 AloePri/A+B 服务器模型上的入口（C06）；KNN 1,000 文本集（C01）；
TFMA 语料 + 合成正控 + SH-P 分组（C10/C12）；IMA 语料与反演器数据管线（C09）。

**Wave 2 — P0 主表（GPU 串行，单卡；CPU 权重面可夜跑）**：
T1.2 VMA-H 四臂全量（C06，4–6h GPU/臂，最大缺口 G1）；
T1.3 VMA-W 全量（C06，六类组合全层 ≥1,000 types，G2）；
T1.4 Gate-IA（C08，G3）；T1.5 Attn-IA（C08，G4）；
T1.8 KNN 论文精确 1,000 文本（C01，G7）；T1.7 ISA 饱和曲线（C05，G6）；
T1.9 TFMA 三种先验 + 饱和 + SH-P/SH-B 对照（C10/C12，G8/G15）；
T1.10 Top-100/CI 回填。GPU 优先级 T1.2/T1.6/T1.8（§8.7）。

**Wave 3 — P1 附录 + P0 收尾（依赖 Wave 2）**：
T1.6 OSNIP-only 第二编码器 + 跨实例 IMA（C09，8–10h GPU，G5）；
T2.1 同钥/跨钥/混合钥 IMA 附录 A（C09）；
T2.2 学习式反演器（C02，G11）；T2.3 SDA 正控修复（C11，修不好维持剔除，G9）；
T2.4 自适应 KNN/IMA Table 8（C01/C09 消融）；
T2.5 输出/PII（C13/C14，G10）；T2.6 跨 checkpoint 迁移（C04，G12）；
BeamClean（C07，成本高，按审稿要求才跑）；Vec2Text（C03，仅引用不跑）。

**Phase 退出门**：P1 退出 = G1–G8、G15 关闭且四臂 × P0 攻击皆有数或明确 N/A、无 `attack_invalid` 数字进主表、
T4.1 逐格签字；P2 退出 = G9–G13 关闭 + 附录 A + Limitations。

## 4. 十四总控任务单（worker 用，逐张独立派工）

### C01 KNN — 嵌入反演（S1/K0，T2 主表）
- **论文**：OSNIP App. C.1 / Table 6；Song & Raghunathan 2020。OSNIP Llama-3.2-1B 论文值 Top-1/5/10 = 5.17/6.37/6.61%。
- **数据**：Alpaca instruction+input，20–512 字符，随机 1,000 条（与训练/VMA 池不交），1 key/文本；补充 200×16×5 key 稳健性单独报。
- **强度**：全词表 128,256（禁 proposal），L2，Top-1/5/10 主 + Top-100 附；部署表（AloePri/A+B 用混淆表）。
- **正控**：干净 Top-1 ≥99%；metric = token 级 ASR@k + bootstrap CI + 词频分桶。
- **现状**：本仓 `nn_knn.euclidean/cosine` + 闭环 `rm_nn_full.py`（plaintext 嵌入层 100%，隐态层随深度下降）；
OSNIP 包 `evaluate_paper_protocol.py` KNN 段。缺口 G7：论文精确 1,000 文本未按现四臂 checkpoint 复核；
AloePri/A+B 用部署表 + 复合链复核（§7.5 若用干净表打混淆 z 则标 ⚠️）。
- **产物**：`reports/privacy_v1/phase1/knn/{plaintext,aloepri,osnip,ab}/result.json`。

### C02 学习式嵌入反演 — EDNN/InvBERT 风格（S1/S2，KI，附录，⬜ 未实现）
- **论文**：TextObfuscator ACL 2023；ObfusLM；RANTEXT BERT 反演器；SGT NN。
- **协议**：公开语料 ≥1M tokens，攻击者实例生成配对，文档级切分，≥3 seeds，early stopping，训练曲线。
- **强度/正控**：同实例域内 Top-1 ≥95% 否则欠拟合，不得报跨实例 0%；指标 Top-1/5/10 + ROUGE-L，与 KNN/ISA 对照。
- **现状**：本仓 `ia.ednn` 是 Lin 2024 原始 EDNN（非学习式反演器），不得冒充；需新建 BERT-base 级反演器（约 3×2h GPU）。
- **产物**：附录一行「学习式反演与 KNN/ISA 结论一致」，不抢主表。

### C03 Vec2Text — 文本级迭代反演（S1/文本，引，P2/⬜）
- **论文**：Morris et al. 2023。**本轮不跑**（Phase 3 按审稿要求）；任务单只要求总控留档引用与成本估算。

### C04 跨 checkpoint/迁移攻击（S1/KI，T9 主/附，⬜）
- **论文**：arXiv 2610.27817；SGT 迁移失效。**引用编号待主控在冻结前核对原文**（禁止编 DOI）。
- **协议**：两个独立 seed checkpoint，同/跨架构两档；可复用 C09 第二实例；同 checkpoint 正控 ≥95%。
- **产物**：`reports/privacy_v1/phase2/cross_checkpoint/*/result.json`。

### C05 ISA — 内部状态攻击（S2/K0，T2/T4 主表）
- **论文**：AloePri App. D.1 + [24]；原始 ER/TBS: Dong et al. 2025。本仓 `isa.er/tbs/aloepri_state` + 闭环 `rm_isa_full.py`（5e4 步）。
- **数据**：200 prompts × 5 keys × 完整序列（≥64 token，目标 512，截断报 `truncated`），层 1/5/10/15。
- **强度**：≥1,000 步、lr 0.05、≥3 restarts（报 best-of + mean），饱和曲线 20/200/1,000/2,000 步。
- **正控**：干净 Top-1 ≥90%（低于则欠预算判无效）；指标 Top-1/5/10/100 + TTRSR + 逐层曲线 + CI。
- **现状**：现有 20 步 smoke（OSNIP 包 `isa_opt`/`isa_fixed.py`）+ 16 token 截断；缺口 G6。

### C06 VMA — VMA-H 隐态词表匹配 + VMA-W 权重置换恢复（S2/S3，主表，⬜/🔶）
- **必须分列两行，禁止合并**（§0.2-1）；OSNIP-only 的 VMA-W 记 **N/A** 不记 0%（§1.6）。
- **VMA-H（S2/K0，OSNIP App. C.2）**：300k 复合池（Alpaca + ARC-Easy + ARC-Challenge + SciQ 选项打乱 + WikiText-103 cap 150k），
200 prompts × 5 keys，层 1/5/10/15，自回归 own-prefix，L1，全词表，Total/Clean（NLTK 停用词）@1/5/10/100 + 候选覆盖。
正控干净 Total ≥99%。**现状：四臂全量未跑 = 最大缺口 G1**；现有主表「VMA」实为 VMA-W，必须替换/分列。
- **VMA-W（S3/KW，AloePri Table 9）**：六类组合（Embedding/Head、Q/K、Gate、Up、Down/Head、Router），全层，多层投票，
全 V² 分块，≥1,000 types。指标：**映射恢复率 + 复合端到端 TTRSR/PIIRSR @1/5/10/100** 分开报；正控明文 100% + known-secret 诊断。
现状：16 types layer 0 诊断（AloePri α\* 90.6/93.8/95.7/98.0%）；A+B 0.0/0.4/0.4/0.8% 待复合链复核。缺口 G2。
本仓 `vma.autoregressive/weight_rowsort/osnip_projection` + 闭环 `rm_vma_full.py / rm_vma_weight_full.py`（明文基线，非四臂）。

### C07 BeamClean — LM 感知 beam 文本重构（S2/KI，附，⬜）
- **论文**：SGT（题名待核对）。协议：白盒 LM + 隐态观测，beam 重构文本，≥1,000 prompts，与 KNN/ISA 同数据；
指标精确/语义匹配 + PII + BLEU；正控明文隐态高重构。成本高，Phase 2 按审稿要求。

### C08 IA — Gate-IA + Attn-IA 权重不变量（S3/KW，T4 附 + 正文汇总行，P0）
- **论文**：AloePri App. D.1。Gate = W_gate 统计不变量；Attn = Q/K 分块不变量（head/block 置换重跑）。
全层、全词表、≥1,000 types、投票，α=0 与 α\* 两档；指标 TTRSR/PIIRSR @1/5/10/100 + 映射恢复率；正控明文 100% + known-secret。
- **现状**：Gate 28.1%（layer 0/32 types）仅诊断；Attn 未跑。缺口 G3/G4。
- **纪律**：与 VMA-W 不变量公式不同故分开；禁止与 C06 合并报。

### C09 IMA — 岭 + 论文版反演器（S1/S2，主表仅 KI 跨实例，其余附录）
- **论文**：AloePri §3.2/Table 2/App. D.1（2 层 8 头 Qwen2 反演器，攻击者用目标机制在公开数据自生成配对）。
- **主表（KI）**：新 seed 自建实例，≥4,096 序列×32 token 或 ≥1M token，4 epochs，双向，两组独立实例对；
Top-1 主 + Top-10 附，正控域内 ≥95%（岭 100%）。现状：AloePri 已跑 0.0/0.0、A+B 岭 0.0/反演器 0.2–1.3%，**OSNIP-only 未测**（需第二编码器 ~6.3h，G5）。
- **附录（KP/EXT-K/KA，非主表）**：同钥 D1/D2（四臂 97–100%）、跨钥 D4、混合钥 D5、自适应 D6。必须写
`attacker_knowledge=KP`、`uses_client_key=true`；主表禁止用部署 key/配对（KI ≠ KP，§0.2-6）。
- **本仓**：`ima.classify/seq2seq/qwen2_aloepri` + 闭环 `rm_ima_full.py`（secret-key 0.0 / known-key 0.63–0.81，密钥持有证明）。

### C10 TFMA — 频率攻击（S4/SH-P，T5 主表）
- **论文**：AloePri Table 5（id 级）+ [27]；A+B/OSNIP 必须先聚类成「嵌入级符号」，标 `attack_variant=embedding_level`。
- **数据**：私密流 ≥1,000 请求 ≥20k 出现 ≥1,000 类型；参考 ≥4.45M tokens ≥50k 类型（≥100× 观测），零文档重叠；
三种先验（zero-knowledge CCI3 / domain-aware MedDialog / distribution-aware Huatuo 子集）全跑，`reference_substitute=true` 若替代。
- **强度**：三种匹配器（rank / independent / global_assignment）全跑；逐钥 K=1/2/4/8/16/32 扫描；SH-P 分组（G15）；
饱和 1M/4.45M/10M × 1k/3k/5k 请求；先验基线并列；出现级 Top-1/5/10/100 + 类型级辅 + 覆盖 + 分桶。
- **正控**：合成已知随机置换 Top-1 ≥90%（低于先加数据）。现状：强口径已跑（A+B 逐钥 6.3–6.6/21.6/28.1/45.3%），
三种论文先验与饱和曲线待补（G8）；若现有数字按 SH-B 产生的，只当下界。
- **本仓**：`tfma.independent/global_assignment` + 闭环 `rm_tfma_full.py`（plaintext 概率制，global 仅前缀 1,398 符号有界）。

### C11 SDA — 统计版 + Transformer 版（S4/K0，附，⚠️）
- **论文**：AloePri Table 5 + [27]；Aldarrab 2021（频次秩 + seq2seq）；Kambhatla 2023（复现编码 + causal LM）。
- **现状**：统计版有；**`sda_transformer` 正控 BLEU-4 失败**（合成已知置换未达显著高）→ 按 §6.8/§7.5 **剔除并披露**，不进主表。
- **任务**：合成已知置换正控先行（BLEU-4 应显著高于论文 ~2）；修不好则维持剔除，禁止用统计版冒充 Transformer 版。

### C12 链接攻击 — 精确 + 近似（S4/SH-P，T5 主表）
- **论文**：本方案扩展（无独立论文，**禁止编造出处**）。
- **数据**：≥1,000 请求 ≥19,805 token ≥580k 同 token 对；Δ 分桶 1/2–3/4–15/16–63/64–127/128–511/≥512。
- **参数**：精确同符号率；近似余弦 T=0.3–0.9 全扫 TPR/FPR；已知共享 token 的最近邻定位；合并类频率攻击。
- **正控**：单钥/明文精确 ≈1.000（合成同 token 对 100%）；定位 ≥95% 才算强。
- **现状**：精确 0.79%（A+B 默认）、T=0.5 TPR 25.2%/FPR 3.3%、定位 99.7% ✅；SH-P 分组待复核（G15）。
- **本仓**：无对应模块（grep 无 linkage）——新建；符号聚类须与 C10 共用符号表，禁止两套符号。

### C13 PII 提取 PUPA（S5，Limitations + 附录，⬜）
- **论文**：AloePri [14]；SGT。≥1,000 prompts；合成 PII（人名/电话/地址/邮箱）或 PUPA；PIIRSR = 完整实体恢复/标注实体
（无实体返回 null 不当 0%）。**禁止写成 A+B 的防护成绩**（只进 Limitations + 附录，D11）。

### C14 GPT-4/LM 重构（S5，外部臂主/附，⬜）
- **论文**：InferDPT/RANTEXT（n-gram/CosSim/Top-K=1/10）；Morris。GPT-4 固定 prompt、temperature 0、3 次；
n-gram/CosSim 同 tokenizer。指标 PIIRSR/TTRSR/CosSim/BLEU-4 + 实体恢复。正控明文高、token 基线显著高于嵌入臂。

## 5. 本机执行约束（总控须据此排期）

- 本机 RTX 4060 Laptop 8GB + Llama-3.2-1B-Instruct 本地权重（`F:\Models\Llama-3.2-1B-Instruct`）；
OSNIP 包四臂（A+B 复合链）的真实验在 `/data2/OSNIP`（Linux 远端，见 `security_attacks_aloepri.py` 硬编码路径），
**本机跑不动四臂全量**（RESOURCE_REQUIREMENTS 已登记 resource_blocked）。本机可跑的是本仓 7 族的 CPU/小模型门禁
（已跑通过 §7）与闭环 plaintext 基线（`reports/closed_loop/summary.md`，明文期望完美 ≠ 防护结论）。
- GPU 串行：训练与攻击不抢同一卡；夜跑队列按 §8.9（`.done/status.txt/timeout/GPU 门控`）。
- subagent 说明：本会话无 LLM 子 Agent 派生工具，**14 总控以本文件为派工合同**——主控负责口径/分工/验收，
各任务单即总控任务（目标+白名单文件+签名+判据+产物路径+自测命令+禁止项，§4 规程），
不另行口头派工；CPU 校验并行以 pwsh 后台 job 执行，GPU 实验走远端队列脚本（`run_*_queue.sh`），
结果经 `summarize_security_attacks.py` 聚合，不手抄。

## 6. 待主控确认（本轮未擅自决定）

1. §0.4 D1–D13（共 13 项）无一签字；本文件按导向默认推荐值起草，签字后以签字版为准（Phase 0 阻塞）。
2.「14 类」口径：本文采用 §3.3 的 IA 合并口径（15 族→14）；若主控另有定义（如仅 P0+P1 的 14 项），总控映射须改。
3. C04（arXiv 2610.27817）、C07（SGT）、C13（[14] PUPA）原题名/版本未在本仓核对，禁止编造，冻结前逐条核对。
4. TFMA 参考语料（CCI3/MedDialog/Huato26M-Lite 完整快照）可得性与替代语料（R3）未定；数据真实库存未清（见 §7）。
5. 本仓 7 族 vs OSNIP 包四臂的归属：导向以 OSNIP 包四臂为准，本仓为算法层复现，两者关系须主控拍板。

## 7. 本轮已核查的基线（可复跑证据）

```
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/verify_fidelity.py        → FIDELITY CHECK PASSED（7 families，ledger 可溯，full 配置零近似）
python scripts/audit_placeholders.py     → PLACEHOLDER AUDIT PASSED
python -m pytest -m "not neural" -q      → 107 passed, 59 deselected in 14.62s
```
另已核查缺口事实：`configs/privacy_protocol_v1.json` 不存在；`data/paper_attack_sets/` 仅 README 空目录；
`data/osnip_sources/raw/` 仅 README；`references/papers/` 仅 OSNIP 两版 PDF/txt（AloePri 报告在 `references/aloepri_code_20261103/` 内 PDF）；
`security_attacks_aloepri.py` 硬编码 `/data2/OSNIP` 远端路径，本机不可跑。闭环数字见 `reports/closed_loop/summary.md`（plaintext 基线）。

## 8. 下游影响点

- C06 VMA-H 入口（T1.1）是 G1 的前置，不通则 T1.2/T1.7/复合链全堵。
- C10↔C12 共用符号聚类与 SH-P 分组：先冻结 G15 再跑 TFMA/链接主表，否则数字全是下界。
- C09 第二实例（T1.6）同时供 C04 复用；GPU 排队冲突时按 T1.2/T1.6/T1.8 优先级。
- 任何 `attack_invalid`/正控失败数字不得进主表；N/A 不得填 0%（§7.5/§1.6）。
