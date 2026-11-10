> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# 14 类攻击状态表（第二轮：打包工件实测）

> 上游：`privacy_test_plan_20261106.md`（导向）→ `docs/ATTACK_ORCHESTRATION_14.md`（派工合同）。
> 对齐审计与逐条判定：`docs/ATTACK_ALIGNMENT_AUDIT_20261106.md`；出处核对：`docs/PROVENANCE_VERIFIED_20261106.md`。
> 第一轮（合成/协议证据）的记录保留在文末「附录：第一轮」。
> **纪律**：`N/A ≠ 0%`；正控失败的数字不进主表；未达强度下限的只能叫「协议证据」。

## 0. 结论先行

- **三臂可在本机真测**：`plaintext`（干净 Llama-3.2-1B）/ `aloepri`（混淆权重 + 混淆 id 流）/
  `ab`（连续 `z`）。第四臂 `osnip` 记 `not_applicable`（编码器是在混淆表上训练的，见 §3）。
- **P0 里已产出真机数字的**：C01（KNN）、C06-W（权重置换）、C08（Gate-IA）、D3（复合链）、
  C10（TFMA-embed，SH-P/SH-B）、C12（链接）、C13（输出侧）。**仍为协议/合成证据的**：
  C02、C03（仅引用）、C04、C05、C06-H、C07、C09、C11、C14（GPT-4 变体）。
- **最重要的四个数字**（已按 2026-11-07 威胁模型修正重挂，见 `docs/THREAT_MODEL_CORRECTION_20261107.md`）：
  ① A+B 的 stage-1 KNN **62.90%**（n=1,000 文本，CI [62.65, 64.13]，**纯 KW**）；
  ② composed 链 **62.71%**（stage-2 = 99.60%，覆盖 100%）—— 端到端被 stage-1 卡死；
  ③ dense-mixing 权重攻击 **99.5%**、Gate-IA **94.1%** —— 与论文 Table 9 同一前提（明文+混淆权重），
  属**主表**；④ RowSort 的 0.39%/3.5% 才是诊断（构造用了 τ）。
  **一句话**：权重面能被打穿，是因为工件跑在**效用工作点**（α=0.2/0.01）且未复现论文 Algorithm 1；
  论文在 α=1.0/0.2 上报 <3% 靠的是噪声，不是隐藏明文权重。
- **轮换确实有效但只对「跨纪元」有效**：同一 key 组内精确链接 Δ=1 达 **97.3%**，Δ≥64 掉到 **0.000**；
  TFMA 的 SH-P **15.6%** vs SH-B **1.1%**，**主表必须 SH-P**。

## 0.5 2026-11-07 威胁模型对齐（必读）

按 AloePri 原文（`2603.01499v2`）§3.2 / §5.1 / App. D.1 Table 9 / §7.3 核对：
**攻击者拥有明文权重与混淆权重、知道机制与公式，不知道 τ、密钥矩阵、噪声种子**；
防 VMA 的是**噪声**（αe=1.0/αh=0.2），不是隐藏明文权重。因此
**C06-W 99.5%、C08 94.1%、composed 62.71% 都回到主表**（`uses_known_pairs=false`、
`uses_plain_and_obfuscated_weights=true`），并附 `paper_working_point`（论文 1.0/0.2 vs
工件 0.2/0.01）；RowSort 的 0.39%/3.5% 仍是诊断（构造用 τ）。
详见 `docs/THREAT_MODEL_ALIGNED_WITH_PAPER_20261107.md`。

## 1. 逐类状态

图例：✅ 真机实测 ｜ 🔶 协议/合成证据（真机受算力/数据限制）｜ ⬜ 未跑 ｜ ⛔ 阻塞。

| 类 | 攻击 | 状态 | 证据（可复跑） | 距论文强度下限 |
|---|---|---|---|---|
| C01 | KNN（S1/K0） | ✅ 真机（达样本下限） | `reports/privacy_v1/full/C01/nn_knn.cosine/{plaintext,ab,aloepri,osnip}/result.json`：明文正控 Top-1 **1.000**（n=1,000 文本 / 16,774 token）；A+B stage-1 **0.6290** [0.6265,0.6413] @1，0.6915 @5，0.7084 @10，0.7356 @100；词频分桶 高 0.948(134)/中 0.800(1,043)/低 0.642(1,541)/参考外 0.612(14,056)；aloepri stage-1 = 1.000（按构造）；osnip = N/A | 全词表 128,256、1 key/文本、bootstrap CI、词频分桶、1,000 文本 —— 均达导向 §6.1 下限 |
| C02 | 学习式嵌入反演（KI） | 🔶 合成 | `ima.cross_instance` + `tests/unit/test_privacy_families_14.py`；域内正控 1.000 | 需 GPU 训练 BERT 级反演器（≥1M token） |
| C03 | Vec2Text | ⬜ 仅引用 | `ATTACK_MODELS["vec2text.iterative"]`（`implemented=false`） | Phase 3，按导向 §4.15 不产数字 |
| C04 | 跨 checkpoint 迁移 | 🔶 合成 | `ima.cross_checkpoint` + 单测 | 出处已更正（无一手论文）；需第二 checkpoint |
| C05 | ISA（S2/K0） | ⬜ 阻塞 | 合成：`tests/mathematical/test_isa_math.py` | 1B fp32 CPU 单步 ~1.3 s；≥1,000 步 × 3 restarts × 200 prompts 不可行 |
| C06 | VMA-H（S2/K0） | ⬜ 阻塞 | 合成：`tests/mathematical/test_vma_math.py`；闭环 `rm_vma_full.py`（明文 ASR@1 = 1.000） | AR own-prefix 每 token 一次前向，200×5×4 层需 GPU |
| C06 | VMA-W（S3） | ✅ 主表（工件变体） | `C06/.../aloepri_weights/result.json`：dense-mixing **0.995**（1,000 部署行，cond 17.5，M 残差 0.0139≈α_h）；`uses_known_pairs=false`、`main_table_eligible=true`、`paper_working_point` 已记录。RowSort **0.0039/0.035** 为诊断 | 论文 Table 9 的六类乘积在 n=128k 时是 131 GB/个，本机不可行；主表须标 `artifact_algorithm1=false` |
| C07 | BeamClean（S1/KI） | 🔶 合成 | gate 臂 0.875/0.750（合成），3 条单测 | 论文用神经 LM 先验；本机无生成式后端 |
| C08 | Gate-IA / Attn-IA（S3） | ✅ Gate 主表 / ⬜ Attn 未跑 | `C08/ia.gate/aloepri_weights/result.json`：Top-1 **0.941**、@10 0.991（1,000 types × 16 层）；`uses_known_pairs=false`、`main_table_eligible=true` | Attn-IA 真权重未跑；α=0 档需另一工件 |
| C09 | IMA（KI/KP） | 🔶 合成 | `tests/mathematical/test_ima_math.py`、`reports/closed_loop/summary.md`（secret-key 0.00 / known-key 0.71） | 2 层 8 头 × ≥4,096 序列需 GPU |
| C10 | TFMA-embed（S4/SH-P） | ✅ 真机（未达下限） | `C10/tfma.independent/SH-P|SH-B/result.json`：出现级 Top-1 **0.1560**（SH-P）vs **0.0110**（SH-B）；合成正控 **0.2427**（<0.90，**未过**，故 `main_table_eligible=false`） | 参考 523k token（下限 4.45M，差 8.6×）；观测 400 请求（下限 1,000）；三先验缺 |
| C11 | SDA（S4） | 🔶 统计版 / ⚠️ transformer 版剔除 | `tests/mathematical/test_sda_math.py`、`tests/regression/test_sda_regression.py` | Transformer 版正控 BLEU-4 未显著 → 维持剔除披露 |
| C12 | 链接（S4/SH-P） | ✅ 真机 | `C12/linkage.exact/ab/result.json`：精确 0.1546（分母 140,166 同 token 对）；Δ 桶 **0.973 / 0.954 / 0.859 / 0.416 / 0.000 / 0.000**（Δ≥64 断崖）；定位正确率 **0.753**（导向要求 ≥0.95 才算强）；T=0.5 TPR 0.195/FPR 0.149，T=0.9 TPR 0.155/FPR 0.00015 | 请求 400（下限 1,000）；定位未达 ≥0.95 |
| C13 | PII 提取（S5） | ✅ 真机（小规模） | `C13/pii.extract/ab_output/result.json`：明文视角 PIIRSR **1.000**（正控过、token 准确率 1.0）；服务器视角 token 准确率 **0.116**（stage-2 覆盖 20k/128,256 类型，覆盖度是主因） | 8 条合成记录（下限 1,000 prompts）；只进 Limitations/附录（D11） |
| C14 | LM/ n-gram 重构（S5） | 🔶 部分 | 同上 payload 的 BLEU-4 / n-gram 字段；`lm_recon.local_lm` 无生成式后端 → N/A | GPT-4 变体需 API |

## 2. 本轮门禁与可复跑证据

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd F:\ConfusionAttack
python -m pytest -m "not neural" -q          # 138 passed
python -m pytest -m "neural" -q              # 59 passed
python scripts\verify_fidelity.py            # FIDELITY CHECK PASSED
python scripts\audit_placeholders.py         # PLACEHOLDER AUDIT PASSED
python scripts\run_full_battery.py --classes C01,C06W,C08,D3,C13 --knn-texts 400 --weight-types 1000 --stage2-types 20000
python scripts\run_full_battery.py --classes C10,C12 --requests 400 --keys 16 --rotate-every 64
```

产物：`reports/privacy_v1/full/<class>/<attack>/<arm>/result.json`（每格一份 App.-A payload，
`schema_valid=true`；`summary_final_*.json` 为汇总）。**13 个 payload 全部通过 schema 校验。**

## 3. 阻塞（环境事实，不是「工程折衷」）

| 阻塞 | 事实 | 影响 |
|---|---|---|
| `osnip` 臂缺失 | 打包编码器 `osnip/config.json` 的 `model_path` 指向混淆 `server/`；换干净表属分布外 | 记 N/A；需第二实例（导向 G5，约 6.3h GPU） |
| 1B CPU 成本 | fp32 前向 ~0.44 s（27 token）、前向+反传 ~1.3 s/步；24 线程 | ISA/VMA-H/IMA/SDA-transformer 的论文预算不可行 |
| 语料规模 | 本机只有 Alpaca 一族（过滤后参考侧 523k token / 22.5k 类型） | TFMA 三先验与 4.45M 参考无法达标（见 F9） |
| 生成式后端 | 无 GPT-4 API | C14 的 GPT-4 变体只能 N/A |

## 4. 与导向决策表的对齐（D1–D13）

| 决策 | 本轮落实 | 证据 |
|---|---|---|
| D1 主表只放 SH | 所有真机 payload 的 `attacker_knowledge ∈ {SH, SH-P}`；KP/KA/EXT-K 全部 `main_table=False` | `tests/unit/test_attack_model_registry.py::test_appendix_only_attack_can_never_be_main_table_eligible` |
| D2 VMA 拆 W/H | 两个独立入口、两个独立 payload；`osnip` 臂的 VMA-W 记 N/A | `C06/*`、`C01/.../osnip` |
| D3 stage-1/2/composed | `C06/vma.weight_rowsort/ab_composed` 三行 + coverage 条件下的 composed | 见 §1 C06/D3 |
| D4 KNN 协议 | 1 key/文本、全词表、部署表；200×5 稳健性属下一轮 | `C01/.../ab` |
| D5 ISA 预算 | 未达 → 不报真机数字 | — |
| D6 VMA-W/IA 规模 | 1,000 types 达标；六类组合/全层投票未跑 | `C06`/`C08` payload 的 `vote_layers` |
| D7 TFMA 强度 | 未达 → `candidate_mode=proposal`、合成正控未过，**不进主表** | `C10/*` |
| D8 IMA 主/附 | 真机未跑；合成里 KP 与 KI 分列 | `ATTACK_MODELS` |
| D9 Top-k 口径 | 全部 top-1/5/10/100 + 随机基线 + 词频分桶 | 各 payload `metrics` |
| D10 声明级阈值 | 无一条主张达标（覆盖率与样本量不足），如实记录 | §1 |
| D11 输出/PII | 只进 Limitations + 附录，`main_table_eligible=false` | `C13` |
| D12 外部臂 | 未接 | — |
| D13 SH-P 主表 / SH-B 敏感性 | TFMA 两条并存；链接按 SH-P | `C10`、`C12` |

## 5. 下一轮（按价值排序）

1. **C01 补齐 1,000 文本**（导向 §6.1 下限；本轮已在跑）并加 200×5 key 稳健性。
2. **C06-W 表 9 六类组合 + 全层投票**：Q/K、Gate、Up、Down、Router 各自的可恢复性会不同（Gate-IA 的 94% 已经提示权重选择决定攻击强度）。
3. **C06-H / C05 放到 GPU 队列**（AR own-prefix 与 ≥1,000 步 ISA）。
4. **TFMA 语料补齐**：CCI3/MedDialog/Huatuo 快照 → 达标后重跑 SH-P 与合成正控。
5. **C08 Attn-IA 真权重**、C09 跨实例 IMA（第二编码器）。
6. **stage-2 提到全词表**（Gate-IA 128k types，约 30–60 min CPU）→ composed 字面值不再受覆盖度限制。

## 附录：第一轮（合成/协议证据）

第一轮的逐类记录（6 类新增 + 7 族控制臂、17 条 gate 臂全部通过、当时认为「四臂不可本机跑」）
保留在 `reports/privacy_v1/phase1/gate_summary.json` 与 git 历史中。**其中「本机跑不动四臂」
这一条已被本轮推翻**（`ab_runnable` 工件 + Alpaca 快照即可覆盖三臂），后续引用请以本文件为准。
