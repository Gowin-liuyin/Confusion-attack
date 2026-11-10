> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# 与 AloePri 论文威胁模型对齐（2026-11-07，取代上一版过度纠正）

> 依据：`C:\Users\yinha\Downloads\2603.01499v2.pdf`（AloePri，`2603.01499v2`），
> 抽取文本见 `.paper_cache/aloepri_2603.01499v2.txt`（逐页可查）。
> **本文件取代 `THREAT_MODEL_CORRECTION_20261107.md` 的第 0–1 节**；上一版把权重组攻击降级为
> "KP 附录"是**过度纠正**，按论文原文应改回主表。

## 0. 论文原文怎么写的（四处直接引文）

| 位置 | 原文 | 含义 |
|---|---|---|
| §3.2 Threat Model（p.4） | "the server is honest-but-curious … **The attacker can access the model** and observe the service's inputs and outputs … 1) **Obfuscation recovery: the attacker knows the obfuscation mechanism** and tries to breach it by exploiting its structural weaknesses" | 半诚实服务器**可以访问模型**、**知道混淆机制**；攻击族之一是"利用结构弱点" |
| §5.1 Overview（p.7） | "The permutation τ is also used to shuffle the embedding and model head at the token level. **To prevent the attacker from recovering τ by comparing plaintext and obfuscated weights** of the embedding and model head, a series of sub-obfuscations … are applied" | 明写"攻击者会拿**明文权重**和**混淆权重**对比"，防护是为了挡住这件事 |
| App. D.1 VMA（p.26） | "**With knowledge of the obfuscated model weights**, attackers can attempt to map each obfuscated token back … recovers permutation matrices Z1 and Z2 from **Y = Z1XZ2 (where X is a known matrix and Y is the observed obfuscated matrix)**"; Table 9 列 "**Known X** / **Observed Y**"，X 例如 `WeWh` | VMA 的输入是**明文权重构造的 X** + **混淆权重构造的 Y**；缺任一项无从谈起 |
| §7.3（p.14） | "**when the noise added to the weights of the embedding or model head is small, each pair of plain and obfuscated weights in Table 9 differs almost only by row-wise and column-wise permutations**. Consequently, VMA can easily recover the secret permutation Π … we recommend setting **αe = 1.0 and αh = 0.2**" | 决定 VMA 强弱的**不是密钥矩阵，而是噪声**；αe=0.5 时 VMA 已恢复 >30% token |
| §8 Protection of Model Weights（p.16） | "AloePri can be further extended to scenarios where **only the server can access the model**. In this case, CPU-TEE or FHE can be leveraged to allow the client to perform obfuscation **without accessing the plaintext model**" | 默认设定里客户端要能读明文权重才能离线混淆；扩展设定里改由 TEE/FHE 让客户端"盲混淆"，而**明文模型仍在服务器侧** → 两种设定下服务器都握有明文权重 |

**结论**：论文威胁模型 = 半诚实服务器 + **明文权重可访问** + **混淆权重可访问** + 知道机制与公式；
**不知道**的是 τ、密钥矩阵 {P̂, Q̂}、噪声种子。

## 1. 因此：上一版的"KP 降级"撤回

| 数字 | 上一版（错） | 本版（按论文） | 理由 |
|---|---|---|---|
| dense-mixing **99.5%** | KP 附录 | ✅ **主表**（`main_table_eligible=true`、`uses_known_pairs=false`） | 与 Table 9 同一前提（Known X / Observed Y）；未用 τ/密钥/种子 |
| Gate-IA **94.1%** | KP 附录 | ✅ **主表** | `Avg(eW_gate)=Avg(ẽW̃_gate)` 本身就是"明文 vs 混淆"两个模型之间的量 |
| composed **62.71%** | KP 附录 | ✅ **主表** | stage-2 同源；stage-1 本来就在主表 |
| C01 KNN **62.90%** | 主表 | ✅ 主表（不变） | 只用部署侧嵌入表 + wire |
| RowSort **0.39% / 3.5%** | 诊断 | ⛔ **诊断**（不变） | 我的构造用 `E_dep[perm[ref]]`，用真值挑行；论文的构造是 `X=WeWh`、`Y=W̃eW̃h`，**不需要 τ** |

## 2. 那"论文数字为什么是那样"——现在可以精确回答

1. **噪声是主防线**：§7.3 明说 αe 小的时候"明文与混淆权重几乎只差行/列置换"，VMA 就能恢复；
   αe=0.5 → >30% 恢复；αe=1.0/αh=0.2 → PIIRSR <3%。
2. **我们的工件是效用工作点**：metadata 显式记录 `alpha_e=0.2, alpha_h=0.01` —— 比论文安全点
   **噪声小 5–20 倍**。在同一攻击下恢复率自然高得多。这正是本方案 §2.2 的提醒：
   "α 工作点不可直接对比，α\* 下 VMA-W 会显著更高（当前 90.6%）"。
   我们 99.5% 与他们的 90.6%（16 types/layer 0 诊断）同量级，不矛盾。
3. **还有一条工件差异**：`client_secret.pt` 的 metadata 写着
   `transform_family="square_monomial_standard_shape"`、`paper_algorithm1_reproduced=false`。
   论文 Algorithm 1 是**逐层两侧**随机密钥矩阵 + 置换 + 噪声；本工件是**单侧稠密基变换**，
   所以 `W'_q = W_q M` 可以被最小二乘直接解出（残差 0.0139 ≈ αh）。
   **这条差异必须写进表注**：这不是"AloePri 被攻破"，而是"该工程变体不在论文的工作点上"。

## 3. 与论文 Table 9 的关系（诚实边界）

- 论文的 VMA 用 **6 类权重组合**（Table 9：Embed/Head、Q/K、Gate、Up、Down/Head、Router），
  每类都是 `Y = Z1 X Z2` 的**乘积**，密钥矩阵在乘积中**相互抵消**，所以 RowSort 能对齐。
- 这些乘积是 **n×n**（n=128,256 → 131 GB/个，float64），本机 CPU 全量不可行；
  我的 dense-mixing 是**另一条路**（解 M 后前向匹配），只适用于"单侧稠密"的工程变体，
  **不能冒充 Table 9 的复现**。
- 因此主表口径应为：`VMA-W（本工件适用变体）= 99.5%`，并标
  `paper_variant=table9_not_feasible_at_full_vocab(131GB)`、`artifact_algorithm1=false`。

## 4. 代码与 payload 落实（本轮已改）

- `run_c06w` / `run_c08` / `run_composed`：`uses_known_pairs=false`、
  `uses_plain_and_obfuscated_weights=true`、`main_table_eligible` 恢复为 true；
  新增 `paper_working_point{paper_alpha_e/h, artifact_alpha_e/h, transform_family,
  paper_algorithm1_reproduced, why_it_matters}`。
- `kw_only` 行保留为**反事实说明**（"若只给混淆权重，则置换不可观测"），不再当作主表前提。
- RowSort 两行保留为诊断，并在 `threat_model` 里写清"构造用 τ"。

## 5. 仍需主控确认（不变的那条）

| # | 问题 | 现状 |
|---|---|---|
| T1 | 本项目的**部署形态**是否等价于论文 LMaaS（服务器拥有并运行模型、客户端只做 token 级混淆）？ | 若等价 → 明文权重可得，本文件口径成立；若是"把混淆模型发给不可信云"，则明文权重不可得，需回到上一版的 KP 口径（两种都已在代码里支持，开关是 `clean_model_public`） |
| T2 | 主表要不要同时列 `paper working point (α=1.0/0.2)` 与 `artifact working point (α=0.2/0.01)` 两行？ | 建议列两行，否则审稿会问"为什么你们 99.5% 而论文 <3%" |
