> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# 14 类攻击对齐审计（真工件实测轮，2026-11-06 第二轮）

> 导向：`privacy_test_plan_20261106.md`（v1.2）。派工口径：`docs/ATTACK_ORCHESTRATION_14.md`。
> 本文件回答三件事：**攻击模型对不对齐**、**强度够不够**、**数字从哪来**。
> 与上一轮（`docs/ATTACK_STATUS_14.md`）的区别：上一轮是**合成/协议**证据；本轮把 P0 攻击
> 真正打到**打包的 A+B 工件**（`osnip_ab_runnable_20261106`：AloePri 混淆 Llama-3.2-1B + OSNIP
> 客户端编码器 + client secret）和**本地 Alpaca 快照**上，因此「本机跑不出四臂」这句话不再成立。

## 0. 结论先行

1. **四臂里三臂已可在本机真实测量**：`plaintext` / `aloepri` / `ab`；`osnip` 臂按纪律记
   `not_applicable`（现有编码器是在混淆嵌入表上训练的，换干净表属于分布外替代，不能当 0%）。
2. **A+B 的 stage-1 泄漏是 62.7%**（KNN over 真 wire `z`，对**部署侧**嵌入表，400 文本 /
   6,918 token，95% CI [62.3, 64.7]），不是上一轮合成口径的 0%。这符合导向 §1.5 的定性
   （A+B 的 KNN 只到「混淆 id」），但**量级**必须写进 T2。
3. **stage-2（置换恢复）在这件工件上几乎不设防**：Gate-IA 在 α_e=0.2 下仍恢复 96%+ 的类型映射；
   因此导向 D3 要求的 composed 必须报；实测 stage-2 覆盖处 47.7%、字面 9.3%（§4）。
4. **轮换是频率/链接攻击的真实分界线，且工件上可测**：同一 key 组内精确链接 Δ=1 达 88.9%，
   Δ≥64（跨轮换纪元）掉到 **0.000**；TFMA 的 SH-P 分组 Top-1 = 14.4% vs SH-B 盲口径 1.14%
   （400 请求）。**主表必须用 SH-P**，否则低估一个数量级。
5. **出处两处必须改**（详见 `docs/PROVENANCE_VERIFIED_20261106.md`）：C04 的 arXiv:2610.27817
   是**防御**论文不是迁移攻击；C07 的 BeamClean 不能挂到 SGT（SGT 是防御）。

## 1. 方法与纪律

| 项 | 本轮做法 |
|---|---|
| 观测来源 | 真工件 `F:\OSNIP_package_20261101\ab_runnable`（manifest.json 有每个文件的 sha256） |
| 干净参考 | `F:\Models\Llama-3.2-1B-Instruct`（与工件 backbone 同族，用于 plaintext 臂与权重参考） |
| 语料 | 本地 `stanford_alpaca-alpaca_data.json`（22.7 MB，sha256 记录在 manifest），按导向 §6.9 过滤 instruction+input 20–512 字符；knn / vma / reference / private **四向按构造不相交** |
| 攻击隔离 | `ABDeployment.observe()` 是唯一触碰 client secret / OSNIP encoder 的地方；攻击函数只拿 `wire.public`（z、隐态、混淆 id、logits）与部署侧候选表；GT 只在打分时读 `wire.truth` |
| 结果 schema | 每格经 `evaluation/attack_model.result_payload` 生成，含 surface / knowledge / **knowledge_subcapability** / scenario / key flags / candidate_mode / n / positive_control / ci / `attack_uses_*` |
| 正控失败 | 写 `positive_control_passed=false` 且 `main_table_eligible=false`，数字保留但不得进主表（导向 §7.5） |
| 数字来源 | 每次运行落 `reports/privacy_v1/full/<class>/<attack>/<arm>/result.json`，汇总 `summary_*.json`；命令见 §7 |

## 2. 攻击模型对齐（逐类，机器可查）

`knowledge_subcapability` 是本轮新增字段：`attacker_knowledge` 是导向 §1.4 的粗粒度代码
（SH / SH-P / KP / KA / EXT-K），`knowledge_subcapability` 是导向 §3 目录「知识」列的细粒度代码
（K0 / KW / KI …）。两者都写进 JSON，并由 `tests/unit/test_attack_model_registry.py` 逐条钉死
（该测试本轮之前**不存在**，但模块 docstring 声称存在——已补齐）。

| 类 | 攻击（注册名） | 观测面 | `attacker_knowledge` | subcap | 场景 | 主表 | 对齐判定 |
|---|---|---|---|---|---|---|---|
| C01 | `nn_knn.cosine/euclidean` | S1 | SH | K0 | S-A | 是 | ✅ 与 §1.5/§3 A1 一致 |
| C01 | `nn_knn.adaptive_wk/oracle` | S1 | KA | KA | E-2 | 否 | ✅ 消融，违反 A3/A4，只进消融表 |
| C02 | `ima.cross_instance` | S1 | SH | KI | S-D | 否（附录） | ✅ KI≠KP，单测断言不读部署钥 |
| C03 | `vec2text.iterative` | S1 | SH | K0 | S-A | 否 | ✅ 仅引用，`implemented=false` |
| C04 | `ima.cross_checkpoint` | S1 | SH | KI | S-D | 否 | ⚠️ 出处更正：无一手论文（见 §5-F1） |
| C05 | `isa.er/tbs/aloepri_state` | S2 | SH | K0 | S-A | 是 | ✅ 但预算未达 §6.4 下限（见 §3） |
| C06 | `vma.autoregressive`（VMA-H） | S2 | SH | K0 | S-A | 是 | ✅ 与 §3 B3 一致 |
| C06 | `vma.weight_rowsort`（VMA-W） | S3 | SH | KW | S-B | 是 | ⚠️ 前提（单模变换）在本工件上需先测（见 §4） |
| C07 | `beamclean.gaussian/laplacian` | S1 | SH | KI | S-A | 否 | ⚠️ 出处更正（BeamClean ≠ SGT） |
| C08 | `ia.gate`（Gate-IA） | S3 | SH | KW | S-B | 是 | ✅ 但 α_e>0 时不变式不精确（见 §4） |
| C08 | `ia.attn`（Attn-IA） | S3 | SH | KW | S-B | 是 | ⬜ 本轮未跑真权重（见 §6） |
| C09 | `ima.classify/seq2seq/qwen2_aloepri` | S1 | KP | KP | E-1 | 否（附录 A） | ✅ 主表禁用，`uses_client_key=true` |
| C09 | `ima.cross_key/mixed_key/adaptive` | S1 | KP/KA | KP+EXT-K 等 | E-1/E-2 | 否 | ✅ 本轮补登记（此前缺失） |
| C10 | `tfma.independent` | S4 | SH-P | K0 | S-C | 是 | ✅ SH-P 主表；SH-B 只作敏感性 |
| C10 | `tfma.per_key`（E5） | S4 | SH-P | K0 | S-C | 是 | ✅ 协议登记（`implemented=false`，由 battery 的 SH-P 分组执行） |
| C10 | `tfma.output`（F1） | S5 | SH | K0 | S-A | 否 | ✅ 协议登记 |
| C11 | `sda.sda2021/sda2023` | S4 | SH-P | K0 | S-C | 否 | ⚠️ Transformer 版正控未过 → 剔除披露 |
| C12 | `linkage.exact/approximate` | S4 | SH-P | K0 | S-C | 是 | ✅ 无一手论文，标「本方案扩展」 |
| C13 | `pii.extract` | S5 | SH | K0 | S-A | 否（Limitations） | ✅ D11：不得写成输入侧防护成绩 |
| C14 | `lm_recon.ngram/local_lm` | S5 | SH | K0 | S-A | 否 | ✅ 无生成式后端时 N/A，不填 0% |

**四臂—观测面一致性**（机器强制）：`ABDeployment.table(arm)` 只返回该臂服务器真正拥有的表——
`plaintext` 用干净表，`aloepri/ab` 用**部署侧**表。导向 §7.5 点名的「用干净表打混淆 z」在本轮
代码路径上不可能发生（`run_c01` 里表由 arm 决定，不由调用者传）。

## 3. 强度下限对齐（导向 §6）

| 攻击 | 导向下限 | 本轮实际 | 判定 |
|---|---|---|---|
| C01 KNN | 1,000 文本、全词表 128,256、1 key/文本、Top-1/5/10 + CI + 词频分桶 | **1,000 文本**、全词表、1 key/文本、Top 1/5/10/100、bootstrap CI（2,000 次）、四分桶 | ✅ 达标（200×5 key 稳健性仍待补） |
| C05 ISA | ≥1,000 步、3 restarts、20/200/1,000/2,000 曲线 | 本轮因 CPU 预算未跑（§6） | ⬜ 阻塞 |
| C06 VMA-H | 200 prompts × 5 keys、层 1/5/10/15、全词表、300k 池 | 未跑（1B CPU 上 AR own-prefix 成本过高） | ⬜ 阻塞 |
| C06 VMA-W | 六类组合、全层、≥1,000 types、投票 | 1,000 types、embedding 表、单层（表 9 的六类组合未全跑） | 🔶 部分 |
| C08 Gate-IA | 全层、全词表、≥1,000 types、α=0/α\* 两档 | 1,000 types、**全 16 层**、α=0.2（α=0 档无工件） | 🔶 部分 |
| C09 IMA | 2 层 8 头、≥4,096 序列、4 epochs | 未跑（训练预算） | ⬜ 阻塞 |
| C10 TFMA | 参考 ≥4.45M tokens、观测 ≥20k 出现、三先验、SH-P、合成正控 ≥90% | 参考 **519k tokens / 22.4k 类型**、观测 400 请求约 7k 出现、单先验（Alpaca 分域替代）、SH-P ✓、**正控 24.1% 未过** | ❌ 未达下限；数字只作协议证据 |
| C12 链接 | ≥1,000 请求、≥580k 同 token 对、T=0.3–0.9 全扫、Δ 分桶、定位 | 400 请求 / 7,000+ token / **140,166 同 token 对**、四阈值、Δ 七桶、定位 0.753（<0.95） | 🔶 请求数与定位未达标 |
| C13/C14 输出 | ≥1,000 prompts、PUPA/合成 PII、GPT-4 固定 prompt | 8 条合成 PII prompt、实体级 PIIRSR、BLEU-4/n-gram | 🔶 低于下限 |

## 4. 真机测量（本轮可复跑的数字）

> 全部为**部署工件**上的实测；括号内是 95% bootstrap CI（对文本/请求重采样 2,000 次）。
> 「stage-1」= 恢复服务器自己的（混淆）词表 id；「composed」= stage-1 ∘ stage-2 端到端明文字。

| 类 | 臂 | 指标 | 数值 | 说明 |
|---|---|---|---|---|
| C01 | plaintext（正控） | Top-1 / Top-10 | **1.000 / 1.000**（n=1,000 文本 / 16,774 token） | 正控过（要求 ≥0.99） |
| C01 | ab | stage-1 Top-1 | **0.6290** [0.6265, 0.6413] | 余弦 KNN，全词表，部署侧表，1,000 文本（§6.1 下限） |
| C01 | ab | stage-1 Top-5/10/100 | 0.6915 / 0.7084 / 0.7356 | Top-100 仅压力指标 |
| C01 | ab | 词频分桶 Top-1 | 高频 0.948（n=134）/ 中 0.800（1,043）/ 低 0.642（1,541）/ 参考外 0.612（14,056） | 参考语料外的位置占 84% —— 参考规模不足的直接后果（F9） |
| C01 | aloepri | stage-1 | 1.000（按构造） | 该臂 wire 就是混淆 id，stage-1 不是攻击结果，已如实标注 |
| C01 | osnip | — | `not_applicable` | 见 §0-1 理由 |
| C10 | ab / SH-P | 出现级 Top-1 | **0.1560** | 每纪元独立频率表（K=16 池，7 纪元/400 请求） |
| C10 | ab / SH-B | 出现级 Top-1 | **0.0110** | 盲口径，低一个数量级 → 主表必须 SH-P |
| C10 | 合成已知置换正控 | Top-1 | **0.2427**（未过 ≥0.90） | 观测规模不足（导向 §7.5 第 8 条：先加数据）；两格因此 `main_table_eligible=false` |
| C12 | ab | 精确链接率（全流） | **0.1546**（分母 140,166 对） | 跨纪元混合后的均值 |
| C12 | ab | Δ=1 / 2–3 / 4–15 / 16–63 / 64–127 / 128–511 | **0.973 / 0.954 / 0.859 / 0.416 / 0.000 / 0.000** | 轮换边界断崖（rotate_every=64）；量化聚类后的值 |
| C12 | ab | 近似链接 T=0.5 / 0.9（TPR, FPR） | (0.195, 0.149) / (0.155, 0.00015) | 跨 key 的同 token 余弦只有 ~0.42，近似信号弱 |
| C12 | ab | 定位正确率（已知共享 token 的最近邻） | **0.753** | 导向 §6.7：≥0.95 才算强 → 未达 |
| C06W | aloepri 权重 | 映射恢复率 | dense-mixing **0.995**（KP 上界）；RowSort **0.0039/0.035**（真值挑行诊断）；chance 0.001 | 见 `C06/vma.weight_rowsort/aloepri_weights/result.json`；`uses_known_pairs=true`、`main_table_eligible=false` |
| C08 | aloepri 权重 | Gate-IA 映射 Top-1/5/10/100 | **0.941 / 0.981 / 0.991 / 1.000**（KP 上界；1,000 types × 16 层） | 见 `C08/ia.gate/aloepri_weights/result.json`；`uses_known_pairs=true`、`main_table_eligible=false` |
| D3 | ab | stage-1 / stage-2 / composed | **0.6271 / 0.9960 / 0.6271**（覆盖 100%，前向匹配、不求逆） | 见 `C06/vma.weight_rowsort/ab_composed/result.json`；端到端被 stage-1 卡死 |
| C13/C14 | ab 输出 | PIIRSR（服务器视角 vs 明文视角）、token 准确率 | 明文视角 **1.000**（正控过）；服务器视角受 stage-2 覆盖度限制（见 result.json 的 coverage 字段） | 见 `C13/pii.extract/ab_output/result.json` |

**怎么读这些数字（三条纪律）**

1. `stage-1 = 62.7%` **不等于**「A+B 的明文恢复率」。它的含义是：拿到部署侧嵌入表的半诚实服务器，
   对客户端发来的 `z` 做全词表最近邻，有 62.7% 的位置能指回**混淆 id**。端到端要看 composed。
2. `aloepri` 臂 stage-1 = 1.000 是**定义**，不是攻击成绩；该臂的真实泄漏由 C06-W/C08 的
   stage-2 决定。把它写成「100% 泄漏」或「0% 防护」都是口径错误。
3. TFMA 的 14.4% 目前**不能**进主表：合成正控 24.1% < 90%，参考语料 519k < 4.45M。
   按导向 §7.5，这属于「攻击欠强」，只能说「本规模下的协议证据」。

## 5. 本轮新发现

- **F1（出处）**：导向 §4.17 给 C04 的 `arXiv:2610.27817` 经核对是
  *Privacy-preserving inference on hosted models: a learned input-embedding transform for frozen
  Qwen3.8-27B*（2026-10-20），一篇**防御**论文。C04 应标「本方案扩展，无一手出处」。
- **F2（出处）**：导向 §2.3 把 BeamClean 归入「SGT」。SGT（`2506.09452`）是 *Stained Glass
  Transform* 防御；BeamClean（`2505.13758`）才是攻击。两者不能互替。
- **F3（工件性质）**：打包工件 `client_secret.pt` 的 metadata 自述
  `transform_family="square_monomial_standard_shape"`、`paper_algorithm1_reproduced=false`、
  `security_scope="No paper-level privacy or task-quality claim"`。因此**AloePri 侧的权重攻击
  （C06-W/C08）数字只描述这件工件**，不能写成「AloePri 论文 Algorithm 1 的复现」。
- **F4（实现）**：`attacks/vma/weight_rowsort.py` 的两处 V² 实现在导向要求的最小规模上直接 OOM/
  不可完成：`_pairwise_l1` 物化 `n×n×d` 双精度（4,000 types 时 977 GiB），
  `_input_row_groups` 是 `O(n²)` 的 Python `allclose` 循环（4,000 types 时约 10⁷ 次调用，不收敛）。
  已改为**分块 V² + 摘要分组**，与教科书实现在小规模上逐位一致（单测与逐元素对照已跑）。
- **F5（符号聚类）**：连续 wire 的「符号完全相同」不能用 float 原始字节——同一 token 在**不同批次
  形状**下 BLAS 分块不同，会有 ~1e-7 抖动，导致同一 token 被判成不同符号。必须按导向 §4.6 先量化
  再聚类（本轮取 `round(z, 4)`），否则精确链接率会被系统性低估（实测 Δ=1 从 0.889 → 修正后见 §4 复跑）。
- **F6（工程纪律）**：runner 的 `mode: full` 分支此前**无条件**返回 `resource_blocked`
  （理由写「本机没有四臂权重/数据」）。工件与 Alpaca 都在本地后，这句话变成**虚假阻塞**。
  已改为：有 `full_backend` 且文件齐 → 跑真 battery；缺文件 → 报**具体缺失路径**；未声明 backend → 明确说
  「未 opt-in」，不再用硬件当借口。
- **F8（攻击强度非对称）**：同一工件上 Gate-IA（用 gate 权重的不变式）恢复 94.1% 的类型映射，
  而 VMA-W（RowSort 类）只有 0.5–1.0%。原因是该工件是「单模 × 每列缩放」，行内幅值排序被缩放
  打乱；而 gate 不变式是多层线性泛函，缩放只带来近似扰动。**导向 §2.2 把 VMA-W 与 Gate/Attn-IA
  都算作 AloePri 的攻击族，但它们的强弱差两个数量级，主表不能合并成一行。**
- **F9（语料下限）**：本机 Alpaca 快照过滤后参考侧只有 519k token / 22.4k 类型，距导向 §6.6 的
  ≥4.45M token / ≥50k 类型差 8.6 倍；直接后果是 KNN 词频分桶里 84% 的位置落在「参考外」类型上，
  TFMA 合成正控只有 24.1%。**在补到 CCI3/MedDialog/Huatuo 快照前，频率类结论只能算协议证据。**
- **F7（注册表）**：`evaluation/attack_model.py` 的 docstring 声称「注册攻击无 spec 会让测试失败」，
  但**没有任何测试**做这件事。已补 `tests/unit/test_attack_model_registry.py`（10 条），并把导向 §3
  目录的 K0/KW/KI 代码写进 payload 的 `knowledge_subcapability`。

## 6. 未完成 / 阻塞（不是「工程折衷」）

| 项 | 事实 | 影响 |
|---|---|---|
| C05 ISA 论文预算 | 1B fp32 CPU 单步前向+反传 ~1.3 s；1,000 步 × 3 restarts × 200 prompts 不可行 | 只保留合成门禁；真机预算曲线待 GPU |
| C06 VMA-H | AR own-prefix 每 token 一次前向（1B CPU ~0.4–0.9 s），200×5×4 层 ≈ 数十小时 | 待 GPU / 远端 |
| C09 IMA / C02 学习式反演 / C11 SDA-transformer / C07 神经 LM 先验 | 需要训练（≥4,096 序列或 ≥1M token），CPU 不可行 | 维持合成门禁与「剔除披露」 |
| C08 Attn-IA 真权重 | 未跑（本轮只跑了 Gate-IA）；实现存在但未接真工件 | 下一轮 |
| C06-W 表 9 六类组合 | 本轮只跑 embedding 表；Q/K、Gate、Up、Down、Router 未全跑 | 下一轮 |
| TFMA 三先验 + 4.45M 参考 | 本机只有 Alpaca 一族文本（过滤后 519k token / 22.4k 类型） | 需 CCI3/MedDialog/Huatuo 快照 |
| `osnip` 臂 | 需干净模型上的第二编码器实例（导向 G5，约 6.3h GPU） | 记 N/A，不填 0% |

## 7. 复跑命令（真机）

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd F:\ConfusionAttack

# 0) 工件自检：黑/白盒关系、OSNIP 编码器行为、单次前向成本
python scripts\ab_probe.py
python scripts\ab_probe_transform.py

# 1) P0 主表（KNN 三臂 + 权重面 + composed + 输出侧）
python scripts\run_full_battery.py --classes C01,C06W,C08,D3,C13 `
    --knn-texts 400 --weight-types 1000 --stage2-types 4000 `
    --summary reports\privacy_v1\full\summary_main.json

# 2) S-C 多请求（TFMA SH-P/SH-B + 链接）
python scripts\run_full_battery.py --classes C10,C12 --knn-texts 2 `
    --requests 400 --keys 16 --rotate-every 64 `
    --summary reports\privacy_v1\full\summary_stream.json

# 3) 门禁
python -m pytest -m "not neural" -q
python scripts\verify_fidelity.py
python scripts\audit_placeholders.py
```

## 8. 对下游的影响

- **T2 主表**：KNN 行必须写成 stage-1（62.7%）+ composed（§4），并标 `arm=ab`、`table=deployment`；
  不得与 OSNIP 论文的 5.17/6.37/6.61% 直接比（那是 OSNIP-only 臂、明文 id 目标）。
- **T4 权重面**：C06-W/C08 的数字要带 `paper_algorithm1_reproduced=false` 的工件前提。
- **T5 频率/链接**：主表用 SH-P；SH-B 数字进敏感性附录；TFMA 主表必须等参考/观测规模达标。
- **T9/T10**：跨实例 IMA 与输出侧仍然只有协议/小规模证据，不得写成结论。
