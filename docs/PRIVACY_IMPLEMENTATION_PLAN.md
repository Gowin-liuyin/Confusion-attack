# 隐私评测代码审计与实施计划

状态：2026-11-07 只读代码审计及后续施工任务，**不是第二份实验规范，也不是修复已经完成的声明**。

唯一实验标准是 [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md)，正式预算和对照组执行其第 8 节。本文只记录当前代码缺口、改动位置、执行依赖和验收证据。参数改变只能修订权威协议。本计划允许维护自身完成状态，不另立攻击强度或安全模型。

配套交接：[分阶段 agent 提示词](PRIVACY_AGENT_PROMPTS.md)。所有路径相对仓库根目录。

## 1. 审核结论与范围

**已有可复用算法和真实模型探针；尚不能称为可直接支撑所有正式对比的可靠评测系统。** 核心问题不是再增加攻击名字，而是正式入口绕过了部分算法实现/隔离约束，真值在候选构造和 LM 输入中渗漏，分母、实验语义、版本绑定和各防御接入尚未统一。

- 本次审核覆盖 `schemes/ab_deployment.py`、`evaluation/real_*.py`、`attack_model.py`、数据划分、运行脚本、结果审计及相应测试；算法模块按其实际调用关系核查，未对所有论文做从零独立数学证明。
- 现有全套测试实跑：`python -m pytest -q`，**205 passed，79.29s**。通过不代表已覆盖下述缺陷。
- 新增只读复现脚本：[reproduce_findings.py](../reports/code_audit_20261107/reproduce_findings.py)，实跑确认 **10 个探针观测**（其中若干属于同一缺陷），输出 [observations.json](../reports/code_audit_20261107/observations.json)。不加载完整模型、不修改算法和既有实验结果。
- [audit_manifest.json](../reports/code_audit_20261107/audit_manifest.json) 保存当时 Git HEAD、163 个代码/配置/测试文件哈希及测试结果。后续 agent 必须先检查源码是否已有变化，复验发现，不能盲按旧行号改。
- 此次未修复生产代码，未重新训练防御/攻击，未执行完整正式评测或正式外部基线复现。不能承诺“无懈可击/完美”；交付目标是权限可审计、实验可复现、实现与算法一致、每项规范有可执行验收，残余限制显式报告。

## 2. 已确认问题与风险排序

P0 阻塞正式实验可信度；P1 阻塞完整覆盖、可扩展性或重复运行。代码行号以审核快照为准，链接直接定位实际源码。

### F01 / P0：Gate、Attn 及输出映射用秘密置换构造候选

- [run_c08](../src/obfuscation_attacks/evaluation/real_battery.py#L608) 用 `E_dep[perm[idx]]`；[recover_mapping_gate_ia](../src/obfuscation_attacks/evaluation/real_battery.py#L705) 也用真实置换建部署候选并写 mapping 的 key；[attention_invariants](../src/obfuscation_attacks/evaluation/real_weight_attacks.py#L55) 同样先取秘密对应行。
- 这不是“只有打分时看真值”：攻击前就知道了评估词在部署表中的准确候选集合，候选范围也不是全词表。`inv_dep` 与 `inv_clean` 按真实配对构造，不能按未知置换恢复实验引用。
- 复现：用禁止 `perm.__getitem__` 的对象调用 Gate/Attn，尚未读取权重即触发秘密访问。现有 isolation 测试主要保护 dense-mixing，不覆盖这两条真实入口。
- 修复：公共输入只包含独立可构造的原表/部署表及公开权重；按公开部署行索引查询完整候选，GT 仅 scorer 持有。Gate/Attn/输出调用全部接同一安全接口。历史结果追加失效审计，不能覆盖旧测量或把它们仍称主表。
- dense-mixing 复合链使用公开矩阵解及全表前向匹配，与本问题不同，不能因为它的隔离测试通过就推断 Gate/Attn 也通过，也不能反过来未经复核撤销所有独立复合结果。

### F02 / P0：LM 重构收到 oracle 还原文本，且误标成输出侧攻击

- [run_c14](../src/obfuscation_attacks/evaluation/real_remaining.py#L273) 实际对输入 wire 做 KNN，随后 `dep.inv_perm` 解码，再喂给 `lm_rewrite`；它既不是攻击者可得到的 LM 输入，也不是服务器生成记录。
- 复现：保持公共 wire 完全相同，只替换秘密 inverse permutation，传入 LM 的字符串随之改变。
- C03/C07 同样存在秘密映射后报告文本指标的路径；即使注释说“上界”，也需独立 `oracle` 输出和明确指标名，不能混进普通 `bleu4`。
- 修复：使用攻击估计 mapping 串联；oracle 输出只能评分端单列。按实际观测面拆输入重构、输出恢复、oracle 诊断。修改读取秘密的入口和调用链，不只修改标签。

### F03 / P0：公共 key ID 可直接还原 key，攻击函数还拿到整套秘密对象

- [ABDeployment._key_for](../src/obfuscation_attacks/schemes/ab_deployment.py#L235) 从整数 seed 生成 key，[observe](../src/obfuscation_attacks/schemes/ab_deployment.py#L299) 将同一个 seed 字符串作为 `WirePublic.key_id`；多数调用使用请求序号作 seed。`build_stream` 也把 seed、明文 ID 和向量放在同一字典。
- 复现：仅用公开 ID 对应整数，调用相同生成函数，得到逐元素一致的 key（探针使用自行构造的假密钥，不打印任何实际秘密）。
- 多数真实攻击 `run_*` 收到完整 `ABDeployment`，因此类型注释称“只看public”并不构成调用边界。
- 修复：公开组标识与秘密无可计算关系；生成端与攻击端资源分离，公开日志/任务缓存不存 seed/key。目标秘密和 scorer 真值单独保存；S-encoder 是否公开按场景授权。Python 逻辑隔离应明确不等于抵御同进程恶意反射；正式 runner 可用序列化公共观察作为进程边界。

### F04 / P0：真实 VMA-H 入口执行的是 observed-prefix 提议变体

- [real_vma](../src/obfuscation_attacks/evaluation/real_vma.py#L26) 复用真实收到的 wire 前缀，并只从前一位置 logits 提议候选；[vma_h_attack](../src/obfuscation_attacks/evaluation/real_vma.py#L83) 跳过首 token。它没有为各层维护独立恢复前缀。
- 使用 observed-prefix 本身可以是半诚实服务器的合法更强观测利用，但不能命名为原文 recovered-prefix 全词表攻击。当前还只在 payload 中取第一个层结果，Clean mask 没有落实完整独立的逐位置 Clean 指标。
- 修复：正式 recovered-prefix 路径接现有原算法与等价缓存回放，另保留明确命名的 observed-prefix 自适应变体；两者都保留，不人为排除对我方更强的合法攻击。首 token、BOS、全词表、各层记录和覆盖率必须可复查。

### F05 / P0：真实 ISA 入口绕过已实现的 ER/TBS 核心

- [real_isa.isa_attack](../src/obfuscation_attacks/evaluation/real_isa.py#L20) 自写 Adam 连续输入优化、随机表行初始化，缺少 TBS 的完整基与重参数化；[真实结果入口](../src/obfuscation_attacks/evaluation/real_battery.py) 仍登记为 `isa.er`。仓库已有 `attacks/isa/er.py`、`tbs.py`，明文脚本能调用它们。
- 单纯把 `steps` 调大不能把这个自定义优化器变成 ER/TBS。checkpoint 的 loss 是更新前值，但候选来自更新后的变量；重启选择也需统一到相同状态和残差。
- 修复：单一真实模型适配器，正式路径直接调用原 ER/TBS；共享评分与预算回调，不维护第三套近似数学。用微型真实前向 fixture 做与原类的差分检查，并在目标模型上记录正式预算曲线。

### F06 / P0：学习式反演删掉 OOV 测试位置后报告 ASR

- [ridge_inverter](../src/obfuscation_attacks/evaluation/real_inversion.py#L106) 和 [mlp_inverter](../src/obfuscation_attacks/evaluation/real_inversion.py#L149) 仅对 `test_keep` 位置计分，训练中没见过的测试标签直接消失。
- 复现：两条测试位置仅一条在训练标签中，正确恢复那一条；当前 Top-1=1.0、denominator=1，真实无条件恢复率应为0.5、denominator=2。当前已报coverage不能替代正确主分母。
- 修复：完整测试位置保留，未覆盖候选填不可恢复；同时给conditional、unconditional和coverage。训练/验证/测试按文档与 key/实例角色分别划分；神经反演输出全词表，不以小头分类器代替正式神经反演。

### F07 / P0：输出/PII 实验未执行真实生成

- [run_c13_c14](../src/obfuscation_attacks/evaluation/real_battery.py#L1433) 将手写 `prompt + answer` 一次上传，用输入 wire 的尾段模拟“输出”。参数 `max_new_tokens` 未决定实际生成过程。
- 当前实体位置计算依赖分别tokenize的长度/偏移，BPE拼接和截断可改变实体边界。不能把这八个teacher-forcing记录称为真实模型生成隐私。
- 修复：按当前正式部署逐步采样、记录真实服务器消息、客户端还原输出；输入与输出各自用字符span/offset mapping评分。脚本答案仅保留为正控。禁止用真正答案构造攻击侧输入。

### F08 / P0：机器 schema 可以接受与事实矛盾的合格标记

- [result_payload/validate_payload](../src/obfuscation_attacks/evaluation/attack_model.py#L602) 允许调用者覆盖 eligibility、仅检查部分枚举；证据文件、正控、预算和信息权限没有形成认证链。
- 复现：`n_prompts=-1`、`positive_control_passed=false`、`main_table_eligible=true` 的payload仍可返回空验证错误。C02 还以“跨实例”名字登记，却实际用部署实例配对训练，类别、权限和执行语义不一致。
- 修复：区分实验事实的schema校验和协议审核，未知/矛盾fail closed；正式结论由只读审计从证据生成，不由run函数自行授权；S-public/S-deploy/S-encoder/S-pair映射落到输入清单，不把KP、公开编码器和知道key混为一谈。

### F09 / P1：结果按固定路径覆写，数据/代码哈希绑定不足

- [write_payload](../src/obfuscation_attacks/evaluation/real_battery.py#L114) 同一类/攻击/臂固定 `result.json`，换seed、数据、模型或预算可覆盖旧运行；schema错误仍写出，不停止。汇总还以class:attack@target作key，可能吞掉多次运行。
- `source_sha256/data_sha256` 目前是对调用者描述对象（路径、数量等）序列化哈希，不必然绑定文件字节和实际内容；脏代码清单/协议版本、逐条预测和训练祖先没有统一记录。
- 修复：内容绑定manifest+唯一run_id，拒绝写已有完成run；临时文件原子提交、job状态及恢复配置核验、全量逐条预测/计数重算。只审计生成新文件，不改原测量。

### F10 / P1：现有协议审计器不是可用于未来正式结果的验收器

- [audit_results_against_protocol](../scripts/audit_results_against_protocol.py#L82) 从硬编码表查预算，忽略实际seed/stream/训练token及step证据；即使记录足够步数仍固定追加“步数不足”；`audited_paper_scale`永远false，不能认出将来完整成功的实验。
- 复现：not_applicable在末尾被gaps覆盖为below_budget；足额step仍被判不足。C06一个floor混合W/H两种不同实验。脚本中的ISA正控阈值和SDA固定BLEU也不完全来自现行协议。
- 修复：审核具体attack variant和profile，读取真实证据及协议绑定，不把“历史上都没完成”写成恒定规则。用真实可满足的完整fixture作为正例、逐项删证据作为反例；预算达标但攻击高恢复率也应通过实验质量审核。

### F11 / P1：流量缓存和链接采样会改变实验含义

- [build_stream](../src/obfuscation_attacks/evaluation/real_battery.py#L919) cache只检查是否有 `stream`，不看key数/轮换/数据/工件；重复扫描复用同cache会返回错误的旧流。
- [run_c12](../src/obfuscation_attacks/evaluation/real_battery.py#L1299) 每个共同token只取首次位置；负对的另一端固定为第二请求首token，位置和词频分布不平衡。只有四个余弦阈值，不是独立验证集确定的固定FPR；缺少KNN对齐、恢复后链接和可见分组多观测聚合。
- 修复：cache以完整公共配置/数据内容作key，key身份与epoch分离；匹配的端点分层抽样，真值制作评分集合但不传入攻击函数；分别报same-key、cross-key、cross-epoch和复用key的流量。

### F12 / P1：点估计与置信区间不是同一个指标

- [bootstrap_ci](../src/obfuscation_attacks/evaluation/real_battery.py#L64) 对每文档ASR平均得到macro区间，而KNN/复合链主点估计按全部token计算micro。长短文档混合时二者不同。
- 复现fixture：两文档分别1/1、0/9命中，micro=0.1，而当前bootstrap中心=0.5。
- 修复：以文档/会话为重采样单位，每次重算 `sum(hits)/sum(tokens)`；另报macro。跨key、训练seed、同文档多观察的聚类层级明确，不将重复token对当独立观测。

### F13 / P1：TFMA 的 Top-100 最多只有七个有效候选

- [_rank_window_candidates](../src/obfuscation_attacks/evaluation/real_battery.py#L973) 默认窗口返回至多7项，外层补`-1`到100，所以Top-10/100不是相应大小的候选搜索。
- 复现：20类型输入的所有符号候选最大长度=7。
- 修复：排名与全局分配规则输出真实有效有序候选，并明确匹配算法；正控、三种先验及规模扩展落实到正式训练/运行，不以pad替代搜索。

### F14 / P1：可选攻击的名字与真实算法不一致且扩量会爆内存

- [vec2text_refine](../src/obfuscation_attacks/evaluation/real_generative.py#L75) 没有训练corrector；从L1最近邻初始化后，只接受更小的同一L1距离，按其目标本来就几乎不能改善，不能称复现Vec2Text。
- [beamclean_recover](../src/obfuscation_attacks/evaluation/real_generative.py#L44) 文档写高斯L2似然，但实现先算L1再平方，且sigma固定；对应算法归属/先验与实际不同。
- 这些路径及简化ISA用 `observations[:,None,:]-table[None,:,:]` 物化三维张量。以6位置、128256词表、2048维FP32计，仅差值约5.87GiB，尚未算临时数组/模型；不能扩到完整序列。
- 修复：可选项退回related-work或明确重命名工程启发式；正式需要时接正确算法。距离/候选分块复用已实现的精确搜索，与小数组穷举一致；在目标尺度测峰值内存，不用减候选绕过。

### F15 / P1：缺少完整多防御/多任务接口，身份与截断不受控

- [ABDeployment](../src/obfuscation_attacks/schemes/ab_deployment.py#L49) 默认原模型路径是Instruct，而元数据指向base，尚无启动时权重血缘认证；`encode`静默截到`max_len`，入口固定max_len=64，可能把正式长文本变成短探针。
- runner固定三臂和单一A+B工件；only-OSNIP缺工件报N/A，实际是尚未提供独立适配。四个新增外部方案、分类/生成任务差异、真实服务器观测及成本边界没有接入。
- 修复：八臂manifest、任务/观测能力声明、准确匹配底座、完整长度记录和逐格适用性；共享编排不虚构统一“向量wire”。不能借口接口缺失把适用对手排除，也不能把不支持的生成编成0分。

## 3. 实施顺序与依赖

| 阶段 | 修复目标 | 主要改动范围 | 依赖 | 完成证据 |
|---|---|---|---|---|
| W0 | 权威协议变成可校验paper profile，建立冻结清单 | `configs/paper/`、CLI/runner预检、schema、对应tests | 无 | 每个冻结字段映射，非法降预算拒绝；探针显式分离；未填身份/来源阻塞 |
| W1 | 秘密隔离与真实部署边界 | `data/contracts.py`、`schemes/base.py`、`ab_deployment.py`、公共观察/评分端接口 | W0 | attack端无生成秘密/真值引用；key_id不能还原key；公开底座和S-encoder授权正确 |
| W2 | 指标、记录、只读验收可信 | `metrics/`、`scoring.py`、`attack_model.py`、审计器/写入器 | W1 | 无条件分母、micro/macro CI、不可覆写run、正反例验收、真实内容绑定 |
| W3 | 单请求与权重攻击真实算法 | `real_vma/isa/weight_attacks`、相关`attacks/`、`real_battery`桥接 | W1+W2 | 无τ构造候选；ER/TBS差分；全词表回放及复合恢复；正控/预算曲线 |
| W4 | 多请求、频率、迁移/聚合 | stream/linkage/tfma/sda、real_inversion/remaining桥接 | W1+W2 | 固定FPR链接、跨key聚合、自建实例与KP隔离、TFMA真Top-k、SDA真正打部署 |
| W5 | 七对照+我方八臂接通 | 各防御adapter、数据/工件manifest、训练与任务适配 | W1+W2接口冻结 | 八臂×任务×观测矩阵，原算法映射、许可证/commit、onlyB独立训练，不支持项有理由 |
| W6 | 真实生成输出审计与效用/效率对齐 | 生成记录、PII/text评分、utility/perf runner | W3+W5 | 真实解码轨迹，实体offset正确；相同工件隐私/效用/成本；正式job计划完整 |
| W7 | 独立验收、冻结正式任务队列 | 集成/回归tests、审计器、run manifests、只读审核 | W3–W6 | 所有协议要求coverage，完整fixture可过、破坏fixture必拒、实际正式预算预检、缺口报告 |

W3/W4/W5可在公共接口稳定后分工，但当前 `real_battery.py` 与CLI共享，必须由单一集成人串行改动；不能同时派多个agent修改同文件。worker只接独占模块，集成人负责桥接和全套回归。W6必须在真实生成协议确定后进行。

## 4. 每阶段必须交付的内容

### W0：冻结配置与预检，不复制实验政策

- 将权威协议第8节展开为可执行配置。YAML/JSON是执行载体，不是独立标准；保存协议hash、规则版本和每项来源锚，增加一致性检查，协议变更使旧profile显式过期。
- 正式profile必须显式提供模型/防御/攻击/场景/数据split/预算；入口缺参数、未知枚举、重复或无效层、负数、候选不足、数据不足、只给mock工件都拒绝正式启动。禁止默默忽略未知class。
- 参数验证结果给出精确短缺与下一步，而不是统称无算力。`--plan-only`只生成完整job矩阵/估算和依赖，不加载/训练模型；`probe`不会被正式聚合器采信。
- 交付 requirements→config→preflight→test 映射表，放在权威协议引用的执行产物中，不新建政策文件。只固定已经核实的revision和license；不可确认就标依赖阻塞，不猜哈希。

### W1：公开视图、生成端、评分端分开

- 优先复用已有AttackContext能力接口，避免另造多层框架。公开视图包含服务器本来可读的权重、完整消息与合法复算函数；同时收紧不应公开的目标secret、truth、encoder。
- 原始/部署权重配对是否合法由场景manifest决定；S-deploy禁读原权重，S-public允许匹配的公开原权重，S-encoder按公开状态提供权重。不要为了“所有attack_uses_encoder=false”削弱真实威胁模型。
- 生成/评分允许持secret，但不可用GT挑攻击候选或喂LM。接口隔离不能只靠函数签名注释；在序列化公共记录上跑攻击和验收，检查闭包/对象属性不再携带秘密。
- 每个真实run路径都要spy/sentinel测试，覆盖Gate/Attn、LM、VMA、IMA、stream，不能只测dense-mixing工具函数。保持public输入相同而任意替换GT时，预测必须完全相同；只有评分可以变。

### W2：统一计分与可追溯产物

- 引入明确的 stage1、estimated-stage2、composed、oracle 命名空间；普通生成/重构字段不得混入oracle。实体分母包含未恢复/未覆盖实体，空集合用null和原因。
- 统计按协议重采样独立单位，存逐条命中/候选/coverage与长度；测试短长混合、OOV、首token、空输入、特殊符号、跨tokenizer、不同seed合并。
- run_id至少区分model/arm/场景/workpoint/seed/数据/profile；manifest采用真实文件流式哈希，hash描述字符串不能冒充内容证明。checkpoint父链/恢复配置核对，恢复不能把零步标完成。
- 审计器只读结果，写独立判定；不依赖run自报“合格”。schema要验证非负计数、分子≤分母、指标重算一致、候选大小真实、场景权限相符、状态不矛盾。所有命令失败/显存不足/中断保留状态，不转换成0泄漏。

### W3/W4：复用正确算法，补真实攻击链

- W3先修F01，再把VMA正式算法、ER/TBS接到真实模型；所有距离分块与缓存优化必须精确等价，保存目标尺度测量。实际查询/更新次数从执行计数得出，不由配置预算直接冒充已完成次数。
- W4训练/验证/测试manifest分开，KI训练集只来自攻击者自己实例；跨key测试key独立，KP用真实配对单列且不等于泄露key。攻击可按验证误差选强配置，不看私有测试答案。
- TFMA/SDA使用同一冻结真实流，不因为SDA正控失败就宣布防御有效；可选启发式退出正式计数。重复观察聚合只使用可见分组，真实同token分组只放oracle对照。
- 不先跑完整预算再修错误。小fixture测逻辑正确属于代码验收，不是低强度正式实验；通过后冻结正式profile，算力不足阻塞排队，不降标准。

### W5：额外混淆方案的接入设计

| 臂 | 必须提供 | 防止的错误 |
|---|---|---|
| 明文 | 每任务同底座/同训练机会参照、原生路径正控 | 不同模型/模板的分数混比 |
| A-only | 与AB准确同一A工件，另外区分完整论文实现和工程变体 | 把自选简化A称为正式AloePri被破解 |
| OSNIP-only | 干净底座独立训练encoder、key/观察manifest | 把AB encoder换干净表当成原方案 |
| SentinelLMs | 正式来源、固定几何变换与原训练适配、真实上传接口、EDNN桥接 | 只跑普通KNN后宣布强防护；把分类方案硬当生成方案 |
| ObfusLM+ | +版的模型/词表混淆、必要微调、真实服务器权重/ID与生成输出 | 只复现弱基础版或用普通置换替代增强机制 |
| InferDPT | 原文扰动和本地提取两模块、真实远端可读文本、两端成本 | 只跑RANTEXT却报完整方案效用；提取端私有明文喂给攻击者 |
| DP-Forward | 正确噪声定义/注入层/预算、训练适配、任务支持声明 | 用DP-SGD代替前向隐私；不同epsilon定义混比 |
| A+B | 同一真实协议的上传/逐步解码/key生命周期/成本；自家工件也过相同验收 | 为我方设私有捷径或少给服务器合法观测 |

- 原文、作者代码、下载工件和license缺失是明确依赖；不得从名字猜实现。每个新方案先通过原论文小型行为一致性，再在冻结的完整任务上复现；小型一致性本身不是正式性能结果。
- 每个adapter暴露正确消息类型和能力，不给无语义的dummy向量统一表面。输入文本、embedding、ID、权重和输出各个攻击按适用性分组；分类独立表有同底座我方参照才允许比较胜负。
- 所有八臂出现在job矩阵，即使暂未集成也写具体阻塞。适用攻击未实现不等于不适用，基线缺代码不等于攻击成功率0。

### W6：效用、效率与输出必须测同一个系统

- 从同一部署adapter获取逐步生成记录，生成中prompt、continuation、实际重传片段分开；保存每步可见metadata/logits或可复算引用。攻击者只能访问公开trace，评分器访问原始输入/最终客户端明文。
- PII使用真实标注集与独立合成补充，字符span驱动token对齐；测试提示尾截断、BPE边界、重复实体、部分恢复、未在输出中出现和EOS提前结束。
- 效用/成本命令共享manifest和profile，不允许一边prompt-only一边完整前缀。结果绑定相同模型/防御/seed/workpoint，批量汇总拒绝拼接不同checkpoint的好效用和好隐私。
- 正式出表保留差的结果和未通过非劣界限的工作点，完整标记失败/缺失。相同epsilon只有相同定义内可解释，不按三角图坐标强称形式化更安全。

### W7：独立验收与交付终点

代码就绪和实验完成分开验收：

1. **代码就绪**：所有P0关闭，相关P1关闭或经协议明确列为非必需；全量tests、算法保真、实际CLI整链fixture与八臂能力矩阵通过；正式profile预检能构造完整job队列，缺失来源/工件明确阻塞；没有一项仅因名字或目录自动合格。
2. **正式实验完成**：每个必需cell实际执行协议预算、正确正控和饱和验证，样本/seed/逐条证据完整，未知/阻塞不冒充0；只读审计通过，已发表基线复现和我方同底座公平性可复核。
3. **结论可写**：统计支持对应隐私/效用主张，无法支持则写边界/折中；攻击高恢复率是有效发现，不是代码验收失败，更不能为了让图好看而修改算法。

最终交付：代码diff、有效测试和实际命令、profile与manifest hash、逐格审计报告、旧结果的追加纠错清单、正式job队列与算力估算、尚未完成的研究限制。未完成正式队列时明确只达到代码就绪，不得写“所有实验完成/绝对安全”。

## 5. 本轮修改边界

本轮只增加代码审计探针/证据、实施计划与交接提示词，并按用户补充在权威协议第8节冻结强度和基线接入要求。**没有修复生产攻击代码，也没有改写已有隐私结果。** 后续实现按W0→W7执行；禁止顺手删除历史结果、把当前观察数字改成“更漂亮”的数字，或启动未经资源预检的长期任务。

## 6. 第 4 轮进度（2026-11-07，W7 真实执行）

本轮把 C01 从正式 CLI 接到真实执行器，并纠正底座血缘判断。证据记录在
[reports/privacy_rebuild_20261107/round4/](../reports/privacy_rebuild_20261107/round4/)。

- **正式 executor**：新增 `src/obfuscation_attacks/evaluation/paper_executor.py`。
  逐格再次校验 manifest 与真实文件（模型 config/tokenizer/weights、工件全部文件、
  语料源哈希与冻结 KNN 文本摘要、协议哈希、模型变体），任何缺项/哈希漂移/不支持的
  arm/scenario 都在构造部署对象前拒绝。支持单元仅
  `knn_l2`/`knn_cosine` × `plaintext`/`aloepri`/`ab` × `S-public` × 1B base。
- **CLI**：`scripts/run_full_battery.py` 新增 `--execute`、`--scenarios`、
  `--root-key-file`、`--result-root`；未传 `--execute` 时不再静默成功，而是逐格
  `resource_blocked`。
- **逐格预检**：`paper_profile.py` 的数据门槛改为按攻击组取（KNN 不再要求 IMA 的
  8192 文档/1M token），`preflight_formal_start` 支持按 job 的模型子集校验；
  faithful 注册支持 scenario。计划矩阵仍保留全部 2688 格，未支持格显式阻塞。
- **结果证据**：`run_c01` 保存逐位置 `predicted/truth/rank`、逐文本分子分母、
  micro/macro 区间、真实数据/工件/模型哈希、执行代码身份；查询矩阵以 npz 证据落盘，
  独立审计可从查询重新跑 KNN。
- **独立审计**：新增 `scripts/audit_paper_run.py`，重算全部 Top-k、核对分母、
  重哈希模型/工件/语料、重跑 KNN、复核正控；篡改逐条预测、预算计数、正控或哈希
  得到 `evidence_mismatch`/`control_failed`。
- **底座血缘**：下载 `unsloth/Llama-3.2-1B`（revision 9535bd9b…）并做 monomial
  列对齐 + 残差噪声比鉴定；A 工件来自 **1B base**（残差/σ=0.1999，α_e=0.2），
  不是 Instruct（0.2928）。上一轮“只有 Instruct”的判断按追加记录纠正。
- **未接通**：VMA/ISA/Gate/Attn/linkage/TFMA/SDA/output 的算法实现保留，但没有
  paper-v1 executor 入口，`paper_runner.UNWIRED_ENTRIES` 逐项记录原因；不把这些
  单元标成 configured。
- **首个真实单元**：C01 S-public 主单元六格（`plaintext`/`aloepri`/`ab` × L2/余弦，
  1000 texts × 1 key，全词表 top-100）与 200×5 fresh-key 稳健性一格已通过正式 CLI
  真实执行；独立审计重算逐条 Top-k、重哈希模型/工件/语料、从查询重跑 KNN、按密钥
  重算上传观测、递归复核正控，结果见
  [round4_report.md](../reports/privacy_rebuild_20261107/round4/round4_report.md)。
  EDNN 子实验未跑，不能称 C01 全组完成。
- **统一方案适配层**：新增 `schemes/adapters.py`、`evaluation/generic_runner.py`、
  `evaluation/classification.py`。八臂各有可执行契约；本地三臂 adapter-ready，
  OSNIP-only 缺干净 encoder，外部四臂在作者 backend/许可证/commit 提供前 fail-closed。
  `scripts/export_adapter_matrix.py` 输出 8×28 完整矩阵（224 格，状态含精确缺项），
  `scripts/run_classification_probe.py` 用真实 DistilBERT SST-2 验证分类面。
- **未完成**：3B/Qwen 正式格、外部四臂、SDA/ISA/VMA 正式预算、效用/效率全套。
  CPU-only 环境与缺失工件仍是明确阻塞；没有任何历史结果因本轮升级为
  `audited_paper_scale`。
