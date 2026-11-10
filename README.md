# CA：Confusion Attack 隐私评测

CA 用于评估 AloePri、OSNIP 及组合方案暴露的模型权重、协议向量和输出记录。仓库将数据与权限约定、算法实现、评分和运行入口分开，可用合成数据验证评测管线。

**[实验评测协议](docs/PAPER_EVALUATION_PROTOCOL.md)** 定义对照组、威胁模型、正式预算和结果验收。下面是实现目录，不是已完成的论文实验清单。原论文设置、针对 AloePri 的适配、针对 OSNIP 的适配和项目扩展应分别报告。

| Family | Attacks (registered name) |
|--------|---------------------------|
| NN / KNN | `nn_knn.euclidean`, `nn_knn.cosine` |
| VMA | `vma.autoregressive` (original), `vma.weight_rowsort` (AloePri), `vma.osnip_projection` (OSNIP) |
| IA | `ia.ednn` (original EDNN), `ia.gate` (AloePri), `ia.attn` (AloePri) |
| ISA | `isa.er`, `isa.tbs` (original), `isa.aloepri_state` (AloePri) |
| IMA | `ima.classify`, `ima.seq2seq` (InvBERT), `ima.qwen2_aloepri` (AloePri, 2 layers / 8 heads) |
| TFMA | `tfma.independent` (AloePri), `tfma.global_assignment` (extension) |
| SDA | `sda.sda2021` (frequency-rank + seq2seq), `sda.sda2023` (recurrence + causal LM) |

14-class implementation catalogue (not a count of completed paper-scale experiments; see the unified protocol):

| Class | Attack (registered name) | Paper / provenance |
|-------|--------------------------|--------------------|
| C02 | `ima.cross_instance` (KI: attacker's own mechanism instance) | InvBERT (arXiv:2109.10104), DeepInvert (arXiv:2608.04477) |
| C04 | `ima.cross_checkpoint` (KI transfer; project extension) | no source paper - the plan's arXiv:2610.27817 was checked and is unrelated |
| C07 | `beamclean.gaussian`, `beamclean.laplacian` | BeamClean (arXiv:2505.13758); SGT (arXiv:2506.09452) is the defense |
| C12 | `linkage.exact`, `linkage.approximate` (SH-P / SH-B) | project extension (plan P0-9) |
| C13 | `pii.extract` (entity-level PIIRSR) | AloePri ref. [14] Papillon/PUPA dataset |
| C14 | `lm_recon.ngram`, `lm_recon.local_lm` | InferDPT / RANTEXT reconstruction metrics |

[ATTACK_PROVENANCE_14.md](docs/ATTACK_PROVENANCE_14.md) records paper anchors and citation corrections.
The registry in `evaluation/attack_model.py` and `scripts/privacy_family_gates.py` checks parts of
the implementation contract; neither certifies all requirements of the unified protocol.

See `docs/FORMULA_LEDGER.md` for paper -> code -> test traceability of every formula and
`docs/FIDELITY.md` for the faithfulness rules.

## 安装和本地检查

使用 Python 3.10+，在独立虚拟环境中执行：

```bash
python -m pip install -e ".[dev,neural]"
python -m pytest -q
python scripts/run_experiment.py --all-smoke
python scripts/verify_fidelity.py
python scripts/audit_placeholders.py
```

核心依赖为 NumPy、SciPy 和 PyYAML。神经模型使用 `neural`，公开语料工具按需安装 `data`。只运行核心测试可安装 `".[dev]"` 并用 `pytest -m "not neural"`。

合成 smoke 的结果写入 `reports/runs/<name>/`，包含解析后的配置、环境、指标、预测和日志。忠实度检查会指出论文配置与可执行实现之间的差异；目前 SDA-2021/2023 的完整规模配置未通过该检查。占位项审计仍报告部分适配器及评测分支抛出 NotImplementedError。合成测试成功不能替代正式结果。

## 正式实验

1. 先按协议锁定威胁模型、模型 revision、输入切分、对照组和预算。
2. 在本地准备匹配的原始模型、A+B 交付工件和有权使用的公开语料。用 `scripts/build_paper_manifest.py --help` 查看如何生成本地身份清单；模型、数据、凭据、机器专用清单与运行产物都不入库。
3. 用 `scripts/run_full_battery.py --profile paper-v1 --plan-only --help` 查看正式计划参数，再带本地清单和相应资源生成计划。正式执行和小预算 `probe` 分开存档。
4. 归档生效配置、代码版本、结果与阻塞原因。新方案可参考 `src/obfuscation_attacks/schemes/custom_example.py` 与 `configs/custom_scheme/example.yaml`，并在观测上下文中明确评测可见的数据。

## 目录结构

```
configs/{original,aloepri,osnip,smoke,custom_scheme}/   experiment configs (full vs smoke separated)
src/obfuscation_attacks/
  data/         contracts (AttackSample / AttackContext / capabilities), synthetic scenarios
  schemes/      ObfuscationScheme base, capability-gated context builder, custom example
  models/       ModelHandle + numpy causal toy + differentiable torch toy
  attacks/      nn_knn/ vma/ ia/ isa/ ima/ tfma/ sda/   (original/AloePri/OSNIP in separate modules)
  metrics/      ASR@k, token P/R/F1, TTRSR, Total/Clean ASR, mapping ASR
  evaluation/   runner + scoring + artifacts
  utils/        seeding, IO, environment snapshot
tests/{unit,mathematical,regression,integration}/       the four test tiers
scripts/        run_experiment.py, verify_fidelity.py, audit_placeholders.py
docs/           PAPER_EVALUATION_PROTOCOL.md (authority); technical references; archive/ (history)
```

## 复现范围

This repository implements the seven attack **routes** and extensions with formula/test traceability.
Implementation tests and small defended-model probes do not establish reproduction of the papers'
numeric tables or paper-scale security. Record blockers per run rather than declaring every experiment
blocked. Implementation deviations are in [DEVIATIONS.md](docs/DEVIATIONS.md); experimental acceptance
is defined only by [PAPER_EVALUATION_PROTOCOL.md](docs/PAPER_EVALUATION_PROTOCOL.md).

A 和 B 的独立组件见 [AloePri-final](https://github.com/Gowin-liuyin/AloePri-final) 和 [OSNIP-final](https://github.com/Gowin-liuyin/OSNIP-final)。
