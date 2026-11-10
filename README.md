# CA：评估推理过程中的隐私泄露

CA 提供一组评测方法，用来测量从混淆后的模型权重、推理向量或输出记录中仍能恢复多少信息。仓库可独立运行：先用合成数据验证流程，无需另外克隆 AloePri、OSNIP 或 A+B，也无需下载真实权重。

## 从合成测试开始

使用 Python 3.10 或更高版本，在独立虚拟环境中执行：

```bash
python -m pip install -e ".[dev,neural]"
python scripts/run_experiment.py --all-smoke
python -m pytest -q
```

结果写入 `reports/runs/<name>/`。先看 `metrics.json`，需要追查运行条件时再看 `config.resolved.yaml` 和日志。只运行不依赖神经模型的测试时，可安装 `".[dev]"` 并执行 `python -m pytest -m "not neural"`。下载公开语料的工具按需安装 `data`。

## 做正式评测

先读[正式评测协议](docs/PAPER_EVALUATION_PROTOCOL.md)，确认评测者能看到哪些信息、用什么对照组，以及什么结果才算通过。再按[运行与结果检查](docs/RUNBOOK.md) 准备模型、匹配工件和语料，生成本地身份清单与执行计划。

实现包括近邻检索、词表与权重匹配、输入恢复、跨请求链接等。查具体配置时，从[实验矩阵](docs/EXPERIMENT_MATRIX.md) 找入口；查原论文与本仓库适配之间的区别时，看[出处核对](docs/ATTACK_PROVENANCE_14.md)和[已知偏差](docs/DEVIATIONS.md)。目录里的方法数不等于已完成的正式实验数。

当前合成用例可以运行。另有两个诊断入口：`verify_fidelity.py` 仍报告 SDA-2021/2023 与论文配置的差异；`audit_placeholders.py` 仍报告部分适配器和评测分支未实现。这些限制不应被写成通过。

## 代码在哪里

| 路径 | 内容 |
| --- | --- |
| [configs/](configs/) | 合成、正式和项目适配配置 |
| [src/obfuscation_attacks/](src/obfuscation_attacks/) | 观测规则、算法、评分和运行逻辑 |
| [scripts/](scripts/) | 运行入口、计划和身份清单生成工具 |
| [tests/](tests/) | 公式、单元、回归和集成测试 |

## 更多文档

- [文档导航](docs/README.md) · [运行与结果检查](docs/RUNBOOK.md)
- [维护与贡献](CONTRIBUTING.md) · [变更记录](CHANGELOG.md)
- [引用与来源](CITATION.md) · [安全问题报告](SECURITY.md)
