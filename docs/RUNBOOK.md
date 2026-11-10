# CA 运行与结果检查

这份说明只介绍脚本和产物。对照组、可见信息、预算及主表准入以 [正式评测协议](PAPER_EVALUATION_PROTOCOL.md) 为准。

## 合成验证

安装方法见 [README](../README.md)。从仓库根目录运行全部 smoke，或先用一个合成配置：

```bash
python scripts/run_experiment.py --config configs/smoke/nn_knn.yaml
python scripts/run_experiment.py --all-smoke
```

`reports/runs/<name>/` 是本地输出。先看 `config.resolved.yaml` 与 `environment.json`，再看 `metrics.json`、预测与运行日志。合成用例只检查流程和指标计算能否运行。

## 正式计划与本地资源

1. 使用 [build_paper_manifest.py](../scripts/build_paper_manifest.py) 记录本地模型、语料、匹配的 A+B 工件及来源信息。通过 `--help` 查看路径和版本参数。新机器显式传入本地资源，不使用历史默认路径。
2. 使用 [run_full_battery.py](../scripts/run_full_battery.py) 先生成计划，确认资源和协议匹配再执行。示例中的清单须由上一步生成：

```bash
python scripts/run_full_battery.py --profile paper-v1 --plan-only --manifest local/manifest.json --plan-out reports/paper-plan.json
```

3. 对照计划检查实际产物中的模型、数据、配置、场景和对照组身份。保存报告和阻塞原因；小预算 `probe` 和正式结果单独归档。

## 解读验证状态

- 有效的 smoke 不代表正式评测已完成。SDA 的实现偏差查 [DEVIATIONS.md](DEVIATIONS.md)，出处与指标查 [公式台账](FORMULA_LEDGER.md)。
- 缺资源或权限导致无法执行时，保留原因。未执行、不适用和真实测得 0 是三种状态。
- 正式身份清单、逐条预测、客户端工件与根密钥都由运行者本地保管。仓库只保留代码、配置与文档。
