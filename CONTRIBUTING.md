# 维护与贡献

- 先读 [README](README.md) 和[运行与结果检查](docs/RUNBOOK.md)，在独立分支提交可复查的改动。
- 先跑 `python -m pytest -q` 与 `python scripts/run_experiment.py --all-smoke`；正式协议相关修改按 [AGENTS.md](AGENTS.md) 验证。
- 修改实验规则只更新 [正式评测协议](docs/PAPER_EVALUATION_PROTOCOL.md) 并同步对应实现和身份指纹。提交说明列出影响的评测路径、测试覆盖和遗留差异。
- 模型、真实数据、凭据与逐条实验结果留在本地；提交示例时使用可再生成的合成数据。
- 文档中的命令从仓库根目录出发。新增参数或改名时，同步更新 README 和运行手册，并检查相对链接。
