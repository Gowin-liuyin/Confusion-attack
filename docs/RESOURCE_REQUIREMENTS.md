# Resource Requirements and Blockers

## Local machine (where this was built)

- OS: Windows (PowerShell). Python 3.13 (base conda at `E:\miniconda`).
- GPU: NVIDIA GeForce RTX 4060 Laptop, **8 GB** VRAM. CPU: AMD Ryzen AI 9 HX 370. RAM: 31 GB.
- PyTorch **2.15.0+cpu** (CPU build; the CUDA wheels conflict with the local driver stack and the
  8 GB card cannot host the target models anyway), transformers **5.21.0**, numpy 2.4.2, scipy 1.17.1.
- OpenMP workaround required: `KMP_DUPLICATE_LIB_OK=TRUE` (set in `tests/conftest.py` and scripts).

## Why the full paper experiments are resource-blocked here

The `mode: full` configs are transcribed faithfully but cannot be executed on this machine:

- **Model scale.** AloePri evaluates Qwen2.5/Qwen3 (14B, 32B), Llama3-8B, R1-Distill (14B/32B),
  Qwen3-MoE-30B, Deepseek-V3.1-Terminus **671B**. OSNIP evaluates Llama-3.2-1B/3B and Qwen3-14B/32B.
  None of the 14B+ models fit in 8 GB VRAM; the 671B model needs a multi-GPU cluster.
- **Attack cost.** Autoregressive VMA replays `|V| x N` forward passes per prompt on 8B models; ISA
  ER/TBS run ~5e4 optimisation steps per prompt with gradients through the full model; the OSNIP VMA
  builds a 300,000-entry candidate pool over seven layers for 1,000 reps per model.
- **Data access / size.** Alpaca, ARC, SciQ, WikiText-103, PUPA, CCI3, Huatuo26M-Lite, MedDialog and a
  Gutenberg-scale corpus are needed at full size; some require download and license acceptance.

Per the Goal's stopping rule, these runs are reported as `resource_blocked` (see
`evaluation/runner.py` and each full run's `fidelity.json`) and the algorithms are instead verified by
mathematical identities and end-to-end synthetic-scenario tests. No smoke result is presented as a
full reproduction.

## To run the full experiments elsewhere

1. Provision multi-GPU hardware (e.g. A100/H100 x several) sized for the target models.
2. Install a CUDA PyTorch build matching the hardware and `pip install -e .[neural,data]`.
3. Download the datasets listed in the `mode: full` configs and point a real data/model backend at
   them (replace the synthetic scenario builders with a loader that materialises `AttackSample`s from
   the obfuscated target model).
4. Run `python scripts/run_experiment.py --config configs/<lineage>/<name>.yaml`.

## Remote / Git delivery blocker (reported, not worked around)

At build time the GitHub CLI token was invalid (`gh auth status` -> "token in keyring is invalid")
and HTTPS to github.com failed TLS negotiation (`git ls-remote` -> schannel handshake failure;
`gh api` -> 403 / EOF). Per the Goal's stopping rule the remote push / PR step is therefore blocked
and reported; it must be completed once `gh` is re-authenticated and network access to github.com is
restored (`gh auth login`, then `git push -u origin <branch>` and `gh pr create --fill`). Local git
history and the feature branch are prepared so the push is a single step afterwards.
