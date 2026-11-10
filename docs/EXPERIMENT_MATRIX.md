# Experiment Matrix

> Technical reference only. Experimental authority: [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md). Historical parameters/results do not define current permissions, budgets or acceptance.

Seven attack families x lineage (original / AloePri / OSNIP). Each row lists the registered
attack name, the full paper config, and the runnable synthetic smoke config.

| Family | Attack name | Lineage | Original full config | Synthetic smoke config |
|--------|-------------|---------|-------------------------------|---------------------|
| NN/KNN | `nn_knn.euclidean`, `nn_knn.cosine` | original (Song 2020); OSNIP C.1; AloePri NN | `configs/original/knn_song.yaml`, `configs/osnip/knn.yaml`, `configs/aloepri/nn.yaml` | `configs/smoke/nn_knn.yaml` |
| VMA (B1) | `vma.autoregressive` | original (Hidden No More) | `configs/original/vma_hidden_no_more.yaml` | `configs/smoke/vma_autoregressive.yaml` |
| VMA (B2) | `vma.weight_rowsort` | AloePri (Table 9) | `configs/aloepri/vma_weight.yaml` | `configs/smoke/vma_weight.yaml` |
| VMA (B3) | `vma.osnip_projection` | OSNIP C.2 | `configs/osnip/vma.yaml` | `configs/smoke/vma_osnip.yaml` |
| IA (EDNN) | `ia.ednn` | original (Lin 2024) | `configs/original/ia_ednn.yaml` | `configs/smoke/ia_ednn.yaml` |
| IA (Gate) | `ia.gate` | AloePri | `configs/aloepri/ia_gate.yaml` | `configs/smoke/ia_gate.yaml` |
| IA (Attn) | `ia.attn` | AloePri | `configs/aloepri/ia_attn.yaml` | `configs/smoke/ia_attn.yaml` |
| ISA (ER/TBS) | `isa.er`, `isa.tbs` | original (Dong 2025) | `configs/original/isa_er_tbs.yaml` | `configs/smoke/isa_er.yaml` |
| ISA (state) | `isa.aloepri_state` | AloePri | `configs/aloepri/isa_state.yaml` | (covered by `isa` scenario) |
| IMA (classify/seq2seq) | `ima.classify`, `ima.seq2seq` | original (InvBERT) | `configs/original/ima_invbert.yaml` | `configs/smoke/ima_classify.yaml` |
| IMA (Qwen2) | `ima.qwen2_aloepri` | AloePri | `configs/aloepri/ima_qwen2.yaml` | (covered by `ima` scenario) |
| TFMA | `tfma.independent` (+ `tfma.global_assignment` ext.) | AloePri | `configs/aloepri/tfma.yaml` | `configs/smoke/tfma.yaml` |
| SDA 2021 | `sda.sda2021` | original (Aldarrab 2021) | `configs/original/sda2021.yaml` | `configs/smoke/sda2021.yaml` |
| SDA 2023 | `sda.sda2023` | original (Kambhatla 2023); AloePri adaptation | `configs/original/sda2023.yaml`, `configs/aloepri/sda.yaml` | `configs/smoke/sda2023.yaml` |

## Usage

This table locates the original lineage configs and smoke implementations. It is not the current
experimental plan or an inventory of completed runs. Use [the unified protocol](PAPER_EVALUATION_PROTOCOL.md)
for attacker knowledge, metrics, budgets and acceptance; use each run's actual artifacts for blockers.
Smoke scenarios do not establish defended-model performance. A config named full must still be checked
against the current protocol before it is used for a paper-scale claim.

The C11 batch probe reads `--sda-config` from either [SDA 2021](../configs/probe/sda2021_stream.yaml)
or [SDA 2023](../configs/probe/sda2023_stream.yaml), or an explicitly configured variant. These are
engineering probes; the formal executor remains separately gated.
