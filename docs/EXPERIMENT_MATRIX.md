# Experiment Matrix

Seven attack families x lineage (original / AloePri / OSNIP). Each row lists the registered
attack name, the full paper config, and the runnable synthetic smoke config.

| Family | Attack name | Lineage | Full config (resource-blocked) | Smoke config (runs) |
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

## Knowledge settings and metrics

- **OSNIP KNN**: Top-1/5/10 ASR over 1,000 Alpaca samples, 4 LLMs, seed 42.
- **OSNIP VMA**: Total/Clean ASR, layer-wise (L1..L39), 300k pool, 200x5 reps, NLTK stop words.
- **AloePri**: TTRSR / PIIRSR / CosSim / BLEU-4; TFMA Top-10/100 in Zero-/Domain-/Distribution-aware settings.
- **InvBERT IMA**: WB / GB / BB knowledge settings.
- **Hidden No More VMA**: permutation types none / sequence-dim / hidden-dim / factorized-2D.

Full configs are **resource-blocked** on this machine (see `RESOURCE_REQUIREMENTS.md`); smoke configs run
end-to-end on synthetic scenarios and are **not** paper reproductions.
