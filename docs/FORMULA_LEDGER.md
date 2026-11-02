# Formula Ledger

Every key formula traced from paper to code to test. Columns: **ID**, **Formula**,
**Paper**, **Section**, **Variant**, **Code**, **Test**. `scripts/verify_fidelity.py`
parses this table and fails if any referenced `.py` path is missing.

| ID | Formula | Paper | Section | Variant | Code | Test |
|----|---------|-------|---------|---------|------|------|
| NN-1 | xhat_i = argmin_v L2(h_i, E_v) | Song & Raghunathan 2020 | NN retrieval | original | `src/obfuscation_attacks/attacks/nn_knn/core.py` | `tests/mathematical/test_nn_math.py` |
| NN-2 | d_cos = 1 - (h.E_v)/(norm h * norm E_v) | Song & Raghunathan 2020 | NN retrieval | original | `src/obfuscation_attacks/attacks/nn_knn/core.py` | `tests/mathematical/test_nn_math.py` |
| NN-3 | white-box continuous relaxation vhat = V softmax(z/T) | Song & Raghunathan 2020 | white-box | original | not implemented (documented paper extension; see DEVIATIONS.md) | n/a |
| METRIC-ASR | ASR@k = mean 1[truth in top-k] | both targets | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| METRIC-TTRSR | positional token recovery ratio | AloePri | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| METRIC-F1 | token precision/recall/F1 (multiset) | shared | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| METRIC-TOTAL-CLEAN | Total vs Clean ASR (stop-word split) | OSNIP | Table 6 | osnip | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| VMA-B1-1 | xhat_i = argmin_v L1(M_le_l([xhat_lt_i, v])_i, h_i) | Hidden No More 2025 | Sec 5 / Alg 1 | original | `src/obfuscation_attacks/attacks/vma/autoregressive.py` | `tests/mathematical/test_vma_math.py` |
| VMA-B1-2 | d_sorted(a,b) = L1(sort a, sort b) | Hidden No More 2025 | 5.2.2 | original | `src/obfuscation_attacks/attacks/vma/autoregressive.py` | `tests/mathematical/test_vma_math.py` |
| VMA-B2-1 | Y = Z1 X Z2 | AloePri | App D.1 | aloepri | `src/obfuscation_attacks/attacks/vma/weight_rowsort.py` | `tests/mathematical/test_vma_math.py` |
| VMA-B2-2 | RowSort(Y) = Z1 . RowSort(X)  (row-sort removes Z2; holds up to row order) | AloePri | App D.1 | aloepri | `src/obfuscation_attacks/attacks/vma/weight_rowsort.py` | `tests/mathematical/test_vma_math.py` |
| VMA-B3-1 | project hidden state to vocab pool, L1 / sorted-L1 | OSNIP | App C.2 | osnip | `src/obfuscation_attacks/attacks/vma/osnip_projection.py` | `tests/mathematical/test_vma_math.py` |
| IA-EDNN-1 | e' = e - 2 (e.l / l.l) l + t, l=1a, t=1b | Lin et al. 2024 | EDNN | original | `src/obfuscation_attacks/attacks/ia/ednn.py` | `tests/mathematical/test_ia_math.py` |
| IA-EDNN-2 | e'[k1]-e'[k2] = e[k1]-e[k2] | Lin et al. 2024 | EDNN | original | `src/obfuscation_attacks/attacks/ia/ednn.py` | `tests/mathematical/test_ia_math.py` |
| IA-EDNN-3 | D_ij = norm((ebar_i - lshift ebar_i) - (e_j - lshift e_j)) | Lin et al. 2024 | EDNN | original | `src/obfuscation_attacks/attacks/ia/ednn.py` | `tests/mathematical/test_ia_math.py` |
| IA-GATE-1 | Avg(e Wgate) = Avg(etilde Wtilde_gate) | AloePri | App D.1 | aloepri | `src/obfuscation_attacks/attacks/ia/gate.py` | `tests/mathematical/test_ia_math.py` |
| IA-ATTN-1 | e (Qj^T Qj)^-1 e^T invariant (whitening) | AloePri | App D.1 | aloepri | `src/obfuscation_attacks/attacks/ia/attn.py` | `tests/mathematical/test_ia_math.py` |
| ISA-1 | xhat = argmin d(h_l^fA(x'), h_l^fV(x)) | Dong et al. 2025 | Alg 1 | original | `src/obfuscation_attacks/attacks/isa/core.py` | `tests/mathematical/test_isa_math.py` |
| ISA-ER-IM | L_im = d(psi_l^fA(what), h_l^fV) | Dong et al. 2025 | Alg 1 | original | `src/obfuscation_attacks/attacks/isa/er.py` | `tests/mathematical/test_isa_math.py` |
| ISA-ER-INV | L_inv = L_im + lambda L_dm (random Gaussian features) | Dong et al. 2025 | Eq 3 | original | `src/obfuscation_attacks/attacks/isa/er.py` | `tests/mathematical/test_isa_math.py` |
| ISA-TBS-SVD | W_in = U Sigma V^T | Dong et al. 2025 | Alg 1 / App A | original | `src/obfuscation_attacks/attacks/isa/tbs.py` | `tests/mathematical/test_isa_math.py` |
| ISA-TBS-REPARAM | what = arctan(z B) | Dong et al. 2025 | App A | original | `src/obfuscation_attacks/attacks/isa/tbs.py` | `tests/mathematical/test_isa_math.py` |
| IMA-CE-1 | L = - sum_t log p_theta(y_t given context) | InvBERT | Sec 4.2 | original | `src/obfuscation_attacks/attacks/ima/classify.py` | `tests/mathematical/test_ima_math.py` |
| IMA-CLS-1 | p = softmax(W MLP(h_t) + b)_{y_t} | InvBERT | Sec 4.2 | original | `src/obfuscation_attacks/attacks/ima/classify.py` | `tests/mathematical/test_ima_math.py` |
| IMA-S2S-1 | P(y given h) = prod_t p_theta(y_t given y_lt_t, h) | InvBERT | Sec 4.2 | original | `src/obfuscation_attacks/attacks/ima/seq2seq.py` | `tests/unit/test_ima.py` |
| IMA-QWEN2 | inverter = Qwen2, 2 decoder layers, 8 heads | AloePri | App D.1 | aloepri | `src/obfuscation_attacks/attacks/ima/qwen2_aloepri.py` | `tests/mathematical/test_ima_math.py` |
| IMA-MSE | optional L_MSE = (1/N) sum norm(ehat_t - e_t)^2 | AloePri (optional) | App D.1 | aloepri | `src/obfuscation_attacks/attacks/ima/qwen2_aloepri.py` | `tests/mathematical/test_ima_math.py` |
| TFMA-1 | f_C(c)=n_C(c)/sum; f_R(t)=n_R(t)/sum | AloePri | Sec 7.6 | aloepri | `src/obfuscation_attacks/attacks/tfma/frequency_matching.py` | `tests/mathematical/test_tfma_math.py` |
| TFMA-2 | pihat(c) = argmin_t abs(f_C(c)-f_R(t)) | AloePri | Sec 7.6 | aloepri | `src/obfuscation_attacks/attacks/tfma/frequency_matching.py` | `tests/mathematical/test_tfma_math.py` |
| TFMA-3 | pihat = argmin_pi sum (f_C(c)-f_R(pi(c)))^2 (Hungarian) | extension | n/a | experimental_completion | `src/obfuscation_attacks/attacks/tfma/frequency_matching.py` | `tests/mathematical/test_tfma_math.py` |
| SDA-2021-ENC | frequency-rank encoding (rank 1 = most frequent) | Aldarrab & May 2021 | Sec 3 | original_2021 | `src/obfuscation_attacks/attacks/sda/encodings.py` | `tests/unit/test_sda.py` |
| SDA-2021-OBJ | L_Seq2Seq = - log P(Y given X) | Aldarrab & May 2021 | Sec 3 | original_2021 | `src/obfuscation_attacks/attacks/sda/sda2021_seq2seq.py` | `tests/mathematical/test_sda_math.py` |
| SDA-2023-ENC | recurrence encoding (ABACAD -> 1 2 1 3 1 4) | Kambhatla et al. 2023 | Eq 1 | original_2023 | `src/obfuscation_attacks/attacks/sda/encodings.py` | `tests/unit/test_sda.py` |
| SDA-2023-OBJ | L_CLM = - log P(X) - log P(Y given X) | Kambhatla et al. 2023 | Eq 2 | original_2023 | `src/obfuscation_attacks/attacks/sda/sda2023_recurrence_clm.py` | `tests/mathematical/test_sda_math.py` |
| VMA-B1-SEQ | sequence-dim permutation reversal (match remaining rows, remove) | Hidden No More 2025 | Alg 4 | original | `src/obfuscation_attacks/attacks/vma/autoregressive.py` | `tests/mathematical/test_vma_math.py` |
| VMA-B1-2D | factorized-2D (rows+cols) reversal with sorted-L1 | Hidden No More 2025 | Alg 6 | original | `src/obfuscation_attacks/attacks/vma/autoregressive.py` | `tests/mathematical/test_vma_math.py` |
| ISA-TBS-ALPHA | what = alpha * arctan(z B), alpha = 5/pi | Dong et al. 2025 | App A Eq 11 | original | `src/obfuscation_attacks/attacks/isa/tbs.py` | `tests/mathematical/test_isa_math.py` |
| METRIC-SER | SER = fraction of positions decoded wrong | SDA 2021/2023 | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| METRIC-COSSIM | token-level recovered/true embedding cosine (proxy) | AloePri | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |
| METRIC-BLEU4 | token-level sentence BLEU-4 with brevity penalty | AloePri | metrics | shared | `src/obfuscation_attacks/metrics/recovery.py` | `tests/unit/test_metrics.py` |

## Notes on faithfulness

- **No metric substitution:** NN uses L2/cosine; autoregressive VMA uses L1 (sorted-L1
  under hidden-dim permutation); weight VMA uses RowSort + fingerprint matching (never a
  decoder); ER/TBS minimise a state distance (never a frequency rank); SDA 2023 uses
  recurrence encoding (never frequency rank). These are pinned by the math tests and by
  `scripts/verify_fidelity.py` runtime invariants.
- **Version separation:** SDA 2021 (frequency-rank + encoder-decoder, `-log P(Y|X)`) and
  SDA 2023 (recurrence + causal LM, `-log P(X) - log P(Y|X)`) are separate modules and
  separate ledger rows.
- **Original vs adaptation:** `original` rows transcribe the source paper; `aloepri` /
  `osnip` rows transcribe the target-paper adaptation. They are never merged.
