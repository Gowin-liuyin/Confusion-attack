# Deviations and Completions

> Technical reference only. Experimental authority: [PAPER_EVALUATION_PROTOCOL.md](PAPER_EVALUATION_PROTOCOL.md). Historical parameters/results do not define current permissions, budgets or acceptance.

Choices the papers did not fully specify (`UNSPECIFIED`) and engineering completions needed to make
a runnable harness (`EXPERIMENTAL_COMPLETION`). None of these change an attack's core mathematics;
none are presented as paper parameters. "Not implemented" items are documented, never faked.

The reproduction keeps the original, AloePri and OSNIP variants in separate modules/configs; the
items below are grouped by family.

## NN / KNN
- **EXPERIMENTAL_COMPLETION — AloePri NN dimension alignment.** AloePri's obfuscated embedding table
  is `W̃_embed = Π W⋆_embed P̂_embed` with `P̂_embed` of shape `d × (d+2h)` (Sec 5.2.2, h=128), so the
  observed rows have `d+2h` columns while the attacker's clean table has `d`. A plain Euclidean lookup
  is therefore not directly runnable; an alignment/projection step is required. Recorded in
  `configs/aloepri/nn.yaml::assumptions`; AloePri Table 2 reports NN ≈ 0% TTRSR precisely because the
  attacker lacks this alignment.
- **Honesty note — `configs/original/knn_song.yaml` reproduces no Song & Raghunathan experiment.**
  Song's inversion recovers the unordered word *set* (precision/recall/F1), with no per-token Top-k NN
  and no Top-k ASR. What runs is the OSNIP-style single-token Euclidean retrieval (NN-1); the config's
  metrics are set-based P/R/F1 and NN-3 (white-box continuous relaxation) stays not-implemented.
- NN now returns the **full per-position ranking** (not truncated to `top_k`), so ASR@k is never capped
  by the attack's requested k; the scorer slices Top-k itself.
- **UNSPECIFIED**: NN/EDNN/OSNIP default `top_k = 10`; AloePri NN `ks` are borrowed from OSNIP (AloePri
  reports no NN Top-k).

## VMA
- **OSNIP (B3) is the autoregressive L1-replay attack** (OSNIP Sec 4.1/App C.2), not an independent
  fixed-pool match: it replays the clean model on `[x̂_<i, v]`, reads the layer-l state and takes the
  L1-nearest to the observed state, feeding the result back. It requires `MODEL_REPLAY`. The App. C.2
  "300,000 pool" is the pool of **text samples** the 200 test instances are drawn from, not candidate
  representations (recorded in `configs/osnip/vma.yaml`).
- **AloePri weight VMA (B2)** now uses RowSort **+ nearest-neighbor** matching (noise-tolerant), since
  AloePri weights carry noise `W⋆ = W_e + α_e·E`. **UNSPECIFIED**: AloePri does not name the matching
  distance. **EXPERIMENTAL_COMPLETION**: L1 (after Thomas et al.) is used, and a Hungarian assignment
  (scipy) completes the column permutation `Z2` and resolves duplicate-row groups. Multi-layer "voting"
  is a **per-row majority vote** across layers (not fingerprint concatenation). Duplicate rows that
  cannot be disambiguated are returned as a **`partial`** result with candidate sets — never a fake
  `verified` and never a wrong `Z1` reported as verified.
- **B1 proposal ordering.** The ε early-accept iterates the vocabulary in a proposal model's order
  (Algorithm 1 line 3). A `proposal_fn`/`proposal_order` hook is provided; if ε is requested without a
  proposal ordering the attack falls back to the pure global argmin (recorded in diagnostics) rather
  than silently using token-id order. The sequence-dimension (Alg 4) and factorized-2D (Alg 6)
  permutation reversals are implemented alongside sorted-L1 (Alg 5).
- **Dense-mixing / least-squares VMA (C06-W, D3).** **EXPERIMENTAL_COMPLETION.** The bundled engineering
  artifact applies a dense hidden-axis re-basing `W' = W M` rather than the monomial permutation of
  AloePri Algorithm 1/2. The attack solves `M` by least squares from the public clean weights and the
  observed deployment weights (both public in S-public), verifies the premise with a leave-one-out
  cross-prediction (an in-sample square fit alone is exact for any matrix and proves nothing), then
  un-mixes the embedding table and matches rows. It is labelled `dense_mixing_inversion`; the
  paper-literal RowSort rows are reported separately as premise-violated diagnostics and never as the
  paper's Table 9 result.
- **Identity note.** `RowSort(Y) = RowSort(X·Z2)` holds only up to row order; the exact relation used
  is `RowSort(Y) = Z1·RowSort(X)` (row-sorting removes the column permutation `Z2`, then the row
  permutation `Z1` permutes the sorted rows). The code was already correct; the ledger/docstring wording
  was fixed.
- The paper's 50-token prompt truncation (Hidden No More) is the **paper's own** setting, recorded as
  such in `configs/original/vma_hidden_no_more.yaml`, not an approximation introduced here.

## IA
- **EXPERIMENTAL_COMPLETION — Attn-IA block index.** The paper writes the invariant with a literal
  per-column `Q[:,j]` whose dimensions do not close; the only consistent reading (verified numerically)
  is the per-RoPE-2×2-block whitening form `q_j G_j^{-1} q_j^T`. The attack also generalizes beyond the
  paper's single-head index-aligned statement: it accepts stacked per-head/per-layer query (and
  optional key) weights and concatenates every valid block into one per-token invariant vector.
- **UNSPECIFIED — RoPE column pairing.** `pairing` defaults to `adjacent` (columns 2j, 2j+1); a
  `half_split` option provides the HuggingFace `rotate_half` layout. A partial final block is rejected,
  not silently used.
- **Gate-IA / Attn-IA precondition.** The invariants are exact only at `α_e = 0` (no embedding noise).
  The full configs run the paper's actual `α_e = 1.0` (Table 10 row "Tab. 2/6/7"), i.e. as an ablation
  against AloePri's noise defense, which is why Table 2 reports IA at ~6% TTRSR. Both the exactness
  precondition (tested at α_e=0) and the degradation at α_e>0 are covered in `test_ia_math.py`.
- Singular/ill-conditioned query Gram blocks are **skipped** (never pseudo-inverted); if none are
  invertible the attack returns `not_applicable`.
- `recovery_accuracy` (EDNN) and `cosine_similarity` (Gate/Attn) in the configs are paper-facing metric
  labels (Lin Table 1 "Recovery acc"; AloePri Table 2 "CosSim"); the local scorer emits
  `asr@1`/`mapping_top1_accuracy` for these resource-blocked runs.

## ISA
- **TBS uses the FULL `d_in` basis** (`z ∈ R^{n×d_in}`), not a low-rank truncation — the earlier `r=8`
  truncation was a fidelity bug and is removed (an optional `r<d_in` remains but is off by default and
  would flip the config's `truncation` flag).
- **Both basis strategies** are implemented: `B=V` (the paper's default "unbiased basis") and `B=Vᵀ`.
  `basis` defaults to `unbiased`.
- **φ_z = α·arctan(z·B) with α = 5/π** (App. A Eq. 11), restored — the earlier plain-`arctan` (α=1) was
  a deviation and is fixed.
- **Optimizer AdamW** (paper) for ER/TBS/AloePri-state. **UNSPECIFIED**: AdamW `weight_decay` (paper
  gives no value; default 0.0, which equals Adam, so the switch is faithful and non-breaking).
- **Default distance MSE** (paper default), with cosine available.
- **ER `L_dm` matches Eq. 3**: Euclidean norm of the difference between batch means, with the
  random Gaussian feature networks and the vocabulary batch resampled each step. The dummy batch
  uses all dummy rows. Older runs with nonzero `lambda_dm` used the squared norm. **EXPERIMENTAL_COMPLETION**:
  the random-feature-network architecture (small Gaussian-weight MLP) is a reproduction choice — the
  paper only names "random Gaussian neural network". `L_dm` is off by default (`λ=0`, the paper's default;
  `λ∈{0,1e-3}`).
- **AloePri ISA `AttnScore` is a separate target, not a hidden-state relabel.** Table 4
  ablates `AttnScore` and `HiddenState`. `target="hidden"` (default) is the appendix
  State1/State2 match. `target="attention"` matches the same decoder block's attention
  map, flattened per token to `[seq, heads*seq]` (`attentions[layer-1]` for
  `hidden_states[layer]`). A model with no attention map returns `not_applicable`.
  The toy model has no attention map, so toy runs stay HiddenState. The loss is still
  `L_im` only (random `X2`, weights fixed) and does **not** reproduce ER/TBS.

## IMA
- **Faithful key-holdout threat model.** The attacker knows the obfuscation *algorithm* (keying it with
  its own sampled keys to synthesise training pairs via `schemes/mechanism.py`) but never the client's
  secret key. The evaluated sample uses a held-out client key, so faithful recovery is ≈ chance —
  matching AloePri Table 2 (IMA TTRSR = 0.0). A separate `attacker_known_key=True` test proves the
  inverter has capacity (it recovers when handed the key). The `ima` smoke scenario remains a known-key
  **capacity demo** (labelled), with the faithful unknown-key model exercised in the regression tests.
- **InvBERT Seq2Seq (BB)** adds a positional encoding on the CTE memory (so it can order tokens),
  predicts over an **attacker-side vocabulary**, and decodes **variable length until EOS** (output length
  is not read from the input). In the synthetic harness the attacker vocabulary shares the integer id
  space only so recovery is measurable (conceptually a custom BPE) — **EXPERIMENTAL_COMPLETION**.
- **InvBERT Classify is GB only.** Classify (the GB attack) supports WB/GB; **BB returns
  `not_applicable`** (BB is Seq2Seq). The earlier invented 64-item BB corpus cap is removed. The WB
  vocabulary probe queries each id on its own length-1 context (per-id), not the whole vocabulary as one
  sequence.
- **AloePri Qwen2** inverts to plaintext token **embeddings** by default (the paper's experiment); a
  cross-entropy token-index head is the clearly-labelled alternative. The embedding path's **MSE loss is
  an implementation choice** (AloePri specifies the *target*, not the loss — never claimed as AloePri's).
  The inverter is fixed at 2 decoder layers / 8 attention heads (asserted). **UNSPECIFIED**:
  `num_key_value_heads` (App. D.1 gives only attention heads; assumed MHA = 8).
- **UNSPECIFIED — InvBERT epochs/learning rate** (the paper states it does no hyperparameter tuning and
  reports none); harness defaults are smoke-only.

## TFMA
- **Method = per-symbol nearest-frequency matching** (`tfma.independent`, AloePri). The one-to-one
  Hungarian assignment (`tfma.global_assignment`, variant `experimental_completion`) is a named,
  separately-reported extension; its candidate list now places the globally-assigned token first so its
  Top-1 matches the assignment.
- **Tie rule (documented):** with a stable argsort the lowest token id wins an exact frequency tie; the
  number of tied symbols is reported in diagnostics. The three knowledge settings (zero-/domain-/
  distribution-aware) differ only in the prior corpus; the corpora themselves are not shipped
  (resource-blocked).
- **Scoring**: mapping metrics count only the cipher symbols actually observed in the ciphertext, not
  vocabulary entries that never appeared.

## SDA
- **2021 frequency-rank encoding is injective per symbol** (paper Fig. 5): distinct cipher symbols get
  distinct codes, ordered by count descending with ties broken by first appearance (key-independent).
  The earlier competition ranking (equal-count symbols sharing a code) was a fidelity bug and is fixed;
  `frequency_rank_info` still reports ties.
- **2021 experiments** (noise substitution/insertion/deletion, cipher-length sweep, space
  enciphered/removed) are implemented as condition generators and scored with SER (and TER for
  length-changing cases).
- **2023 homophonic** ciphers are generated and evaluated end-to-end (cipher→plaintext is a deterministic
  many-to-one map, so the deterministic-permutation capability holds).
- **EXPERIMENTAL_COMPLETION — Algorithm 1 `char_weight`.** The literal `int(freq/n)` is 0 for every
  character (freq ≤ n) and would yield no homophones; the code reads `char_weight = freq/n` as a fraction
  (the only reading consistent with the paper's frequency-flattening description), keeping `int(·)` and
  the `wsym==0→1` floor verbatim.
- **Honesty note.** On the tiny synthetic smoke language the 2023 recurrence model trails classical
  frequency-matching on short (<128) 1:1 ciphers (the paper itself notes recurrence needs longer
  context) but beats it decisively on homophonic ciphers; the 2021 model matches the frequency baseline.
  These historical smoke-scale observations used the former majority-vote decoding; they do not establish
  the behavior of the current raw-beam and constrained-beam implementations.
- **UNSPECIFIED** (follow the cited char-NMT setup but are not restated): 2021 `d_model, nhead, dim_ff,
  dropout, learning_rate, warmup, optimizer_betas`; 2023 `d_model` and learning-rate schedule remain unverified; Table 8 gives dropout 0.1, attention dropout 0.1, peak learning rate 0.0005 and batch size 32000; AloePri SDA `num_train, epochs, model dims`.
- **AloePri SDA version mixing.** AloePri *cites* Aldarrab & May 2021 but its appendix *describes*
  recurrence encoding + a causal LM (Kambhatla et al. 2023). `configs/aloepri/sda.yaml` runs the 2023
  method (G2) to match the appendix algorithm and records the citation/appendix mismatch; both lineages
  are implemented separately.

## Metrics
- Added `symbol_error_rate` (SDA), `bleu4` (AloePri BLEU-4, token-level), and `recovered_cosine_similarity`.
  **EXPERIMENTAL_COMPLETION**: CosSim is a **token-level** proxy (mean per-position cosine between the
  recovered and true token embeddings), not a sentence-encoder similarity as in AloePri.
- Special-token positions are excluded uniformly from every token metric (including token P/R/F1 and
  exact-sequence match), via the scorer.

## Shared / infrastructure
- **`ObfuscationMechanism`** (`schemes/mechanism.py`) separates the public algorithm from the secret key;
  `AloePriToyMechanism` is a toy permutation + covariant map + optional embedding noise used by the IMA
  key-holdout tests. The toy models (`CausalToyModel`, `TorchToyModel`) and small synthetic scenarios are
  verification stand-ins, never paper reproductions.
- **Toy/smoke architectures.** Smoke configs/tests use tiny models (few layers, small `d_model`); paper
  `mode: full` configuration text alone does not establish execution at paper scale. SDA 2023 defaults to
  integer lookup embeddings; `decimal_char_cnn` is a separately identified experimental recurrence-input
  variant with paper-described filter widths and highway layers, not a verified author implementation.

## Environment deviation
- **OpenMP duplicate runtime.** The conda MKL + pip torch combination raises `OMP: Error #15`. The
  accepted workaround `KMP_DUPLICATE_LIB_OK=TRUE` is set in `tests/conftest.py` and all scripts. Recorded
  in [the historical environment record](archive/RESOURCE_REQUIREMENTS.md); verify the actual environment per run.


## SDA 2023 source verification (2026-11-09)

- [The paper, §4.2 and Table 8](https://aclanthology.org/2023.findings-eacl.160.pdf) specifies the
  12-layer/12-head model, CNN filter sizes, character dimension and two highway layers. It gives
  dropout 0.1, attention dropout 0.1, batch size 32000 and peak learning rate 0.0005. The main text
  says approximately 30 epochs while Table 8 lists maximum 20; this conflict is unresolved.
- The paper-linked [author repository at commit 3525d314984ec107206acfff358d108c2e74901c](https://github.com/protonish/decipher_symbol_recurrence/tree/3525d314984ec107206acfff358d108c2e74901c)
  contains only a README announcing code and data; no implementation or license was available there
  for differential testing at the time of inspection. This does not prove that no other release exists.
- `representation=decimal_char_cnn` composes decimal recurrence numbers with digit embeddings,
  CNN max pooling and highway layers, then projects to model width. Decimal digit assignment, padding,
  projection, plaintext/tag treatment and output head are engineering decisions pending author-code
  verification. The default remains `lookup`; neither is certified as a full reproduction.
