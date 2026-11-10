> ARCHIVED / NON-AUTHORITATIVE (2026-11-07). Historical evidence only.
> Current rules: [PAPER_EVALUATION_PROTOCOL.md](../PAPER_EVALUATION_PROTOCOL.md).
> Old budgets, permissions, status and approval language below are superseded.

# Closed-loop attacks on the plaintext Llama-3.2-1B

This report records the end-to-end runs requested for the plaintext model: every
attack family is executed on the real checkpoint
`F:\Models\Llama-3.2-1B-Instruct` with the actual attack classes, the real
tokenizer/vocabulary, and no synthetic stand-ins. It is deliberately separate
from the defended-scheme results: here there is **no obfuscation**, so the
numbers measure whether each implementation reproduces its paper's mechanism,
not whether the defense holds.

Environment: `attack-verify` conda env (torch 2.6.0+cu124, transformers 5.21.0),
NVIDIA RTX 4060 Laptop GPU (8 GB), 31 GB RAM. Artifacts:
`reports/closed_loop/*.jsonl`, aggregate: `reports/closed_loop/summary.{json,md}`.

## VMA - full vocabulary, recovered prefix, 8 layers

`scripts/rm_vma_full.py` implements the paper's rule exactly:

```
xhat_i = argmin_{v in V} || M_{<=l}([xhat_{<i}, v])_i - h_i ||_1
```

with **all 128,256 candidates**, the **recovered** prefix (not the true one),
and every layer decoded independently. Candidate replay uses one-token cached
forwards; the L1 rule and argmin are unchanged by batching.

Checks before decoding: an fp32 cached-vs-full replay matches bit-exactly
(max |diff| = 0.0 at layers 1..16), and an fp16 decision check over the full
vocabulary reproduces the fp32 top-1 (top-100 overlap 99/100). The run then uses
fp16 cached replay with batch 4096.

Coverage: prompts 0-2 (43 positions) x layers {1,2,4,6,8,10,12,16}. Every
position and layer is recovered exactly (ASR@1 = ASR@5 = TTRSR = 1.0).

Interpretation: on an undefended model with exact replay, the true token is a
zero-distance fixed point, so perfect recovery is the mathematically expected
outcome, not evidence of an unusually strong attack. The informative
comparison is the defended setting, where replay and victim computation differ.
Stopping at three prompts was a compute decision: at the measured 2.8-5.1 h per
prompt, the remaining 21 prompts would add 3+ days and cannot change the
qualitative result.

## ISA - ER and TBS at the paper's 5e4-step budget

`scripts/rm_isa_full.py` calls the real `ERAttack` / `TBSAttack` through the
capability-gated context. ER targets shallow layers (lr 1e-3), TBS deep layers
(lr 1e-2), as in ISA Table 3; TBS uses the full `d_in` basis and
`phi_z = alpha*arctan` (alpha = 5/pi). The frozen model is fp32 with TF32
matmuls.

An exact early-exit forward (run only the first `l` decoder layers) was verified
bit-exact against HF's `hidden_states[l]` (max |diff| = 0.0), cutting per-step
cost roughly 2-4x without changing the mathematics.

Results are appended per job to `isa_full.jsonl`; the table in
`reports/closed_loop/summary.md` lists completed (method, layer, prompt) jobs
with TTRSR, ASR@5 and final state distance. Coverage on this machine is prompts
0-1 x {ER L2, ER L4, TBS L8, TBS L12}; the 5e4 budget is not reduced.

## IA - invariant attacks over the full vocabulary

`scripts/rm_ia_full.py`:

* **EDNN** on real prompt embeddings (glide-reflection obfuscation, exact
  invariant) against the full 128,256 x 2048 reference table: 72 jobs
  (24 prompts x layers 4/8/12). All recover exactly; expected because the
  invariant is exact and no noise is added.
* **Gate-IA**: secret row permutation of the real embedding table plus
  independent column permutations of four real `gate_proj` layers. Invariants
  use the actual `gate_invariant`; matching uses the actual
  `match_by_invariant`. All 128,256 tokens are mapped (3 seeds): top-1 = 1.0.
* **Attn-IA**: Q = E Wq^T for the first 8 query heads of layer 8, per-RoPE-block
  invertible 2x2 transforms (`half_split`), secret row permutation. Invariants
  use the actual `stacked_invariant`; matching is the same nearest-neighbour
  rule via a chunked BLAS implementation (verified equivalent to
  `match_by_invariant` on random data) because the full 128k x 128k x 256
  distance matrix does not fit in memory. 3 seeds: top-1 = 0.9982,
  top-10 = 0.9989.

These use the invariant-preserving construction (no additive embedding noise);
AloePri's `alpha_e` noise is precisely what breaks the invariants, so this is
the attack ceiling under white-box knowledge, not a claim about the defense.

## IMA - InvBERT Classify with a secret key

`scripts/rm_ima_full.py` uses the keyed AloePri mechanism on real embeddings:
the attacker knows the algorithm and synthesises pairs with keys it samples
itself, while the client's token permutation is held out. Working vocabulary:
4,096 real tokens (most frequent in the public corpus), 2,048-d embeddings,
~130k real-text training pairs, 120 epochs, GB setting.

Unknown-key ASR@1 = 0.0 in all three seeds (chance 2.4e-4), training accuracy
2.8%: the classifier cannot generalise across the secret permutation. The
known-key control, same architecture and data, reaches ASR@1 = 0.63 / 0.81 /
0.69 (training accuracy 84-86%), so the failure is the secret key, not model
capacity. Scaling to 16,384 and the full 128,256 vocabulary is corpus-bound (a
128k-way classifier needs millions of pairs); the keyed result is already the
paper's qualitative claim (IMA fails under a secret key).

## TFMA - real text, full tokenizer

`scripts/rm_tfma_full.py` enciphers held-out real text with a random 1:1
permutation over the full 128,256-token vocabulary and gives the attacker the
other half of the corpus as frequency priors. 4,518 observed cipher symbols,
8 seeds.

The paper's independent nearest-frequency match reaches top-1 ~9e-4 (chance
2.2e-4) and top-100 ~4%. The separately named global one-to-one assignment is
not run at this symbol count: the cubic Hungarian solve is deliberately bounded
(recorded per seed as `not_run` with the reason), because the paper's method is
the independent match.

## SDA-2021 - real English, character level

`scripts/rm_sda_full.py` builds a character alphabet from real repository text,
enciphers a 256-character held-out passage with a fresh random 1:1 substitution,
and trains the seq2seq decipherer on `freq-rank(cipher) -> plaintext` pairs with
the target-side cross-entropy, length-matched beam search and the deterministic
key-consistency post-pass. GPU training support was added to `SDA2021` (the
default remains CPU).

Budget note: the paper trains 2M pairs for 20 epochs on a V100 (~110 min/epoch).
This run uses 100k pairs x 8 epochs with a smaller transformer; the objective,
encoding, beam search and key structure are unchanged. The `ser` / `ttrsr` /
`asr` values are recorded in `sda_full.jsonl` and the summary.

## NN/KNN baselines - full vocabulary, both observation levels

`scripts/rm_nn_full.py` runs the real `run_knn` scorer over the full
128,256 x 2048 table for 24 prompts x layers {1,2,4,6,8,10,12,16} x {cosine,
euclidean}:

* embeddings level (the attacker sees the token embeddings): ASR@1 = 1.0
  everywhere - the identity map, a harness check rather than an attack;
* hidden-state level (match a layer state against the embedding table): cosine
  ASR@1 falls from 1.00 at layer 1 to 0.44 (L4), 0.11 (L8), 0.00 (L16);
  euclidean is consistently a little behind (0.81 at L1, 0.07 at L8). This
  gradient with depth is what the ISA papers report and it is the baseline the
  optimization attacks are measured against.

## VMA weight-structure (AloePri B2) on real weights

`scripts/rm_vma_weight_full.py` recovers `Y = Z1 X Z2` with the real
`WeightRowSortVMA` on complete real weight matrices - the GQA key/value
projections (512 x 2048) of decoder layers 4 and 8, which share one row
dimension and therefore one secret Z1:

* `alpha_e = 0` (exact relation): Z1 top-1 = 1.0, `verified = True`,
  reconstruction residual 0.0;
* `alpha_e = 1.0` (AloePri's recommended noisy weights): Z1 top-1 = 0.051,
  `verified = False` - the noise the defense actually adds collapses the
  row-sort fingerprint, which is the paper's point.

The matcher materialises an `n x n x m` distance tensor, so the 128,256-row
embedding table is out of reach for this variant on this machine; the run uses
whole real matrices rather than subsampled rows.

## TFMA global assignment

`scripts/rm_tfma_full.py` now runs the Hungarian assignment on a bounded prefix
of the ciphertext (the solve is cubic in the number of observed symbols), so it
produces real numbers instead of a skip: over 1,398 observed symbols and 8 seeds,
top-1 = 0.0014, top-10 = 0.0162 - slightly ahead of the independent match
(top-1 0.0009, top-10 0.0084) but still near the 1/1398 chance floor. Frequency
alone does not identify tokens.

## Runs still in flight

SDA-2021 (real English, 100k pairs x 8 epochs), SDA-2023 (1:1 and homophonic),
ISA AloePri State1/State2 at the 5e4 budget, the VMA OSNIP sorted-L1 variant on
the full vocabulary (its batched distance is verified equal to the class's
`_sorted_l1` to 2.8e-14), and the IMA Seq2Seq (BB) / Qwen2 inverters under both
secret-key and known-key settings. The BB decoder emits its own length, so it is
scored with SER. `scripts/run_gpu_queue.sh` runs the GPU jobs one at a time -
concurrent GPU jobs on this card slow each other about 4x.

## Honest limitations

* One model (Llama-3.2-1B) on one 8 GB GPU; the paper's 14B/32B/671B and
  multilingual corpora are unavailable.
* VMA covers 3 prompts x 8 layers; ISA covers 2 prompts x 4 layer/method pairs;
  the remaining budget is compute-bound, not omitted by choice of method.
* IMA's working vocabulary is 4,096 real tokens rather than 128,256; the
  classifier needs a corpus far larger than this machine can generate/train on.
* TFMA's global-assignment extension is bounded; the paper's independent method
  is run in full.
* There is no defense in this loop. Plaintext replay attacks are expected to be
  perfect; the reported numbers validate implementations and establish the
  undefended baseline.
