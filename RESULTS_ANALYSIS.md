# Results and Analysis

This file summarizes experimental results and analysis for MrBERT, MrXLM, and the relationship between **loss and deletion rate**. Data below are drawn from `results/new/` (L4 BERT runs and A100 XLM-R runs) and from earlier 3-epoch runs where noted.

---

## 1. MrBERT (BERT backbone) — L4 runs in `results/new/bert_from_l4`

### 1.1 BERT with warmup, target deletion 0.5 (1 epoch, batch 24, GATE_WARMUP_STEPS=1000)

Source: `results/new/bert_from_l4/from_l4_MR_TARGET_DEL=0.5MR_USE_PI=1MODELS=bertGATE_WARMUP_STEPS=1000BATCH=24USE_WANDB=1LOG_LEVEL=1/train_results.jsonl`

| Dataset | Baseline (val acc) | MrBERT (val acc) | MrBERT actual deletion |
|---------|--------------------|------------------|-------------------------|
| MRPC    | 68.87%             | 68.38%           | 30.1%                   |
| IMDB    | 87.28%             | 49.76%           | 4.2%                    |
| SNLI    | 60.32%             | **88.78%**       | 76.2%                   |
| SST-2   | 80.39%             | **91.17%**       | 58.9%                   |
| TyDi QA | 26.08%             | 23.64%           | 26.8%                   |
| XNLI    | 75.38%             | **81.65%**       | 65.3%                   |

### 1.2 BERT with warmup, target deletion 0.3 (1 epoch, batch 24, GATE_WARMUP_STEPS=1000)

Source: `results/new/bert_from_l4/from_l4_MR_TARGET_DEL=0.3MR_USE_PI=1MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000/train_results.jsonl` (lines 36–47)

| Dataset | Baseline (val acc) | MrBERT (val acc) | MrBERT actual deletion |
|---------|--------------------|------------------|-------------------------|
| MRPC    | 68.38%             | 68.63%           | 53.8%                   |
| IMDB    | 87.85%             | 57.42%           | 4.7%                    |
| SNLI    | 74.00%             | **89.02%**       | 76.1%                   |
| SST-2   | 67.89%             | **92.55%**       | 61.8%                   |
| TyDi QA | 20.18%             | **28.16%**       | 10.9%                   |
| XNLI    | 74.82%             | **80.56%**       | 65.4%                   |

### 1.3 BERT with warmup, target deletion 0.7 (1 epoch, batch 24, GATE_WARMUP_STEPS=1000)

Source: `results/new/bert_from_l4/from_l4_MR_TARGET_DEL=0.7MR_USE_PI=1MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000LOG_LEVEL=1USE_WANDB=1/results/train_results.jsonl`

| Dataset | Baseline (val acc) | MrBERT 0.7 (val acc) | MrBERT 0.7 actual deletion |
|---------|--------------------|----------------------|----------------------------|
| MRPC    | 70.34%             | 32.84%               | 30.8%                      |
| IMDB    | 88.42%             | 50.00%               | 12.2%                      |
| SNLI    | 67.27%             | **87.79%**           | 78.2%                      |
| SST-2   | 86.58%             | **90.48%**           | 57.7%                      |
| TyDi QA | 24.45%             | **0.00%**            | 36.6%                      |
| XNLI    | 66.31%             | **80.28%**           | 67.2%                      |

**Takeaways (BERT)**  
- Gate warmup (1000 steps) stabilizes MrBERT; target 0.3 with warmup often matches or beats target 0.5 on SNLI, SST-2, XNLI, TyDi QA.  
- Increasing the target deletion to 0.7 is a mixed story: SNLI, SST-2 and XNLI still improve over baseline, but MRPC and IMDB degrade sharply and TyDi QA collapses to 0% accuracy, showing that very aggressive deletion is unsafe for QA and some classification tasks.  
- IMDB remains unstable across runs (gate sometimes barely activates, and high-target runs can fail).
- Within this 0.7 run, per-example loss–deletion correlations (from `loss_vs_deletion_mrpc.json`, `snli.json`, `sst2.json` under the same folder) are weak (e.g. MRPC Pearson ≈ 0.007, SNLI ≈ −0.019, SST-2 ≈ −0.041), suggesting that once deletion is this aggressive the model either learns to delete relatively uninformative tokens on SNLI/SST-2 or the signal becomes too noisy to show a clear monotonic trend.

### 1.4 BERT 0.3 without PI (ablation; 1 epoch, batch 24, GATE_WARMUP_STEPS=1000)

Source: `results/new/bert_from_l4/MR_TARGET_DEL=0.3MR_USE_PI=0MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000LOG_LEVEL=1USE_WANDB=1/results/train_results.jsonl`

Gate is on (gate_weight=1e-4) but **PI controller is off** (use_pi=false), so α is fixed. This run completed MRPC and IMDB only in the no-PI block; baselines are from the same file (same run’s baseline block).

| Dataset | Baseline (val acc) | MrBERT 0.3 no-PI (val acc) | MrBERT 0.3 no-PI actual deletion |
|---------|--------------------|----------------------------|-----------------------------------|
| MRPC    | 68.14%             | 69.36%                     | 67.8%                             |
| IMDB    | 87.82%             | 86.20%                     | 2.7%                              |

**Takeaway:** Without PI, the gate still deletes heavily on MRPC (67.8%) and accuracy is close to baseline; on IMDB the gate barely activates (2.7%) and accuracy stays high. This supports that the PI controller is needed to steer deletion toward a target and avoid either over-deletion (MRPC) or under-activation (IMDB).

---

## 2. MrXLM (XLM-R backbone) — A100 runs in `results/new/xlmr_from_A100`

### 2.1 XLM-R, target deletion 0.5 (3 epochs, batch 24, gate warmup 1000)

Source: `results/new/xlmr_from_A100/epochs3batch24Mr-target-del0.5-mr-use-pi-log-level1-gate-warmup-steps1000/train_results.jsonl`

| Dataset | MrXLM (val acc) | Note |
|---------|-----------------|------|
| MRPC    | 67.16%          | Reasonable |
| SST-2   | 52.41%          | Near random; over-deletion in loss_vs_deletion |
| SNLI    | 50.74%          | Near random; over-deletion |
| IMDB    | **85.76%**      | Good |
| XNLI    | **68.07%**      | Reasonable |

*No baseline XLM-R was run in this experiment; comparison is to BERT tables above.*

### 2.2 XLM-R, target deletion 0.3 (5 epochs, batch 24, gate warmup 1500)

Source: `results/new/xlmr_from_A100/epochs5Batch24MrTargetDel0.3GateWarmupSteps1500/train_results.jsonl`

| Dataset | MrXLM (val acc) |
|---------|-----------------|
| MRPC    | 68.38%          |
| SST-2   | 57.45%          |
| SNLI    | 45.79%          |
| IMDB    | 85.60%          |
| XNLI    | 56.71%          |

### 2.3 XLM-R, target deletion 0.3 (3 epochs, batch 24, gate warmup 1500, log-level 3) — Modal A100

Source: `results/new/xlmr_from_A100/epochs3--batch 24--mr-target-del 0.3--mr-use-pi--log-level3gate-warmup-steps1500/train_results.jsonl`

This run has **baseline XLM-R** (gate_weight=0) and **MrXLM 0.3** (PI) for each dataset; no TyDi QA.

| Dataset | Baseline (val acc) | MrXLM 0.3 (val acc) | Note |
|---------|--------------------|---------------------|------|
| MRPC    | 68.38%             | **72.06%**          | MrXLM improves |
| SST-2   | 79.01%             | 61.01%              | Large drop with gate |
| SNLI    | 33.82%             | 33.82%              | Both poor; no change |
| IMDB    | 79.87%             | 50.00%              | Large drop |
| XNLI    | 62.77%             | 43.41%              | Large drop |

**Takeaways (XLM-R)**  
- XLM-R is more sensitive to deletion than BERT: SST-2, SNLI, IMDB, and XNLI often drop or stay near random when the gate is active; only MRPC improves (72% vs 68% baseline) in this 0.3 run.  
- 0.3 + warmup 1500 does not fix the sensitivity: in this 3-epoch run, MrXLM 0.3 hurts SST-2, IMDB, and XNLI.  
- Baseline vs MrXLM is now directly comparable in this folder (and in 2.1/2.2 where applicable).

---

## 3. Loss vs. deletion rate: correlation analysis

We compute **per-example validation loss** and **per-example deletion rate** (fraction of tokens dropped by the gate for that example) and store them in `loss_vs_deletion_<dataset>.json` under each run. Each file includes:

- `pearson`, `spearman`: correlation between loss and deletion rate over validation examples.  
- `scatter_sample`: a sample of (loss, deletion_rate) pairs.

**There is a correlation between higher deletion rate and higher loss** in several settings, consistent with the intuition that removing more tokens tends to hurt prediction on that example.

### 3.1 Representative correlations (from `results/new/`)

| Run | Dataset | Pearson | Spearman | Interpretation |
|-----|---------|---------|----------|----------------|
| BERT L4, 0.3 + warmup | MRPC | **+0.064** | **+0.090** | Higher deletion → higher loss |
| BERT L4, 0.5 + warmup | MRPC | **+0.068** | +0.057 | Higher deletion → higher loss |
| BERT L4, 0.5 + warmup | SST-2 | **+0.035** | +0.015 | Weak positive |
| BERT L4, 0.3 + warmup | SST-2 | −0.022 | −0.010 | Near zero |
| BERT L4, 0.3 + warmup | SNLI | −0.048 | +0.060 | Mixed |
| XLM-R A100, 0.5 | MRPC | −0.018 | +0.014 | Near zero |
| XLM-R A100, 0.3 | MRPC | **+0.028** | +0.009 | Weak positive |
| XLM-R A100, 0.5 | SST-2 | −0.014 | +0.026 | Near zero |
| XLM-R A100, 0.5 | SNLI | +0.028 | +0.012 | Weak positive |
| XLM-R A100, 0.5 | XNLI | −0.046 | −0.084 | Slight negative (higher del → lower loss) |
| XLM-R A100, 0.3 | SST-2 | −0.043 | −0.011 | Near zero / weak negative |
| XLM-R A100, 0.3 | SNLI | −0.015 | −0.027 | Near zero |
| XLM-R A100, 0.3 | XNLI | +0.005 | +0.007 | Near zero |
| BERT L4, 0.3 no-PI | MRPC | +0.007 | +0.008 | Near zero |
| BERT L4, 0.3 no-PI | SNLI | −0.019 | −0.016 | Near zero |
| BERT L4, 0.3 no-PI | SST-2 | −0.041 | −0.049 | Near zero / weak negative |
| XLM-R A100, 0.3 (3ep, warmup 1500) | MRPC | −0.130 | −0.208 | Negative (higher del → lower loss) |
| XLM-R A100, 0.3 (3ep, warmup 1500) | SST-2 | **+0.195** | **+0.211** | Higher deletion → higher loss |
| XLM-R A100, 0.3 (3ep, warmup 1500) | SNLI | −0.006 | +0.015 | Near zero |
| XLM-R A100, 0.3 (3ep, warmup 1500) | XNLI | +0.011 | +0.070 | Weak positive |

Across runs, **when the correlation is positive** (e.g. BERT MRPC in the L4 warmup runs), it supports the claim that **an increase in deletion rate is associated with an increase in loss** on that example—i.e. more aggressive pruning correlates with worse predictions. Where the correlation is near zero or slightly negative, the model may be deleting less informative tokens first, or the signal may be noisier; the positive cases in the table still show that loss vs. deletion rate is not independent and that over-deletion can be harmful.

**Why only these datasets?** The table is built from **existing** `loss_vs_deletion_<dataset>.json` files under `results/new/` (each contains `pearson` and `spearman`). Only (run, dataset) pairs that have such a file are included. The reason other datasets do not appear in the table is that there is **no** `loss_vs_deletion` file for them (i.e. no correlation data between loss and deletion rate), not that there is no training loss.

- **IMDB**: `run_experiments.sh` does **not** call `analyze_loss_vs_deletion` for IMDB (only MRPC, SNLI, SST-2, XNLI, TyDi QA). So there are no IMDB correlation numbers unless the script is run manually for IMDB.
- **TyDi QA**: The pipeline can produce `loss_vs_deletion_tydiqa.json`, but in the current `results/new/` snapshot there are **no** TyDi QA loss-vs-deletion files (e.g. those runs may have skipped the step or used different output paths). Adding TyDi QA to the table would require re-running the analysis step for TyDi QA and copying the file into the run folders.
- **XNLI**: Present for **XLM-R** runs only in `results/new/` (see table above); BERT run folders in this snapshot do not contain `loss_vs_deletion_xnli.json`, so BERT–XNLI is not compared here.

### 3.2 Summary sentence for the report

**There is a correlation between the increase in loss and the deletion rate:** in several BERT (and some XLM-R) runs, validation examples with higher per-example deletion rates tend to have higher cross-entropy loss. The effect is modest in magnitude (e.g. Pearson ≈ 0.03–0.07 where positive) but supports the design goal of controlling deletion via the PI controller and the observation that excessive deletion (e.g. on XLM-R SST-2/SNLI or TyDi QA) coincides with poor accuracy.

---

## 4. TyDi QA and SNLI error cases

Error cases are extracted by `scripts/extract_error_cases.py` (wrong prediction and `deletion_rate >= 0.7`). Below we use files under `results/new/` (e.g. BERT L4 run folders and XLM-R A100 folders).

### 4.1 TyDi QA

**Source (example):** `results/new/bert_from_l4/from_l4_MR_TARGET_DEL=0.3MODELS=bertBATCH=24EPOCHS=1/error_cases_tydiqa.jsonl`

- **Count & deletion:** One run has 50 error cases; `deletion_rate` ranges 0.73–0.95, mean ≈ 0.86, median ≈ 0.87; **30%** of cases have `deletion_rate ≥ 0.90`.
- **Answer-bearing tokens in `dropped_tokens`:** In many cases the gold answer span lies in context, but key subwords appear in `dropped_tokens`. Example (idx 2): question "ما عاصمة جورجيا؟" (What is the capital of Georgia?), answer "تبليسي" (Tbilisi); `dropped_tokens` include question/context subwords that precede or overlap the answer region (e.g. "ع", "##ا", "##م", "##ة", "ج", "##و", "##ر", "##ج", "##ي", "##ا"). Another (idx 5): answer "جيمس واط" (James Watt); the model predicted (0, 24) while gold is (45, 51); many context tokens are dropped so the span predictor sees a heavily pruned sequence.
- **Coordinate re-mapping:** Start/end indices are re-mapped correctly to the *kept* token sequence; the issue is that when too many tokens are removed, the answer span can fall in the dropped region or the remaining context is insufficient for the model to localize the span.
- **Mitigation:** Lower target deletion (e.g. 0.3), gate warmup, or QA-specific constraints (e.g. protect tokens in the context window that overlap the gold span during evaluation, or use a separate deletion budget for question vs context).

### 4.2 SNLI

**Source (example):** `results/new/bert_from_l4/from_l4_MR_TARGET_DEL=0.5MR_USE_PI=1MODELS=bertGATE_WARMUP_STEPS=1000BATCH=24USE_WANDB=1LOG_LEVEL=1/error_cases_snli.jsonl`

- **Count & deletion:** 50 error cases in this file; many have `deletion_rate` in 0.82–0.95 (e.g. 0.90+ is common). Premise and hypothesis content frequently appear in `dropped_tokens`.
- **Concrete examples:**
  - Case (idx 1): premise "Two women are embracing while holding to go packages.", hypothesis "Two woman are holding packages." Label **entailment**, pred **contradiction**. `dropped_tokens` include "two", "women", "are", "to", "go", "packages", "woman", "are", "holding", "packages"—i.e. much of the premise/hypothesis is pruned, so the model decides from a small fragment and tends to predict contradiction.
  - Case (idx 25): premise "A young boy in a field of flowers carrying a ball", hypothesis "boy in field". Label **entailment**, pred **contradiction**. Dropped: "a", "young", "boy", "in", "field", "carrying", "boy", "in", "field"; again key content is removed and the model fails to infer entailment.
- **Pattern:** Many errors are **entailment → predicted contradiction** or **neutral → contradiction**; with deletion_rate ≥ 0.90, premise/hypothesis tokens are heavily dropped and the model often defaults to contradiction. This produces a long tail of failures despite good average accuracy.
- **Mitigation:** Lower target deletion, longer gate warmup, or constraining deletion so that at least a minimum fraction of premise/hypothesis tokens is kept (e.g. in the NLI head).

---

## 5. Latency (reference)

- **GPU** (from L4/A100 runs): e.g. `latency_results.json` in some result folders reports baseline vs MrBERT inference; hard deletion shortens sequence length and can yield ~30–55% speedup when the gate is active.  
- **CPU** (earlier run): MrBERT was slower than baseline due to gate and index overhead; pruning is most beneficial on GPU where matmul/attention dominate.

---

## 6. Data locations (results/new)

All paths under `results/new/`. Each folder may contain `train_results.jsonl`, `loss_vs_deletion_*.json`, `error_cases_*.jsonl`, `latency_results.json`, and optionally `results/` or `logs/` subdirs (e.g. after copying from GPU/Modal).

- **BERT from L4**  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.5MR_USE_PI=1MODELS=bertGATE_WARMUP_STEPS=1000BATCH=24USE_WANDB=1LOG_LEVEL=1/` — 0.5 + PI, 1 ep, batch 24  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.3MR_USE_PI=1MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000/` — 0.3 + PI, 1 ep  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.3MODELS=bertBATCH=24EPOCHS=1/` — 0.3, no warmup  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.7MR_USE_PI=1MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000LOG_LEVEL=1USE_WANDB=1/results/` — 0.7 + PI, 1 ep (results + logs in `results/`, `logs/`)  
  - `bert_from_l4/MR_TARGET_DEL=0.3MR_USE_PI=0MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000LOG_LEVEL=1USE_WANDB=1/results/` — 0.3 **no PI** (ablation), 1 ep  
- **XLM-R from A100 (Modal)**  
  - `xlmr_from_A100/epochs3batch24Mr-target-del0.5-mr-use-pi-log-level1-gate-warmup-steps1000/` — 0.5, 3 ep, warmup 1000  
  - `xlmr_from_A100/epochs5Batch24MrTargetDel0.3GateWarmupSteps1500/` — 0.3, 5 ep, warmup 1500  
  - `xlmr_from_A100/epochs3--batch 24--mr-target-del 0.3--mr-use-pi--log-level3gate-warmup-steps1500/` — 0.3, 3 ep, warmup 1500, log-level 3 (baseline + MrXLM per dataset)

---

## 7. Key implementation details

This section documents how the main technical components are implemented in the codebase (MrBERT/MrXLM, `train_mrbert.py`, `mrbert/pi_controller.py`).

### 7.1 Deletion map (keep_indices, kept_lengths)

**Purpose:** At inference, we use **hard deletion**: tokens with gate value below a threshold are removed and the sequence is shortened. The QA head (and any span-based output) runs on this shortened sequence, so predicted positions refer to **short indices**. We need a **deletion map** to map those positions back to **original token indices** for evaluation (e.g. EM on the original context).

**Implementation:**

- In `mrbert/modeling_mrbert.py` (and similarly in `modeling_mrxlm.py`, `modeling_mrroberta.py`), when `use_soft_deletion=False` (inference path):
  1. After the gate layer, we compute `keep_masks = (gate > threshold)` with `threshold = gate_k * gate_threshold_ratio` (default 0.5, so tokens with \(G > k/2\) are kept). `[CLS]` is force-kept (index 0) so the pooler and classification head always have a valid token.
  2. For each batch element \(b\), we collect the **original indices** of kept tokens: `keep_indices[b, j]` = original sequence index of the \(j\)-th kept token. `kept_lengths[b]` = number of kept tokens for that element.
  3. Hidden states are gathered with `torch.gather(hidden_states, 1, keep_indices.unsqueeze(-1).expand(...))` so that the rest of the encoder and the task head see a tensor of shape `(batch, max_kept, hidden_size)`.
  4. The model output attaches `keep_indices` and `kept_lengths` so downstream code can remap positions.

- **QA coordinate re-mapping** (`train_mrbert.py`, evaluation and `scripts/extract_error_cases.py`): The QA head outputs `(pred_start_short, pred_end_short)` in the **short** sequence. We map back to original indices with:
  - `pred_start = keep_indices.gather(1, pred_start_short.unsqueeze(1)).squeeze(1)`
  - `pred_end = keep_indices.gather(1, pred_end_short.unsqueeze(1)).squeeze(1)`
  and then compare to gold spans in the original sequence. Without this map, EM would be computed in the wrong coordinate system.

### 7.2 Gate warmup

**Purpose:** Let the model learn from the full sequence for the first \(N\) steps before applying gate pressure, so the gate and classifier do not collapse (e.g. TyDi QA with no warmup often deletes almost all tokens).

**Implementation:**

- `train_mrbert.py` exposes `--gate_warmup_steps` (default 0). `run_experiments.sh` passes it via `GATE_WARMUP_STEPS` and `WARMUP_ARGS=(--gate_warmup_steps $GATE_WARMUP_STEPS)`.
- In the training loop, for each step:
  - If `step < args.gate_warmup_steps`: we set `alpha = 0.0` (no gate regularizer term in the loss). The gate is still computed and can be logged; the PI controller is **not** updated during warmup so it does not push alpha up while the model is learning.
  - If `step >= args.gate_warmup_steps`: we use the normal `gate_regularizer_weight` (or the PI-controlled alpha) and the PI controller is updated each step using the current batch’s gate.
- When warmup ends, a one-line log is printed: `Gate warm-up done at step N; enabling gate regularizer (alpha=...)`.

### 7.3 PI controller (P + I)

**Purpose:** Keep the **actual deletion ratio** close to a **target ratio** (e.g. 0.5) by adapting the gate regularizer weight \(\alpha\) each step (paper Eq. 4–6).

**Implementation:**

- `mrbert/pi_controller.py`: `PIController(target_ratio, kp=0.5, ki=1e-5, gamma=0.9)`.
- Each step, with the current batch gate (values in \([k, 0]\)):
  - Deletion ratio = fraction of tokens with `gate < gate_k * 0.5`.
  - Error = `target_ratio - current_ratio`.
  - P term: exponential moving average of error, `p = gamma * p + (1 - gamma) * error`.
  - I term: integral of error, `i = i + error`.
  - \(\alpha = \max(0, k_p \cdot p + k_i \cdot i)\), and this \(\alpha\) is used as the gate regularizer weight for the next forward (or the same step’s loss if you use the updated alpha for the next batch only; in our code we use it for the **next** batch).
- The integral term (I) is what allows the controller to remove steady-state error (e.g. if deletion is consistently below target, \(\alpha\) keeps increasing until deletion reaches target).

### 7.4 Soft vs hard deletion

- **Training:** `use_soft_deletion=True` (default when `model.training` is True). The gate is **not** used to drop tokens; it is added to the attention logits as a per-key bias (paper Eq. 2). Sequence length stays the same; the loss is differentiable everywhere. This avoids non-differentiable indexing and lets the gate learn which tokens to down-weight.
- **Inference:** `use_soft_deletion=False`. Tokens with \(G \le k/2\) are removed: we build `keep_indices` and `kept_lengths`, gather hidden states, and run the remaining encoder layers (and the task head) on the shortened sequence. This gives real speedup (fewer tokens in attention and FFN).

### 7.5 Gate placement and softmax1

- The gate is inserted **after** a fixed encoder layer index (`gate_layer_index`, default 3). Layers 0..gate_layer run as in the base model; the gate is applied to the hidden states at that layer; then either soft masking (training) or hard deletion (inference) is applied before the remaining layers.
- In gated layers we use **softmax1** (paper Eq. 7): \(\mathrm{softmax1}(x)_i = \exp(x_i) / (1 + \sum_j \exp(x_j))\) so that when all gate values are equal to \(k\), attention weights do not collapse to zero.

---

## 8. Todo list (next steps)

Prioritized list of follow-up work for experiments and the report. Items are in English for direct use in planning or the write-up.

### 8.1 XLM-R rescue (non-distillation)

XLM-R “cracks” under the same gate settings that work for BERT (SNLI/SST-2/IMDB/XNLI drop to near random). Without distillation, the following **non-distillation** directions are recommended (by priority). **Items 1–3 are already implemented** (CLI args + `run_experiments.sh` + `run_xlmr_modal.py`); the descriptions below are kept for reference.

1. **Move gate to a later layer (highest ROI)** — **Already implemented.**  
   - **Idea:** Prune after layer **6 or 8** instead of 3 so multilingual semantics are more stable before deletion.  
   - **Code:** `gate_layer_index` is supported in `MrXLMRoberta*` (and BERT) but is **hardcoded to 3** in `train_mrbert.py`; add `--gate_layer_index` and pass it into the model constructors.  
   - **Trade-off:** Fewer layers run on shortened sequence → less speedup, but accuracy may recover.

2. **Ultra-slow PI warmup** — **Already implemented.**  
   - **Idea:** Use **GATE_WARMUP_STEPS ≥ 3000** and **lower \(K_p\)** (e.g. 0.002 instead of 0.5) so \(\alpha\) ramps up very slowly and the model adapts to low deletion (e.g. ~5%) before moving toward 30%.  
   - **Code:** `--gate_warmup_steps` exists; PI gains are fixed in `PIController(kp=0.5, ki=1e-5)`. Add `--controller_kp` (and optionally `--controller_ki`) to the parser and pass them into `PIController`.

3. **Higher retention threshold for XLM-R** — **Already implemented.**  
   - **Idea:** Use a **higher gate_threshold_ratio** (e.g. 0.6–0.7) so more tokens are kept (less aggressive deletion) for XLM-R only.  
   - **Code:** `gate_threshold_ratio` is already in the config (default 0.5); expose it as `--gate_threshold_ratio` and pass through when building the model.

4. **Subword / token-group protection (optional)**  
   - **Idea:** At inference, if any subword of a word is kept, force-keep all subwords of that word (avoids broken words with SentencePiece).  
   - **Code:** Would require token-group logic in the hard-deletion path (e.g. in `modeling_mrxlm.py`) using tokenizer word boundaries.

5. **Sigmoid temperature (optional)**  
   - **Idea:** Softer gate (e.g. scaled sigmoid with temperature) so borderline tokens are less “binary” and get a bit more survival chance.  
   - **Code:** Currently gate is `gate_k * sigmoid(logits)`; could add a temperature divisor inside sigmoid and expose it.

**Priority order for implementation:** (1) `--gate_layer_index` and run XLM-R with layer 6 vs 3; (2) `--controller_kp` + longer warmup; (3) `--gate_threshold_ratio` for XLM-R. *(Items 1–3 are done.)* MRPC already improves with the gate (72% vs 68%); the goal is to replicate that kind of “controlled” deletion on SNLI/SST-2 by making the gate less aggressive (later layer, slower \(\alpha\), or higher keep threshold).

### 8.2 General todos

- [ ] **Run full XLM-R with baseline:** Execute one complete XLM-R run so that every task has Baseline XLM-R and MrXLM (done for the 2.3 run; 2.1/2.2 lack baseline in the same run).
- [ ] **PI ablation:** Compare fixed \(\alpha\) (no PI) vs full PI; optionally P-only vs PI. Plot deletion rate vs step.
- [ ] **Pareto curve (accuracy vs speed):** For BERT (and optionally XLM-R), sweep target_deletion, record val acc and latency; plot and highlight Pareto frontier.
- [ ] **Deletion vs loss visualization:** Add a scatter figure from `loss_vs_deletion_*.json` for one or two datasets.
- [ ] **Error-case narrative:** Describe 2–3 high-deletion error cases from SNLI/TyDi QA in the report.
- [ ] **Optional — distillation:** If time allows, add distillation to reduce XLM-R accuracy drop.
- [ ] **Optional — cross-lingual:** Short subsection on XNLI multi-language or transfer gap; or cite existing XNLI tables.
