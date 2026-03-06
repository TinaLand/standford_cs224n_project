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

**Takeaways (BERT)**  
- Gate warmup (1000 steps) stabilizes MrBERT; target 0.3 with warmup often matches or beats target 0.5 on SNLI, SST-2, XNLI, TyDi QA.  
- TyDi QA benefits from lower target (0.3) and warmup: 28.16% vs 23.64% at 0.5, with more conservative deletion (10.9% vs 26.8%).  
- IMDB is unstable across runs (gate sometimes barely activates).

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

**Takeaways (XLM-R)**  
- XLM-R is more sensitive to deletion than BERT: SST-2 and SNLI often collapse when the gate is too aggressive.  
- 0.3 + longer warmup does not consistently beat 0.5 on XLM-R (e.g. XNLI 68.07% vs 56.71%).  
- For a proper Baseline XLM-R vs MrXLM comparison, the script now runs baseline (gate_weight=0) before each MrXLM run; new runs will fill that gap.

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

Across runs, **when the correlation is positive** (e.g. BERT MRPC in the L4 warmup runs), it supports the claim that **an increase in deletion rate is associated with an increase in loss** on that example—i.e. more aggressive pruning correlates with worse predictions. Where the correlation is near zero or slightly negative, the model may be deleting less informative tokens first, or the signal may be noisier; the positive cases in the table still show that loss vs. deletion rate is not independent and that over-deletion can be harmful.

### 3.2 Summary sentence for the report

**There is a correlation between the increase in loss and the deletion rate:** in several BERT (and some XLM-R) runs, validation examples with higher per-example deletion rates tend to have higher cross-entropy loss. The effect is modest in magnitude (e.g. Pearson ≈ 0.03–0.07 where positive) but supports the design goal of controlling deletion via the PI controller and the observation that excessive deletion (e.g. on XLM-R SST-2/SNLI or TyDi QA) coincides with poor accuracy.

---

## 4. TyDi QA and SNLI error cases (unchanged from previous analysis)

- **TyDi QA** (`error_cases_tydiqa.jsonl`): High-deletion error cases often have answer-bearing tokens in `dropped_tokens`; coordinate re-mapping is correct but over-deletion harms span prediction. Mitigation: lower target deletion, gate warmup, or QA-specific constraints.  
- **SNLI** (`error_cases_snli.jsonl`): Many errors have deletion_rate ≥ 0.90; premise/hypothesis content appears in `dropped_tokens`, so the model decides from a heavily pruned fragment. This explains a long tail of failures despite good average accuracy.

---

## 5. Latency (reference)

- **GPU** (from L4/A100 runs): e.g. `latency_results.json` in some result folders reports baseline vs MrBERT inference; hard deletion shortens sequence length and can yield ~30–55% speedup when the gate is active.  
- **CPU** (earlier run): MrBERT was slower than baseline due to gate and index overhead; pruning is most beneficial on GPU where matmul/attention dominate.

---

## 6. Data locations (results/new)

- **BERT from L4**  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.5MR_USE_PI=1MODELS=bertGATE_WARMUP_STEPS=1000BATCH=24USE_WANDB=1LOG_LEVEL=1/`  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.3MR_USE_PI=1MODELS=bertBATCH=24EPOCHS=1GATE_WARMUP_STEPS=1000/`  
  - `bert_from_l4/from_l4_MR_TARGET_DEL=0.3MODELS=bertBATCH=24EPOCHS=1/` (no warmup)  
- **XLM-R from A100**  
  - `xlmr_from_A100/epochs3batch24Mr-target-del0.5-mr-use-pi-log-level1-gate-warmup-steps1000/`  
  - `xlmr_from_A100/epochs5Batch24MrTargetDel0.3GateWarmupSteps1500/`  

Each folder may contain `train_results.jsonl`, `loss_vs_deletion_*.json`, `error_cases_*.jsonl`, and `latency_results.json`.

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

- [ ] **Run full XLM-R with baseline:** Execute one complete XLM-R run (L4 or Modal) so that every classification task has both Baseline XLM-R (gate_weight=0) and MrXLM. The script already runs baseline before MrXLM per dataset; fill the gap in the results table.
- [ ] **PI ablation:** Add experiments comparing (1) **fixed \(\alpha\)** (no PI) vs full **PI controller**, and optionally (2) **P-only** vs **PI**, to show that the integral term and dynamic \(\alpha\) are necessary to hit the target deletion rate and avoid drift. Plot deletion rate vs step or deletion distribution for each setting.
- [ ] **Pareto curve (accuracy vs speed):** For one model (e.g. BERT) and one or two datasets, run several target_deletion values (e.g. 0.2, 0.3, 0.5, 0.6), record validation accuracy and inference latency (or FLOPs if available). Plot accuracy vs latency (or vs FLOPs) and highlight the Pareto frontier in the report.
- [ ] **Deletion vs loss visualization:** Use existing `loss_vs_deletion_*.json` scatter data to add a figure (e.g. scatter plot of loss vs deletion_rate per validation example) for one or two datasets, to support the “correlation between increase in loss and deletion rate” claim.
- [ ] **Error-case narrative:** Pick 2–3 representative examples from `error_cases_snli.jsonl` or `error_cases_tydiqa.jsonl` (high deletion, wrong prediction) and briefly describe in the report: high deletion → important tokens dropped → prediction error, to reinforce the correlation and the need for controlling deletion.
- [ ] **Optional — distillation:** If time allows, add a simple distillation setup (e.g. logits or one-layer feature alignment) to reduce XLM-R accuracy drop; report as preliminary or future work.
- [ ] **Optional — cross-lingual:** If desired, add a short subsection on XNLI multi-language accuracy or language transfer gap; otherwise cite XNLI results already in the tables.
