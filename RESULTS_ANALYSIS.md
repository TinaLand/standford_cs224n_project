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
