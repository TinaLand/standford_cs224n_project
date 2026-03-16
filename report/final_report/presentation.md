GitHub:
- **repo1: Alina Huang (this repo)** `https://github.com/TinaLand/standford_cs224n_project`
- **repo2: Hiva Zaad**: `https://github.com/HivaMohammadzadeh1/CS224N-project`

---

## 1. Key gate design decisions
| Decision | Choice | Where |
|----------|--------|--------|
| **Gate position** | After a fixed encoder layer (default layer 3). | `MrBertConfig.gate_layer_index`; `_encoder_forward_with_gate` runs layers 0..L then gate then rest. |
| **Softmax1** | Use softmax1 in gated layers so attention can sum to &lt;1. | `use_softmax1=True` in config; `_run_layer_with_soft_gate` calls `softmax1(attention_scores)`. |
| **PI controller** | Optional; targets a deletion ratio instead of fixed α. | `train_mrbert.py` with `--use_pi`; `PIController.step()` each batch. |
| **Gate warmup** | Delay gate regularizer for N steps to stabilize. | `--gate_warmup_steps` (e.g. 1000); α set to 0 until step N. |
| **QA** | Span prediction on shortened sequence; map back to original indices. | `keep_indices` / `kept_lengths` in `MrBertForQuestionAnswering` / `MrXLMRobertaForQuestionAnswering` for span coordinate remapping. |

## 2. What Matches (repo2 ↔ repo1)

At a high level, our implementation matches Hiva's report in four areas: **model structure, control rules, QA handling, and tasks/metrics**.

| Category | Repo2 | Repo1 | Note |
|----------|---------------------|---------------------|------|
| **Model + gate** | Delete gate after encoder layer 3; LayerNorm → Linear(768→1) → Sigmoid · k (k = −30); soft deletion = add G as attention bias; hard deletion = remove tokens with G < k/2; keep [CLS]/[SEP], delete [PAD]. | `DeleteGate` in `mrbert/modeling_mrbert.py` with `gate_layer_index=3`, `gate_k=-30`; `_run_layer_with_soft_gate` adds gate to attention scores; hard path in `_encoder_forward_with_gate` thresholds at `gate_k * gate_threshold_ratio` (k/2), keeps CLS, gathers kept tokens, rebuilds the attention mask. | **Core mechanism is fully aligned**; differences come later from init / PI hyperparameters / QA blending. |
| **PI controller + loss** | L = L_task + α·L_deletion, L_deletion = mean(G); α_t updated by a PI controller: error e_t = δ − r_t, P/I updates, α_t = max(0, p_t + i_t); γ=0.9 for EMA. | `get_gate_regularizer_loss(gate)=gate.mean()`; `PIController.step()` implements the same rule: compute error from current deletion ratio, update P/I, set α = max(0, kp*P + ki*I) with default `pi_gamma=0.9`. | **Control logic matches**; the only difference is the default `kp` (report: 0.01, this repo: 0.5). |
| **QA span remap + ragged batch** | After hard deletion, predict spans on the **shortened sequence**, then map back to original token indices via keep indices / lengths; batch items have different lengths, handled via ragged / gather-style ops. | `MrBertForQuestionAnswering` / `MrXLMRobertaForQuestionAnswering` output `keep_indices` and `kept_lengths` for span remap; after hard deletion, `max_kept + torch.gather + length mask` build a variable-length batch, matching the "QA span coordinate remapping / ragged tensors" description. | **QA pipeline design is aligned**; the main difference is that this repo currently **does not implement pre-deletion blending**. |
| **Tasks & metrics** | SNLI, SST-2, MRPC, IMDB, TyDi QA; plus XNLI for XLM-R; classification evaluated with validation accuracy, QA with EM; backbones are BERT-base / XLM-R-base. | `train_mrbert.py --dataset` supports mrpc, imdb, sst2, snli, xnli, tydiqa; `--backbone` selects `bert` / `xlmr`, corresponding to `MrBertModel` / `MrXLMRobertaModel`; classification uses accuracy and TyDi uses EM, consistent with the report. | **Tasks and metrics are aligned**, so numerical comparisons are meaningful. |

---

## 3. Mismatches

### 3.1 Pre-deletion blending (in report, not in code)

| Item | Repo2 | Repo1 |
|------|--------|------|
| **Formula** | w_i = clamp(-g_i/30, 0, 1); h̃_i = (1−w_i)·h_i^(L) + w_i·h_i^(gate) | **Not implemented**: no `blend` in repo |
| **Role** | Blend deleted tokens with pre-gate representation to avoid QA representation collapse; TyDi EM 0.10 → 0.35 | TyDi in code uses only span remapping + gate warmup; no blending. TyDi in `RESULTS_ANALYSIS`/combined report ~0.28 EM (0.35 from report/other runs) |
| **Impact** | Report lists “Pre-deletion blending” as a contribution and critical for SQuAD/TyDi | To match report’s QA experiments, blending must be added to this repo |

### 3.2 Gate initialization

| Item | Repo2 | Repo1 |
|------|--------|------|
| **Linear layer** | W ~ N(0, 0.02), b = 10.0; initial g≈-0.001, almost no deletion | `DeleteGate`: Linear uses default PyTorch init; **bias is `nn.Parameter(torch.zeros(1))` (b=0)**, not 10 |
| **Impact** | Report uses b=10 for very conservative initial deletion; code may delete more initially, relying on warmup/PI to recover | To match report: init linear with N(0, 0.02) and set bias=10 in `DeleteGate` |

### 3.3 PI hyperparameters

| Item | Repo2 | Repo1 |
|------|--------|--------------|
| **k_p, k_i** | k_p = 0.01, k_i = 10^−5 | `train_mrbert.py`: `--controller_kp` default **0.5**, `--controller_ki` default 1e-5 |
| **Impact** | Report uses smaller k_p; code default k_p is larger, so PI reacts more aggressively | Can align with report via `--controller_kp 0.01` |

### 3.4 Training and data settings

| Item | Repo2 | Repo1 |
|------|--------|--------------------------------|
| **Max length** | SNLI 128; SQuAD/TyDi 384 stride 128; IMDB 512 | `--max_length` default **128**; no per-task 384/512 |
| **Batch** | 32 (classification), 16 (QA) | Default **16**; not task-specific |
| **Epoch** | 3 (5 for MRPC) | Default **3**; no special case for MRPC |
| **Gate lr** | Gate lr = 10^−4 (separate from backbone) | **Not implemented**: optimizer does not use a separate lr for gate parameters |
| **Regulariser delay** | 1000 steps (100 for TyDi) | `--gate_warmup_steps` exists, default 0; can set 1000/100 |
| **Hardware / env** | A100 via Modal, fp16 | This repo’s BERT: **L4**; XLM-R: **A100** (Modal). Latency: T4. |

### 3.5 Experiment and result sources

| Item | Repo2 | Repo1 |
|------|--------|----------------|
| **SNLI table** | A100: baseline 90.48%, MrBERT-30% 90.21%, 1.89× speedup, etc. | Our BERT: **L4** (1 ep, warmup); see Section 6 for side-by-side numbers. |
| **SQuAD** | Report includes SQuAD curves and “without blending does not converge” | This repo’s main results do not include SQuAD; combined report states SQuAD was not run |
| **TyDi + blending** | Layer 9 + blending → EM 0.35 | No blending in code; TyDi here from span remap + warmup, ~0.28 EM |
| **GitHub** | Report: Code: https://github.com/HivaMohammadzadeh1/CS224N-project/tree/main | Whether this is the same repo is unclear; if same, report may refer to a different branch or script set |

---

## 4. Concrete data comparison

### 4.1 BERT (MrBERT): Repo2 vs Repo1

Our BERT runs are on **L4** (1 epoch, batch 24, gate warmup 1000, target 0.3 or 0.5). Repo2 uses **A100** (3 epochs, batch 32, gate lr 10⁻⁴, weight decay 0.01, etc.). **Accuracy is not determined by GPU** (same model → same accuracy on same inputs); the differences below come from **training setup** (epochs, init, blending, data length, etc.).

| Dataset | Repo1 (L4): baseline | Repo1 (L4): MrBERT | Repo1 actual del% | Repo2 (A100): baseline | Repo2 (A100): MrBERT-30% | Repo2 actual del% |
|---------|--------------------------|------------------------|------------------------|------------------------|--------------------------|-------------------|
| **SNLI** | 74.00% (0.3 run) | **89.02%** | 76.1% | **90.48%** | **90.21%** | 30.8% |
| **SNLI** (0.5) | 60.32% | **88.78%** | 76.2% | — | — | — |
| **SST-2** | 67.89% (0.3) | **92.55%** | 61.8% | **97.29%** | **96.67%** | 47.4% |
| **SST-2** (0.5) | 80.39% | **91.17%** | 58.9% | — | — | — |
| **MRPC** | 68.38% (0.3) | 68.63% | 53.8% | **86.03%** | **68.71%** | 5.7% |
| **MRPC** (0.5) | 68.87% | 68.38% | 30.1% | — | — | — |
| **IMDB** | 87.85% (0.3) | 57.42% | 4.7% | **93.97%** | **93.03%** | 43.7% (avg seq) |
| **TyDi QA (EM)** | 20.18% (0.3) | **28.16%** | 10.9% | 0.40 (no blend) | **0.35** (blend L9) | ~24.7% |
| **XNLI** | 74.82% (0.3) | **80.56%** | 65.4% | — | — | — |

**Conclusion (6.1):** Baseline gaps (our L4 1 ep vs Hiva A100 3–5 ep) are **+16 to +29 pp** on SNLI/SST-2/MRPC—due to **epochs/schedule**, not GPU. **Delta from baseline:** we get **+15 to +25 pp** on SNLI/SST-2 (gate corrects weak baseline); Hiva gets **−0.27 to −0.62 pp** (gate preserves strong baseline). MRPC: we are flat; Hiva −17.32 pp despite only 5.7% deletion → task/setup sensitivity. TyDi: we **+7.98 pp EM** without blending (~0.28); Hiva 0.35 EM with blending → **blending + L9** explains the gap. *From Hiva’s report (SNLI ablations):* MrBERT-0% 90.50%, Random 30% 87.26%, No-PI 30% 86.47% (88.9% del)—PI is needed; random deletion underperforms.

### 4.2 XLM-R (MrXLM): Direct comparison (both A100)

**Both repo1 and repo2 use A100** (ours: `results/new/xlmr_from_A100/`). So we can compare **accuracy and deletion** directly; any difference is from **implementation and hyperparameters** (blending, gate init, gate lr, warmup, epochs, etc.), not from GPU.

| Dataset | Repo1 (A100): baseline | Repo1 (A100): MrXLM 0.3 | Note (codebase) | Repo2 (A100): baseline | Repo2 (A100): MrXLMR-30% | Note (Hiva) |
|---------|----------------------------|----------------------------|-----------------|------------------------|--------------------------|-------------|
| **MRPC** | 68.38% | **72.06%** | MrXLM improves | — | — | — |
| **SST-2** | 79.01% | 61.01% | Large drop with gate | — | — | — |
| **SNLI** | 33.82% | 33.82% | Both poor; no change | **89.85%** | **82.27%** | −7.58pp |
| **IMDB** | 79.87% | 50.00% | Large drop | — | — | — |
| **XNLI** | 62.77% | 43.41% | Large drop | — | — | — |

**Conclusion (6.2):** Same GPU (A100) but **baseline levels differ sharply** (e.g. SNLI 33.82% vs 89.85% → 56 pp gap): our XLM-R run is underconverged or different split/settings. **Gate effect:** we get **+3.68 pp** on MRPC; **−18 to −30 pp** on SST-2/IMDB/XNLI; Repo2 SNLI **−7.58 pp**. So **XLM-R + gate is task-dependent and more fragile than BERT**; only MRPC (our run) shows a clear gain.

### 4.3 Inference latency (GPU does matter here)

| Source | GPU | Baseline | MrBERT-30% | Speedup |
|--------|-----|----------|------------|---------|
| **Repo1** | **T4** (batch 16, seq 256) | ~95 ms/batch | ~43 ms/batch | **~30–55%** |
| **Repo2 (report)** | **A100** | 1.440 ms/sample | 0.763 ms/sample | **1.89×** |

**Conclusion (6.3):** **GPU determines latency**, not accuracy. A100 is much faster than T4; both speedup numbers are valid for their hardware. For the same model and batch/seq, relative speedup would be comparable; absolute ms/sample lower on A100.

### 4.4 Detailed analysis tables

#### 4.4.1 Baseline gap (codebase L4 vs Hiva A100)

| Dataset | Repo1 baseline (L4, 1 ep) | Repo2 baseline (A100) | Gap (pp) | Interpretation |
|---------|------------------------------|------------------------|----------|----------------|
| SNLI    | 74.00%                       | 90.48%                 | **+16.48** | 1 epoch vs 3 epochs (and possibly different data order/seed). Most of the SNLI gap is from **training length**, not GPU. |
| SST-2   | 67.89%                       | 97.29%                 | **+29.40** | Very large: our 1-ep baseline is far from converged; Hiva's is near ceiling. |
| MRPC    | 68.38%                       | 86.03%                 | **+17.65** | Hiva uses 5 epochs for MRPC; we use 1–3. Small dataset (3.7K) benefits from more epochs. |
| IMDB    | 87.85%                       | 93.97%                 | **+6.12**  | Both reasonable; we use 1 ep, Hiva 3. Long-doc task needs more steps. |

**Takeaway:** The **16–29 pp** baseline gaps on SNLI/SST-2/MRPC are overwhelmingly due to **epoch count and training schedule** (1 ep vs 3–5 ep), not L4 vs A100. Comparing **gated accuracy** across setups is only meaningful after acknowledging baseline strength; comparing **delta from baseline** (gated − baseline) is more comparable.

#### 4.4.2 Delta from baseline (gated − baseline)

| Dataset | Repo1 (L4): Δ (pp) | Repo2 (A100): Δ (pp) | Comment |
|---------|------------------------|----------------------|---------|
| SNLI    | **+15.02** (74→89.02)  | **−0.27** (90.48→90.21) | We gain a lot from the gate over a weak baseline; Hiva keeps a strong baseline almost unchanged. Same mechanism, different starting point. |
| SST-2   | **+24.66** (67.89→92.55) | **−0.62** (97.29→96.67) | Same pattern: we gain 24.66 pp; Hiva loses 0.62 pp. |
| MRPC    | +0.25 (68.38→68.63)   | **−17.32** (86.03→68.71) | We are flat; Hiva has a large drop. MRPC is sensitive: with a strong baseline and same nominal "30% target," her setup deletes only 5.7% but accuracy collapses—likely **task + small data + gate dynamics**. |
| IMDB    | **−30.43** (87.85→57.42) | −0.94 (93.97→93.03) | We **collapse** (gate barely activates, 4.7% del; run unstable). Hiva stays near baseline. Our IMDB is an outlier; her setup (longer seq, more ep) is more stable. |
| TyDi QA  | **+7.98** (20.18→28.16% EM) | −0.05 (0.40→0.35 with blend L9) | We gain ~8 pp EM without blending; Hiva stays near BERT with blending. Our 0.28 EM is below her 0.35—**blending** recovers the gap. |

**Takeaway:** On **SNLI and SST-2**, our **positive Δ** shows the gate helps when the baseline is underconverged; Hiva's **small negative Δ** shows the gate preserves a strong baseline. So the **gate is not hurting** in either case; the apparent "difference" is baseline strength. **MRPC** is the only task where Hiva's gated model drops a lot (−17.32 pp) despite low actual deletion (5.7%)—suggesting **dataset size and paraphrase sensitivity** interact badly with her gate/controller in that run. **IMDB** on our side is unstable (one run collapses); Hiva's longer sequences and epochs keep accuracy. **TyDi:** Our +7.98 pp is without blending; her 0.35 EM with blending shows **blending is the main lever** for closing the QA gap.

#### 4.4.3 Actual deletion rate vs accuracy

| Dataset | Repo1 actual del% | Repo1 gated acc | Repo2 actual del% | Repo2 gated acc | Observation |
|---------|----------------------|--------------------|------------------|----------------|-------------|
| SNLI    | 76.1%                | 89.02%             | 30.8%            | 90.21%         | We delete **much more** (76% vs 31%) but still reach 89%; Hiva deletes less and keeps 90%. So **higher deletion can still retain high accuracy** on SNLI (redundancy). |
| SST-2   | 61.8%                | 92.55%             | 47.4%            | 96.67%         | Same pattern: we delete more, accuracy slightly lower in absolute terms but both are high. |
| MRPC    | 53.8%                | 68.63%             | **5.7%**         | 68.71%         | Hiva's model **barely deletes** (5.7%) yet **drops 17 pp**; we delete 54% and are flat. Suggests on MRPC, **deletion rate alone does not explain the drop**—likely controller/init or data sensitivity. |
| TyDi QA | 10.9%                | 28.16% EM          | ~24.7%           | 0.35 EM        | We keep deletion low (11%) and get 0.28 EM; Hiva targets ~25% with **blending** and reaches 0.35. **Controlled deletion + blending** beats "low deletion, no blend." |

**Takeaway:** **SNLI / SST-2** tolerate **high actual deletion** (47–76%) with small accuracy cost—consistent with **task redundancy**. **MRPC (Hiva):** Low deletion (5.7%) but large accuracy drop → points to **optimization or regularization** (e.g. gate init, weight decay, small dataset) rather than deletion magnitude. **TyDi:** Blending + moderate deletion (25%) yields better EM than no blending with low deletion (11%) → **mechanism (blending) matters more than deletion rate alone** for QA.

#### 4.4.4 XLM-R (both A100): direct comparison with Δ

Same GPU (A100), so differences are **purely from setup** (epochs, data, seed, gate/controller settings).

| Dataset | Repo1 baseline | Repo1 MrXLM 0.3 | Repo1 Δ (pp) | Repo2 baseline | Repo2 MrXLMR-30% | Repo2 Δ (pp) |
|---------|--------------|---------------|------------|---------------|-----------------|-------------|
| MRPC    | 68.38%       | 72.06%        | **+3.68**   | —             | —               | —           |
| SST-2   | 79.01%       | 61.01%        | **−18.00**  | —             | —               | —           |
| SNLI    | 33.82%       | 33.82%        | 0.00        | 89.85%        | 82.27%          | **−7.58**   |
| IMDB    | 79.87%       | 50.00%        | **−29.87**  | —             | —               | —           |
| XNLI    | 62.77%       | 43.41%        | **−19.36**  | —             | —               | —           |

**Findings:** (1) Our XLM-R baseline on SNLI is **33.82%** vs Hiva's **89.85%**—a **56 pp** gap (underconverged or different splits). (2) With a healthy baseline (Hiva SNLI), adding the gate costs **−7.58 pp**—clear **XLM-R fragility**. (3) On our runs: gate helps on MRPC (+3.68 pp), hurts on SST-2/IMDB/XNLI (−18 to −30 pp). (4) **XLM-R + gate** is task-dependent and more fragile than BERT; only MRPC (our run) shows a clear gain.

#### 4.4.5 TyDi QA: no blending vs blending

| Config              | Baseline EM | Gated EM | Actual del% | Note                          |
|---------------------|-------------|----------|-------------|--------------------------------|
| Codebase (no blend) | 20.18%      | **28.16%** | 10.9%     | Span remap + warmup only.      |
| Hiva (no blend, L3) | 0.40        | **0.10**   | 60.7%     | Gate collapses; EM collapses. |
| Hiva (blend, L3)    | —           | **0.30**   | ~0%      | Blending stabilises; gate stops deleting. |
| Hiva (blend, L9)    | 0.40        | **0.35**   | ~24.7%   | Best: near-baseline EM with ~25% deletion. |

**Analysis:** Without blending, Hiva's gate at L3 **over-deletes** (60.7%) and EM drops to **0.10**; with blending at L3, the gate **stops deleting** (~0%) and EM recovers to **0.30**. Moving the gate to **L9 with blending** gives **0.35 EM** with **~25% deletion**—best tradeoff. Our codebase reaches **0.28 EM** with **10.9% deletion** and **no blending**; the **0.35 − 0.28 ≈ 0.07 EM** gap is attributable to **blending + deeper gate (L9)**, not GPU.

---

## 7. Repo 1 and Repo2 comparation

### 7.2 Method / setup differences (why numbers differ)

| Aspect | Codebase (Alina) | Hiva (report) |
|--------|-------------------|---------------|
| **Hardware (training)** | BERT: **L4**; XLM-R: **A100** (Modal) | **A100** via Modal |
| **Hardware (latency)** | **T4** (batch 16, seq 256) | **A100** (e.g. 1.44 ms baseline, 0.76 ms MrBERT-30%) |
| **Pre-deletion blending** | **No** | **Yes** (Eq.3); TyDi QA, SQuAD |
| **Gate init** | bias=0, default Linear | W~N(0,0.02), **b=10** |
| **Gate lr** | Same as backbone | **10⁻⁴** (separate) |
| **Weight decay** | Not set | **0.01** |
| **PI kp** | Default **0.5** | **0.01** |
| **Batch (classification)** | **16 or 24** | **32** |
| **Max length** | **128** (TyDi/XNLI 256 in some runs) | SNLI 128; SQuAD/TyDi **384**; IMDB **512** |
| **Regulariser delay** | `--gate_warmup_steps` (e.g. 1000); default 0 | 1000 steps (100 for TyDi) |
| **Epochs** | 1–3 | 3 (5 for MRPC) |
| **Hard-train** | No | Yes (e.g. MrBERT-30% hard-train) |

### 7.3 Summary: what differs and why

| Metric / aspect | Codebase (Alina) | Hiva (report) | Main reason for gap |
|-----------------|------------------|---------------|----------------------|
| **SNLI baseline** | 74% (1 ep, L4) | 90.48% (A100) | Epochs / training setup |
| **SNLI MrBERT** | 89.02% | 90.21% | Baseline + setup |
| **SST-2 baseline** | 67–80% | 97.29% | Epochs / setup |
| **MRPC baseline** | ~68% | 86% | Epochs / setup (e.g. 5 ep MRPC) |
| **TyDi QA EM (gated)** | ~0.28 (no blend) | 0.35 (blend, L9) | **Pre-deletion blending** + gate layer |
| **XLM-R SNLI (A100)** | 33.82% bl / 33.82% MrXLM | 89.85% bl / 82.27% MrXLM | Different baselines (setup); both show fragility |
| **Latency** | T4, 30–55% speedup | A100, 1.89× | **GPU** + same idea |

### 7.4 Short answers

1. **Compare codebase vs Hiva data and results**  
   - **BERT:** Our L4 runs (1 ep, warmup, no blending) give strong **relative** gains (e.g. SNLI 74%→89%, SST-2 67%→92.55%, TyDi 20%→28% EM). Hiva’s A100 runs use stronger baselines and report small drops (e.g. SNLI 90.48%→90.21%, SST-2 97.29%→96.67%). Absolute numbers differ mainly because of **baselines and setup**, not GPU.  
   - **TyDi QA:** The **0.35 vs ~0.28 EM** gap is from **pre-deletion blending** (and layer-9 gate) in Hiva’s setup.  
   - **XLM-R (both A100):** Direct comparison in Section 6.2; baseline levels differ a lot (e.g. SNLI 33% vs 90%) → different training; both show XLM-R fragility with the gate.  
   - **Latency:** We report T4 (30–55% speedup); Hiva reports A100 (1.89×). **GPU** explains the different absolute and relative speedups.

2. **Does different GPU only affect inference time, not accuracy?**  
   **Yes.** For the same model (same weights), same inputs, and same precision, **accuracy is the same** on L4, T4, or A100. Only **inference time** (and training wall time) change with the GPU.

---

## 8. Deep data analysis — synthesis

The **conclusions** are given **immediately after each table** in Section 6 (Conclusion 6.1, 6.2, 6.3), and **detailed tables** are in Section 6.4. Summary of what the data support:

1. **Baseline strength dominates absolute accuracy.** Differences in SNLI/SST-2/MRPC (e.g. 74% vs 90%) come from **epochs and schedule**, not L4 vs A100.
2. **Delta from baseline is more comparable.** Our gate adds **+15 to +25 pp** on SNLI/SST-2 over a weak baseline; Hiva's gate keeps a strong baseline almost unchanged (−0.27 to −0.62 pp). So the **same mechanism** behaves as "corrector" in our setting and "preserver" in Hiva's.
3. **Task sensitivity.** BERT: SNLI/SST-2/XNLI tolerate high deletion; MRPC/IMDB/TyDi are sensitive. XLM-R: only MRPC shows a clear gain with the gate (our A100); SST-2/SNLI/IMDB/XNLI drop.
4. **Blending is critical for QA.** TyDi 0.10 (no blend) → 0.30 (blend L3) → 0.35 (blend L9) in Hiva's data; we get 0.28 without blending. The **0.07 EM gap** is explained by **pre-deletion blending + gate layer**, not hardware.
5. **Actual deletion rate does not alone predict accuracy.** MRPC (Hiva): 5.7% deletion but −17 pp; SNLI (us): 76% deletion but 89% accuracy. So **task structure and optimization** matter as much as deletion magnitude.
6. **XLM-R on A100:** Same GPU, different baselines and outcomes—confirms that **training setup** (epochs, data, seed, gate/controller) drives the numbers; GPU does not change accuracy.
