GitHub:
- **repo1: Alina Huang (this repo)** `https://github.com/TinaLand/standford_cs224n_project`
- **repo2: repo2**: `https://github.com/repo2Mohammadzadeh1/CS224N-project`

---
## Summary

The **conclusions** are given **immediately after each table** in Section 6 (Conclusion 6.1, 6.2, 6.3), and **detailed tables** are in Section 6.4. Summary of what the data support:

1. **Baseline strength dominates absolute accuracy.** Differences in SNLI/SST-2/MRPC (e.g. 74% vs 90%) come from **epochs and schedule**.
2. **Delta from baseline is more comparable.** Our gate adds **+15 to +25 pp** on SNLI/SST-2 over a weak baseline; repo2's gate keeps a strong baseline almost unchanged (−0.27 to −0.62 pp). So the **same mechanism** behaves as "corrector" in our setting and "preserver" in repo2's.
3. **Task sensitivity.** BERT: SNLI/SST-2/XNLI tolerate high deletion; MRPC/IMDB/TyDi are sensitive. XLM-R: only MRPC shows a clear gain with the gate (our A100); SST-2/SNLI/IMDB/XNLI drop.
4. **Blending is critical for QA.** TyDi 0.10 (no blend) → 0.30 (blend L3) → 0.35 (blend L9) in repo2's data; we get 0.28 without blending. The **0.07 EM gap** is explained by **pre-deletion blending + gate layer**, not hardware.
5. **Actual deletion rate does not alone predict accuracy.** MRPC (repo2): 5.7% deletion but −17 pp; SNLI (us): 76% deletion but 89% accuracy. So **task structure and optimization** matter as much as deletion magnitude.
6. **XLM-R on A100:** Same GPU, different baselines and outcomes—confirms that **training setup** (epochs, data, seed, gate/controller) drives the numbers; GPU does not change accuracy.

## 1. Key gate design decisions
| Decision | Choice | Where |
|----------|--------|--------|
| **Gate position** | After a fixed encoder layer (default layer 3). | `MrBertConfig.gate_layer_index`; `_encoder_forward_with_gate` runs layers 0..L then gate then rest. |
| **Softmax1** | Use softmax1 in gated layers so attention can sum to &lt;1. | `use_softmax1=True` in config; `_run_layer_with_soft_gate` calls `softmax1(attention_scores)`. |
| **PI controller** | Optional; targets a deletion ratio instead of fixed α. | `train_mrbert.py` with `--use_pi`; `PIController.step()` each batch. |
| **Gate warmup** | Delay gate regularizer for N steps to stabilize. | `--gate_warmup_steps` (e.g. 1000); α set to 0 until step N. |
| **QA** | Span prediction on shortened sequence; map back to original indices. | `keep_indices` / `kept_lengths` in `MrBertForQuestionAnswering` / `MrXLMRobertaForQuestionAnswering` for span coordinate remapping. |

## 2. What Matches (repo2 ↔ repo1)

At a high level, our implementation matches repo2's report in four areas: **model structure, control rules, QA handling, and tasks/metrics**.

| Category | Repo2 | Repo1 | Note |
|----------|---------------------|---------------------|------|
| **Model + gate** | Delete gate after encoder layer 3; LayerNorm → Linear(768→1) → Sigmoid · k (k = −30); soft deletion = add G as attention bias; hard deletion = remove tokens with G < k/2; keep [CLS]/[SEP], delete [PAD]. | `DeleteGate` in `mrbert/modeling_mrbert.py` with `gate_layer_index=3`, `gate_k=-30`; `_run_layer_with_soft_gate` adds gate to attention scores; hard path in `_encoder_forward_with_gate` thresholds at `gate_k * gate_threshold_ratio` (k/2), keeps CLS, gathers kept tokens, rebuilds the attention mask. | **Core mechanism is fully aligned**; differences come later from init / PI hyperparameters / QA blending. |
| **PI controller + loss** | L = L_task + α·L_deletion, L_deletion = mean(G); α_t updated by a PI controller: error e_t = δ − r_t, P/I updates, α_t = max(0, p_t + i_t); γ=0.9 for EMA. | `get_gate_regularizer_loss(gate)=gate.mean()`; `PIController.step()` implements the same rule: compute error from current deletion ratio, update P/I, set α = max(0, kp*P + ki*I) with default `pi_gamma=0.9`. | **Control logic matches**; the only difference is the default `kp` (report: 0.01, this repo: 0.5). |
| **QA span remap + ragged batch** | After hard deletion, predict spans on the **shortened sequence**, then map back to original token indices via keep indices / lengths; batch items have different lengths, handled via ragged / gather-style ops. | `MrBertForQuestionAnswering` / `MrXLMRobertaForQuestionAnswering` output `keep_indices` and `kept_lengths` for span remap; after hard deletion, `max_kept + torch.gather + length mask` build a variable-length batch, matching the "QA span coordinate remapping / ragged tensors" description. | **QA pipeline design is aligned**; the main difference is that this repo currently **does not implement pre-deletion blending**. |
| **Tasks & metrics** | SNLI, SST-2, MRPC, IMDB, TyDi QA; plus XNLI for XLM-R; classification evaluated with validation accuracy, QA with EM; backbones are BERT-base / XLM-R-base. | `train_mrbert.py --dataset` supports mrpc, imdb, sst2, snli, xnli, tydiqa; `--backbone` selects `bert` / `xlmr`, corresponding to `MrBertModel` / `MrXLMRobertaModel`; classification uses accuracy and TyDi uses EM, consistent with the report. | **Tasks and metrics are aligned**, so numerical comparisons are meaningful. |

---

## 3. Mismatches

Single overview: Repo2 (repo2/report) vs Repo1 (this codebase).

| Category | Feature / Parameter | Repo 2 (Report / Ideal) | Repo 1 (Current Code) | Impact of the Choice | Actual Data / Result Impact |
|----------|---------------------|--------------------------|------------------------|----------------------|-----------------------------|
| Algorithm | Pre-deletion Blending | Enabled (L9/L3) | Disabled (no blending; soft delete only, then hard delete) | Without blending, deleted tokens' information is lost, causing QA boundaries to collapse. | TyDi QA EM: **0.35** (Repo 2, with blending) vs **0.28** (Repo 1, no blending) → blending adds ~**7 pp** EM. |
| Init | Gate Bias (b) | b = 10.0 (conservative, almost no early deletion) | b = 0.0 (default / random) | Low bias allows more random deletions early on, destabilizing pre-trained weights. | IMDB stability issue: one Repo 1 run collapsed to **57.42%** accuracy; repo2’s IMDB stays near **94%**. |
| Control | PI Gain (k_p) | k_p = 0.01 (stable, slow controller) | k_p = 0.5 (aggressive default) | High k_p makes the controller react too sharply, causing deletion-rate oscillations. | Repo 1’s deletion rate fluctuates more before reaching steady state compared to Repo 2’s smoother traces. |
| Opt | Gate Learning Rate | Separate gate lr = 10^−4 | Shared with backbone | A shared high LR can make the small gate parameter “over-shoot” and fail to converge cleanly. | Observed task-sensitive deletion rates (e.g., MRPC / IMDB) in Repo 1, consistent with more fragile gate optimization. |
| Data | Max Sequence Length | Task-specific (128–512, e.g., IMDB 512) | Mostly fixed 128 (some runs 256) | Fixed short length loses context on long-document tasks like IMDB. | IMDB baseline gap: Repo 2 ≈ **94%** vs Repo 1 ≈ **88%**. |
| Schedule | Regularizer Warmup | 1000 steps (100 for TyDi) | 0 steps by default (immediate regularizer) | Immediate deletion pressure prevents the gate from first learning token importance properly. | Contributes to under-convergence in Repo 1 when combined with 1-epoch training. |
| Hardware | Compute Platform | A100 (high-end) | L4 / T4 (mid-range) | GPU affects **throughput and latency**, not accuracy, for the same model and data. | Latency: Repo 2 reports **1.89×** speedup (A100); Repo 1 sees ~**30–55%** speedup on T4 for MrBERT-30%. |

---

## 4. Concrete data comparison

### 4.1 BERT (MrBERT): Repo2 vs Repo1

Our BERT runs are on **L4** (1 epoch, batch 24, gate warmup 1000, target 0.3 or 0.5). Repo2 uses **A100** (3 epochs, batch 32, gate lr 10⁻⁴, weight decay 0.01, etc.). 

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

**Conclusion (6.1):**

- **Baseline gap (absolute accuracy).** Baseline gaps (our L4 1 ep vs repo2 A100 3–5 ep) are **+16 to +29 pp** on SNLI/SST-2/MRPC—driven by **epochs / training schedule**, not GPU type.
- **Delta from baseline (relative gain).** We get **+15 to +25 pp** on SNLI/SST-2 (gate corrects a weak baseline); repo2 gets **−0.27 to −0.62 pp** (gate preserves a strong baseline). **Same mechanism**, but in our setting it behaves like a *corrector*, in repo2 like a *preserver*.
- **MRPC sensitivity.** On MRPC we are essentially flat; repo2 drops **−17.32 pp** despite only **5.7%** deletion → indicates **task / small-data / controller dynamics** sensitivity rather than pure deletion rate.
- **TyDi QA and blending.** We reach **~0.28 EM** (+7.98 pp) on TyDi **without blending**; repo2 reaches **0.35 EM** with **pre-deletion blending + L9 gate**, and the **0.07 EM gap** is explained by that mechanism choice, not hardware.
- **SNLI ablations and PI.** *From repo2’s report (SNLI ablations):* MrBERT-0% 90.50%, Random 30% 87.26%, No-PI 30% 86.47% (88.9% del)—showing that **random deletion underperforms** and **removing PI degrades performance**, so the **PI controller is necessary** for stable deletion targeting.

### 4.2 XLM-R (MrXLM): Direct comparison (both A100)

**Both repo1 and repo2 use A100** (ours: `results/new/xlmr_from_A100/`). So we can compare **accuracy and deletion** directly; any difference is from **implementation and hyperparameters** (blending, gate init, gate lr, warmup, epochs, etc.), not from GPU.

| Dataset | Repo1 (A100): baseline | Repo1 (A100): MrXLM 0.3 | Note (codebase) | Repo2 (A100): baseline | Repo2 (A100): MrXLMR-30% | Note (repo2) |
|---------|----------------------------|----------------------------|-----------------|------------------------|--------------------------|-------------|
| **MRPC** | 68.38% | **72.06%** | MrXLM improves | — | — | — |
| **SST-2** | 79.01% | 61.01% | Large drop with gate | — | — | — |
| **SNLI** | 33.82% | 33.82% | Both poor; no change | **89.85%** | **82.27%** | −7.58pp |
| **IMDB** | 79.87% | 50.00% | Large drop | — | — | — |
| **XNLI** | 62.77% | 43.41% | Large drop | — | — | — |

**Conclusion (6.2):**

- **Same GPU, very different baselines.** Both sides use **A100**, but baseline accuracies differ a lot (e.g. SNLI **33.82% vs 89.85%**, a **56 pp** gap) → our XLM-R baseline is underconverged or uses different splits/settings compared to repo2.
- **Gate effect by task.** With that caveat, the gate gives us **+3.68 pp** on MRPC, but **−18 to −30 pp** on SST-2/IMDB/XNLI; on repo2 SNLI it costs **−7.58 pp**.
- **Fragility vs BERT.** These patterns show that **XLM-R + gate is task-dependent and more fragile than BERT**: only MRPC (our run) shows a clear gain, while several other tasks degrade, especially when the baseline is already strong.

### 4.3 Inference latency (GPU does matter here)

| Source | GPU | Baseline | MrBERT-30% | Speedup |
|--------|-----|----------|------------|---------|
| **Repo1** | **T4** (batch 16, seq 256) | ~95 ms/batch | ~43 ms/batch | **~30–55%** |
| **Repo2 (report)** | **A100** | 1.440 ms/sample | 0.763 ms/sample | **1.89×** |

**Conclusion (6.3):**

- **Accuracy is GPU-invariant.** For the same model, data, and precision, changing from T4 to A100 does **not** change accuracy—only wall-clock time.
- **Latency is GPU-dependent.** A100 is much faster than T4; both reported speedups (our ~30–55% on T4, repo2’s 1.89× on A100) are valid **for their hardware**.
- **Relative vs absolute speedup.** For the same model and batch/sequence length, the **relative speedup** from token deletion should be similar across GPUs, while **absolute ms/sample** will always be lower on faster hardware like A100.

### 4.4 Detailed analysis tables

#### 4.4.1 Baseline gap (codebase L4 vs repo2 A100)

| Dataset | Repo1 baseline (L4, 1 ep) | Repo2 baseline (A100) | Gap (pp) | Interpretation |
|---------|------------------------------|------------------------|----------|----------------|
| SNLI    | 74.00%                       | 90.48%                 | **+16.48** | 1 epoch vs 3 epochs (and possibly different data order/seed). Most of the SNLI gap is from **training length**, not GPU. |
| SST-2   | 67.89%                       | 97.29%                 | **+29.40** | Very large: our 1-ep baseline is far from converged; repo2's is near ceiling. |
| MRPC    | 68.38%                       | 86.03%                 | **+17.65** | repo2 uses 5 epochs for MRPC; we use 1–3. Small dataset (3.7K) benefits from more epochs. |
| IMDB    | 87.85%                       | 93.97%                 | **+6.12**  | Both reasonable; we use 1 ep, repo2 3. Long-doc task needs more steps. |

**Takeaway:**

- **Baseline gaps come from training, not GPU.** The **16–29 pp** baseline gaps on SNLI/SST-2/MRPC are overwhelmingly due to **epoch count and training schedule** (1 ep vs 3–5 ep), not L4 vs A100.
- **Compare deltas, not raw numbers.** Comparing **gated accuracy** across setups is only meaningful after acknowledging baseline strength; comparing **delta from baseline** (gated − baseline) is a more reliable way to compare methods.

#### 4.4.2 Delta from baseline (gated − baseline)

| Dataset | Repo1 (L4): Δ (pp) | Repo2 (A100): Δ (pp) | Comment |
|---------|------------------------|----------------------|---------|
| SNLI    | **+15.02** (74→89.02)  | **−0.27** (90.48→90.21) | We gain a lot from the gate over a weak baseline; repo2 keeps a strong baseline almost unchanged. Same mechanism, different starting point. |
| SST-2   | **+24.66** (67.89→92.55) | **−0.62** (97.29→96.67) | Same pattern: we gain 24.66 pp; repo2 loses 0.62 pp. |
| MRPC    | +0.25 (68.38→68.63)   | **−17.32** (86.03→68.71) | We are flat; repo2 has a large drop. MRPC is sensitive: with a strong baseline and same nominal "30% target," her setup deletes only 5.7% but accuracy collapses—likely **task + small data + gate dynamics**. |
| IMDB    | **−30.43** (87.85→57.42) | −0.94 (93.97→93.03) | We **collapse** (gate barely activates, 4.7% del; run unstable). repo2 stays near baseline. Our IMDB is an outlier; her setup (longer seq, more ep) is more stable. |
| TyDi QA  | **+7.98** (20.18→28.16% EM) | −0.05 (0.40→0.35 with blend L9) | We gain ~8 pp EM without blending; repo2 stays near BERT with blending. Our 0.28 EM is below her 0.35—**blending** recovers the gap. |

**Takeaway:**

- **SNLI / SST-2: gate helps weak baselines, preserves strong ones.** On SNLI and SST-2, our **positive Δ** shows the gate helps when the baseline is underconverged; repo2's **small negative Δ** shows the gate preserves a strong baseline. The **gate is not hurting** in either case—the apparent difference is **baseline strength**, not the mechanism.
- **MRPC: small-data, paraphrase-sensitive, and fragile.** MRPC is the only task where repo2's gated model drops a lot (−17.32 pp) despite low actual deletion (5.7%), suggesting **dataset size and paraphrase sensitivity** interact badly with the gate/controller in that run.
- **IMDB: instability vs stability.** IMDB on our side is unstable (one run collapses), while repo2's longer sequences and more epochs keep accuracy high—pointing again to **training schedule and input length**, not GPU.
- **TyDi: blending is the key lever.** Our +7.98 pp gain is without blending; repo2's 0.35 EM with blending shows that **pre-deletion blending (plus L9 gate)** is the main lever for closing the QA gap, beyond just deletion rate.

#### 4.4.3 Actual deletion rate vs accuracy

| Dataset | Repo1 actual del% | Repo1 gated acc | Repo2 actual del% | Repo2 gated acc | Observation |
|---------|----------------------|--------------------|------------------|----------------|-------------|
| SNLI    | 76.1%                | 89.02%             | 30.8%            | 90.21%         | We delete **much more** (76% vs 31%) but still reach 89%; repo2 deletes less and keeps 90%. So **higher deletion can still retain high accuracy** on SNLI (redundancy). |
| SST-2   | 61.8%                | 92.55%             | 47.4%            | 96.67%         | Same pattern: we delete more, accuracy slightly lower in absolute terms but both are high. |
| MRPC    | 53.8%                | 68.63%             | **5.7%**         | 68.71%         | repo2's model **barely deletes** (5.7%) yet **drops 17 pp**; we delete 54% and are flat. Suggests on MRPC, **deletion rate alone does not explain the drop**—likely controller/init or data sensitivity. |
| TyDi QA | 10.9%                | 28.16% EM          | ~24.7%           | 0.35 EM        | We keep deletion low (11%) and get 0.28 EM; repo2 targets ~25% with **blending** and reaches 0.35. **Controlled deletion + blending** beats "low deletion, no blend." |

**Takeaway:**

- **SNLI / SST-2 are highly redundant.** These tasks tolerate **high actual deletion** (47–76%) with only a small accuracy cost, consistent with substantial **token-level redundancy**.
- **MRPC (repo2) shows optimization issues, not just deletion.** MRPC on repo2 has **low deletion (5.7%)** but a **large accuracy drop**, pointing toward **optimization / regularization** effects (e.g., gate init, weight decay, small dataset) rather than deletion magnitude alone.
- **TyDi: mechanism > deletion rate.** On TyDi, **blending + moderate deletion (~25%)** yields better EM than **no blending with low deletion (~11%)**, showing that the **mechanism (pre-deletion blending) matters more than raw deletion rate** for QA performance.

#### 4.4.4 XLM-R (both A100): direct comparison with Δ

Same GPU (A100), so differences are **purely from setup** (epochs, data, seed, gate/controller settings).

| Dataset | Repo1 baseline | Repo1 MrXLM 0.3 | Repo1 Δ (pp) | Repo2 baseline | Repo2 MrXLMR-30% | Repo2 Δ (pp) |
|---------|--------------|---------------|------------|---------------|-----------------|-------------|
| MRPC    | 68.38%       | 72.06%        | **+3.68**   | —             | —               | —           |
| SST-2   | 79.01%       | 61.01%        | **−18.00**  | —             | —               | —           |
| SNLI    | 33.82%       | 33.82%        | 0.00        | 89.85%        | 82.27%          | **−7.58**   |
| IMDB    | 79.87%       | 50.00%        | **−29.87**  | —             | —               | —           |
| XNLI    | 62.77%       | 43.41%        | **−19.36**  | —             | —               | —           |

**Findings:**

- **Huge baseline gap on SNLI.** Our XLM-R baseline on SNLI is **33.82%** vs repo2's **89.85%**—a **56 pp** gap, indicating our run is underconverged or uses different splits/settings.
- **Gate hurts when baseline is strong.** With a healthy baseline (repo2 SNLI), adding the gate costs **−7.58 pp**, a clear sign of **XLM-R fragility** under strong baselines.
- **Mixed results across tasks.** In our runs, the gate **helps on MRPC** (+3.68 pp) but **hurts on SST-2/IMDB/XNLI** (−18 to −30 pp).
- **Overall: XLM-R + gate is fragile.** Compared to BERT, **XLM-R + gate** is more task-dependent and fragile; only MRPC (our run) shows a robust gain.

#### 4.4.5 TyDi QA: no blending vs blending

| Config              | Baseline EM | Gated EM | Actual del% | Note                          |
|---------------------|-------------|----------|-------------|--------------------------------|
| Codebase (no blend) | 20.18%      | **28.16%** | 10.9%     | Span remap + warmup only.      |
| repo2 (no blend, L3) | 0.40        | **0.10**   | 60.7%     | Gate collapses; EM collapses. |
| repo2 (blend, L3)    | —           | **0.30**   | ~0%      | Blending stabilises; gate stops deleting. |
| repo2 (blend, L9)    | 0.40        | **0.35**   | ~24.7%   | Best: near-baseline EM with ~25% deletion. |

**Analysis:**

- **No blending at L3 collapses TyDi.** Without blending, repo2's gate at L3 **over-deletes** (60.7%) and EM drops to **0.10**—the QA representation collapses.
- **Blending at L3 stabilises but stops deletion.** With blending at L3, the gate **stops deleting** (~0%) and EM recovers to **0.30**, but we lose any compression benefit.
- **Blending at L9 gives the best tradeoff.** Moving the gate to **L9 with blending** yields **0.35 EM** with **~25% deletion**, which is the best accuracy–compression tradeoff in repo2.
- **Our codebase is close but slightly worse.** Our codebase reaches **0.28 EM** with **10.9% deletion** and **no blending**; the **0.35 − 0.28 ≈ 0.07 EM** gap is attributable to **blending + deeper gate (L9)**, not hardware (GPU).

---

### 4.5 Summary: what differs and why

| Metric / aspect | Codebase (Alina) | repo2 (report) | Main reason for gap |
|-----------------|------------------|---------------|----------------------|
| **SNLI baseline** | 74% (1 ep, L4) | 90.48% (A100) | Epochs / training setup |
| **SNLI MrBERT** | 89.02% | 90.21% | Baseline + setup |
| **SST-2 baseline** | 67–80% | 97.29% | Epochs / setup |
| **MRPC baseline** | ~68% | 86% | Epochs / setup (e.g. 5 ep MRPC) |
| **TyDi QA EM (gated)** | ~0.28 (no blend) | 0.35 (blend, L9) | **Pre-deletion blending** + gate layer |
| **XLM-R SNLI (A100)** | 33.82% bl / 33.82% MrXLM | 89.85% bl / 82.27% MrXLM | Different baselines (setup); both show fragility |
| **Latency** | T4, 30–55% speedup | A100, 1.89× | **GPU** + same idea |

