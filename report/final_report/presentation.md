GitHub:
- **repo1: Alina Huang (this repo)** `https://github.com/TinaLand/standford_cs224n_project`
- **repo2 (comparison codebase)**: `https://github.com/HivaMohammadzadeh1/CS224N-project`

---
## Summary

The **conclusions** are given **immediately after each table** in Section 6 (Conclusion 6.1, 6.2, 6.3), and **detailed tables** are in Section 6.4. Summary of what the data support:

1. **Baseline strength dominates absolute accuracy.** Differences in SNLI/SST-2/MRPC (e.g. 74% vs 90%) come from **epochs and schedule**.
2. **Delta from baseline is more comparable.** Our gate adds **+15 to +25 pp** on SNLI/SST-2 over a weak baseline; repo2's gate keeps a strong baseline almost unchanged (−0.27 to −0.62 pp). So the **same mechanism** behaves as "corrector" in our setting and "preserver" in repo2's.
3. **Task sensitivity.** BERT: SNLI/SST-2/XNLI tolerate high deletion; MRPC/IMDB/TyDi are sensitive. XLM-R: only MRPC shows a clear gain with the gate (our A100); SST-2/SNLI/IMDB/XNLI drop.
4. **Blending is critical for QA.** In repo2's ablations: TyDi 0.10 (no blend, L3) → 0.30 (blend L3) → 0.35 (blend L9). We implement **fixed** pre-deletion blending and a **learnable** variant (learnable denominator in the blend weight). **Evidence from `modal_logs` (Modal A100, TyDi validation EM, 3984 examples, gate L3, 3 ep):** (i) **`[Training args]` present:** no pre-deletion blend → **0.3471 EM** (~25.3% train del); **learnable blend** (`use_pre_deletion_blend=true`, `use_learnable_pre_deletion_blend=true`, init scale 30) → **0.3562 EM** (~28.6% del)—about **+0.9 pp** vs that no-blend run, with **higher deletion** (not a pure single-switch ablation). (ii) **Legacy runs (no args block):** another pair showed **0.3778 vs 0.3471** (~**+3.1 pp**) but **W&B names and flags disagree** on one job—treat as **supporting** fixed blending, not a nailed-down replicate until re-run with `[Training args]`. **L4 / 1 ep** without blending: **~0.28 EM** at ~11% deletion—**schedule, deletion level, and mechanism** all shift EM; GPU alone does not explain gaps vs repo2's **~0.35**. **Do not** use **20.18%** (weak L4 no-gate baseline) as the anchor for **pre-deletion blending** claims; blending must be argued from **matched Modal** pairs (**0.3471 vs 0.3562**, etc.—see §4.1 TyDi note below).
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
| **QA span remap + ragged batch** | After hard deletion, predict spans on the **shortened sequence**, then map back to original token indices via keep indices / lengths; batch items have different lengths, handled via ragged / gather-style ops. | `MrBertForQuestionAnswering` / `MrXLMRobertaForQuestionAnswering` output `keep_indices` and `kept_lengths` for span remap; after hard deletion, `max_kept + torch.gather + length mask` build a variable-length batch, matching the "QA span coordinate remapping / ragged tensors" description. | **QA pipeline design is aligned.** Repo1 implements **pre-deletion blending** (`--use_pre_deletion_blend`) and an optional **learnable blend scale** (`--use_learnable_pre_deletion_blend`, `--pre_deletion_blend_init_scale`), on top of repo2-style rehydration. |
| **Tasks & metrics** | SNLI, SST-2, MRPC, IMDB, TyDi QA; plus XNLI for XLM-R; classification evaluated with validation accuracy, QA with EM; backbones are BERT-base / XLM-R-base. | `train_mrbert.py --dataset` supports mrpc, imdb, sst2, snli, xnli, tydiqa; `--backbone` selects `bert` / `xlmr`, corresponding to `MrBertModel` / `MrXLMRobertaModel`; classification uses accuracy and TyDi uses EM, consistent with the report. | **Tasks and metrics are aligned**, so numerical comparisons are meaningful. |

---

## 3. Ablation Experiment

Single overview: Repo2 (repo2/report) vs Repo1 (this codebase).

| Category | Feature / Parameter | Repo 2 (Report / Ideal) | Repo 1 (Current Code) | Impact of the Choice | Actual Data / Result Impact |
|----------|---------------------|--------------------------|------------------------|----------------------|-----------------------------|
| Algorithm | Pre-deletion Blending | Enabled (L9/L3) | Repo1: fixed blend + optional **learnable** blend scale; A/B via `--no-use_pre_deletion_blend` / `--use_learnable_pre_deletion_blend`. | Without blending, hard deletion can hurt span features; learnable scale can interact with PI and deletion rate. | **Repo2:** TyDi **~0.35** EM (blend + L9). **Repo1 L4, 1 ep:** **~0.28** EM, ~11% del. **Repo1 Modal (`modal_logs`):** no blend **0.3471**; learnable blend **0.3562** (+**~0.9 pp**, del **~28.6%** vs **~25.3%**). Legacy **0.3778 vs 0.3471** suggests **~+3.1 pp** fixed blend but **W&B/flag mismatch**—re-run with `[Training args]` for a clean cite. |
| Init | Gate Bias (b) | b = 10.0 (conservative, almost no early deletion) | b = 0.0 (default / random) | Low bias allows more random deletions early on, destabilizing pre-trained weights. | IMDB stability issue: one Repo 1 run collapsed to **57.42%** accuracy; repo2’s IMDB stays near **94%**. |
| Control | PI Gain (k_p) | k_p = 0.01 (stable, slow controller) | k_p = 0.5 (aggressive default) | High k_p makes the controller react too sharply, causing deletion-rate oscillations. | Repo 1’s deletion rate fluctuates more before reaching steady state compared to Repo 2’s smoother traces. |
| Opt | Gate Learning Rate | Separate gate lr = 10^−4 | Shared with backbone | A shared high LR can make the small gate parameter “over-shoot” and fail to converge cleanly. | Observed task-sensitive deletion rates (e.g., MRPC / IMDB) in Repo 1, consistent with more fragile gate optimization. |
| Data | Max Sequence Length | Task-specific (128–512, e.g., IMDB 512) | Mostly fixed 128 (some runs 256) | Fixed short length loses context on long-document tasks like IMDB. | IMDB baseline gap: Repo 2 ≈ **94%** vs Repo 1 ≈ **88%**. |
| Schedule | Regularizer Warmup | 1000 steps (100 for TyDi) | 0 steps by default by command support for warm up parameter(immediate regularizer) | Immediate deletion pressure prevents the gate from first learning token importance properly. | Contributes to under-convergence in Repo 1 when combined with 1-epoch training. |
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
| **TyDi QA (EM)** | 20.18% (0.3; weak L4 baseline†) | **28.16%** (L4, 1 ep); Modal A100 **0.3471** (no pre-deletion blend); **0.3562** (learnable blend); **0.3499** / **0.3778** (legacy, no `[Training args]`); **0.4000 peak / 0.3861 final** (post-fix L9/384 rerun) | 10.9% (L4); Modal **~25.3%** / **~28.6%** / ~24.3% / ~25.9% / **~48.0%** | 0.40 (no blend) | **0.35** (blend L9) | ~24.7% |
| **XNLI** | 74.82% (0.3) | **80.56%** | 65.4% | — | — | — |

**TyDi QA — two baselines (do not conflate):** **†20.18% EM** is from **`results/new/bert_from_l4/`** with **`gate_weight=0`**, **1 epoch**, L4-style training—appropriate only for **“gated vs no-gate under our weak L4 setup”** (e.g. vs **28.16%**). It is **not** the same protocol as Modal TyDi (3 ep, L3/L9, batch 16, etc.) and **must not** be presented as the baseline for **pre-deletion blending** ablations. For blending, cite **matched Modal** runs: **no pre-deletion blend vs blend** on the same hyperparameters (authoritative pair with `[Training args]`: **0.3471 vs 0.3562**; legacy **0.3778 vs 0.3471** with the flag/name caveats elsewhere in this report).

**Conclusion (4.1):**

- **Baseline gap (absolute accuracy).** Baseline gaps (our L4 1 ep vs repo2 A100 3–5 ep) are **+16 to +29 pp** on SNLI/SST-2/MRPC—driven by **epochs / training schedule**, not GPU type.
- **Delta from baseline (relative gain).** We get **+15 to +25 pp** on SNLI/SST-2 (gate corrects a weak baseline); repo2 gets **−0.27 to −0.62 pp** (gate preserves a strong baseline). **Same mechanism**, but in our setting it behaves like a *corrector*, in repo2 like a *preserver*.
- **MRPC sensitivity.** On MRPC we are essentially flat; repo2 drops **−17.32 pp** despite only **5.7%** deletion → indicates **task / small-data / controller dynamics** sensitivity rather than pure deletion rate.
- **TyDi QA and blending.** **L4 (1 ep):** **~0.28 EM** gated vs weak baseline. **Modal A100 (`modal_logs`):** the run with **`[Training args]`** shows **no pre-deletion blend → 0.3471 EM** and **learnable blend → 0.3562 EM** (+**~0.9 pp**, but train deletion **~28.6% vs ~25.3%**). **Legacy** Modal rows (**0.3778** vs **0.3471**) support **~+3.1 pp** for turning blending on, yet **W&B names/flags disagree** on one job—quote legacy as **hypothesis-consistent**, not ground truth. **Post-fix L9/384 rerun** reaches **0.4000 peak EM (epoch 5)** and **0.3861 final** at ~48% logged deletion, showing the earlier 0.089 collapse was implementation-related rather than an L9 dead end.
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
| **Repo1** | **L4** (batch 16, seq 256) | ~95 ms/batch | ~43 ms/batch | **~30–55%** |
| **Repo2 (report)** | **A100** | 1.440 ms/sample | 0.763 ms/sample | **1.89×** |

**Conclusion (6.3):**

- **Accuracy is GPU-invariant.** For the same model, data, and precision, changing from T4 to A100 does **not** change accuracy—only wall-clock time.
- **Latency is GPU-dependent.** A100 is much faster than T4; both reported speedups (our ~30–55% on T4, repo2’s 1.89× on A100) are valid **for their hardware**.
- **Relative vs absolute speedup.** For the same model and batch/sequence length, the **relative speedup** from token deletion should be similar across GPUs, while **absolute ms/sample** will always be lower on faster hardware like A100.

### 4.4 Detailed analysis tables

#### 4.4.1 Delta from baseline (gated − baseline)

| Dataset | Repo1 (L4): Δ (pp) | Repo2 (A100): Δ (pp) | Comment |
|---------|------------------------|----------------------|---------|
| SNLI    | **+15.02** (74→89.02)  | **−0.27** (90.48→90.21) | We gain a lot from the gate over a weak baseline; repo2 keeps a strong baseline almost unchanged. Same mechanism, different starting point. |
| SST-2   | **+24.66** (67.89→92.55) | **−0.62** (97.29→96.67) | Same pattern: we gain 24.66 pp; repo2 loses 0.62 pp. |
| MRPC    | +0.25 (68.38→68.63)   | **−17.32** (86.03→68.71) | We are flat; repo2 has a large drop. MRPC is sensitive: with a strong baseline and same nominal "30% target," her setup deletes only 5.7% but accuracy collapses—likely **task + small data + gate dynamics**. |
| IMDB    | **−30.43** (87.85→57.42) | −0.94 (93.97→93.03) | We **collapse** (gate barely activates, 4.7% del; run unstable). repo2 stays near baseline. Our IMDB is an outlier; repo2's longer max sequence length and more epochs make their run much more stable. |
| TyDi QA  | **+7.98** (20.18→28.16% EM, L4/no blend; **weak L4 baseline only**) | −0.05 (0.40→0.35 with blend L9) | **+7.98 pp** is **not** comparable to blending gains: it mixes **L4 1 ep / no gate** vs **L4 gated**. **Blending:** use Modal **0.3562 vs 0.3471** (+**~0.9 pp**, deletion not matched). Legacy **+3.1 pp** (**0.3778 vs 0.3471**)—**needs `[Training args]` replicate**. **Post-fix L9/384** **0.4000 peak / 0.3861 final**. |

**Takeaway:**

- **SNLI / SST-2: gate helps weak baselines, preserves strong ones.** On SNLI and SST-2, our **positive Δ** shows the gate helps when the baseline is underconverged; repo2's **small negative Δ** shows the gate preserves a strong baseline. The **gate is not hurting** in either case—the apparent difference is **baseline strength**, not the mechanism.
- **MRPC: small-data, paraphrase-sensitive, and fragile.** MRPC is the only task where repo2's gated model drops a lot (−17.32 pp) despite low actual deletion (5.7%), suggesting **dataset size and paraphrase sensitivity** interact badly with the gate/controller in that run.
- **IMDB: instability vs stability.** IMDB on our side is unstable (one run collapses), while repo2's longer sequences and more epochs keep accuracy high—pointing again to **training schedule and input length**, not GPU.
- **TyDi: blending matters; learnable is mixed.** L4: +7.98 pp gated vs **weak L4 no-gate** (20.18%)—**separate axis** from blending. Modal blending story: **learnable blend vs logged no-blend** **~0.9 pp** (**0.3562 vs 0.3471**) at **higher deletion**; legacy **~+3.1 pp** (**0.3778 vs 0.3471**)—**re-run with `[Training args]`**. repo2's **0.35** (L9) shows **gate depth + blending** still matter.

#### 4.4.2 Actual deletion rate vs accuracy

| Dataset | Repo1 actual del% | Repo1 gated acc | Repo2 actual del% | Repo2 gated acc | Observation |
|---------|----------------------|--------------------|------------------|----------------|-------------|
| SNLI    | 76.1%                | 89.02%             | 30.8%            | 90.21%         | We delete **much more** (76% vs 31%) but still reach 89%; repo2 deletes less and keeps 90%. So **higher deletion can still retain high accuracy** on SNLI (redundancy). |
| SST-2   | 61.8%                | 92.55%             | 47.4%            | 96.67%         | Same pattern: we delete more, accuracy slightly lower in absolute terms but both are high. |
| MRPC    | 53.8%                | 68.63%             | **5.7%**         | 68.71%         | repo2's model **barely deletes** (5.7%) yet **drops 17 pp**; we delete 54% and are flat. Suggests on MRPC, **deletion rate alone does not explain the drop**—likely controller/init or data sensitivity. |
| TyDi QA | 10.9% (**L4**) / ~25–29% (**Modal L3**) / **~48.0% (Modal L9/384 post-fix)** | 28.16% EM (L4); **0.3471** / **0.3562** / 0.3499–0.3778 EM (Modal L3); **0.4000 peak / 0.3861 final** (Modal L9/384 post-fix) | ~24.7%           | 0.35 EM        | **No blend 0.3471** vs **learnable 0.3562**: blend helps slightly but **deletion differs**. Legacy **0.378 vs 0.347** at ~26% / ~25% del supports strong **fixed** blend effect—verify with args print. Post-fix L9/384 can match or exceed repo2-level EM in this setup. |

**Takeaway:**

- **SNLI / SST-2 are highly redundant.** These tasks tolerate **high actual deletion** (47–76%) with only a small accuracy cost, consistent with substantial **token-level redundancy**.
- **MRPC (repo2) shows optimization issues, not just deletion.** MRPC on repo2 has **low deletion (5.7%)** but a **large accuracy drop**, pointing toward **optimization / regularization** effects (e.g., gate init, weight decay, small dataset) rather than deletion magnitude alone.
- **TyDi: mechanism, level, and learning dynamics.** **Learnable blend** versus **no blend** (+**~0.9 pp**) comes with **higher deletion** (~29% vs ~25%). **Legacy fixed-blend** gap (+**~3.1 pp**) is the cleaner “same regime” story once replicated. **L4 low deletion + no blend** (**0.28**) shows **epochs and deletion** also dominate. The earlier **L9 / 384 / `gate_threshold_ratio=0.8`** failure (**~0.089 EM**, `ap-c5H2J…`) was due to **misaligned PI deletion metrics** (fixed k/2 vs model 0.8·k); after the fix, rerun **`ap-Qyiv...`** reaches **0.4000 peak / 0.3861 final**, validating the L9 path.

#### 4.4.3 XLM-R (both A100): direct comparison with Δ

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

#### 4.4.4 TyDi QA: no blending vs blending

**Baseline column semantics:** The **20.18%** row is a **no-gate, 1-epoch L4** checkpoint—useful for **coarse “weak baseline vs gated MrBERT”** stories only. Claims that **pre-deletion blending** improves TyDi must compare **no-blend vs blend under matched Modal (or otherwise matched) training** (e.g. **0.3471 vs 0.3562**), not **20.18% → …**.

| Config              | Baseline EM | Gated EM | Actual del% | Note                          |
|---------------------|-------------|----------|-------------|--------------------------------|
| Codebase (L4, no blend) | 20.18% (weak L4; not Modal blend baseline) | **28.16%** | 10.9%     | Span remap + warmup; 1 ep.   |
| **Modal A100, no pre-deletion blend, L3** (`[Training args]`) | — | **34.71%** (0.3471) | **~25.28%** | Log: `modal_logs/tydiqa_modal_EM03471_ap-4FzTHi10GW6ipDKX7fZ9TC.log`. W&B: `tydiqa-no-blend-l3-30pct`. |
| **Modal A100, learnable blend, L3** (`[Training args]`) | — | **35.62%** (0.3562) | **~28.64%** | `use_pre_deletion_blend=true`, `use_learnable_pre_deletion_blend=true`, `pre_deletion_blend_init_scale=30`. Log: `modal_logs/tydiqa_modal_EM03562_ap-G88KtoH6OWW4NMrsTl89YX_learnableblend.log`. **~+0.9 pp** vs row above; **deletion not matched**. |
| Modal A100, legacy “with-blend” name (no args block) | — | **34.99%** (0.3499) | **~24.34%** | `modal_logs/tydiqa_modal_EM03499_ap-rqoNveoPynOFoc2iKRSCnA.log`. |
| Modal A100, legacy high EM (W&B “no-blend” name) | — | **37.78%** (0.3778) | **~25.90%** | `modal_logs/tydiqa_modal_EM03778_ap-IEe2gcBxFgDvH62acNX22v.log`. **~+3.1 pp** vs **0.3471** if both were true fixed A/B—**flag/name inconsistency**; replicate with `[Training args]`. |
| **Modal A100, “limit” run (L9, 384, 6 ep, fixed blend)** | — | **8.87%** (0.0887; peak **13.12%** ep 5) | **~88.75% logged** | `modal_logs/tydiqa_modal_ap-c5H2JBykE62ASkj1prfX7M.log`, App `ap-c5H2JBykE62ASkj1prfX7M`. Intended: `gate_threshold_ratio=0.8`, `target_deletion=0.2`, `kp=0.1`. **Not a valid L9 ablation:** PI + W&B deletion metric used **fixed k/2** while hard-delete used **0.8·k**, so PI drove **α→0** and logged del% was **not** aligned with actual keep/drop. **Code fixed** so PI and logs use `--gate-threshold-ratio`. **Re-run** after fix to assess L9+384. |
| **Modal A100, post-fix rerun (L9, 384, 6 ep, fixed blend)** | — | **40.00% peak** (ep5), **38.61% final** (0.3861) | **~47.99%** | `modal_logs/tydiqa_modal_EM03861_ap-Qyiv6SzrZ7T2OgtJwL7LWl.log`, App `ap-Qyiv6SzrZ7T2OgtJwL7LWl`, W&B `aronima7-stanford-university/alina` run `u03dgn34`. Confirms the PI-threshold fix restored this setting to competitive performance. |
| repo2 (no blend, L3) | 0.40        | **0.10**   | 60.7%     | Gate collapses; EM collapses. |
| repo2 (blend, L3)    | —           | **0.30**   | ~0%      | Blending stabilises; gate stops deleting. |
| repo2 (blend, L9)    | 0.40        | **0.35**   | ~24.7%   | Best reported tradeoff in repo2. |

**Analysis:**

- **No blending at L3 collapses TyDi (repo2's extreme case).** Without blending, repo2's gate at L3 **over-deletes** (60.7%) and EM drops to **0.10**—the QA representation collapses.
- **Blending at L3 stabilises but can stop deletion (repo2).** With blending at L3, repo2's gate **stops deleting** (~0%) and EM recovers to **0.30**, but compression is lost in that report row.
- **Blending at L9 gives the best tradeoff (repo2).** **L9 with blending** yields **0.35 EM** with **~25% deletion** in repo2's table.
- **Repo1: authoritative Modal pair (args logged).** **No pre-deletion blend (0.3471)** vs **learnable blend (0.3562)** shows a **modest EM gain** with **higher train deletion**—not a single-variable ablation. **Legacy 0.3778 vs 0.3471** is **stronger** support for “blending on helps” at **similar** ~25–26% deletion, but **must be re-run** with `[Training args]` for a defensible fixed-blend row.
- **Failed “limit” Modal run (L9, 384, `gate_threshold_ratio=0.8`).** Final EM **~0.089** with **α→0** and inflated logged deletion: **PI and train-loop deletion % used k/2 while the gate used 0.8·k**, so the controller mis-estimated deletion vs `target_deletion=0.2`. **Repository fix:** `PIController.step(..., gate_threshold_ratio)` and `train_mrbert.py` now use `args.gate_threshold_ratio` for PI and logged rates. **Do not** cite this run as evidence that L9 hurts TyDi until re-running with the fix.
- **Post-fix rerun validates L9+384.** With the fix applied, the same high-capacity setting reaches **0.4000 peak EM** (epoch 5) and **0.3861 final** at ~48% logged deletion (`ap-Qyiv...`), showing the previous collapse was implementation-induced, not an inherent failure of L9.
- **Why `alpha_final = 0` is still plausible in a successful run.** In this repo, PI uses `alpha = max(0, kp * P + ki * I)`. Late in training, when current deletion stayed above the target (`target_deletion=0.2` vs logged ~0.48), the error term (`target - current`) remained negative, so the P/I accumulator drove `kp * P + ki * I` below zero and the clamp set `alpha` to 0. This means PI stopped adding *extra* gate regularization pressure; it does **not** mean the gate stops deleting. Deletion can remain high because gate behavior is still shaped by task loss and model dynamics. After the threshold-alignment fix, this is interpreted as late-stage controller saturation, not a PI bug.
- **Pulling Modal logs:** use `modal app logs ap-… --timestamps` (see `RUN_EXPERIMENTS_README.md` §6); some CLI builds omit `--tail` / `--since`.

Commands (from repo root; Modal + `wandb-api-key` secret):

```bash
# No pre-deletion blend (matches 0.3471 log pattern)
modal run --detach run_mrbert_tydi_modal.py \
  --no-use-pre-deletion-blend \
  --no-use-learnable-pre-deletion-blend \
  --wandb-project mrbert-tydiqa --wandb-run-name tydiqa-no-blend-l3-30pct \
  --batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 \
  --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 \
  --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5

# Fixed pre-deletion blend, learnable scale off (for clean vs learnable / vs no-blend)
modal run --detach run_mrbert_tydi_modal.py \
  --use-pre-deletion-blend \
  --no-use-learnable-pre-deletion-blend \
  --wandb-project mrbert-tydiqa --wandb-run-name tydiqa-fixedblend-l3-30pct \
  --batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 \
  --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 \
  --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5

# Learnable blend (matches 0.3562 log)
modal run --detach run_mrbert_tydi_modal.py \
  --use-pre-deletion-blend \
  --use-learnable-pre-deletion-blend \
  --pre-deletion-blend-init-scale 30 \
  --wandb-project mrbert-tydiqa --wandb-run-name tydiqa-learnableblend-l3-30pct \
  --batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 \
  --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 \
  --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5
```

---

### 4.5 Summary: what differs and why

| Metric / aspect | repo1 | repo2 | Main reason for gap |
|-----------------|------------------|---------------|----------------------|
| **SNLI baseline** | 74% (1 ep, L4) | 90.48% (A100) | Epochs / training setup |
| **SNLI MrBERT** | 89.02% | 90.21% | Baseline + setup |
| **SST-2 baseline** | 67–80% | 97.29% | Epochs / setup |
| **MRPC baseline** | ~68% | 86% | Epochs / setup (e.g. 5 ep MRPC) |
| **TyDi QA EM (gated)** | ~0.28 (L4); **20.18%** = weak L4 **no-gate** only (not blend baseline); Modal **0.3471** (no blend, args); **0.3562** (learnable blend, args); **0.3778** / **0.3499** (legacy logs); **0.4000 peak / 0.3861 final** (post-fix L9/384); **~0.089** pre-fix “limit” run **invalid** | ~0.35 (blend L9) | **Blending + epochs + gate depth**; blend A/B on **matched Modal** (**0.3471 vs 0.3562**); learnable +**~0.9 pp** vs logged no-blend but **higher del**; post-fix L9/384 reaches repo2-level or better EM; pre-fix 0.089 = bug record only |
| **XLM-R SNLI (A100)** | 33.82% bl / 33.82% MrXLM | 89.85% bl / 82.27% MrXLM | Different baselines (setup); both show fragility |
| **Latency** | T4, 30–55% speedup | A100, 1.89× | **GPU** + same idea |

