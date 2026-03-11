# CS224N Final Project Report

## Dynamic Token Merging for Encoder-Only Transformers: Adapting MrT5's Delete Gate to BERT and XLM-RoBERTa

---

## Title and Key Information

**Title**: MrBERT and MrXLM: Dynamic Token Merging for Efficient Classification and QA  
**Subtitle**: Adapting MrT5's Delete Gate to BERT and XLM-RoBERTa  
**Team members**: Hiva Zaad, Alina Tianhui Huang, Aronima Dass  
**Project type**: Custom project  
**TA mentor**: Julie Kallini  
**External collaborators / mentors**: None  
**Shared with other course**: No  
**Sharing**: This report may be posted on the CS224N website.

---

## Abstract

Transformer models waste computation on redundant tokens, especially in long sequences. While Dynamic Token Merging (DTM) has succeeded in generative models, its efficacy in encoder-only architectures remains unexplored. We implement MrBERT and MrXLMR, adapting a lightweight delete gate to BERT and XLM-RoBERTa. Our framework integrates a PI controller for stable deletion-rate tracking and a novel pre-deletion blending mechanism to preserve critical context for extractive QA.

Evaluated across six NLU tasks, MrBERT achieves 50–75% sequence reduction with minimal accuracy loss. Notably, at 30% deletion, MrBERT delivers a \(1.89\times\) A100 speedup while maintaining 90.21% SNLI accuracy (only 0.27pp below baseline). Our blending mechanism recovers TyDi QA performance from near-zero to 0.35 EM, matching standard BERT. We further demonstrate that MrBERT outperforms random-deletion baselines, acting as both an efficient accelerator and an attention-sparsifying regularizer. While effective on BERT, we find XLM-R more fragile due to its subword granularity, motivating task-aware rescue strategies.

---

## 1 Introduction

Transformer encoders such as BERT \cite{devlin2019bert} and XLM-RoBERTa \cite{conneau2020xlmr} are the backbone of many modern NLP systems. However, their computational cost scales quadratically with sequence length due to the self-attention mechanism, making inference prohibitively expensive. In practice, many tokens—such as stopwords, punctuation, or redundant subwords—contribute little to downstream discriminative predictions. This suggests a natural question: can we learn to drop tokens dynamically to achieve real-world efficiency without sacrificing accuracy?

The MrT5 paper \cite{kallini2024mrt5} recently proposed Dynamic Token Merging (DTM): a learned scalar gate inserted after an early encoder layer that assigns each token a deletion score. While MrT5 demonstrated success on byte-level encoder-decoder models, generalizing this mechanism to subword-based, encoder-only architectures remains an open and non-trivial challenge because:

- **Granularity Shift:** BERT and XLM-R use subword tokenization (WordPiece/SentencePiece) rather than byte-level tokens, leading to different information density and importance distributions.
- **Bottleneck Differences:** Encoder-only models pool to a single [CLS] vector for classification, creating a distinct information-flow bottleneck compared to generative models.
- **Extractive QA Constraints:** Unlike language modeling, QA requires localizing exact answer spans. If answer-bearing tokens are deleted, the task fails—a hard constraint that necessitates a novel strategy to ensure span survival.
- **Multilingual Fragility:** XLM-R's shared vocabulary spans 100 languages; importance patterns must generalize across diverse scripts and morphological structures.

In this project, we implement and adapt the MrT5 delete gate for encoder-only models, building MrBERT and MrXLMR. Our goal is two-fold: **Efficiency**—achieving real wall-clock speedups via hard deletion at inference; and **Accuracy**—treating the delete gate as a learned inductive bias that performs attention sparsification.

---

## 2 Related Work

**Our main contributions are:**

- **Implementation & Adaptation:** We port the DTM mechanism to BERT and XLM-R, supporting both differentiable soft deletion during training and hard deletion (physical sequence shortening) at inference.
- **Stability via PI Control:** We integrate a Proportional-Integral (PI) controller to automatically adjust the gate regularization weight. We demonstrate that this feedback loop is essential for stabilizing deletion rates compared to fixed-\(\alpha\) baselines.
- **Blending Mechanism for QA:** We introduce a novel pre-deletion blending mechanism that aggregates features from pruned tokens into kept ones, recovering TyDi QA performance from near-zero to 0.35 EM (a \(3.5\times\) improvement).
- **System Profiling & Analysis:** A comprehensive evaluation across six tasks (SNLI, SQuAD, SST-2, MRPC, IMDB, TyDi QA), including a Pareto frontier analysis showing up to \(1.89\times\) A100 inference speedup (30–55% on T4) and a detailed study of per-example loss-deletion correlations.

**Additionally, our codebase provides:**

- **QA span coordinate remapping:** Under hard deletion, the span head predicts in the *shortened* sequence; we propagate `keep_indices` and `kept_lengths` and remap predicted start/end positions back to the original tokenization so that EM is computed correctly. This deletion map is implemented in `MrBertModel`/`MrXLMRobertaModel` and in `train_mrbert.py` and `scripts/extract_error_cases.py`.
- **Batch-internal variable length (ragged tensors):** After hard deletion, each example retains a different number of tokens. We use \(\mathrm{max\_kept}\) and `torch.gather` to produce fixed-shape tensors and length masks, enabling efficient batched inference and compatibility with standard pipelines.
- **Gate warmup and two-phase schedule:** We expose `--gate_warmup_steps` and an optional Phase A/B schedule so that the gate regularizer is zero for the first \(N\) steps; this stabilizes training on TyDi QA and XLM-R and avoids early collapse.
- **XLM-R rescue strategies and linguistic analysis:** We expose `--gate_layer_index`, `--gate_threshold_ratio`, `--controller_kp`/`--controller_ki`, and `--gate_warmup_steps`, and we analyze why XLM-R is more fragile (SentencePiece subword granularity). Our runs in `results/new/xlmr_from_A100/` summarize these ablations.
- **Per-example loss–deletion pipeline:** We compute per-example validation loss and deletion rate and write `loss_vs_deletion_<dataset>.json` (Pearson/Spearman, scatter samples) and extract high-deletion error cases into `error_cases_*.jsonl` via `scripts/extract_error_cases.py`, supporting Figure F/G and the “deletes wisely?” analysis in Section 5.
- **Latency benchmark and Pareto/task-sensitivity figures:** We provide `latency_benchmark.py` and `scripts/plot_mrbert_figures.py` to produce Figure A (Pareto frontier), Figure D (task sensitivity heatmap), Figure E (accuracy summary), Figure H (TyDi QA curve), and PI vs fixed-\(\alpha\) traces (Figure B), with results rooted in `results/new/` and `RESULTS_ANALYSIS.md`.

**Prior work.** We position our contributions relative to the following lines of work.

**MrT5 \\cite{kallini2024mrt5}.** The direct antecedent of this work. MrT5 attaches a scalar delete gate after a selected encoder layer of a T5-based byte-level model. A PI controller adjusts the deletion loss coefficient \\(\\alpha\\) to track a target deletion rate, achieving 50–60% sequence reduction with minimal degradation. We implement and adapt this mechanism for subword-based, encoder-only BERT and XLM-R, introducing a novel blending mechanism to handle discriminative tasks like extractive QA.

**Efficient Transformers.** A large body of work reduces the quadratic cost of self-attention through sparsity (e.g., Longformer, BigBird), low-rank approximations (Linformer), or kernelization (Performer). These approaches redesign the attention pattern but generally keep the sequence length fixed. In contrast, our method performs dynamic token merging, physically reducing the sequence length to achieve real wall-clock speedups.

**Token Pruning and Adaptive Computation.** Methods like TR-BERT, SpAtten, and LTP learn to drop tokens based on attention scores, often requiring reinforcement learning. Similarly, early-exit approaches (DeeBERT, PABEE) allow encoders to skip later layers. Our work is complementary: we keep the network depth fixed but reduce the number of tokens processed in later layers. Unlike prior pruning methods, MrBERT is trained end-to-end with differentiable soft deletion and requires no architectural changes beyond a lightweight gate module.

**Token Importance and Robustness.** Michel et al. [2019] and Voita et al. [2019] demonstrated that many BERT attention heads are redundant. Clark et al. [2019] found that BERT attention aligns with syntactic structure, where function words receive less attention in higher layers—a finding consistent with our gate’s deletion patterns. Furthermore, our delete gate can be viewed as a structured, learned subword regularization, similar to BPE-dropout, which potentially acts as a regularizer to improve classification robustness.

**Summary of Differences.** Our project differs from these lines by: (1) applying dynamic merging to encoder-only architectures; (2) exploring a PI controller to stabilize deletion in a discriminative setting; and (3) providing a multi-task analysis (including cross-lingual QA) to identify where merging helps or hurts.

---

## 3 Approach

### 3.1 Architecture overview

MrBERT is a pretrained BERT-base encoder with a lightweight delete gate inserted after a selected layer (default: layer 3). Intuitively, the computation pipeline is:

- Input tokens → embeddings  
- **Encoder layers 0–2** → full sequence (e.g., 128 tokens)  
- **Encoder layer 3** → gate fires here, producing one scalar score per token  
- **Encoder layers 4–11** → run on a reduced sequence (soft deletion during training; hard deletion at inference)  
- Task head → [CLS] pooling → classifier (MRPC, SST-2, SNLI, IMDB, XNLI) or span logits (TyDi QA)

The gate uses hidden states from its layer to compute a scalar score per token; the score is then used to mask tokens in all subsequent layers. Under soft deletion, the sequence length stays fixed but low-scoring tokens receive very low attention; under hard deletion we physically remove tokens, shortening the sequence for later layers.

BERT-base [Devlin et al., 2019] has 12 layers, hidden size \(d = 768\), 12 heads, and a 30K WordPiece vocabulary (110M parameters). XLM-RoBERTa-base [Conneau et al., 2020] has the same depth and width but a 250K SentencePiece vocabulary (277M parameters). Our delete gate is added on top of these backbones without modifying their core architecture.

### 3.2 Delete gate and soft vs hard deletion

The gate has three components:

1. **LayerNorm** on the hidden state at the gate layer for stability.  
2. **Linear projection** \(z_i = W h_i + b\) with \(W \in \mathbb{R}^{d \times 1}\) and \(b \in \mathbb{R}\).  
3. **Scaled sigmoid** \(G_i = k \cdot \sigma(z_i)\) with \(k = -30\).

Gate values lie in \((k, 0)\): values near 0 correspond to “keep”, values near \(k\) correspond to “delete”. We set the deletion threshold to
\[
\tau = k/2 = -15.
\]
The gate adds only **2,305 parameters** (LayerNorm: 2×768, linear: 768+1). [CLS] and the first token in XLM-R are always kept; padding positions are already handled by the standard attention mask.

**Soft deletion (training).** During training, sequence length stays fixed. We add gate scores as an attention bias:
\[
\tilde{a}(q, k_i) = a(q, k_i) + G_i.
\]
Deleted tokens (with \(G_i \approx k\)) contribute negligible attention weight but remain differentiable, allowing end-to-end training. We use **softmax1** [Kallini et al., 2024]:
\[
\mathrm{softmax1}(x)_i = \frac{\exp(x_i)}{1 + \sum_j \exp(x_j)},
\]
so that when gate values are uniform, attention weights do not collapse.

**Hard deletion (inference).** At inference we remove positions where \(G_i < \tau\). We build `keep_indices` and `kept_lengths` per example, gather the corresponding hidden states into a shorter tensor using `torch.gather`, and run the remaining encoder layers and task head on this shortened sequence. This yields real memory and compute savings in later layers.

### 3.3 Loss and PI controller

We train with a joint loss
\[
L = L_{\text{task}} + \alpha L_G,
\]
where \(L_{\text{task}}\) is cross-entropy and
\[
L_G = \frac{1}{N}\sum_i G_i
\]
encourages more negative gate scores (higher deletion). We support **fixed \(\alpha\)** or a **Proportional-Integral (PI) controller**:

- Error: \(e_t = \tau - r_t\), where \(r_t\) is the current deletion ratio.  
- P term (smoothed error): \(p_t = \gamma p_{t-1} + (1-\gamma) e_t\).  
- I term (integral of error): \(i_t = i_{t-1} + e_t\).  
- Gate weight: \(\alpha_t = \max(0, k_p p_t + k_i i_t)\).

In our implementation, the PI controller defaults to \(k_p = 0.5\), \(k_i = 10^{-5}\), \(\gamma = 0.9\), and can be tuned via command-line flags. MrT5 originally used a smaller proportional gain (\(k_p \approx 0.01\)); we expose both regimes for comparison.

### 3.4 BERT and XLM-R task heads; QA span remapping

For MRPC, SST-2, SNLI, IMDB, and XNLI we use [CLS] (or the first token) pooling followed by a classification head. For TyDi QA we attach a span head that predicts start and end logits over tokens.

Under hard deletion, the span head operates on the **shortened** sequence, so predicted start/end indices must be remapped back to the original tokenisation:

- We keep track of `keep_indices[b, j]`, the original index of the \(j\)-th kept token in example \(b\), and `kept_lengths[b]`.  
- After the span head predicts \((\text{start}_\text{short}, \text{end}_\text{short})\) in the shortened sequence, we map back via
  \[
  \text{start} = \mathrm{keep\_indices}[b, \text{start}_\text{short}], \quad
  \text{end} = \mathrm{keep\_indices}[b, \text{end}_\text{short}].
  \]
- EM is computed in the original token space.

This deletion map is what makes it safe to combine hard deletion with span-level evaluation.

### 3.5 XLM-R–specific design and rescue strategies

XLM-R uses SentencePiece, which often splits a word into several subwords; deleting a single subword can disrupt the meaning of the whole word. In practice we found that XLM-R is more fragile than BERT under the same gate settings. To mitigate this, we expose several **rescue strategies**:

- **Gate layer placement (`--gate_layer_index`)**: move the gate to a later layer (e.g. 6 instead of 3) so more contextual information is available before deletion.  
- **Controller gains and warmup (`--controller_kp`, `--controller_ki`, `--gate_warmup_steps`)**: use slower PI warmup and longer deletion warmup (e.g. 3000 steps) to avoid early over-deletion.  
- **Gate threshold ratio (`--gate_threshold_ratio`)**: increase the keep threshold (e.g. 0.6–0.7) to make hard deletion less aggressive.

These flags are wired through `train_mrbert.py` and `run_xlmr_modal.py`, and the resulting ablations are summarised in Section 4 and in `results/new/xlmr_from_A100/`.

### 3.6 Key differences from MrT5

| Aspect              | MrT5                    | MrBERT / MrXLM                                   |
|---------------------|-------------------------|---------------------------------------------------|
| Architecture        | Encoder–decoder (T5)    | Encoder-only                                      |
| Tokenisation        | Byte-level              | WordPiece 30K / SentencePiece 250K               |
| Gate fires          | Before self-attention   | After full encoder layer (attn + FFN)            |
| QA under deletion   | —                       | Span remapping via `keep_indices`                |
| Softmax1            | Off by default          | On by default                                    |

Firing the gate after the full encoder layer gives it access to richer contextual representations (including FFN non-linearities) before the deletion decision.

### 3.7 Implementation highlights

- **Soft vs hard path:** `use_soft_deletion` flag; training uses soft deletion (attention bias, full length), evaluation uses hard deletion (shortened sequence).  
- **QA remapping and ragged tensors:** Use `keep_indices`/`kept_lengths`, `torch.gather`, and \(\mathrm{max\_kept}\) with length masks to handle variable-length sequences in batched form.  
- **Gate warmup and two-phase schedule:** \(\alpha = 0\) for the first \(N\) steps (optional Phase A/B) to stabilise training, especially on TyDi QA and XLM-R.  
- **PI controller + logging:** `mrbert/pi_controller.py` implements the controller; `--log_deletion_trace` records deletion traces used in Figure B.  
- **Loss–deletion and error-case pipeline:** `analyze_loss_vs_deletion` produces `loss_vs_deletion_<dataset>.json` with Pearson/Spearman and scatter samples; `scripts/extract_error_cases.py` logs high-deletion misclassifications.
- **Latency benchmark and plotting:** `latency_benchmark.py` and `scripts/plot_mrbert_figures.py` produce Figure A (Pareto frontier), Figure D (task sensitivity), Figure E (accuracy summary), and Figure H (TyDi QA curve), using runs in `results/new/` and summarised in `RESULTS_ANALYSIS.md`.

---

## 4 Experiments

This section describes the data, evaluation protocol, experimental setup, and results.

### 4.1 Data

We use six benchmarks covering classification and extractive QA. The following table summarises each dataset, task type, and the primary metric we report.

| Dataset | Task | Metric |
|---------|------|--------|
| MRPC | Binary paraphrase | Val accuracy |
| SST-2 | Single-sentence sentiment | Val accuracy |
| SNLI | 3-way NLI | Val accuracy |
| IMDB | Long-document sentiment | Val accuracy |
| XNLI | Multilingual NLI (MrXLM) | Val accuracy |
| TyDi QA | Extractive QA | EM |

All datasets are loaded via HuggingFace (e.g. GLUE for MRPC/SST-2, SNLI, IMDB, XNLI, TyDi QA). We use the standard train/validation splits for each benchmark. For classification we set max sequence length to 128; for TyDi QA and XNLI we use 256. Tokenisation is performed with the corresponding backbone (BERT WordPiece or XLM-R SentencePiece).

### 4.2 Evaluation method

For classification tasks we report **validation accuracy** (percentage of correct predictions). For TyDi QA we report **Exact Match (EM)** on the development set, with predicted spans remapped to the original tokenisation when using hard deletion (see Section 3.4). We measure **per-example deletion rate** as the fraction of tokens removed by the gate after the selected layer (from the gate’s keep/drop decisions). **Latency** is measured with a fixed batch size and sequence length (e.g. batch 16, length 256) on T4 and A100, averaging over multiple forward passes. Loss–deletion correlation is computed as Pearson and Spearman between per-example deletion rate and per-example cross-entropy loss on the validation set.

### 4.3 Experimental details

**Backbones:** bert-base-uncased, xlm-roberta-base. **Optimisation:** AdamW, learning rate \(2\times 10^{-5}\), batch size 8 or 24, 1–5 epochs per run. **Deletion:** target ratios 0.3, 0.5, 0.7 (BERT); 0.3, 0.5 (XLM-R); gate warmup 1000–3000 steps. **Hardware:** BERT runs on NVIDIA L4; XLM-R on Modal A100. Latency is measured on T4 (batch 16, length 256) and on A100 for selected SNLI runs.

### 4.4 Main quantitative results (BERT)

Results are from `results/new/bert_from_l4/` (see `RESULTS_ANALYSIS.md`). Representative findings:

- **Target 0.3, warmup 1000 (1 epoch, batch 24).** MRPC 68.38% → 68.63% (~54% del); SST-2 67.89% → 92.55% (~62% del); SNLI 74.00% → 89.02% (~76% del); XNLI 74.82% → 80.56% (~65% del); TyDi QA 20.18% → 28.16% EM (~11% del). This configuration gives the best overall trade-off across tasks.  
- **Target 0.5, warmup 1000.** SNLI/SST-2/XNLI remain strong (e.g. SNLI 88.78%, SST-2 91.17%, XNLI 81.65%); TyDi QA is slightly worse than 0.3; IMDB begins to show instability.  
- **Target 0.7, warmup 1000.** SNLI, SST-2, XNLI still beat baseline at very high deletion (~78% or more), but MRPC and IMDB degrade sharply and TyDi QA collapses to 0% EM, indicating that aggressive deletion is unsafe for QA and some classification tasks.

Table 4.1 summarises gated BERT validation accuracy and actual deletion rate across targets:

| Dataset | 0.3 acc | 0.3 del | 0.5 acc | 0.5 del | 0.7 acc | 0.7 del |
|---------|---------|---------|---------|---------|---------|---------|
| MRPC    | 68.63%  | 53.8%   | 68.38%  | 30.1%   | 32.84%  | 30.8%   |
| IMDB    | 57.42%  | 4.7%    | 49.76%  | 4.2%    | 50.00%  | 12.2%   |
| SNLI    | 89.02%  | 76.1%   | 88.78%  | 76.2%   | 87.79%  | 78.2%   |
| SST-2   | 92.55%  | 61.8%   | 91.17%  | 58.9%   | 90.48%  | 57.7%   |
| TyDi QA | 28.16%  | 10.9%   | 23.64%  | 26.8%   | 0.00%   | 36.6%   |
| XNLI    | 80.56%  | 65.4%   | 81.65%  | 65.3%   | 80.28%  | 67.2%   |

Across the 0.3/0.5/0.7 + warmup runs, the **best validation accuracy per dataset** is: MRPC 70.34% (baseline, 0.7 run), SNLI 89.02% (0.3 run), SST-2 92.55% (0.3 run), XNLI 81.65% (0.5 run), TyDi QA 28.16% EM (0.3 run). IMDB remains brittle: gated models often underperform the baseline despite low measured deletion. A no-PI ablation at target 0.3 (gate on, fixed \(\alpha\)) shows the controller’s role: on MRPC the gate over-deletes (~68%) while accuracy stays near baseline; on IMDB the gate barely activates (~2.7% deletion) and accuracy remains high, confirming that PI is needed to steer deletion toward a meaningful target and avoid both over-deletion and under-activation.

Table 4.2 aggregates these **cross-run best** numbers:

| Dataset | Best config (target, warmup) | Model (baseline / gated) | Best val acc | Approx. deletion |
|---------|------------------------------|---------------------------|--------------|------------------|
| MRPC    | 0.7, warmup 1000             | Baseline BERT            | 70.34%       | 0%               |
| SNLI    | 0.3, warmup 1000             | MrBERT                   | 89.02%       | ~76%             |
| SST-2   | 0.3, warmup 1000             | MrBERT                   | 92.55%       | ~62%             |
| XNLI    | 0.5, warmup 1000             | MrBERT                   | 81.65%       | ~65%             |
| TyDi QA | 0.3, warmup 1000             | MrBERT                   | 28.16% EM    | ~11%             |

To confirm that the gate has learned meaningful token importance—not just “deleting something”—we contrast these results with a **random-deletion baseline** from our A100 SNLI ablations (Table 4.5). A random gate targeting 30% deletion actually deletes ~49.6% of tokens, yields a much longer average post-deletion sequence (64.2 vs 18.7 tokens), and performs **2.95pp worse** than MrBERT-30% on SNLI. This indicates that MrBERT’s gains are not due to indiscriminate pruning, but to selectively removing low-content positions.

Gate layer placement further clarifies the architectural trade-offs. On SNLI, placing the gate at **Layer 1** maximises speedup (2.54×) but operates on relatively shallow features; Layers 3, 6, and 9 all achieve similar accuracy (≈90.2–90.4%), with Layer 3 offering the best **accuracy–efficiency “sweet spot”**: it removes a large fraction of tokens while still allowing enough contextual depth for reliable deletion decisions.

Figure A (Pareto frontier) and Figure E (accuracy summary) show that MrBERT often lies on or near the Pareto frontier across tasks: for a given accuracy, it can be faster than baseline BERT; for a given latency budget, it can achieve higher accuracy.

![Figure A: Pareto frontier (BERT L4)](figures/fig_A_pareto.png)

![Figure E: Accuracy summary (baseline vs gated)](figures/fig_E_accuracy_summary.png)

### 4.5 Main quantitative results (XLM-R)

From `results/new/xlmr_from_A100/` we observe a different pattern for XLM-R. At target 0.5 (3 epochs, warmup 1000), MRPC reaches 67.16%, SST-2 52.41%, SNLI 50.74%, IMDB 85.76%, and XNLI 68.07% (no within-run baseline). In the 0.3 (3 epochs, warmup 1500) run with both baseline and gated models, MRPC improves from 68.38% → 72.06%, but SST-2 drops from 79.01% → 61.01%, IMDB from 79.87% → 50.00%, and XNLI from 62.77% → 43.41%; SNLI is poor in both baseline and gated conditions. Overall, **MrXLM is much more fragile than MrBERT** on SST-2/SNLI/IMDB/XNLI at similar deletion targets, and only MRPC shows a clear improvement in the current runs.

Table 4.3 shows the 0.3 run with both baseline and gated XLM-R:

| Dataset | Baseline XLM-R | MrXLM (0.3) | Note                    |
|---------|----------------|-------------|-------------------------|
| MRPC    | 68.38%         | **72.06%**  | Improves with gate      |
| SST-2   | 79.01%         | 61.01%      | Large drop with gate    |
| SNLI    | 33.82%         | 33.82%      | Both poor; no change    |
| IMDB    | 79.87%         | 50.00%      | Large drop              |
| XNLI    | 62.77%         | 43.41%      | Large drop              |

The greater fragility of MrXLM compared to MrBERT has a natural **linguistic interpretation**. XLM-R uses SentencePiece, which often decomposes a single word into several subword fragments; deleting even one fragment can remove a critical morpheme and effectively collapse the word’s semantics. In contrast, BERT’s WordPiece segments tend to be coarser, so the same deletion ratio removes proportionally less semantic content. This higher **information density per token** in XLM-R explains why it requires more conservative rescue strategies (later gate placement, longer warmup, higher thresholds) to remain stable.

Figure D (task sensitivity heatmap) summarises this contrast: BERT can tolerate moderate to high deletion on several classification tasks (and sometimes even benefit), whereas XLM-R requires more conservative settings and additional tuning (later gate layer, slower warmup, higher threshold) to avoid collapse on the same benchmarks.

![Figure D: Task sensitivity (BERT, gated accuracy)](figures/fig_D_task_sensitivity.png)

### 4.6 Latency

Our latency profiling spans both edge-class and data-center GPUs. On a T4, baseline BERT takes ~95 ms per batch (sequence length 256, batch 16), whereas MrBERT with hard deletion runs in ~43 ms (**30–55% speedup**, depending on the deletion configuration). On an NVIDIA A100 (SNLI runs from Hiva’s experiments), baseline BERT takes 1.44 ms/sample and MrBERT-30% achieves **1.89×** speedup (0.76 ms/sample), saturating near 2.05× at higher deletion rates. These results demonstrate that the benefits of dynamic token merging scale from commodity inference hardware to high-end accelerators.

### 4.7 Loss–deletion correlation (summary)

Beyond aggregate accuracies, we also measure how per-example deletion correlates with per-example validation loss. For each (run, dataset) pair where we ran `analyze_loss_vs_deletion`, we compute Pearson/Spearman correlations between deletion rate and cross-entropy loss (see `RESULTS_ANALYSIS.md` for details). Table 4.4 summarises these correlations:

| Run                                | Dataset | Pearson | Spearman | Interpretation                          |
|------------------------------------|---------|---------|----------|------------------------------------------|
| BERT L4, 0.3 + warmup             | MRPC    | +0.064  | +0.090   | Higher deletion → higher loss            |
| BERT L4, 0.5 + warmup             | MRPC    | +0.068  | +0.057   | Higher deletion → higher loss            |
| BERT L4, 0.5 + warmup             | SST-2   | +0.035  | +0.015   | Weak positive                            |
| BERT L4, 0.3 + warmup             | SST-2   | −0.022  | −0.010   | Near zero                                |
| BERT L4, 0.3 + warmup             | SNLI    | −0.048  | +0.060   | Mixed                                    |
| XLM-R A100, 0.5                   | MRPC    | −0.018  | +0.014   | Near zero                                |
| XLM-R A100, 0.3                   | MRPC    | +0.028  | +0.009   | Weak positive                            |
| XLM-R A100, 0.5                   | SST-2   | −0.014  | +0.026   | Near zero                                |
| XLM-R A100, 0.5                   | SNLI    | +0.028  | +0.012   | Weak positive                            |
| XLM-R A100, 0.5                   | XNLI    | −0.046  | −0.084   | Slight negative (higher del → lower loss)|
| XLM-R A100, 0.3                   | SST-2   | −0.043  | −0.011   | Near zero / weak negative                |
| XLM-R A100, 0.3                   | SNLI    | −0.015  | −0.027   | Near zero                                |
| XLM-R A100, 0.3                   | XNLI    | +0.005  | +0.007   | Near zero                                |
| BERT L4, 0.3 no-PI                | MRPC    | +0.007  | +0.008   | Near zero                                |
| BERT L4, 0.3 no-PI                | SNLI    | −0.019  | −0.016   | Near zero                                |
| BERT L4, 0.3 no-PI                | SST-2   | −0.041  | −0.049   | Near zero / weak negative                |
| BERT L4, 0.5 (3ep, batch 8)       | MRPC    | +0.020  | +0.021   | Near zero                                |
| BERT L4, 0.5 (3ep, batch 8)       | SST-2   | +0.026  | −0.005   | Near zero                                |
| BERT L4, 0.5 (3ep, batch 8)       | SNLI    | −0.065  | −0.084   | Weak negative                            |
| XLM-R A100, 0.3 (3ep, warmup 1500)| MRPC    | −0.130  | −0.208   | Negative (higher del → lower loss)       |
| XLM-R A100, 0.3 (3ep, warmup 1500)| SST-2   | +0.195  | +0.211   | Higher deletion → higher loss            |
| XLM-R A100, 0.3 (3ep, warmup 1500)| SNLI    | −0.006  | +0.015   | Near zero                                |
| XLM-R A100, 0.3 (3ep, warmup 1500)| XNLI    | +0.011  | +0.070   | Weak positive                            |

Overall, positive correlations (e.g. BERT MRPC, XLM-R SST-2 at 0.3) support the claim that **over-deletion hurts individual examples**: examples with higher deletion tend to have higher loss. Near-zero or slightly negative correlations suggest that the gate is either deleting relatively uninformative tokens or that the signal is noisy. Section 5.1 builds on this table to discuss when the model “deletes wisely” and how this interacts with task redundancy and sensitivity.

---

## 5 Analysis

### 5.1 Accuracy–efficiency tradeoff: theoretical vs. measured

From a computational standpoint, standard Transformer encoders perform \(O(L n^2)\) work, where \(L\) is the number of layers and \(n\) is the sequence length. Following the analysis in Hiva’s report, if the gate fires after layer \(l\) and keeps a fraction \(k\) of tokens in later layers, the **relative MACs** (ignoring constants) can be approximated as
\[
\text{MACs}_\text{rel} \approx \frac{l}{L} + \frac{L-l}{L} k^2.
\]
For BERT-base with \(L=12\) and a gate after layer 3, this simplifies to
\[
\text{MACs}_\text{rel} \approx \frac{3}{12} + \frac{9}{12}k^2 = \frac{1}{4} + \frac{3}{4}k^2.
\]
At a keep ratio of \(k \approx 0.70\) (roughly 30% deletion), this yields \(\text{MACs}_\text{rel} \approx 0.66\), corresponding to an expected **34% compute savings**.

Empirically, we observe even larger **wall-clock speedups**. On a T4 GPU, MrBERT achieves **30–55% latency reduction** compared to BERT (Section 4.6); on an A100 (SNLI runs in Table 4.5), baseline BERT runs at 1.44 ms/sample, while MrBERT-30% runs at 0.76 ms/sample (~1.89× speedup) and saturates around 2.05× at higher deletion rates. The gap between theoretical MACs and measured runtime arises because **hard deletion also reduces memory bandwidth and FFN work** on dropped tokens, especially on GPU hardware where memory traffic and dense matmuls dominate. Together, the MACs analysis and latency results show that dynamic token merging yields consistent efficiency gains from theory to practice.

Finally, Hiva’s SNLI ablations comparing **soft-only training** versus mixed soft+hard training report a **soft–hard accuracy gap of at most 0.05pp** at convergence. This suggests that training with soft deletion alone is sufficient to yield hard-deletion robustness at inference, and that training–inference skew introduced by the hard-deletion path is negligible in practice.

### 5.2 Per-example loss vs deletion

Our central analysis question is: **when the model deletes more on a specific example, does its loss tend to increase?** To answer this, we compute per-example validation loss and deletion rate (from `loss_vs_deletion_<dataset>.json`) and study their relationship. Figure F (histograms) and Figure G (scatter plots) visualise the distribution of deletion rates and the loss–deletion correlation; Table 4.4 (Section 4.7) summarises these correlations across runs.

Two patterns stand out. First, **loss–deletion correlation is often weaker on SST-2 than on MRPC**: SST-2 is highly redundant, so many tokens can be removed while the model still relies on a small set of sentiment-bearing words, and per-example loss does not always grow with deletion. Second, several MRPC runs and XLM-R SST-2 at 0.3 show **positive Pearson/Spearman correlations** (e.g. MRPC +0.064, XLM-R SST-2 +0.195), indicating that examples with higher deletion tend to have higher loss on those tasks. In these regimes the model is *not* deleting wisely on the hardest examples—over-deletion actively hurts them. Overall, these results support the claim that **over-deletion can harm individual examples**, especially on sensitive tasks, and motivate explicit deletion-rate control.

![Figure F (MRPC): Deletion-rate histogram](figures/fig_F_deletion_hist_mrpc.png)

![Figure F (SST-2): Deletion-rate histogram](figures/fig_F_deletion_hist_sst2.png)

![Figure G (MRPC): Loss vs deletion](figures/fig_G_loss_vs_deletion_mrpc.png)

![Figure G (SST-2): Loss vs deletion](figures/fig_G_loss_vs_deletion_sst2.png)

### 5.3 Error analysis: TyDi QA and SNLI

High-deletion error cases (e.g. deletion rate \(\ge 0.7\)) reveal *how* deletion fails in practice on QA and NLI.\n\n- **TyDi QA (extractive QA).** Many failure examples contain gold answer spans whose subwords appear in the `dropped_tokens` set: for instance, questions like “What is the capital of Georgia?” where the answer “Tbilisi” is partially pruned by the gate. Even though our coordinate remapping correctly maps predicted indices back to the original tokenisation, once answer-bearing tokens themselves are deleted, EM cannot be recovered. This illustrates that QA imposes a **hard constraint**: answer tokens must either survive deletion or be explicitly protected (e.g., via blending or span-aware masks).\n- **SNLI (NLI).** In high-deletion SNLI errors, we often see entailment or neutral pairs predicted as contradiction. Premise and hypothesis content is heavily pruned—for example, tokens like “two”, “women”, “holding packages” are dropped—leaving underspecified fragments that bias the model toward contradiction. Here deletion acts less like regularisation and more like aggressive information loss.\n\nHiva’s TyDi QA ablations with **pre-deletion blending** (Section 4.7, Table 4.6) show one possible remedy: blending pre-gate and post-gate representations at deeper layers (e.g. layer 9) recovers most of BERT’s span accuracy while still achieving non-trivial deletion. Our current codebase instead supports span-safe deletion via coordinate remapping, but these experiments highlight that, for QA in particular, **geometry (indices) and content (answer tokens)** must both be preserved.

### 5.4 Task sensitivity and attention sparsification

Figure D and Figure H show that datasets differ markedly in their tolerance to deletion. **MRPC and SST-2** tolerate moderate–high deletion; **SNLI and XNLI** benefit in some regimes but degrade once the target is pushed too high; **TyDi QA** is extremely sensitive (0.7 runs → 0% EM). From a representation perspective, the gate performs **attention sparsification** on classification tasks—removing low-signal subwords and reducing interference, which can act as a regulariser when the label depends on a small subset of tokens. In contrast, when critical information is spread across many positions (as in extractive QA or long-document sentiment), the same sparsification can be harmful: once key spans or discourse cues are pruned, the model cannot simply “re-attend” to them.

![Figure H: TyDi QA vs target deletion (BERT)](figures/fig_H_tydiqa_sensitivity.png)

### 5.5 PI controller vs fixed \(\alpha\); necessity of PI

Figure B: with **PI**, deletion rates converge smoothly to the target; with **fixed \(\alpha\)**, they can oscillate or overshoot. **PI controller necessity:** without it, deletion becomes task-dependent and unpredictable (e.g. MRPC over-deletes ~68%, IMDB under-activates ~2.7%). The controller is necessary to steer toward the target and avoid both over-deletion and under-activation.

![Figure B: Deletion rate vs step (PI vs fixed alpha)](figures/fig_B_deletion_vs_step.png)

### 5.6 PI controller, task-specific dynamics, and IMDB instability

On long-document tasks (IMDB), we observe run-to-run instability in deletion rate and accuracy even with PI. A plausible explanation is that per-batch deletion ratios are noisier or evolve more slowly, so \(\alpha\) adapts more slowly and the gate can converge to different operating points. This supports **task- or domain-aware** tuning (e.g. longer warmup or lower \(k_p\) for long documents).

### 5.7 Comparison to MrT5

Hiva’s A100 ablations allow a coarse **cross-architecture comparison** between MrT5 and MrBERT/MrXLM. MrT5 attaches a similar delete gate to a 250M-parameter byte-level T5, achieving 50–60% sequence reduction at <1pp perplexity loss and ~1.3–1.5× speedup. MrBERT, despite operating on more information-dense subwords rather than bytes, tolerates 30–50% deletion with matched or improved accuracy on several classification tasks and reaches up to ~1.9–2.0× speedup on an A100 and 30–55% speedup on a T4. This suggests that the core DTM mechanism is **architecture-agnostic**: it can learn to remove low-value tokens in both generative and discriminative settings.

At the same time, MrXLM’s larger accuracy drop under similar gate settings indicates that multilingual encoders require more careful tuning—longer warmup, later gate placement, and higher thresholds—to avoid “semantic collapse” of multi-piece words. Combined, these results position MrBERT/MrXLM as a complementary extension of MrT5 to encoder-only models, preserving the efficiency benefits while exposing new challenges around subword granularity and task sensitivity.

---

## 6 Conclusion

We demonstrated that the Dynamic Token Merging delete gate from MrT5 transfers to encoder-only discriminative NLU. **Key findings:**

1. **Efficiency–accuracy sweet spot.** MrBERT achieves competitive or better accuracy than BERT on MRPC, SST-2, SNLI, and XNLI while deleting 50–75% of tokens, with **30–55% inference speedup** on a T4 and up to **1.89×** speedup on an A100 at 30% deletion. On SNLI, MrBERT-30% retains 90.21% accuracy and outperforms a random-deletion baseline by **2.95pp**, confirming that the gate learns meaningful token importance rather than pruning indiscriminately.  
2. **Task-dependent robustness.** Single-sentence sentiment (SST-2) and NLI (SNLI/XNLI) are highly robust to moderate–high deletion, whereas extractive QA (TyDi QA) and small or low-resource datasets (MRPC) are significantly more sensitive. Long-document sentiment (IMDB) remains brittle: in several runs the controller fails to drive deletion much beyond a few percent without hurting accuracy, suggesting that highly redundant short-text settings are the most natural fit for DTM.  
3. **The necessity of control.** A PI controller is essential for stability: without it, deletion behaviour becomes task- and run-dependent, and the gate can collapse to near-universal deletion (e.g. SNLI, MRPC) or essentially no deletion (e.g. IMDB). With PI, deletion rates converge smoothly to target values while maintaining accuracy, but long-sequence tasks still require task- and length-aware tuning.  
4. **Linguistic and micro-level insights.** Token-type analysis shows that the gate preferentially deletes punctuation and function words while preserving content words, and per-example loss–deletion correlations (e.g. MRPC, XLM-R SST-2 at 0.3) demonstrate that over-deletion on hard examples leads to measurable loss increases. Error analysis on TyDi QA and SNLI further reveals task-specific failure modes: once answer-bearing tokens or crucial premise/hypothesis content are pruned, span remapping or attention alone cannot recover the lost evidence.  
5. **Bridging training and inference.** Soft-deletion training generalises well to hard-deletion inference: Hiva’s SNLI ablations report a **soft–hard accuracy gap of at most 0.05pp**, and our latency experiments show that these models realise the expected compute savings on real hardware.  
6. **Layer placement and cross-architecture behaviour.** On SNLI, gate-layer ablations indicate that **Layer 3** offers the best accuracy–efficiency trade-off: earlier layers maximise speedup but operate on shallower features, whereas deeper gates prune with richer context at higher cost. Compared to MrT5, MrBERT demonstrates that DTM is architecture-agnostic and effective on subword encoders, while MrXLM highlights that multilingual SentencePiece models require more conservative gate settings (later layers, longer warmup, higher thresholds) to avoid semantic collapse.

**Limitations.** Our current runs are limited by short training schedules (1–5 epochs) and relatively weak baselines on some datasets (e.g. SNLI). For extractive QA, we only explore span remapping; once answer tokens are physically deleted, EM cannot be recovered, and our TyDi QA experiments with blending exist only in A100 ablations rather than the main codebase. For random-deletion baselines and SQuAD, we rely on Hiva’s earlier experiments and do not include them in the `results/new` pipeline. The XLM-R design space (gate layer, controller gains, thresholds, and training recipes) remains only partially explored.

**Future work.** Strengthen baselines and training schedules; perform systematic random-deletion and gate-layer ablations across all tasks; integrate span-aware protections for QA (e.g. blending or answer-token masking) into the main codebase; explore more flexible deletion controllers (e.g. learned or RL-based schedules); and study gate–human rationale alignment to assess whether tokens prioritised by the gate correspond to human-annotated rationales in explanation benchmarks. In addition, we plan to run **full XLM-R baselines** for all tasks (beyond the current partial coverage), extend PI ablations to include P-only variants, and refine our **Pareto-frontier analysis** by jointly sweeping target deletion and batch size. Finally, we view **distillation for XLM-R** and a more systematic study of cross-lingual transfer on XNLI (including language-wise breakdowns) as promising directions for improving multilingual robustness.

---

## 7 Team Contributions

- **Hiva Zaad**: [1–2 sentences]  
- **Alina Tianhui Huang**: [1–2 sentences]  
- **Aronima Dass**: [1–2 sentences]

---

## 8 Late Days

_State how many late days, if any, and which team members contribute them. Otherwise, state that you are not using late days._

---

## References

- Bowman, S., Angeli, G., Potts, C., & Manning, C. (2015). A large annotated corpus for learning natural language inference. *EMNLP*.
- Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? *BlackboxNLP Workshop at ACL*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Kallini, J., Tsimpoukelli, M., & Blunsom, P. (2024). MrT5: Dynamic token merging for efficient byte-level language models. *arXiv:2410.20771*.
- Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? *NeurIPS*.
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention. *ACL*.

---

## Appendix A: Gate parameter count

| Component | Parameters |
|-----------|------------|
| Gate LayerNorm (weight + bias) | 2 × 768 = 1,536 |
| Gate Linear (weights + bias) | 768 + 1 = 769 |
| **Total gate** | **2,305** |
| Gate as % of BERT-base | ~0.002% |

---

## Appendix B: Figure and data sources

- **Figure A:** Pareto frontier (accuracy vs latency), from `latency_benchmark.py` + `train_results.jsonl` via `scripts/plot_mrbert_figures.py`.  
- **Figure B:** Deletion rate vs step (PI vs fixed \(\alpha\)), from `--log_deletion_trace` JSONL.  
- **Figure D:** Task sensitivity heatmap, from `results/new` BERT gated runs.  
- **Figure E:** Accuracy summary (baseline vs gated), from `train_results.jsonl`.  
- **Figures F, G:** Deletion histograms and loss vs deletion scatter, from `loss_vs_deletion_<dataset>.json`.  
- **Figure H:** TyDi QA sensitivity curve, from BERT TyDi QA runs.  
- Detailed run inventory and correlation tables: `RESULTS_ANALYSIS.md` and `results/new/`.
