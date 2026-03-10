## 2 Approach

We adapt the **MrT5-style Dynamic Token Merging** mechanism to the encoder-only **BERT** architecture. Our model, **MrBERT**, introduces a learned delete gate inside `BertModel` that dynamically prunes redundant tokens, with the goal of saving computation while preserving task performance.

### 2.1 Model and Gate Mechanism

We follow the core design of MrT5 but instantiate it in BERT:

- **Delete Gate.** After encoder layer \(l = 3\), we insert a lightweight scalar gate that produces one score \(G_i \in [k, 0]\) per token, with \(k = -30\):
\[
G = k \cdot \sigma(\mathrm{LayerNorm}(H_l)W + b),
\]
where \(H_l\) is the hidden state at layer \(l\) and \(W \in \mathbb{R}^{d\times 1}\). This adds only \(2d_{\text{model}} + 1\) parameters and matches the parameterization in MrT5.

- **Soft vs. hard deletion.**
  - **Training (soft deletion).** We keep the sequence length fixed and add gate scores to the attention logits of subsequent layers:
  \[
  \mathrm{scores} = \frac{QK^\top}{\sqrt{d}} + \mathrm{mask} + \mathbf{1}G^\top.
  \]
  This emulates deletion while remaining fully differentiable.
  - **Inference (hard deletion).** Tokens with \(G_i < k \cdot \text{threshold\_ratio}\) (we use 0.5) are physically removed. We construct a shorter sequence and new attention mask, and run only the remaining encoder layers on the kept tokens, which can yield real latency gains.

- **Softmax1.** In gated layers we replace standard softmax with **softmax1**:
\[
\mathrm{softmax1}(x)_i = \frac{e^{x_i}}{1 + \sum_j e^{x_j}},
\]
which prevents attention weights from collapsing even when many tokens receive large negative gate scores.

- **Force-keep \[CLS\].** Because BERT uses `[CLS]` for pooled representations, the hard-deletion path always keeps position 0 regardless of its gate value.

### 2.2 Task Heads and Training Objective

We implement two task-specific wrappers on top of MrBERT:

- **MrBertForSequenceClassification** for MRPC, IMDB, SST-2, and SNLI, using a standard pooled `[CLS]` representation followed by dropout and a linear head.
- **MrBertForQuestionAnswering** for TyDi QA, using a span head that predicts start and end logits over tokens.

In both cases the training loss is
\[
L = L_{\text{task}} + \alpha L_G,
\]
where \(L_{\text{task}}\) is the standard cross-entropy task loss and
\[
L_G = \frac{1}{N}\sum_i G_i
\]
encourages lower gate scores (stronger deletion). The coefficient \(\alpha\) is either a fixed **gate\_weight** or is updated online by the PI controller (Section 2.3).

### 2.3 PI Controller and Regularization

To automatically regulate the deletion rate toward a target (e.g., 50%), we implement the **PI controller** described in MrT5:

- We track the **actual deletion rate** during training and compute an error term relative to a target deletion.
- The controller updates \(\alpha\) based on proportional and integral components of this error, increasing \(\alpha\) when we delete too few tokens and decreasing it when we delete too many.

This makes the deletion behavior adaptive to the dataset and training dynamics instead of relying on a fixed regularization weight.

### 2.4 Baselines and Code Dependencies

- **Baselines.** Our primary baseline is standard **BERT** (`bert-base-uncased`) using HuggingFace `BertForSequenceClassification` and `BertForQuestionAnswering` with the gate disabled (i.e., no deletion). Architecturally, the baselines share the same task heads and optimizer settings as MrBERT.
- **External code.** We build on the HuggingFace Transformers and Datasets libraries for model and data loading. All dynamic token merging, gating, PI controller, and diagnostics logic is implemented by us in the `mrbert` package (`modeling_mrbert.py`, `configuration_mrbert.py`, `pi_controller.py`, `diagnostics.py`).

### 2.5 Diagnostics and Interpretability

We implemented a diagnostics module (`mrbert/diagnostics.py`) to better understand the behavior of the delete gate:

- **Parameter counts** (total, trainable, and gate-only) to quantify the overhead introduced by the gate.
- **Deletion statistics**: actual deletion rate, gate mean and standard deviation, and final \(\alpha\).
- Optional **per-token gate visualization** and summary statistics by token type (word / subword / punctuation / special tokens), providing a handle on what the gate tends to remove.
- **Theoretical compute savings**: we approximate MACs saved by comparing self-attention cost before and after hard deletion across layers, enabling us to relate deletion rate to theoretical efficiency.

These tools are used both during experiments and for planned error/ablation analysis.

---

## 3 Experiments

### 3.1 Data

We evaluate MrBERT on five publicly available benchmarks loaded via HuggingFace:

- **MRPC (GLUE)**: Microsoft Research Paraphrase Corpus; sentence-pair paraphrase detection; ~3.7k train / 408 validation examples.
- **IMDB**: Movie review sentiment classification; ~25k train / 25k validation examples.
- **SST-2 (GLUE)**: Stanford Sentiment Treebank 2; single-sentence sentiment classification; ~67k train / 872 validation examples.
- **SNLI**: Stanford Natural Language Inference; three-way NLI (entailment, neutral, contradiction); ~550k train / 10k validation examples (filtered).
- **TyDi QA (secondary task)**: Extractive QA; ~50k train / 5k validation examples (SQuAD-style secondary task subset).

All datasets are standard English benchmarks; we use default splits and preprocessing from HuggingFace (`glue/mrpc`, `imdb`, `glue/sst2`, `snli`, and the TyDi QA secondary-task subset).

### 3.2 Evaluation Method

- **Classification tasks (MRPC, IMDB, SST-2, SNLI)**: We report **validation accuracy**.
- **TyDi QA**: We report **Exact Match (EM)**, defined as both start and end token indices matching the ground-truth span.
- **Efficiency metrics**:
  - **Actual deletion rate** (% of tokens removed after the gate).
  - Gate statistics (mean, std, final \(\alpha\)).
  - **Training time** per run (seconds).
  - **Latency** (ms/forward) from a dedicated benchmark script.

### 3.3 Experimental Details

- **Model configuration.** We use `bert-base-uncased` as the backbone, with the delete gate inserted after layer 3, \(k = -30\), target deletion ~50%, and hard-deletion threshold ratio 0.5. Max sequence length is 128 for classification tasks and 256 for TyDi QA.
- **Optimization.** We train with AdamW, learning rate \(2\times10^{-5}\), batch size 8, for **1 epoch** for all reported runs. For MrBERT, we either fix `gate_weight` or enable the PI controller to adjust \(\alpha\) online.
- **Hardware.** All experiments are run on a single **NVIDIA T4 GPU** VM on GCP. Latency is measured with `latency_benchmark.py` at batch size 16 and fixed sequence lengths.
- **Implementation.** Baselines use HuggingFace `BertForSequenceClassification` / `BertForQuestionAnswering`. MrBERT uses our custom `MrBertModel` and heads, sharing optimizer and scheduler settings with the baseline.

### 3.4 Results

#### 3.4.1 Classification performance

**Table 1: Classification results (1 epoch, batch 8)**

| Model         | Dataset | Deletion | Actual Del% | Val Acc | Avg Loss | Time (s) | Alpha   |
|---------------|---------|----------|-------------|---------|----------|----------|---------|
| Baseline BERT | MRPC    | 0%       | 40.45%      | 69.85%  | 0.6039   | 356.6    | —       |
| MrBERT        | MRPC    | ~50%     | 74.45%      | 71.81%  | 0.5949   | 364.8    | 0.00e+00 |
| Baseline BERT | SNLI    | 0%       | 65.27%      | 37.56%  | 0.9766   | 1195.4   | —       |
| MrBERT        | SNLI    | ~50%     | 80.71%      | 69.74%  | 0.8330   | 1106.4   | 0.00e+00 |
| Baseline BERT | TYDIQA  | 0%       | 13.82%      | 1.28%   | 4.6695   | 308.3    | —       |
| MrBERT        | TYDIQA  | ~50%     | 51.18%      | 0.00%   | 4.5464   | 252.5    | 1.74e-02 |
| Baseline BERT | IMDB    | 0%       | 4.40%       | 86.60%  | 0.4049   | 878.3    | —       |
| MrBERT        | IMDB    | ~50%     | 65.65%      | 80.16%  | 0.4256   | 841.2    | 0.00e+00 |
| Baseline BERT | SST-2   | 0%       | 0.03%       | 85.32%  | 0.2290   | 1756.3   | —       |
| MrBERT        | SST-2   | ~50%     | 68.90%      | 89.11%  | 0.2206   | 1755.1   | 0.00e+00 |

**Analysis.**

- **MRPC** and **SST-2**: MrBERT slightly **improves accuracy** (e.g., 69.85% → 71.81% on MRPC and 85.32% → 89.11% on SST-2) while deleting 70%+ of tokens, suggesting that aggressive token merging acts as a useful regularizer on shorter classification tasks.
- **SNLI**: Our 1-epoch baseline is weak (37.56%), but MrBERT reaches 69.74% with ~81% deletion—a large gain over the baseline. We hypothesize that the delete gate acts as an **information bottleneck**, aggressively down-weighting redundant premise/hypothesis tokens and forcing the model to focus on core entailment cues earlier in training. The comparison is somewhat unfair due to the undertrained baseline; strengthening the baseline is part of our planned work.
- **IMDB**: On long documents, MrBERT **trades accuracy for deletion**: it retains a reasonable 80.16% but underperforms the 86.60% baseline. This suggests that if the gate is not carefully tuned, it can prematurely merge sentiment-bearing tokens in long reviews.
- **TyDi QA**: Both baseline and MrBERT have very low EM (≤1.3%). MrBERT reaches the target deletion rate (~51%) but EM collapses to 0%. We believe this is due to a combination of: (1) only 1 epoch of training on a challenging QA task, and (2) hard deletion occasionally removing answer tokens. In addition, our current span head does **not** remap predicted start/end indices from the shortened sequence back to original document indices, which can further hurt EM. These runs primarily validate our QA pipeline and highlight TyDi QA as a clear failure mode that we plan to address.

#### 3.4.2 Latency and efficiency

Using `latency_benchmark.py`, we evaluate inference latency on GPU (batch 16):

**Table 2: Latency vs. sequence length (GPU, T4)**

| Setting          | Seq length | Avg time (ms) |
|------------------|------------|---------------|
| Baseline BERT    | 256        | 186.86        |
| MrBERT (soft)    | 256        | —             |
| MrBERT (hard)    | 120        | 132.72        |

Soft deletion (training mode) preserves the full sequence length to keep gradients flowing through all tokens and therefore does not yield latency gains; it is used only for optimization. MrBERT with **hard deletion** (inference mode) achieves a **29% speedup** over the baseline at the same batch size (256 → 120 effective tokens). This matches our theoretical MACs savings from shortening later layers and demonstrates that, once we actually run only on kept tokens, dynamic token merging can translate into real wall-clock efficiency gains on GPU.

---

## 4 Future Work

1. **Strengthen baselines and training.**  
   Train more epochs and tune hyperparameters on SNLI, IMDB, SST-2, and especially TyDi QA. Sweep target deletion ratios (e.g., 0.3–0.7) and plot **accuracy vs. compute (MACs)** curves to identify task-specific Pareto-optimal trade-offs between efficiency and performance.

2. **Error analysis.**  
   Use `scripts/extract_error_cases.py` to collect “wrong + high-deletion” examples on SNLI and TyDi QA. Compare deletion rates and token categories (word, subword, punctuation, special) between correct and incorrect predictions to understand when merging is helpful vs. harmful.

3. **Ablations and robustness.**  
   Vary the **gate layer** (e.g., layers 1, 3, 6) and gate strength to justify our design choices. Evaluate robustness to noise (e.g., misspellings, subword perturbations) and test whether the gate learns to drop noisy or uninformative tokens first.

4. **TyDi QA improvements.**  
   Improve preprocessing and training schedule (more epochs, better learning rate schedule) and, critically, add a **coordinate remapping** mechanism to translate predicted spans on the shortened sequence back to original document indices. This should address the current EM collapse under hard deletion.

5. **Efficiency improvements.**  
   Extend latency measurements to more sequence lengths and batch sizes, and further optimize the hard-deletion path (e.g., avoiding redundant computation before deletion, fusing operations) so that the observed 29% speedup can be maintained or improved across tasks and deployment settings.

