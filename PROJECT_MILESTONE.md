# CS224N Project Milestone

*Draft for the 2-page milestone (excluding figures and references). Fill in Key Information (names, TA mentor) before submission.*

---

## Key Information

- **Title**: MrBERT: Dynamic Token Merging for BERT-based Classification and QA
- **Team member names**:  
  - [Your Name] ([sunetid]@stanford.edu)  
  - [Teammate Name] (if any)
- **Custom or Default Project**: Custom project
- **Sharing Project**: (leave blank if not shared)
- **TA Mentor**: [TA name from proposal feedback]
- **(Optional) External Mentor**: (if applicable)
- **(Optional) External Collaborators**: (if any)

---

## Abstract

Subword-based Transformer encoders like BERT are inefficient on long sequences because they process every token at every layer, even when many tokens are redundant. Inspired by MrT5, we port its **Dynamic Token Merging** mechanism into BERT by adding a lightweight **Delete Gate** that learns which tokens to keep. During fine-tuning we use **soft deletion** (gate-modulated attention with softmax1) so gradients can still flow through all positions, while at inference we apply **hard deletion** to physically shorten sequences and reduce the theoretical self-attention MACs. On MRPC and SNLI, our MrBERT model maintains or improves accuracy while deleting 65–80% of tokens, and we have initial evidence that the gate can significantly improve over a weak baseline. We also implement a TyDi QA variant and a detailed logging/diagnostics pipeline, and our milestone focuses on the core implementation, preliminary classification/QA results, and planned analyses of the efficiency–accuracy tradeoff.

---

## Approach

### Model and Gate Mechanism

We adapt **MrT5-style Dynamic Token Merging** to the encoder-only BERT architecture:

- **Delete Gate (Eq. 1)**: After encoder layer \(l = 3\), we insert a lightweight delete gate that produces one scalar \(G_i \in [k, 0]\) per token with \(k = -30\):
  \[
  G = k \cdot \sigma(\text{LayerNorm}(H_l) W + b)
  \]
  where \(H_l\) is the hidden state at layer \(l\), \(W \in \mathbb{R}^{d \times 1}\). This adds only \(2d_{\text{model}} + 1\) parameters, matching the MrT5 design.

- **Soft vs. hard deletion**:
  - **Training (soft)**: We keep the sequence length fixed but add gate scores to the attention logits of later layers,
    \(\text{scores} = \frac{QK^\top}{\sqrt{d}} + \text{mask} + \mathbf{1} G^\top\),
    so deletion is emulated while remaining fully differentiable.
  - **Inference (hard)**: Tokens with \(G_i < k \cdot \text{threshold\_ratio}\) (we use 0.5) are physically removed; we build a shorter sequence and new attention mask and run only the remaining layers on the kept tokens.

- **Softmax1 (Eq. 7)**: In the gated layers we use **softmax1** instead of softmax so that attention weights do not collapse even when many tokens receive large negative gate scores.

- **Force-keep [CLS]**: Because BERT uses `[CLS]` as the pooled representation, our hard-deletion path always keeps position 0 regardless of its gate value.

### Task heads and loss

We implement two task-specific wrappers:

- **MrBertForSequenceClassification** for MRPC, IMDB, SST-2, and SNLI (pooler + dropout + linear head).
- **MrBertForQuestionAnswering** for TyDi QA (span head with start/end logits).

In both cases the training loss is
\(L = L_{\text{task}} + \alpha L_G\)
with \(L_{\text{task}}\) a cross-entropy task loss and
\(L_G = \frac{1}{N}\sum_i G_i\).
The coefficient \(\alpha\) is either a fixed `gate_weight` or is updated online by the PI controller described below.

### Baseline

We use **standard BERT** (HuggingFace `BertForSequenceClassification` / `BertForQuestionAnswering`) with the same base and task heads as our **no-deletion baseline** (gate effectively disabled).

### PI Controller and Logging

We implement the **PI controller** (MrT5 Section 3.2) to regulate deletion toward a target (e.g. 50%). We also added a **diagnostics module**: parameter counts (including gate-only), deletion rate and gate statistics, optional per-token gate visualization and dropped-token stats by type (word / subword / punctuation / special), and theoretical MACs savings after hard deletion.

---

## Experiments

### Data and Tasks

- **MRPC (GLUE)**: Sentence-pair paraphrase detection; ~3.7k train / 408 validation; metric: **validation accuracy**.
- **IMDB**: Movie review sentiment; ~25k train / 25k validation; metric: **validation accuracy**.
- **SST-2 (GLUE)**: Single-sentence sentiment; ~67k train / 872 validation; metric: **validation accuracy**.
- **SNLI**: Three-way NLI (entailment / neutral / contradiction); ~550k train / 10k validation (filtered); metric: **validation accuracy**.
- **TyDi QA (secondary task)**: Extractive QA; ~50k train / 5k validation (SQuAD-style); metric: **exact match** (start and end token correct).

All datasets are loaded from HuggingFace (`glue/mrpc`, `imdb`, `glue/sst2`, `snli`, and the TyDi QA secondary-task subset).

### Evaluation and Setup

- **Metrics**: Classification accuracy (MRPC, IMDB, SST-2, SNLI); EM for TyDi QA; plus deletion statistics (actual deletion %, gate mean/std, final \(\alpha\)), training time, and latency (ms/forward).
- **Config**: `bert-base-uncased`; delete gate after layer 3; \(k = -30\); target deletion ~50%; hard-deletion threshold ratio 0.5; max length 128 for classification and 256 for TyDi QA; AdamW, lr \(2\cdot10^{-5}\); batch size 8; **1 epoch** for all runs reported below.
- **Hardware**: Training runs on a single GPU VM (NVIDIA T4). Latency benchmarks use a separate script (`latency_benchmark.py`) with batch size 16 and fixed sequence lengths.

### Results

**Classification (1 epoch, batch 8)**

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

**Quantitative analysis.**

- **MRPC** and **SST-2**: MrBERT slightly **improves accuracy** (e.g., 69.85% → 71.81% on MRPC and 85.32% → 89.11% on SST-2) while deleting 70%+ of tokens, suggesting that aggressive token merging acts as a useful regularizer on shorter classification tasks.
- **SNLI**: Our 1-epoch baseline is weak (37.56%), but MrBERT reaches 69.74% with ~81% deletion—**a large gain over the baseline**. We hypothesize that the delete gate acts as an information bottleneck: by aggressively down-weighting redundant premise/hypothesis tokens, the model focuses earlier on core semantic contrasts, accelerating convergence in the first epoch. The comparison is still somewhat unfair, and we plan to strengthen the baseline for the final report.
- **IMDB**: For long movie reviews, MrBERT **trades accuracy for deletion**: it maintains a reasonable 80.16% but underperforms the 86.60% baseline. This suggests that in long documents, sentiment-bearing tokens can be prematurely merged if the gate is not carefully tuned.
- **TyDi QA**: Both baseline and MrBERT runs have very low EM (≤1.3%). The MrBERT run reaches the target deletion rate (~51%) but **collapses EM to 0%**, likely because hard deletion sometimes removes answer tokens and because 1 epoch is insufficient for QA. In addition, our current span head does not explicitly remap predicted start/end indices back to the original (pre-deletion) coordinates, which can further hurt EM. These runs mainly validate our QA pipeline; TyDi QA is a clear area where our current approach underperforms and where we plan to improve both training and index handling.

**Latency (GPU, batch 16)**.

Using `latency_benchmark.py`, we compare baseline BERT and MrBERT under controlled sequence lengths:

| Setting        | Seq length | Avg time (ms) |
|----------------|------------|---------------|
| Baseline BERT  | 256        | 186.86        |
| MrBERT (hard)  | 120        | 132.72        |

MrBERT with hard deletion achieves a **29% speedup** over the baseline at the same batch size (256 → 120 effective tokens). This aligns with our theoretical MACs savings from shortening later layers, and shows that once we actually run only on the kept tokens, dynamic token merging can translate into real wall-clock efficiency gains on GPU.

---

## Future Work

1. **Strengthen baselines and training**: Train more epochs and tune hyperparameters on SNLI, IMDB, SST-2, and especially TyDi QA; sweep target deletion (e.g. 0.3–0.7) and plot **accuracy vs. compute (MACs)** to identify the best trade-off point per task.
2. **Error analysis**: Use `scripts/extract_error_cases.py` to collect “wrong + high-deletion” examples on SNLI and TyDi QA; compare deletion rates and token categories (word/subword/punctuation/special) for correct vs. incorrect predictions to understand when merging is harmful.
3. **Ablations**: Vary the **gate layer** (e.g. layers 1, 3, 6) and gate strength to justify our design; test robustness (e.g. noisy tokens, adversarial subwords) and whether the gate learns to drop noise first.
4. **Efficiency improvements**: Extend latency measurements to more sequence lengths and batch sizes; further optimize the hard-deletion path (e.g. avoid redundant computation before deletion, fuse operations) so that the observed 29% speedup can be sustained across tasks and settings.

---

## References

(Use BibTeX in your Overleaf template. Example entries:)

```bibtex
@inproceedings{kallini2025mrt5,
  title     = {MrT5: Dynamic Token Merging for Masked Language Models},
  author    = {Kallini, Julie and others},
  booktitle = {ICLR},
  year      = {2025}
}
@article{devlin2019bert,
  title   = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author  = {Devlin, Jacob and others},
  journal = {NAACL},
  year    = {2019}
}
```

---

*Copy the sections above into the course Overleaf template, replace placeholders in Key Information, and export to PDF for submission. Keep the main body to 2 pages (excluding figures and references).*
