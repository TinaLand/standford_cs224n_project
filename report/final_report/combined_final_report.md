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

Transformer-based language models apply uniform computation to every input token, regardless of token informativeness. The MrT5 paper [Kallini et al., 2024] introduced a Dynamic Token Merging (DTM) delete gate for encoder–decoder models, achieving substantial sequence reduction with minimal quality degradation. **We ask: does this mechanism generalise to subword-based encoder-only architectures used for discriminative NLU?** We implement MrBERT and MrXLM, adapting the MrT5 delete gate to BERT-base and XLM-RoBERTa-base respectively. Our gate fires after encoder layer 3 (out of 12), uses a PI controller to track a target deletion ratio, and relies on coordinate remapping for extractive QA under hard deletion. Evaluated across six tasks—MRPC, SST-2, SNLI, IMDB, XNLI, and TyDi QA—MrBERT consistently matches or improves validation accuracy over BERT while deleting 50–75% of tokens on several runs. A GPU latency benchmark on a T4 shows hard deletion yields **30–55%** inference speedup at fixed batch size. We perform per-example loss–deletion analysis and error case study, showing that higher deletion often correlates with higher loss on difficult examples and that over-deletion can remove answer-bearing tokens. MrXLM works well on MRPC but is less robust on SST-2, SNLI, IMDB, and XNLI, motivating gate placement, warmup, and threshold as rescue strategies. Overall, dynamic token merging is a promising direction for encoder efficiency but requires careful control and task-aware tuning.

---

## 1 Introduction

Transformer encoders such as BERT and XLM-R are the backbone of many modern NLP systems, but they remain computationally expensive. For a sequence of \(n\) tokens and \(L\) layers, a standard encoder performs \(O(L n^2)\) attention operations: every token attends to every other token at every layer. In practice, many tokens (e.g., stopwords, punctuation, subwords inside multi-token words) contribute little to downstream predictions, especially after the model has already integrated local context in earlier layers. This suggests a natural question: **can we learn to drop tokens dynamically, without sacrificing too much accuracy?**

The MrT5 paper [Kallini et al., 2024] proposes Dynamic Token Merging (DTM): a learned scalar gate inserted after an early encoder layer that assigns each token a deletion score. Low-scoring tokens are masked out of subsequent attention (soft deletion) or physically removed (hard deletion). MrT5 demonstrates this on a byte-level T5 model with minimal perplexity cost. However, MrT5 focuses on encoder–decoder models and does not explore pure encoders like BERT or multilingual encoders like XLM-R that are widely used for classification and extractive QA.

**The transfer to encoder-only models is non-trivial** because: (1) BERT and XLM-R use **subword** tokenisation (WordPiece / SentencePiece), so token granularity and importance distributions differ from byte-level models. (2) Encoder-only models pool to a single **[CLS]** vector for classification, creating a different information-flow bottleneck than encoder–decoder models. (3) **Extractive QA** requires localising answer spans—predicted spans must be remapped to the original tokenisation for EM evaluation. (4) **XLM-R** spans many languages with a shared SentencePiece vocabulary; importance patterns must generalise across scripts.

In this project we implement and adapt the MrT5 delete gate for **encoder-only models**, building **MrBERT** and **MrXLM**. Our goal is two-fold: **Efficiency**—achieve real wall-clock speedups by shortening sequences via hard deletion at inference; **Accuracy**—maintain or improve task performance. Our main contributions are: MrBERT for English classification (MRPC, SST-2, SNLI, IMDB, XNLI) and TyDi QA with a delete gate after layer 3, soft deletion during training, and hard deletion at inference; a **PI controller** that stabilises deletion rate vs fixed \(\alpha\); extension to **XLM-R** (MrXLM) with analysis and rescue strategies (later gate layer, slower warmup, higher threshold); and detailed results analysis including Pareto frontier (Figure A), task sensitivity heatmap (Figure D), per-example loss–deletion correlations (Figure G), deletion histograms (Figure F), TyDi QA curve (Figure H), and PI vs fixed-\(\alpha\) traces (Figure B).

---

## 2 Related Work

**MrT5 [Kallini et al., 2024].** The direct antecedent of this work. MrT5 attaches a scalar delete gate after a selected encoder layer of a T5-based byte-level model. A PI controller adjusts the deletion loss coefficient \(\alpha\) to track a target deletion rate. The gate adds fewer than 3K parameters, achieving 50–60% sequence reduction on language modelling with minimal degradation. We implement and adapt this mechanism for encoder-only BERT and XLM-R.

**Efficient Transformers.** Sparse attention (Longformer, BigBird), low-rank approximations (Linformer), and kernelization (Performer) redesign attention but generally keep sequence length fixed. Dynamic token merging reduces **sequence length** by dropping tokens. Early-exit approaches (DeeBERT, PABEE) skip later layers; our method keeps depth fixed and reduces tokens in later layers.

**Token pruning and merging.** Token pruning methods (TR-BERT, SpAtten, LTP) learn to drop tokens based on attention scores or importance weights, often with RL or auxiliary losses. Unlike these, our method is trained end-to-end with differentiable soft deletion and requires no architectural changes beyond the gate module.

**Token importance in BERT.** Michel et al. [2019] and Voita et al. [2019] showed many BERT attention heads are redundant. Clark et al. [2019] found BERT attention aligns with syntactic structure; function words receive less attention in higher layers, consistent with our gate’s deletion patterns (Section 5.1–5.2).

**Subword regularization.** Prior work on BPE-dropout and data augmentation shows that subword-level perturbations can improve robustness. Our delete gate can be viewed as a structured, learned perturbation that removes low-utility subwords, potentially acting as a regulariser.

Our project differs by (1) applying dynamic token merging to encoder-only BERT and XLM-R, (2) exploring a PI controller in this setting, and (3) performing a detailed multi-task analysis including cross-lingual QA.

---

## 3 Approach

### 3.1 Architecture overview

MrBERT is a pretrained BERT-base encoder with a lightweight delete gate inserted after a selected layer (default: layer 3). The gate uses hidden states from that layer to compute a scalar score per token; the score is used to mask tokens in all subsequent layers. Encoder layers 0–2 run on the full sequence; the gate fires after layer 3; encoder layers 4–11 run on the reduced sequence (soft deletion: same length, masked attention; hard deletion: shortened sequence). The task head uses [CLS] pooling for classification or span logits for QA. BERT-base: 12 layers, \(d=768\), 110M parameters, WordPiece 30K. XLM-RoBERTa-base: same dimensions, 277M parameters, SentencePiece 250K.

### 3.2 Delete gate and soft vs hard deletion

The gate has three components: (1) **LayerNorm** on the hidden state for stability; (2) **linear projection** \(z_i = W h_i + b\) with \(W \in \mathbb{R}^{d \times 1}\); (3) **scaled sigmoid** \(G_i = k \cdot \sigma(z_i)\) with \(k = -30\). Gate values lie in \((k, 0)\): near 0 means keep, near \(k\) means delete. Deletion threshold \(\tau = k/2 = -15\). The gate adds only **2,305 parameters** (LayerNorm 2×768 + linear 768+1). [CLS] and the first token in XLM-R are always kept.

**Soft deletion (training).** Sequence length stays fixed. We add gate scores as an attention bias: \(\tilde{a}(q, k_i) = a(q, k_i) + G_i\). Deleted tokens contribute negligible weight; gradients flow through all tokens. We use **softmax1** [Kallini et al., 2024]: \(\mathrm{softmax1}(x)_i = \exp(x_i) / (1 + \sum_j \exp(x_j))\), so attention weights do not collapse when gate values are equal.

**Hard deletion (inference).** We remove positions where \(G_i < \tau\), build `keep_indices` and `kept_lengths`, and run remaining layers and the task head on the shortened sequence. This yields real memory and compute savings.

### 3.3 Loss and PI controller

Total loss: \(L = L_{\text{task}} + \alpha L_G\), where \(L_G = \frac{1}{N}\sum_i G_i\) over non-padding tokens. We support **fixed \(\alpha\)** or a **PI controller**: \(e_t = \tau - r_t\), \(p_t = \gamma p_{t-1} + (1-\gamma) k_p e_t\), \(i_t = i_{t-1} + k_i e_t\), \(\alpha_t = \max(0, k_p p_t + k_i i_t)\). Defaults: \(k_p = 0.5\), \(k_i = 10^{-5}\), \(\gamma = 0.9\) (or as in MrT5: \(k_p = 0.01\), \(k_i = 10^{-5}\)). The controller drives the actual deletion rate toward the target \(\tau\).

### 3.4 BERT and XLM-R task heads; QA span remapping

We use [CLS] (or first token) pooling + classifier for MRPC, IMDB, SST-2, SNLI, XNLI; span head for TyDi QA. For QA, we propagate `keep_indices` and `kept_lengths` and **remap** predicted start/end indices from the shortened sequence back to original token indices before computing EM. This coordinate remapping is essential for meaningful TyDi QA evaluation under hard deletion.

### 3.5 XLM-R–specific design and rescue strategies

XLM-R uses SentencePiece, yielding finer subwords than BERT’s WordPiece; deleting one of several subwords per word can disrupt semantics. We expose **rescue strategies**: `--gate_layer_index` (e.g. layer 6), `--controller_kp` / `--controller_ki`, `--gate_warmup_steps` (e.g. 3000), `--gate_threshold_ratio` (e.g. 0.6–0.7).

### 3.6 Key differences from MrT5

| Aspect | MrT5 | MrBERT / MrXLM |
|--------|------|----------------|
| Architecture | Encoder–decoder (T5) | Encoder-only |
| Tokenisation | Byte-level | WordPiece 30K / SentencePiece 250K |
| Gate fires | Before self-attention | After full encoder layer (attn + FFN) |
| QA under deletion | — | Span remapping via `keep_indices` |
| Softmax1 | Off by default | On by default |

Firing the gate after the full encoder layer gives it access to richer contextual representations (including FFN) before the deletion decision.

### 3.7 Implementation highlights

- **Soft vs hard:** `use_soft_deletion` flag; training = soft (attention bias), evaluation = hard (shortened sequence).  
- **QA remapping:** Compute keep mask → `keep_indices`, `kept_lengths` → `torch.gather` for shortened hidden states → remap span predictions via `keep_indices` before EM.  
- **Ragged tensors:** Use \(\mathrm{max\_kept}\) and padding so batched sequences have fixed shape; length mask for attention.  
- **Gate warmup:** \(\alpha = 0\) for first \(N\) steps; optional two-phase schedule.  
- **PI controller:** `mrbert/pi_controller.py`; `--log_deletion_trace` for Figure B.  
- **Loss–deletion pipeline:** `loss_vs_deletion_<dataset>.json`, `error_cases_*.jsonl` for analysis.  
- **Latency:** `latency_benchmark.py` + `plot_mrbert_figures.py` for Pareto (Figure A) and accuracy summary (Figure E).

---

## 4 Experiments

### 4.1 Datasets and evaluation

| Dataset | Task | Metric |
|---------|------|--------|
| MRPC | Binary paraphrase | Val accuracy |
| SST-2 | Single-sentence sentiment | Val accuracy |
| SNLI | 3-way NLI | Val accuracy |
| IMDB | Long-document sentiment | Val accuracy |
| XNLI | Multilingual NLI (MrXLM) | Val accuracy |
| TyDi QA | Extractive QA | EM |

Classification: max length 128; TyDi QA / XNLI: 256. HuggingFace datasets; BERT or XLM-R tokeniser.

### 4.2 Training setup

**Backbones:** bert-base-uncased, xlm-roberta-base. **Optimisation:** AdamW, lr \(2\times 10^{-5}\), batch 8 or 24, 1–5 epochs per run. **Deletion:** target ratios 0.3, 0.5, 0.7 (BERT); 0.3, 0.5 (XLM-R); gate warmup 1000–3000 steps. **Hardware:** BERT on NVIDIA L4; XLM-R on Modal A100. Latency measured on T4, batch 16, length 256.

### 4.3 Main quantitative results (BERT)

Results are from `results/new/bert_from_l4/` (see `RESULTS_ANALYSIS.md`). Representative findings:

- **Target 0.3, warmup 1000:** MRPC 68.38% → 68.63% (~54% del); SST-2 67.89% → 92.55% (~62% del); SNLI 74.00% → 89.02% (~76% del); XNLI 74.82% → 80.56% (~65% del); TyDi QA 20.18% → 28.16% EM (~11% del). Best overall trade-off.  
- **Target 0.5:** SNLI/SST-2/XNLI remain strong; TyDi QA slightly worse; IMDB brittle.  
- **Target 0.7:** SNLI, SST-2, XNLI still beat baseline at very high deletion; MRPC and IMDB degrade; TyDi QA collapses to 0% EM.

Figure A (Pareto frontier) and Figure E (accuracy summary) show MrBERT often on or near the Pareto frontier across tasks.

![Figure A: Pareto frontier (BERT L4)](figures/fig_A_pareto.png)

![Figure E: Accuracy summary (baseline vs gated)](figures/fig_E_accuracy_summary.png)

### 4.4 Main quantitative results (XLM-R)

From `results/new/xlmr_from_A100/`: at target 0.5 (3 epochs), MRPC 67.16%, SST-2 52.41%, SNLI 50.74%, IMDB 85.76%, XNLI 68.07%. At target 0.3 (3 epochs, baseline vs MrXLM): MRPC 68.38% → 72.06%; SST-2 79.01% → 61.01%; IMDB 79.87% → 50.00%; XNLI 62.77% → 43.41%. MrXLM is much more fragile than MrBERT on SST-2/SNLI/IMDB/XNLI at similar settings. Figure D (task sensitivity heatmap) summarises this.

![Figure D: Task sensitivity (BERT, gated accuracy)](figures/fig_D_task_sensitivity.png)

### 4.5 Latency

T4 benchmark: baseline BERT ~95 ms/batch; MrBERT with hard deletion ~43 ms/batch in example runs (**~55% speedup**). Pareto frontier demonstrates 30–55% inference speedup at fixed batch size while maintaining or improving accuracy on several tasks.

---

## 5 Analysis

### 5.1 Per-example loss vs deletion

We compute per-example validation loss and deletion rate (`loss_vs_deletion_<dataset>.json`). Figure F (histograms) and Figure G (scatter) show loss–deletion relationship. **Loss–deletion correlation is often weaker on SST-2 than on MRPC** (SST-2 has higher redundancy). Key observations: positive Pearson on several MRPC runs (e.g. +0.064) and on XLM-R SST-2 0.3 run (~+0.195), indicating over-deletion hurts those examples. **Positive correlation supports the claim that over-deletion hurts individual examples** and reinforces the need for careful deletion-rate control.

![Figure F (MRPC): Deletion-rate histogram](figures/fig_F_deletion_hist_mrpc.png)

![Figure F (SST-2): Deletion-rate histogram](figures/fig_F_deletion_hist_sst2.png)

![Figure G (MRPC): Loss vs deletion](figures/fig_G_loss_vs_deletion_mrpc.png)

![Figure G (SST-2): Loss vs deletion](figures/fig_G_loss_vs_deletion_sst2.png)

### 5.2 Error analysis: TyDi QA and SNLI

High-deletion error cases (e.g. deletion rate \(\ge 0.7\)) often show answer-bearing tokens dropped (TyDi QA) or premise/hypothesis content heavily pruned leading to spurious contradiction (SNLI). This highlights that deletion can remove crucial evidence; span remapping alone cannot recover when answer tokens are deleted.

### 5.3 Task sensitivity and attention sparsification

Figure D and Figure H show: MRPC and SST-2 tolerate moderate–high deletion; SNLI and XNLI benefit in some regimes but degrade when target is too high; TyDi QA is very sensitive (0.7 runs → 0% EM). **Why accuracy can improve:** the gate performs **attention sparsification**, removing low-signal subwords and reducing interference; this bottleneck can act as a regulariser on classification. It is less beneficial or harmful when critical information is spread (QA, long documents).

![Figure H: TyDi QA vs target deletion (BERT)](figures/fig_H_tydiqa_sensitivity.png)

### 5.4 PI controller vs fixed \(\alpha\); necessity of PI

Figure B: with **PI**, deletion rates converge smoothly to the target; with **fixed \(\alpha\)**, they can oscillate or overshoot. **PI controller necessity:** without it, deletion becomes task-dependent and unpredictable (e.g. MRPC over-deletes ~68%, IMDB under-activates ~2.7%). The controller is necessary to steer toward the target and avoid both over-deletion and under-activation.

![Figure B: Deletion rate vs step (PI vs fixed alpha)](figures/fig_B_deletion_vs_step.png)

### 5.5 PI controller, task-specific dynamics, and IMDB instability

On long-document tasks (IMDB), we observe run-to-run instability in deletion rate and accuracy even with PI. A plausible explanation is that per-batch deletion ratios are noisier or evolve more slowly, so \(\alpha\) adapts more slowly and the gate can converge to different operating points. This supports **task- or domain-aware** tuning (e.g. longer warmup or lower \(k_p\) for long documents).

### 5.6 Comparison to MrT5

MrBERT tolerates 30–50% subword deletion with matched or improved accuracy on several classification tasks, despite subwords being more informationally dense than bytes—suggesting the mechanism is architecture-agnostic. Our T4 benchmark reports 30–55% inference speedup. MrXLM’s larger drop under the same settings indicates XLM-R needs different hyperparameters (longer warmup, later gate, higher threshold), consistent with SentencePiece granularity.

---

## 6 Conclusion

We demonstrated that the Dynamic Token Merging delete gate from MrT5 transfers to encoder-only discriminative NLU. **Key findings:**

1. **Efficiency–accuracy tradeoff:** MrBERT achieves competitive or better accuracy than BERT on MRPC, SST-2, SNLI, XNLI while deleting 50–75% of tokens, with **30–55% inference speedup** on a T4.  
2. **Task-dependent robustness:** SST-2 and SNLI are highly robust; MRPC and TyDi QA more sensitive; IMDB shows run-to-run instability.  
3. **PI controller is essential:** Without it, deletion is unpredictable; the controller steers the gate toward the target.  
4. **Per-example analysis:** Positive loss–deletion correlation and error cases show over-deletion can hurt individual examples and remove answer-bearing or evidence tokens.  
5. **XLM-R requires tuning:** Same settings as BERT are too aggressive; rescue strategies (later gate, higher threshold, longer warmup) and SentencePiece granularity are discussed.

**Limitations:** Short training schedules (often one epoch), weak baselines on some datasets, incomplete XLM-R design-space exploration. We do not report a random-deletion baseline or SQuAD in the current `results/new` runs.

**Future work:** Stronger baselines; more flexible deletion controllers; span-aware protections for QA; random-deletion ablation; additional languages and domains.

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
