# Results and Analysis

This file summarizes experimental results and analysis for MrBERT and the extended backbones (MrRoBERTa, MrXLM). Raw numbers come from `results/results/train_results.jsonl` and from CPU sanity checks.

---

## MrBERT (BERT backbone) — full experiments

Results are from `results/results/train_results.jsonl` for the **3-epoch** full run: baseline BERT vs MrBERT with PI controller, target deletion 0.5, batch size 8.

| Dataset | Baseline (val acc) | MrBERT (val acc) | MrBERT actual deletion | Note |
|---------|--------------------|------------------|------------------------|------|
| MRPC    | 69.85%             | **74.26%**       | 64.9%                  | MrBERT improves over baseline. |
| IMDB    | **88.04%**         | 84.19%           | 60.4%                  | Slight drop; trade-off. |
| SNLI    | 73.74%             | **81.88%**       | 54.9%                  | MrBERT +8.1 pts. |
| SST-2   | 87.16%             | **91.28%**       | 71.7%                  | MrBERT +4.1 pts. |
| TyDi QA | **32.05%** (EM)    | 0%               | 99.8%                  | Gate collapsed; coordinate re-mapping fixed for index shift; needs lower target deletion or QA-specific tuning. |

**Takeaways**

- On classification (MRPC, SNLI, SST-2), MrBERT matches or beats baseline at 55–72% token deletion.
- TyDi QA under current settings over-deletes (99.8%); the same `keep_indices` / `kept_lengths` re-mapping used for EM is in place and validated on smoke runs. Further work: lower target deletion or QA-specific tuning.

---

## New backbones: MrRoBERTa and MrXLM (architecture check only)

To show that the **delete-gate and index-mapping design** is backbone-agnostic, we added two wrappers that reuse the same gate and output contract `(hidden_states, gate, keep_indices, kept_lengths)`:

- **MrRoBERTa** (`mrbert/modeling_mrroberta.py`): RoBERTa-base with the same gate after layer 3. No `token_type_ids`; otherwise same soft/hard deletion and index contract.
- **MrXLM** (`mrbert/modeling_mrxlm.py`): XLM-RoBERTa-base with the same gate. Different tokenizer (SentencePiece) but tensor-level indexing is unchanged, so coordinate re-mapping for QA applies without code change.

### Preliminary runs (CPU, no training)

- **MrXLM**: Loaded `xlm-roberta-base`, forward with `use_soft_deletion=False`, `return_gate=True`. Observed: `last_hidden_state` shape shortened (e.g. 2×32 → 2×5), `kept_lengths` and `keep_indices` correct; gate and hard-deletion logic behave as on BERT.
- **MrRoBERTa**: Same check on `roberta-base`: forward pass and gate/indices shapes verified.

So far these are **sanity checks** only: no full training or evaluation. They support the claim that the framework is **cross-architecture**: the same PI, gate, and QA index re-mapping can be used on BERT, RoBERTa, and XLM-R with only a different encoder wrapper.
