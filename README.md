# MrBERT: Dynamic Token Merging on BERT

This repository implements **MrBERT**, an adaptation of the **dynamic token merging** mechanism from **MrT5 (MergeT5): Dynamic Token Merging for Efficient Byte-Level Language Models** (Kallini et al., ICLR 2025) onto **BERT**. The goal is to make BERT more efficient by learning to drop less important tokens after a fixed encoder layer while keeping the rest of the architecture unchanged.

---

## Paper Summary (MrT5)

- **Problem** (Section 1–2): Byte-level models like ByT5 have long sequences and are slow; subword tokenization is sensitive to spelling noise and has uneven compression across languages.
- **Idea** (Section 3): Insert a **delete gate** after a **fixed encoder layer** `l`. The gate outputs a score `G_i ∈ [k, 0]` per token (with `k = -30`). Tokens with `G_i` near `k` are treated as “deleted.”
  - **Training**: **Soft deletion** — add `G` to the attention logits of *subsequent* layers so that “deleted” positions are softly masked; the process stays differentiable (Section 3.1, Figure 1).
  - **Inference**: **Hard deletion** — remove token columns where `G_i` is below a threshold (e.g. `k/2 = -15`), then run the rest of the encoder on the shortened sequence to save compute (Section 3.1, Figure 1).
- **Formulas** (Section 3.1–3.2):
  - **Gate** (Eq. 1): `G = k · σ(LayerNorm(H_l) W + 1_N b)`  
    Only `2d_model + 1` extra parameters.
  - **Soft-deletion attention** (Eq. 2): `attention_scores = QK^T / √d + 1_N G^T`, then softmax (or softmax1) and multiply by `V`.
  - **Gate regularizer** (Eq. 3): `L_G = (1/N) Σ_i G_i`; total loss `L = L_CE + α · L_G`.  
    Larger `α` encourages more deletion.
  - **PI controller** (optional, Eq. 4–6): To hit a target deletion ratio `δ`, the paper updates `α` with a proportional–integral rule.
  - **Softmax1** (Eq. 7): `softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))`.  
    Used so that when all `G_i = k`, the attention weights do not collapse.
- **Gate placement** (Section 3, 7): A single gate is placed after one fixed layer (e.g. layer 3 in the paper) to limit overhead and to allow early layers to build context before deletion (Figure 4).

---

## What We Changed (and Why, with Paper References)

Below is a concise list of **what was implemented** and **where in the paper** it comes from.

### 1. Delete gate (paper Section 3.1, Eq. (1))

- **What**: After a fixed encoder layer `gate_layer_index` (default 3), we compute  
  `G = k · σ(LayerNorm(H) W + b)` with `k = -30`.
- **Where in code**: `mrbert/modeling_mrbert.py` — `DeleteGate` class; `MrBertModel._encoder_forward_with_gate` calls it after the first `gate_layer_index + 1` BERT layers.
- **Why**: Paper Eq. (1). We use standard LayerNorm (BERT uses it); the paper uses RMSNorm in T5.

### 2. Soft deletion during training (paper Section 3.1, Eq. (2))

- **What**: For encoder layers *after* the gate, we do **not** call the normal BERT layer. Instead we compute Q, K, V from that layer, then  
  `attention_scores = QK^T/sqrt(d) + attention_mask + gate_broadcast`,  
  then softmax1 (or softmax) and multiply by V, then pass through the rest of the layer (output projection, residual, FFN).
- **Where in code**: `_run_layer_with_soft_gate()` in `modeling_mrbert.py`; it is used for layers `gate_layer_index+1` through the last when `use_soft_deletion=True`.
- **Why**: Paper Eq. (2): the gate is added to the key dimension so that positions with `G_j ≈ k` get strongly down-weighted. We broadcast `G` as `(batch, 1, 1, seq_len)` so it is added to every query position.

### 3. Hard deletion at inference (paper Section 3.1)

- **What**: After the gate layer, we compute a boolean mask “keep” where `G > k · gate_threshold_ratio` (default 0.5, i.e. `k/2 = -15`). We keep only those positions, pad to a common length per batch, and run the remaining BERT layers on this shortened sequence with a new attention mask.
- **Where in code**: `_encoder_forward_with_gate()` when `use_soft_deletion=False` (e.g. `model.eval()`): we slice hidden states and build a new attention mask for the kept positions.
- **Why**: Paper Section 3.1 and Figure 1: “hard deletion via column removal” at test time to reduce sequence length and gain speed.

### 4. Gate regularizer (paper Section 3.2, Eq. (3))

- **What**: Loss term `L_G = (1/N) Σ_i G_i`. We add `α · L_G` to the cross-entropy loss; `α` is set by the user (e.g. `gate_regularizer_weight=1e-4`) or by the PI controller.
- **Where in code**: `MrBertModel.get_gate_regularizer_loss(gate)`; `MrBertForSequenceClassification.forward(..., gate_regularizer_weight=...)` adds it to the total loss.
- **Why**: Paper Eq. (3): “encourages them to be more negative (i.e. closer to k)”, so the model learns to delete more when `α` is larger.

### 5. Softmax1 (paper Section 3.2, Eq. (7))

- **What**: In layers that use the gate, we use  
  `softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))` instead of standard softmax on attention scores.
- **Where in code**: `softmax1()` in `modeling_mrbert.py`; used in `_run_layer_with_soft_gate()` when `use_softmax1=True`.
- **Why**: Paper Section 3.2: if all `G_i = k`, adding the same constant to every logit does not change standard softmax; with softmax1, very negative scores yield near-zero attention and avoid the “all deleted” failure.

### 6. PI controller for target deletion ratio (paper Section 3.2, Eq. (4)–(6))

- **What**: To steer the fraction of deleted tokens toward a target `δ`, we update `α` each step using a proportional–integral rule with an exponential moving average on the P term.
- **Where in code**: `mrbert/pi_controller.py` — `PIController.step(gate, gate_k)`; used in `run_mrbert_example.py` and `train_mrbert.py` when `--use_pi` is set.
- **Why**: Paper Eq. (4)–(6): “optimize for a specific ratio of deleted tokens” without hand-tuning `α`.

### 7. Config and gate placement (paper Section 3, 7)

- **What**: `MrBertConfig` adds: `gate_layer_index` (default 3), `gate_k` (-30), `gate_threshold_ratio` (0.5), `use_softmax1` (True). The gate is placed after layer `gate_layer_index` (0-based).
- **Where in code**: `mrbert/configuration_mrbert.py`; used everywhere the model is built.
- **Why**: Paper Section 3 uses one fixed layer; Section 7 / Figure 4 reports layer 3 as a good trade-off for their setup. We keep the same default for BERT.

### 8. BERT-specific choices (no direct paper equivalent)

- **Encoder-only**: BERT has no decoder/cross-attention. We only modify **encoder** self-attention (add gate in subsequent layers). No cross-attention gate.
- **Layer output type**: In transformers 5.x, `BertLayer.forward` returns a single tensor, not a tuple. We use `hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs` so it works for both.
- **Attention implementation**: We use `attn_implementation="eager"` when loading so that our custom attention path (with gate and softmax1) is used; SDPA would require different mask handling.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `transformers`, and (for training scripts) `datasets`.

---

## Usage

### 1. Load pretrained BERT as MrBERT and train with gate regularizer

```python
from mrbert import MrBertForSequenceClassification

model = MrBertForSequenceClassification.from_bert_pretrained(
    "bert-base-uncased",
    num_labels=2,
    gate_layer_index=3,
    gate_k=-30.0,
    attn_implementation="eager",
)
model.train()
out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    gate_regularizer_weight=1e-4,
)
loss = out["loss"]
```

### 2. MrBertModel only (e.g. for custom heads)

```python
from mrbert import MrBertModel

model = MrBertModel.from_pretrained_bert("bert-base-uncased", gate_layer_index=3)
outputs, gate = model(input_ids=..., attention_mask=..., return_gate=True, use_soft_deletion=True)
gate_loss = model.get_gate_regularizer_loss(gate)
```

### 3. Inference with hard deletion (faster)

```python
model.eval()
with torch.no_grad():
    outputs = model(input_ids=..., attention_mask=..., return_gate=False)
# use_soft_deletion=False is used automatically when model is in eval mode
```

### 4. Train on a real dataset (MRPC / IMDB / SST-2)

These datasets are **not** from the MrT5 paper; they are common English binary classification benchmarks used here for convenience. The paper evaluates on XNLI, TyDi QA, and character-level tasks (Spelling Correction, Word Search).

```bash
# MRPC (sentence-pair paraphrase, ~3.7k train)
python train_mrbert.py --dataset mrpc --epochs 3 --batch_size 16 --gate_weight 1e-4
# With PI controller for target deletion ratio 0.5
python train_mrbert.py --dataset mrpc --epochs 3 --use_pi --target_deletion 0.5
# IMDB sentiment
python train_mrbert.py --dataset imdb --epochs 1 --batch_size 16
```

### 5. PI controller in your own training loop

```python
from mrbert.pi_controller import PIController

controller = PIController(target_ratio=0.5, kp=0.5, ki=1e-5)
# After each step:
gate_regularizer_weight = controller.step(gate, gate_k=-30.0)
```

### 6. Inspect data (see what is loaded for training)

```bash
python inspect_data.py --dataset mrpc
```

### 7. Run all experiments and update results table

```bash
# Run baseline + MrBERT on MRPC, IMDB (and optionally SNLI). Results appended to results/train_results.jsonl.
./run_experiments.sh

# Optional: skip SNLI (faster)
SKIP_SNLI=1 ./run_experiments.sh

# Regenerate RESULTS.md (reads train_results.jsonl + results/latency_results.json if present)
python scripts/aggregate_results.py

# Optional: run latency benchmark and write to results/latency_results.json (then re-run aggregate_results)
python latency_benchmark.py --output_result results/latency_results.json
```

### 8. Gate interpretability (which tokens are deleted?)

```bash
# Per-token gate score and KEEP/DEL
python gate_interpretability.py

# Deletion rate by token type (word / subword / punctuation)
python gate_interpretability.py --stats
```

---

## Report outline / Code walkthrough

For a written or oral report, you can structure the implementation as follows:

1. **DeleteGate placement**  
   The gate is inserted after **layer 3** (0-based). The first 4 layers run normally; then we compute `G = k·σ(LayerNorm(H)W+b)` and use it in all subsequent layers. Early placement maximizes compute savings (paper Section 3, Figure 4).

2. **Soft vs hard deletion**  
   **Training**: `use_soft_deletion=True` — gate is added to attention logits; sequence length unchanged; fully differentiable.  
   **Inference**: `use_soft_deletion=False` — tokens with `G < k/2` are removed; hidden states and attention mask are shortened; rest of the encoder runs on fewer tokens.

3. **Softmax1**  
   In gated layers we use `softmax1` instead of softmax so that when all gates are `k`, attention does not collapse (paper Eq. (7)).

---

## Project Structure

| Path | Role |
|------|------|
| `mrbert/configuration_mrbert.py` | `MrBertConfig`: extends `BertConfig` with `gate_layer_index`, `gate_k`, `gate_threshold_ratio`, `use_softmax1`. |
| `mrbert/modeling_mrbert.py` | `DeleteGate`, `MrBertModel` (soft/hard deletion in encoder), `MrBertForSequenceClassification`; gate regularizer and optional gate return. |
| `mrbert/pi_controller.py` | PI controller for target deletion ratio (paper Eq. (4)–(6)). |
| `run_mrbert_example.py` | Minimal demo: 2 sentences, 3 steps, no real dataset. |
| `train_mrbert.py` | Training on HuggingFace datasets (mrpc / imdb / sst2 / snli). |
| `run_experiments.sh` | Run baseline + MrBERT on MRPC, IMDB, SNLI; append results to `results/train_results.jsonl`. |
| `scripts/aggregate_results.py` | Read JSONL and update `RESULTS.md` comparison table. |
| `gate_interpretability.py` | Per-token gate scores and deletion-by-token-type stats. |
| `inspect_data.py` | Print dataset description and sample rows. |
| `RESULTS.md` | Accuracy and latency comparison table (filled by aggregate_results.py). |
| `DATA_README.md` | Data source and label meaning for mrpc / imdb / sst2. |
| `dynamic_token.pdf` | MrT5 paper (ICLR 2025). |

---

## Datasets: Paper vs This Repo

- **Paper (MrT5)**: XNLI (cross-lingual NLI), TyDi QA (QA), Spelling Correction with Context, Word Search (character-level). No MRPC, IMDB, or SST-2.
- **This repo**: Training scripts use **MRPC, IMDB, SST-2** as easy-to-use binary classification benchmarks for BERT-style models. They are **not** the paper’s benchmarks; they are only for validating the implementation and running examples.

---

## References

- **Paper**: Kallini et al., *MrT5: Dynamic Token Merging for Efficient Byte-Level Language Models*, ICLR 2025.
- **Official MrT5 code**: https://github.com/jkallini/mrt5
# standford_cs224n_project
