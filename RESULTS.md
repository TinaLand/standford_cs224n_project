# Experiment Results

Comparison: **Baseline BERT** (no gate) vs **MrBERT** (gate + PI, target deletion ~50%).

## Accuracy

| Model | Dataset | Deletion | Val Accuracy |
|-------|---------|----------|--------------|
| Baseline BERT | MRPC | 0% | *(run baseline to fill)* |
| MrBERT | MRPC | ~50% | 68.4% |
| Baseline BERT | IMDB | 0% | *(run baseline to fill)* |
| MrBERT | IMDB | ~50% | 84.8% |

*After running `./run_experiments.sh` and then `python scripts/aggregate_results.py`, this table is updated from `results/train_results.jsonl`.*

## Latency (from latency_benchmark.py)

| Setting | Seq length (after gate) | Avg time (CPU) |
|---------|-------------------------|----------------|
| Soft deletion (train) | 512 | — |
| Hard deletion (eval)  | 275 | ~1504 ms |
| Baseline BERT         | 512 | ~1322 ms |

On CPU, MrBERT hard-deletion is currently slower due to custom attention path; GPU benchmark expected to show speedup.
