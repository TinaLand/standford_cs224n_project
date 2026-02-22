# Experiment Results

Comparison: Baseline BERT (no gate) vs MrBERT (gate + PI, target deletion ~50%).

| Model | Dataset | Deletion | Actual Del% | Val Acc | Avg Loss | Time(s) | Alpha |
|-------|---------|----------|-------------|--------|----------|---------|-------|
| Baseline BERT | MRPC | 0% | 40.45% | 69.85% | 0.6039 | 356.6 | — |
| MrBERT | MRPC | ~50% | 74.45% | 71.81% | 0.5949 | 364.8 | 0.00e+00 |
| Baseline BERT | SNLI | 0% | 65.27% | 37.56% | 0.9766 | 1195.4 | — |
| MrBERT | SNLI | ~50% | 80.71% | 69.74% | 0.8330 | 1106.4 | 0.00e+00 |
| MrBERT | IMDB | ~50% | — | 84.78% | — | — | — |
| Baseline BERT | IMDB | 0% | — | *(run baseline)* | — | — | — |

## Latency & sequence length (from latency_benchmark.py)

| Setting | Seq length | Avg time (ms) |
|---------|------------|---------------|
| Baseline BERT | 512 | 1310.57 |
| MrBERT (soft) | 512 | — |
| MrBERT (hard) | 359 | 1853.48 |

**Speedup:** -41.4% (positive = MrBERT faster).

Run: `python latency_benchmark.py --output_result results/latency_results.json` to refresh.
