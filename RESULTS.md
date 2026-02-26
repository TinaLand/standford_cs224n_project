# Experiment Results

Comparison: Baseline BERT (no gate) vs MrBERT (gate + PI, target deletion ~50%).

| Model | Dataset | Deletion | Actual Del% | Val Acc | Avg Loss | Time(s) | Alpha |
|-------|---------|----------|-------------|--------|----------|---------|-------|
| Baseline BERT | MRPC | 0% | 40.45% | 69.85% | 0.6039 | 356.6 | — |
| MrBERT | MRPC | ~50% | 74.45% | 71.81% | 0.5949 | 364.8 | 0.00e+00 |
| Baseline BERT | SNLI | 0% | 65.27% | 37.56% | 0.9766 | 1195.4 | — |
| MrBERT | SNLI | ~50% | 80.71% | 69.74% | 0.8330 | 1106.4 | 0.00e+00 |
| Baseline BERT | TYDIQA | 0% | 13.82% | 1.28% | 4.6695 | 308.3 | — |
| MrBERT | TYDIQA | ~50% | 51.18% | 0.00% | 4.5464 | 252.5 | 1.74e-02 |
| Baseline BERT | IMDB | 0% | 4.40% | 86.60% | 0.4049 | 878.3 | — |
| MrBERT | IMDB | ~50% | 65.65% | 80.16% | 0.4256 | 841.2 | 0.00e+00 |
| Baseline BERT | SST2 | 0% | 0.03% | 85.32% | 0.2290 | 1756.3 | — |
| MrBERT | SST2 | ~50% | 68.90% | 89.11% | 0.2206 | 1755.1 | 0.00e+00 |

## Latency & sequence length (from latency_benchmark.py)

| Setting | Seq length | Avg time (ms) |
|---------|------------|---------------|
| Baseline BERT | 256 | 186.86 |
| MrBERT (soft) | 256 | — |
| MrBERT (hard) | 120 | 132.72 |

**Speedup:** 29.0% (positive = MrBERT faster).

Run: `python latency_benchmark.py --output_result results/latency_results.json` to refresh.
