# run_experiments.sh Guide

## 1. What the script runs

### Normal mode `./run_experiments.sh`

| Step | Task | Dataset | Notes |
|------|------|---------|------|
| 1 | Baseline BERT | **MRPC** | gate_weight=0, results appended to `results/train_results.jsonl` |
| 2 | Baseline BERT | **IMDB** | gate_weight=0 |
| 3 | MrBERT (~50% deletion) | **MRPC** | gate_weight=1e-4, use_pi, target_deletion=0.5 |
| 4 | MrBERT (~50% deletion) | **IMDB** | same as above |
| 5 | Baseline BERT | **SNLI** | only if `SKIP_SNLI=1` is not set |
| 6 | MrBERT | **SNLI** | same as above |
| 7 | Baseline BERT | **SST-2** | only if `SKIP_SST2=1` is not set |
| 8 | MrBERT | **SST-2** | same as above |
| 9 | Baseline BERT | **TyDi QA** | only if `SKIP_TYDIQA=1` is not set |
| 10 | MrBERT | **TyDi QA** | same as above |

- **Environment variables**: `EPOCHS`, `BATCH` can be overridden (defaults 1 and 8); `SKIP_SNLI=1`, `SKIP_SST2=1`, `SKIP_TYDIQA=1` skip the corresponding datasets.
- **Not run automatically**: `scripts/aggregate_results.py` — run it yourself after experiments to update RESULTS.md.

### Quick mode `QUICK=1 ./run_experiments.sh`

| Step | Task | Notes |
|------|------|-------|
| 1 | Print CUDA device | Check that GPU is used |
| 2 | MrBERT | **First 200 MRPC samples**, 1 epoch, batch 8, writes `train_results.jsonl` |
| 3 | latency_benchmark.py | batch 16, seq 256, 20 steps, writes `results/latency_results.json` |

---

## 2. Scripts and data used

**Called by run_experiments.sh:**

- `train_mrbert.py` — training (MRPC / IMDB / SNLI / SST-2 / TyDi QA)
- `latency_benchmark.py` — only when QUICK=1

**Not called by run_experiments.sh (run separately if needed):**

- `scripts/aggregate_results.py` — reads `train_results.jsonl` and `latency_results.json`, generates **RESULTS.md**
- `scripts/extract_error_cases.py` — extracts "wrong prediction + high deletion rate" cases (SNLI/TyDi QA)
- `python3 -m mrbert.diagnostics` — gate interpretability (single sentence / stats)
- `inspect_data.py` — inspect dataset samples
- `run_mrbert_example.py` — minimal example

---

## 3. Dataset coverage

| Dataset | Supported by train_mrbert.py | Run by run_experiments.sh |
|---------|------------------------------|---------------------------|
| **MRPC** | Yes | Yes (baseline + MrBERT) |
| **IMDB** | Yes | Yes (baseline + MrBERT) |
| **SNLI** | Yes | Yes, optional (baseline + MrBERT), skip with SKIP_SNLI=1 |
| **SST-2** | Yes | Yes, optional (baseline + MrBERT), skip with SKIP_SST2=1 |
| **TyDi QA** | Yes | Yes, optional (baseline + MrBERT), skip with SKIP_TYDIQA=1 |

---

## 4. After running experiments

1. Update RESULTS.md:
   ```bash
   python3 scripts/aggregate_results.py
   ```
2. (Optional) Copy results from VM to your machine:
   ```bash
   gcloud compute scp --recurse mrbert-gpu:~/cs224n_project/results ./results --zone=YOUR_ZONE
   ```

---

## 5. Running SST-2 or TyDi QA only

To run them manually from project root:

```bash
# SST-2 baseline + MrBERT
python3 train_mrbert.py --dataset sst2 --epochs 1 --batch_size 8 --gate_weight 0.0 --output_result results/train_results.jsonl
python3 train_mrbert.py --dataset sst2 --epochs 1 --batch_size 8 --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result results/train_results.jsonl

# TyDi QA (use --max_length 256; optionally --max_train_samples for a shorter run)
python3 train_mrbert.py --dataset tydiqa --epochs 1 --batch_size 8 --max_length 256 --gate_weight 0.0 --output_result results/train_results.jsonl
python3 train_mrbert.py --dataset tydiqa --epochs 1 --batch_size 8 --max_length 256 --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result results/train_results.jsonl
```

Then run `python3 scripts/aggregate_results.py` again to update RESULTS.md.
