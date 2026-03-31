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
| **10b** | **MrXLM (XLM-R)** | **MRPC** | `--backbone xlmr`, same PI/gate; results include `"backbone":"xlmr"` |

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

---

## 6. Modal A100 TyDi QA (`modal_logs/`)

These files are **local copies** of Modal App stdout (not `scp` from a VM). Download with:

`modal app logs <APP_ID> --timestamps > modal_logs/<name>.log`

Full parameter snapshots appear in new runs as **`[Training args]`** after `Device:` (see `train_mrbert.py`).

| Local log file | Modal App ID | Final validation EM | Train del% (logged) | Intended launcher flags |
|----------------|--------------|----------------------|----------------------|-------------------------|
| `modal_logs/tydiqa_modal_EM03471_ap-4FzTHi10GW6ipDKX7fZ9TC.log` | `ap-4FzTHi10GW6ipDKX7fZ9TC` | **0.3471** | ~25.28% | `run_mrbert_tydi_modal.py` + `--no-use-pre-deletion-blend` + shared hyperparameters below |
| `modal_logs/tydiqa_modal_EM03778_ap-IEe2gcBxFgDvH62acNX22v.log` | `ap-IEe2gcBxFgDvH62acNX22v` | **0.3778** | ~25.90% | Same + `--use-pre-deletion-blend` (**~+3.1 pp** vs row above; W&B run name may not match—trust App ID + `[Training args]`) |
| `modal_logs/tydiqa_modal_EM03499_ap-rqoNveoPynOFoc2iKRSCnA.log` | `ap-rqoNveoPynOFoc2iKRSCnA` | **0.3499** | ~24.34% | Overlapping TyDi job from the same experiment batch; use logs / W&B for exact config |

**Shared hyperparameters** (both primary runs): `--batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5` and `--wandb-project mrbert-tydiqa` with distinct `--wandb-run-name` values.

**Example commands** (from repo root):

```bash
# A: with pre-deletion blending
modal run --detach run_mrbert_tydi_modal.py \
  --use-pre-deletion-blend \
  --wandb-project mrbert-tydiqa --wandb-run-name tydiqa-with-blend-l3-30pct \
  --batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 \
  --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 \
  --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5

# B: without blending
modal run --detach run_mrbert_tydi_modal.py \
  --no-use-pre-deletion-blend \
  --wandb-project mrbert-tydiqa --wandb-run-name tydiqa-no-blend-l3-30pct \
  --batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 \
  --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 \
  --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5
```

See **`report/final_report/presentation.md`** §4.4.4 for analysis and interpretation.
