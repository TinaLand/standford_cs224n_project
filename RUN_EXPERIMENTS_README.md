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

- **Environment variables**: `EPOCHS`, `BATCH` can be overridden (defaults 1 and 8); `SKIP_SNLI=1`, `SKIP_SST2=1`, `SKIP_TYDIQA=1` skip the corresponding datasets. With `USE_WANDB=1`, you can set **`WANDB_PROJECT`** and **`WANDB_ENTITY`** so every `train_mrbert.py` invocation in the script logs to that W&B project (same charts as gated runs: `train/*`, `val/*`, etc.).
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

### 5.1 Weights & Biases: baseline logs charts too

`train_mrbert.py` uses the same W&B path for **`gate_weight=0` (baseline)** and for MrBERT/MrXLM: pass `--use_wandb` plus optional **`--wandb_project`** and **`--wandb_entity`**. Default project name if omitted is `mrbert`. Run names default to `{dataset}_baseline` or `{dataset}_mrbert` unless you set `--wandb_run_name`.

**Batch script** (`run_experiments.sh`): enable logging and send all runs to team project `alina`.

Minimal (only sets W&B destination; `MODELS` / `EPOCHS` / `BATCH` use script defaults):

```bash
USE_WANDB=1 WANDB_PROJECT=alina WANDB_ENTITY=aronima7-stanford-university ./run_experiments.sh
```

Example with XLM-R + longer training (same W&B target):

```bash
USE_WANDB=1 \
WANDB_PROJECT=alina \
WANDB_ENTITY=aronima7-stanford-university \
MODELS=xlmr EPOCHS=5 BATCH=24 \
./run_experiments.sh
```

(Requires `wandb login` or `WANDB_API_KEY` in the environment. TyDi post-fix runs used the same entity/project: `wandb.ai/aronima7-stanford-university/alina`.)

### 5.2 XLM-R SNLI baseline → W&B `alina` (recommended `max_length` / epochs)

`train_mrbert.py` flags use **underscores** (not kebab-case):

```bash
python3 train_mrbert.py \
  --dataset snli \
  --backbone xlmr \
  --epochs 5 \
  --batch_size 24 \
  --max_length 256 \
  --lr 2e-5 \
  --gate_weight 0.0 \
  --gate_layer_index 3 \
  --gate_threshold_ratio 0.5 \
  --log_level 1 \
  --use_wandb \
  --wandb_project alina \
  --wandb_entity aronima7-stanford-university \
  --wandb_run_name xlmr-snli-baseline-m256-ep5 \
  --output_result results/train_results.jsonl
```

No changes to `train_mrbert.py` are required for baseline W&B curves.

### 5.3 Modal A100: XLM-R SNLI baseline only

Use `run_xlmr_snli_modal.py` (same W&B defaults as above: project `alina`, entity `aronima7-stanford-university`). Requires secret `wandb-api-key`. From **repo root**:

```bash
modal run --detach run_xlmr_snli_modal.py
```

Optional flags (OOM 时减小 batch):

```bash
modal run --detach run_xlmr_snli_modal.py --batch-size 16 --wandb-run-name xlmr-snli-baseline-try2
```

Disable W&B:

```bash
modal run --detach run_xlmr_snli_modal.py --no-use-wandb
```

Training appends **`train_results.jsonl` directly under the Volume** at `train_results.jsonl` (and `commit()` at the end). Pull everything locally:

```bash
modal volume get xlmr-results ./results_from_modal
# jsonl path: ./results_from_modal/train_results.jsonl
```

If an **older** run only wrote under `/workspace/results` and depended on a copy step, still try `modal volume get` — the job may have copied before shutdown.

**Full run logs (not SCP):** Modal logs are fetched with the **App ID** from the run’s page in the dashboard (string starting with `ap-`, not the Container ID `ta-…` or other internal ids). From repo root:

```bash
modal app logs ap-YOUR_APP_ID_HERE --timestamps > modal_logs/xlmr_snli_baseline_FULL.log
```

If `modal app list` is empty, use the same Modal account/workspace as in the browser, or open **App** → **Logs** → export from the UI.

---

## 6. Modal A100 TyDi QA (`modal_logs/`)

These files are **local copies** of Modal App stdout (not `scp` from a VM). Download with:

`modal app logs <APP_ID> --timestamps > modal_logs/<name>.log`

**If the saved file is empty or missing training lines:** Some Modal CLI versions only return a **bounded** log window (older docs mentioned the last 100 lines). Try:

```bash
# Works on all recent CLIs (from repo root):
modal app logs ap-XXXXXXXX --timestamps > modal_logs/my_run.log

# If your `modal app logs -h` lists --tail / --since, use them for long runs:
# modal app logs ap-XXXXXXXX --timestamps --tail 20000 > modal_logs/my_run.log

# Wrong Modal environment shows no apps / empty logs — match the dashboard:
modal app list --json
modal app logs ap-XXXXXXXX -e main --timestamps

# Open the same page as the browser (then copy-paste if CLI still fails)
modal app dashboard ap-XXXXXXXX
```

**PI vs `--gate-threshold-ratio` (important):** Through 2026-03-31, `PIController` and the train-loop “deletion %” used a **fixed `gate_k * 0.5`** while the model’s hard-delete threshold followed **`--gate-threshold-ratio`**. Using e.g. `0.8` made PI think deletion was far above `target_deletion`, pushed **α → 0**, and produced misleading logged del%. **This is fixed** in `mrbert/pi_controller.py` + `train_mrbert.py` (PI and logs now use `args.gate_threshold_ratio`). Re-run any experiment that mixed non-0.5 ratios with PI after pulling the fix.

Full parameter snapshots appear in new runs as **`[Training args]`** after `Device:` (see `train_mrbert.py`).

**TyDi “baseline” vs blending comparisons:** The **~20.18%** validation EM in `results/new/bert_from_l4/` (**no gate**, **1 epoch**, L4 batch/schedule) is a **weak L4 baseline** for “gated vs no-gate” narratives (paired with **~28.16%** gated L4). It is **not** the right reference for **pre-deletion blending** ablations, which must use **matched** runs—typically the Modal rows below (**e.g. 0.3471** without pre-deletion blend vs **0.3562** with learnable blend, same shared hyperparameters).

| Local log file | Modal App ID | Final validation EM | Train del% (logged) | Intended launcher flags |
|----------------|--------------|----------------------|----------------------|-------------------------|
| `modal_logs/tydiqa_modal_EM03471_ap-4FzTHi10GW6ipDKX7fZ9TC.log` | `ap-4FzTHi10GW6ipDKX7fZ9TC` | **0.3471** | ~25.28% | `run_mrbert_tydi_modal.py` + `--no-use-pre-deletion-blend` + shared hyperparameters below |
| `modal_logs/tydiqa_modal_EM03778_ap-IEe2gcBxFgDvH62acNX22v.log` | `ap-IEe2gcBxFgDvH62acNX22v` | **0.3778** | ~25.90% | Same + `--use-pre-deletion-blend` (**~+3.1 pp** vs row above; W&B run name may not match—trust App ID + `[Training args]`) |
| `modal_logs/tydiqa_modal_EM03499_ap-rqoNveoPynOFoc2iKRSCnA.log` | `ap-rqoNveoPynOFoc2iKRSCnA` | **0.3499** | ~24.34% | Overlapping TyDi job from the same experiment batch; use logs / W&B for exact config |
| `modal_logs/tydiqa_modal_ap-c5H2JBykE62ASkj1prfX7M.log` | `ap-c5H2JBykE62ASkj1prfX7M` | **0.0887** (peak **0.1312** ep 5) | ~88.75% **(misleading; see below)** | “Limit” run: L9, `max_length` 384, 6 ep, fixed blend, `target_deletion` 0.2, `gate_threshold_ratio` **0.8**, `controller_kp` 0.1. **Failed** mainly because PI + logged del% used **k/2** while hard-delete used **0.8·k**—α collapsed to 0. **Fixed in code** (PI/logs now follow `gate_threshold_ratio`); treat this log as invalid for comparing L9 vs L3 until re-run. W&B: `tydiqa-lim-l9-m384-ep6-t020-gt08-kp01-fixedblend`. |
| `modal_logs/tydiqa_modal_EM03861_ap-Qyiv6SzrZ7T2OgtJwL7LWl.log` | `ap-Qyiv6SzrZ7T2OgtJwL7LWl` | **0.3861** (peak **0.4000** ep 5) | ~47.99% | **Post-fix rerun** of the L9/384 setting (fixed blend, `target_deletion` 0.2, `gate_threshold_ratio` 0.8, `kp` 0.1). Shows the PI/threshold fix recovered the run from failure to competitive EM. W&B (team project): `aronima7-stanford-university/alina`, run `u03dgn34`. |

**Shared hyperparameters** (both primary runs): `--batch-size 16 --epochs 3 --max-length 256 --lr 2e-5 --target-deletion 0.3 --gate-layer-index 3 --gate-threshold-ratio 0.5 --gate-warmup-steps 1000 --use-pi --controller-kp 0.5 --controller-ki 1e-5` and `--wandb-project mrbert-tydiqa` with distinct `--wandb-run-name` values.

**Why `alpha_final` can still be 0 in successful runs:** PI only controls *additional* deletion pressure (`L = L_task + alpha * L_gate`, with `alpha = max(0, kp*P + ki*I)`). If current deletion stays above target late in training, PI clamps `alpha` to 0. This does **not** force deletion to 0; task gradients through the gate can still keep substantial deletion (e.g., ~48% in `ap-Qyiv...`).

For the post-fix run `ap-Qyiv6SzrZ7T2OgtJwL7LWl` specifically: `target_deletion=0.2` while logged train deletion remained ~0.48 near the end. The error term (`target - current`) therefore stayed negative, the PI P/I state accumulated negative values, and `kp * P + ki * I` became negative; after clamping, `alpha_final` is reported as 0. This is expected under the current controller design and should be read as late-stage saturation of PI, not an implementation error.

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
