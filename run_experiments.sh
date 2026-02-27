#!/usr/bin/env bash
# Run baseline and MrBERT experiments, appending results to results/train_results.jsonl.
# Usage:
#   ./run_experiments.sh              # full run (MRPC, IMDB, SNLI, SST-2, TyDi QA)
#   QUICK=1 ./run_experiments.sh      # quick: MRPC 200 samples + latency only
#   SKIP_SNLI=1 ./run_experiments.sh  # skip SNLI (faster)
#   SKIP_SST2=1 ./run_experiments.sh  # skip SST-2
#   SKIP_TYDIQA=1 ./run_experiments.sh  # skip TyDi QA (saves time)
#   EPOCHS=3 BATCH=16 ./run_experiments.sh  # custom epochs/batch

# sample command
#   MR_TARGET_DEL=0.4 MR_USE_PI=1 EPOCHS=3 BATCH=8 ./run_experiments.sh
set -e
cd "$(dirname "$0")"
mkdir -p results
RESULTS_FILE="results/train_results.jsonl"

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-8}

# MrBERT deletion settings (can be overridden via env):
#   MR_TARGET_DEL: target_deletion for PI controller (default 0.5)
#   MR_USE_PI: 1 = use_pi, 0 = no PI (only gate_weight) (default 1)
MR_TARGET_DEL=${MR_TARGET_DEL:-0.5}
MR_USE_PI=${MR_USE_PI:-1}
if [ "$MR_USE_PI" = "1" ]; then
  MR_PI_ARGS=(--use_pi --target_deletion "$MR_TARGET_DEL")
else
  MR_PI_ARGS=()
fi

if [ "${QUICK:-0}" = "1" ]; then
  echo "=== QUICK mode: GPU smoke test (MRPC 200 samples + latency) ==="
  python3 -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
  echo ""
  python3 train_mrbert.py --dataset mrpc --epochs 1 --batch_size 8 --max_train_samples 200 \
    --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"
  python3 latency_benchmark.py --batch_size 16 --seq_length 256 --steps 20 --output_result results/latency_results.json
  echo "=== Aggregating results into RESULTS.md ==="
  python3 scripts/aggregate_results.py || echo "aggregate_results failed (QUICK mode)"
  echo "Done."
  exit 0
fi

echo "=== 1. Baseline BERT: MRPC ==="
python3 train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 2. Baseline BERT: IMDB ==="
python3 train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 3. MrBERT (~50% deletion): MRPC ==="
python3 train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"

echo "=== 4. MrBERT (~50% deletion): IMDB ==="
python3 train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"

if [ "${SKIP_SNLI:-0}" != "1" ]; then
  echo "=== 5. Baseline BERT: SNLI ==="
  python3 train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"
  echo "=== 6. MrBERT: SNLI ==="
  python3 train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"
fi

if [ "${SKIP_SST2:-0}" != "1" ]; then
  echo "=== 7. Baseline BERT: SST-2 ==="
  python3 train_mrbert.py --dataset sst2 --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"
  echo "=== 8. MrBERT: SST-2 ==="
  python3 train_mrbert.py --dataset sst2 --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"
fi

if [ "${SKIP_TYDIQA:-0}" != "1" ]; then
  echo "=== 9. Baseline BERT: TyDi QA ==="
  python3 train_mrbert.py --dataset tydiqa --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 0.0 --output_result "$RESULTS_FILE"
  echo "=== 10. MrBERT: TyDi QA ==="
  python3 train_mrbert.py --dataset tydiqa --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 1e-4 "${MR_PI_ARGS[@]}" --output_result "$RESULTS_FILE"
fi

echo "=== 11. Aggregating results into RESULTS.md ==="
python3 scripts/aggregate_results.py || echo "aggregate_results failed"

echo "=== 12. Extracting high-deletion error cases (SNLI / TyDi QA) ==="
if [ "${SKIP_SNLI:-0}" != "1" ]; then
  python3 -m scripts.extract_error_cases --dataset snli --max_samples 5000 --output results/error_cases_snli.jsonl || echo "extract_error_cases (SNLI) failed"
fi
if [ "${SKIP_TYDIQA:-0}" != "1" ]; then
  python3 -m scripts.extract_error_cases --dataset tydiqa --max_samples 1000 --output results/error_cases_tydiqa.jsonl || echo "extract_error_cases (TyDi QA) failed"
fi

echo "=== Done. Results in $RESULTS_FILE; error cases in results/error_cases_*.jsonl ==="
