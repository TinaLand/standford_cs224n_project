#!/usr/bin/env bash
# Run baseline and MrBERT experiments, appending results to results/train_results.jsonl.
# Usage: ./run_experiments.sh   (or: bash run_experiments.sh)
# Optional: set SKIP_SNLI=1 to skip SNLI (faster); set EPOCHS=1 and BATCH=8 for quick run.

set -e
cd "$(dirname "$0")"
mkdir -p results
RESULTS_FILE="results/train_results.jsonl"

# Clear previous results (optional; comment out to append)
# : > "$RESULTS_FILE"

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-8}

echo "=== 1. Baseline BERT (no deletion): MRPC ==="
python train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 2. Baseline BERT (no deletion): IMDB ==="
python train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 3. MrBERT (deletion ~50%): MRPC ==="
python train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"

echo "=== 4. MrBERT (deletion ~50%): IMDB ==="
python train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"

if [ "${SKIP_SNLI:-0}" != "1" ]; then
  echo "=== 5. Baseline BERT: SNLI ==="
  python train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"
  echo "=== 6. MrBERT: SNLI ==="
  python train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"
fi

echo "=== Done. Results appended to $RESULTS_FILE ==="
echo "Run: python scripts/aggregate_results.py  to update RESULTS.md"
