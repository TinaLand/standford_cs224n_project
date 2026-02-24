#!/usr/bin/env bash
# Run baseline and MrBERT experiments, appending results to results/train_results.jsonl.
# Usage:
#   ./run_experiments.sh              # full run (MRPC, IMDB, optional SNLI)
#   QUICK=1 ./run_experiments.sh      # quick: MRPC 200 samples + latency only
#   SKIP_SNLI=1 ./run_experiments.sh  # skip SNLI (faster)
#   EPOCHS=3 BATCH=16 ./run_experiments.sh  # custom epochs/batch

set -e
cd "$(dirname "$0")"
mkdir -p results
RESULTS_FILE="results/train_results.jsonl"

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-8}

if [ "${QUICK:-0}" = "1" ]; then
  echo "=== QUICK mode: GPU smoke test (MRPC 200 samples + latency) ==="
  python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
  echo ""
  python train_mrbert.py --dataset mrpc --epochs 1 --batch_size 8 --max_train_samples 200 \
    --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"
  python latency_benchmark.py --batch_size 16 --seq_length 256 --steps 20 --output_result results/latency_results.json
  echo "Done. Run: python scripts/aggregate_results.py"
  exit 0
fi

echo "=== 1. Baseline BERT: MRPC ==="
python train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 2. Baseline BERT: IMDB ==="
python train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"

echo "=== 3. MrBERT (~50% deletion): MRPC ==="
python train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"

echo "=== 4. MrBERT (~50% deletion): IMDB ==="
python train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"

if [ "${SKIP_SNLI:-0}" != "1" ]; then
  echo "=== 5. Baseline BERT: SNLI ==="
  python train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 --output_result "$RESULTS_FILE"
  echo "=== 6. MrBERT: SNLI ==="
  python train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 --use_pi --target_deletion 0.5 --output_result "$RESULTS_FILE"
fi

echo "=== Done. Results in $RESULTS_FILE ==="
echo "Run: python scripts/aggregate_results.py  to update RESULTS.md"
