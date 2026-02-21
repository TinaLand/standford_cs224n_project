#!/usr/bin/env bash
# Quick GPU smoke test: small data, 1 epoch. Run on a GCP GPU VM or locally with CUDA.
# Usage: ./run_gpu_small.sh

set -e
cd "$(dirname "$0")"
mkdir -p results

echo "=== Checking for GPU ==="
python -c "
import torch
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
else:
    print('WARNING: No CUDA. Training will run on CPU.')
"

echo ""
echo "=== MrBERT quick run: MRPC, 200 samples, 1 epoch, batch 8 ==="
python train_mrbert.py \
  --dataset mrpc \
  --epochs 1 \
  --batch_size 8 \
  --max_train_samples 200 \
  --gate_weight 1e-4 \
  --use_pi \
  --target_deletion 0.5 \
  --output_result results/train_results.jsonl

echo ""
echo "=== Latency benchmark (GPU if available) ==="
python latency_benchmark.py --batch_size 16 --seq_length 256 --steps 20 --output_result results/latency_results.json

echo ""
echo "Done. Check results/train_results.jsonl and terminal for val accuracy and speedup."
