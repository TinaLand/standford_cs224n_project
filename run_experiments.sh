#!/usr/bin/env bash
# Run baseline and MrBERT experiments, appending results to results/train_results.jsonl.
# Scripts run automatically after training:
#   Step 11: scripts/aggregate_results.py
#   Step 12: scripts/extract_error_cases.py (SNLI, TyDi QA)
#   Step 10b–10h: MrXLM (XLM-R) on MRPC, SST-2, SNLI, IMDB, XNLI, TyDi QA (--backbone xlmr)
#   Step 13: scripts/analyze_loss_vs_deletion.py (mrpc, snli, sst2, tydiqa)
#   Step 14: scripts/roberta_pruning_demo.py, scripts/xlm_pruning_demo.py
#
# Usage:
#   ./run_experiments.sh              # full run (BERT + XLM-R, all datasets)
#   MODELS=bert ./run_experiments.sh  # only BERT / MrBERT experiments
#   MODELS=xlmr ./run_experiments.sh  # only XLM-R / MrXLM experiments
#   MODELS=bert,xlmr ./run_experiments.sh  # both (default)
#   QUICK=1 ./run_experiments.sh      # quick: MRPC 200 samples + latency only
#   SKIP_SNLI=1 ./run_experiments.sh  # skip SNLI (faster)
#   SKIP_SST2=1 ./run_experiments.sh  # skip SST-2
#   SKIP_XNLI=1 ./run_experiments.sh  # skip XNLI (cross-lingual NLI)
#   SKIP_TYDIQA=1 ./run_experiments.sh  # skip TyDi QA (saves time)
#   EPOCHS=3 BATCH=16 ./run_experiments.sh  # custom epochs/batch (L4: try BATCH=24 or 32)
#   GATE_WARMUP_STEPS=1000 ./run_experiments.sh  # gate warmup (recommended for TyDi QA)
#   LOG_LEVEL=1 ./run_experiments.sh        # write logs for every run (1=minimal, 2=+PI, 3=+gate details)
#   USE_WANDB=1 ./run_experiments.sh       # log metrics to Weights & Biases (wandb)

# sample commands
#   MODELS=bert ./run_experiments.sh
#   MR_USE_PI=0 EPOCHS=3 BATCH=8 LOG_LEVEL=1 MODELS=bert ./run_experiments.sh
#   GATE_WARMUP_STEPS=1000 BATCH=24 EPOCHS=3 MODELS=bert ./run_experiments.sh   # TyDi QA rescue + L4
set -e
cd "$(dirname "$0")"
mkdir -p results
mkdir -p logs
RESULTS_FILE="results/train_results.jsonl"

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-8}
LOG_LEVEL=${LOG_LEVEL:-0}
if [ "$LOG_LEVEL" != "0" ]; then
  LOG_ARGS=(--log_level "$LOG_LEVEL")
else
  LOG_ARGS=()
fi
if [ "${USE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS=(--use_wandb)
else
  WANDB_ARGS=()
fi

# Which backbones to run: comma-separated, e.g. bert, xlmr. Default: bert,xlmr.
MODELS_RAW="${MODELS:-bert,xlmr}"
MODELS_LIST=$(echo "$MODELS_RAW" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
run_bert() {
  case ",${MODELS_LIST}," in
    *,bert,*) return 0 ;;
    *) return 1 ;;
  esac
}
run_xlmr() {
  case ",${MODELS_LIST}," in
    *,xlmr,*) return 0 ;;
    *) return 1 ;;
  esac
}
echo "=== MODELS: $MODELS_LIST (bert=$(run_bert && echo on || echo off), xlmr=$(run_xlmr && echo on || echo off)) ==="

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

# Gate placement and threshold (XLM-R rescue: try GATE_LAYER_INDEX=6, GATE_THRESHOLD_RATIO=0.6)
GATE_LAYER_INDEX=${GATE_LAYER_INDEX:-3}
GATE_THRESHOLD_RATIO=${GATE_THRESHOLD_RATIO:-0.5}
GATE_LAYER_ARGS=(--gate_layer_index "$GATE_LAYER_INDEX" --gate_threshold_ratio "$GATE_THRESHOLD_RATIO")
# PI controller gains (XLM-R rescue: try CONTROLLER_KP=0.002, CONTROLLER_KI=5e-6)
CONTROLLER_KP=${CONTROLLER_KP:-0.5}
CONTROLLER_KI=${CONTROLLER_KI:-1e-5}
CONTROLLER_ARGS=(--controller_kp "$CONTROLLER_KP" --controller_ki "$CONTROLLER_KI")

# Gate warmup (Priority 1: rescue TyDi QA). Passed to train_mrbert.py for MrBERT runs.
#   GATE_WARMUP_STEPS: first N steps alpha=0, then enable gate/PI (default 0 = off)
#   PHASE1_STEPS, PHASE1_GATE_WEIGHT: optional phase with lighter gate pressure after warmup
GATE_WARMUP_STEPS=${GATE_WARMUP_STEPS:-0}
WARMUP_ARGS=()
if [ "${GATE_WARMUP_STEPS}" -gt 0 ]; then
  WARMUP_ARGS=(--gate_warmup_steps "$GATE_WARMUP_STEPS")
  if [ "${PHASE1_STEPS:-0}" -gt 0 ]; then
    WARMUP_ARGS+=(--phase1_steps "$PHASE1_STEPS" --phase1_gate_weight "${PHASE1_GATE_WEIGHT:-1e-3}")
  fi
  echo "=== GATE_WARMUP_STEPS=$GATE_WARMUP_STEPS (PI enabled after warmup) ==="
fi
if [ "${GATE_LAYER_INDEX}" != "3" ]; then
  echo "=== GATE_LAYER_INDEX=$GATE_LAYER_INDEX (gate after layer $GATE_LAYER_INDEX) ==="
fi

if [ "${QUICK:-0}" = "1" ]; then
  echo "=== QUICK mode: GPU smoke test (MRPC 200 samples + latency) ==="
  python3 -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
  echo ""
  python3 train_mrbert.py --dataset mrpc --epochs 1 --batch_size 8 --max_train_samples 200 \
    --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
  python3 latency_benchmark.py --batch_size 16 --seq_length 256 --steps 20 --output_result results/latency_results.json
  echo "=== Aggregating results into RESULTS.md ==="
  python3 scripts/aggregate_results.py || echo "aggregate_results failed (QUICK mode)"
  echo "=== Loss vs deletion (MRPC, quick) ==="
  python3 -m scripts.analyze_loss_vs_deletion --dataset mrpc --max_samples 200 --output results/loss_vs_deletion_mrpc.json || echo "analyze_loss_vs_deletion failed (QUICK)"
  echo "Done."
  exit 0
fi

if run_bert; then
  echo "=== 1. Baseline BERT: MRPC ==="
  python3 train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"

  echo "=== 2. Baseline BERT: IMDB ==="
  python3 train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"

  echo "=== 3. MrBERT (~50% deletion): MRPC ==="
  python3 train_mrbert.py --dataset mrpc --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"

  echo "=== 4. MrBERT (~50% deletion): IMDB ==="
  python3 train_mrbert.py --dataset imdb --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"

  if [ "${SKIP_SNLI:-0}" != "1" ]; then
    echo "=== 5. Baseline BERT: SNLI ==="
    python3 train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
    echo "=== 6. MrBERT: SNLI ==="
    python3 train_mrbert.py --dataset snli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
  fi

  if [ "${SKIP_SST2:-0}" != "1" ]; then
    echo "=== 7. Baseline BERT: SST-2 ==="
    python3 train_mrbert.py --dataset sst2 --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
    echo "=== 8. MrBERT: SST-2 ==="
    python3 train_mrbert.py --dataset sst2 --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
  fi

  if [ "${SKIP_TYDIQA:-0}" != "1" ]; then
    echo "=== 9. Baseline BERT: TyDi QA ==="
    python3 train_mrbert.py --dataset tydiqa --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
    echo "=== 10. MrBERT: TyDi QA ==="
    python3 train_mrbert.py --dataset tydiqa --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
  fi

  if [ "${SKIP_XNLI:-0}" != "1" ]; then
    echo "=== 8b. Baseline BERT: XNLI (en) ==="
    python3 train_mrbert.py --dataset xnli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
    echo "=== 8c. MrBERT: XNLI (en) ==="
    python3 train_mrbert.py --dataset xnli --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE"
  fi
fi

if run_xlmr; then
  echo "=== 10a. Baseline XLM-R: MRPC ==="
  python3 train_mrbert.py --dataset mrpc --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R MRPC failed"
  echo "=== 10b. MrXLM (XLM-R): MRPC ==="
  python3 train_mrbert.py --dataset mrpc --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) MRPC failed"

  if [ "${SKIP_SST2:-0}" != "1" ]; then
    echo "=== 10c. Baseline XLM-R: SST-2 ==="
    python3 train_mrbert.py --dataset sst2 --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R SST-2 failed"
    echo "=== 10c2. MrXLM (XLM-R): SST-2 ==="
    python3 train_mrbert.py --dataset sst2 --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) SST-2 failed"
  fi
  if [ "${SKIP_SNLI:-0}" != "1" ]; then
    echo "=== 10d. Baseline XLM-R: SNLI ==="
    python3 train_mrbert.py --dataset snli --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R SNLI failed"
    echo "=== 10d2. MrXLM (XLM-R): SNLI ==="
    python3 train_mrbert.py --dataset snli --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) SNLI failed"
  fi
  echo "=== 10e. Baseline XLM-R: IMDB ==="
  python3 train_mrbert.py --dataset imdb --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R IMDB failed"
  echo "=== 10e2. MrXLM (XLM-R): IMDB ==="
  python3 train_mrbert.py --dataset imdb --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) IMDB failed"

  if [ "${SKIP_XNLI:-0}" != "1" ]; then
    echo "=== 10f. Baseline XLM-R: XNLI (en) ==="
    python3 train_mrbert.py --dataset xnli --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R XNLI failed"
    echo "=== 10f2. MrXLM (XLM-R): XNLI (en) ==="
    python3 train_mrbert.py --dataset xnli --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) XNLI failed"
  fi
  if [ "${SKIP_TYDIQA:-0}" != "1" ]; then
    echo "=== 10g. Baseline XLM-R: TyDi QA ==="
    python3 train_mrbert.py --dataset tydiqa --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 0.0 "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "Baseline XLM-R TyDi QA failed"
    echo "=== 10h. MrXLM (XLM-R): TyDi QA ==="
    python3 train_mrbert.py --dataset tydiqa --backbone xlmr --epochs "$EPOCHS" --batch_size "$BATCH" --max_length 256 --gate_weight 1e-4 "${MR_PI_ARGS[@]}" "${LOG_ARGS[@]}" "${WANDB_ARGS[@]}" "${WARMUP_ARGS[@]}" "${GATE_LAYER_ARGS[@]}" "${CONTROLLER_ARGS[@]}" --output_result "$RESULTS_FILE" || echo "MrXLM (XLM-R) TyDi QA failed"
  fi
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

echo "=== 13. Loss vs deletion correlation (per-example) ==="
python3 -m scripts.analyze_loss_vs_deletion --dataset mrpc --max_samples 500 --output results/loss_vs_deletion_mrpc.json || echo "analyze_loss_vs_deletion (MRPC) failed"
if [ "${SKIP_SNLI:-0}" != "1" ]; then
  python3 -m scripts.analyze_loss_vs_deletion --dataset snli --max_samples 500 --output results/loss_vs_deletion_snli.json || echo "analyze_loss_vs_deletion (SNLI) failed"
fi
if [ "${SKIP_SST2:-0}" != "1" ]; then
  python3 -m scripts.analyze_loss_vs_deletion --dataset sst2 --max_samples 500 --output results/loss_vs_deletion_sst2.json || echo "analyze_loss_vs_deletion (SST-2) failed"
fi
if [ "${SKIP_XNLI:-0}" != "1" ]; then
  python3 -m scripts.analyze_loss_vs_deletion --dataset xnli --max_samples 500 --output results/loss_vs_deletion_xnli.json || echo "analyze_loss_vs_deletion (XNLI) failed"
fi
if [ "${SKIP_TYDIQA:-0}" != "1" ]; then
  python3 -m scripts.analyze_loss_vs_deletion --dataset tydiqa --max_length 256 --max_samples 500 --output results/loss_vs_deletion_tydiqa.json || echo "analyze_loss_vs_deletion (TyDi QA) failed"
fi

echo "=== 14. Architecture demos (MrRoBERTa / MrXLM) ==="
if run_bert; then
  python3 -m scripts.roberta_pruning_demo || echo "roberta_pruning_demo failed"
fi
if run_xlmr; then
  python3 -m scripts.xlm_pruning_demo || echo "xlm_pruning_demo failed"
fi

echo "=== Done. Results in $RESULTS_FILE; error cases in results/error_cases_*.jsonl; loss vs deletion in results/loss_vs_deletion_*.json ==="
