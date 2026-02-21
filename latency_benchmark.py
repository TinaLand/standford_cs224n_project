#!/usr/bin/env python3
"""
Latency and hard-deletion sanity benchmark for MrBERT vs baseline BERT.

This script checks:
1. Shape / sequence-length behavior:
   - soft deletion (training-style): sequence length stays the same
   - hard deletion (inference-style): sequence length becomes shorter
2. Inference latency comparison (CPU or GPU):
   - baseline: BertForSequenceClassification
   - mrbert:   MrBertForSequenceClassification (eval mode, hard deletion)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from mrbert import MrBertForSequenceClassification


def build_batch(tokenizer, batch_size: int, seq_length: int, device: torch.device):
    """Construct a batch of long dummy sentences to stress attention."""
    # Roughly make text longer than seq_length so truncation kicks in.
    base = "This is a long example sentence for MrBERT latency benchmarking. "
    repeat = max(1, seq_length // 8)
    texts = [base * repeat for _ in range(batch_size)]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


def time_model(model, input_ids, attention_mask, num_steps: int) -> float:
    """Return average forward time (seconds) over num_steps."""
    model.eval()
    device = input_ids.device
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / num_steps


def main():
    parser = argparse.ArgumentParser(description="MrBERT hard-deletion latency benchmark")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for benchmarking")
    parser.add_argument("--seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--steps", type=int, default=50, help="Number of timed forward passes")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda, default: auto")
    parser.add_argument("--output_result", type=str, default=None, help="Write latency and shape stats to this JSON file (e.g. results/latency_results.json)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_ids, attention_mask = build_batch(tokenizer, args.batch_size, args.seq_length, device)
    print(f"Batch shape: input_ids={tuple(input_ids.shape)}, attention_mask={tuple(attention_mask.shape)}")

    # 1) Baseline BERT
    print("\nLoading baseline BertForSequenceClassification...")
    bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    bert = bert.to(device)

    # 2) MrBERT (hard deletion in eval mode)
    print("Loading MrBertForSequenceClassification (from bert-base-uncased)...")
    mrbert = MrBertForSequenceClassification.from_bert_pretrained(
        "bert-base-uncased",
        num_labels=2,
        gate_layer_index=3,
        gate_k=-30.0,
        attn_implementation="eager",
    )
    mrbert = mrbert.to(device)

    # ---- Shape / sequence-length sanity check ----
    print("\n=== Hard deletion shape check (MrBERT encoder only) ===")
    mrbert.mrbert.train()
    with torch.no_grad():
        soft_outputs, gate = mrbert.mrbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_gate=True,
            use_soft_deletion=True,
        )
    soft_hidden = soft_outputs.last_hidden_state
    print(f"Soft deletion: last_hidden_state shape = {tuple(soft_hidden.shape)}")

    mrbert.mrbert.eval()
    with torch.no_grad():
        hard_outputs = mrbert.mrbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_gate=False,
            use_soft_deletion=False,
        )
    hard_hidden = hard_outputs.last_hidden_state
    seq_len_soft = soft_hidden.shape[1]
    seq_len_hard = hard_hidden.shape[1]
    print(f"Hard deletion: last_hidden_state shape = {tuple(hard_hidden.shape)}")

    # ---- Latency benchmark ----
    print("\n=== Latency benchmark ===")
    print(f"Running {args.steps} forward passes (batch_size={args.batch_size}, seq_length={args.seq_length})...")

    t_bert = time_model(bert, input_ids, attention_mask, args.steps)
    print(f"Baseline BERT avg time: {t_bert * 1000:.2f} ms per forward")

    # MrBERT in eval mode uses hard deletion inside the encoder
    t_mrbert = time_model(mrbert, input_ids, attention_mask, args.steps)
    print(f"MrBERT (hard deletion) avg time: {t_mrbert * 1000:.2f} ms per forward")

    speedup_pct = None
    if t_bert > 0:
        speedup_pct = (t_bert - t_mrbert) / t_bert * 100
        print(f"Speedup: {speedup_pct:.1f}% (positive means MrBERT is faster)")

    if args.output_result:
        Path(args.output_result).parent.mkdir(parents=True, exist_ok=True)
        latency_row = {
            "baseline_ms": round(t_bert * 1000, 2),
            "mrbert_ms": round(t_mrbert * 1000, 2),
            "speedup_pct": round(speedup_pct, 2) if speedup_pct is not None else None,
            "seq_len_original": args.seq_length,
            "seq_len_soft": seq_len_soft,
            "seq_len_hard": seq_len_hard,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "device": str(device),
        }
        with open(args.output_result, "w") as f:
            json.dump(latency_row, f, indent=2)
        print(f"Latency result written to {args.output_result}")

    print("Done.")


if __name__ == "__main__":
    main()

