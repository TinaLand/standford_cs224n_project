#!/usr/bin/env python3
"""
Extract "typical error cases" for SNLI or TyDi QA: samples that are both
**wrong** (pred != label) and **high deletion rate** (likely over-deletion).
Useful for analyzing when the model drops too many tokens and fails.

Usage:
  python scripts/extract_error_cases.py --dataset snli --checkpoint path/to/model.pt --min_deletion_rate 0.7
  python scripts/extract_error_cases.py --dataset tydiqa --max_samples 1000 --output results/error_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

from mrbert import MrBertForSequenceClassification, MrBertForQuestionAnswering
from mrbert.diagnostics import analyze_dropped_tokens

# SNLI label names
SNLI_LABELS = ("entailment", "neutral", "contradiction")

# TyDi QA: reuse preprocessing from train_mrbert
def _prepare_tydiqa_example(ex, tokenizer, max_length):
    question = ex["question"]
    context = ex["context"]
    answers = ex["answers"]
    if not answers or not answers["text"] or not answers["answer_start"]:
        return None
    answer_start_char = answers["answer_start"][0]
    answer_text = answers["text"][0]
    answer_end_char = answer_start_char + len(answer_text)
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    offset_mapping = inputs.pop("offset_mapping")
    token_type_ids = inputs.get("token_type_ids")
    start_token = -1
    end_token = -1
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        if token_type_ids and token_type_ids[i] == 0:
            continue
        if s <= answer_start_char < e:
            start_token = i
        if s < answer_end_char <= e:
            end_token = i
            break
    if start_token < 0 or end_token < 0 or end_token < start_token:
        return None
    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    inputs["_question"] = question
    inputs["_context"] = context
    inputs["_answer_text"] = answer_text
    return inputs


def load_snli_val(tokenizer, max_length: int):
    ds = load_dataset("snli")
    for split in list(ds.keys()):
        ds[split] = ds[split].filter(lambda x: x["label"] >= 0)
    raw_val = list(ds["validation"])
    def tokenize(ex):
        return tokenizer(
            ex["premise"],
            ex["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
    val_remove = [c for c in ds["validation"].column_names if c not in ("label", "idx")]
    val_ds = ds["validation"].map(tokenize, batched=True, remove_columns=val_remove)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return val_ds, raw_val


def load_tydiqa_val(tokenizer, max_length: int):
    ds = load_dataset("tydiqa", "secondary_task")
    val_raw = ds["validation"]
    from datasets import Dataset
    val_list = []
    for i in range(len(val_raw)):
        ex = val_raw[i]
        row = _prepare_tydiqa_example(ex, tokenizer, max_length)
        if row is not None:
            val_list.append({
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "token_type_ids": row["token_type_ids"],
                "start_positions": row["start_positions"],
                "end_positions": row["end_positions"],
                "_question": row["_question"],
                "_context": row["_context"],
                "_answer_text": row["_answer_text"],
            })
    val_ds = Dataset.from_list(val_list)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"])
    return val_ds, val_list


def main():
    p = argparse.ArgumentParser(description="Extract wrong + high-deletion error cases for SNLI or TyDi QA")
    p.add_argument("--dataset", type=str, required=True, choices=["snli", "tydiqa"])
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model state_dict (optional)")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--min_deletion_rate", type=float, default=0.7,
                   help="Minimum deletion rate to count as 'over-deletion' (default 0.7)")
    p.add_argument("--max_samples", type=int, default=None, help="Cap validation samples (for speed)")
    p.add_argument("--max_cases", type=int, default=50, help="Max number of error cases to output")
    p.add_argument("--max_dropped_tokens", type=int, default=25, help="Max dropped tokens to store per case")
    p.add_argument("--output", type=str, default=None, help="Save cases to JSONL file")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    gate_k, threshold_ratio = -30.0, 0.5
    threshold = gate_k * threshold_ratio

    if args.dataset == "snli":
        val_ds, raw_val = load_snli_val(tokenizer, args.max_length)
        is_qa = False
        num_labels = 3
    else:
        val_ds, raw_val = load_tydiqa_val(tokenizer, args.max_length)
        is_qa = True
        num_labels = None

    if args.max_samples is not None:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))
    raw_val = raw_val[: len(val_ds)]

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if is_qa:
        model = MrBertForQuestionAnswering.from_bert_pretrained(
            "bert-base-uncased", gate_layer_index=3, gate_k=gate_k, attn_implementation="eager",
        )
    else:
        model = MrBertForSequenceClassification.from_bert_pretrained(
            "bert-base-uncased", num_labels=num_labels, gate_layer_index=3, gate_k=gate_k, attn_implementation="eager",
        )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No --checkpoint: using randomly initialized gate (for deletion-rate stats only).")
    model = model.to(device)
    model.eval()

    error_cases = []
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if is_qa:
                start_pos = batch["start_positions"].to(device)
                end_pos = batch["end_positions"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    gate_regularizer_weight=0.0,
                )
                logits_s = out["start_logits"]
                logits_e = out["end_logits"]
                # Map predictions from shortened sequence back to original indices if hard deletion was used
                pred_start_short = logits_s.argmax(dim=1)
                pred_end_short = logits_e.argmax(dim=1)
                keep_indices = out.get("keep_indices")
                kept_lengths = out.get("kept_lengths")
                if keep_indices is not None and kept_lengths is not None:
                    max_valid = (kept_lengths - 1).clamp(min=0)
                    pred_start_short = torch.minimum(pred_start_short, max_valid)
                    pred_end_short = torch.minimum(pred_end_short, max_valid)
                    pred_start = keep_indices.gather(1, pred_start_short.unsqueeze(1)).squeeze(1)
                    pred_end = keep_indices.gather(1, pred_end_short.unsqueeze(1)).squeeze(1)
                else:
                    pred_start, pred_end = pred_start_short, pred_end_short
                correct = (pred_start == start_pos) & (pred_end == end_pos)
            else:
                labels = batch["label"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    gate_regularizer_weight=0.0,
                )
                preds = out["logits"].argmax(dim=1)
                correct = preds == labels
            gate = out.get("gate")
            if gate is None:
                break
            # Per-sample deletion rate: (batch, seq_len) -> (batch,)
            del_rate = (gate < threshold).float().mean(dim=1)
            B = gate.size(0)
            for i in range(B):
                if correct[i].item():
                    continue
                dr = del_rate[i].item()
                if dr < args.min_deletion_rate:
                    continue
                idx = global_idx + i
                if args.dataset == "snli":
                    r = raw_val[idx]
                    text = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
                    label = int(labels[i].item())
                    pred = int(preds[i].item())
                    label_str = SNLI_LABELS[label] if label < 3 else str(label)
                    pred_str = SNLI_LABELS[pred] if pred < 3 else str(pred)
                else:
                    row = raw_val[idx]
                    text = {"question": row["_question"], "context": row["_context"][:300] + "...", "answer": row["_answer_text"]}
                    label = (int(start_pos[i].item()), int(end_pos[i].item()))
                    pred = (int(pred_start[i].item()), int(pred_end[i].item()))
                    label_str = str(label)
                    pred_str = str(pred)
                keep_mask = (gate[i : i + 1] > threshold).cpu()
                stats = analyze_dropped_tokens(tokenizer, batch["input_ids"][i : i + 1], keep_mask, batch_index=0)
                dropped = stats["dropped_tokens"][: args.max_dropped_tokens]
                case = {
                    "idx": idx,
                    "dataset": args.dataset,
                    "deletion_rate": round(dr, 4),
                    "label": label_str,
                    "pred": pred_str,
                    "text": text,
                    "dropped_tokens": dropped,
                    "dropped_by_type": stats["dropped_by_type"],
                }
                error_cases.append(case)
                if len(error_cases) >= args.max_cases:
                    break
            global_idx += B
            if len(error_cases) >= args.max_cases:
                break

    print(f"Found {len(error_cases)} error cases (wrong + deletion_rate >= {args.min_deletion_rate:.0%})")
    for i, c in enumerate(error_cases[:10]):
        print(f"\n--- Case {i+1} (idx={c['idx']}, del_rate={c['deletion_rate']:.1%}) ---")
        print(f"  Label: {c['label']}  Pred: {c['pred']}")
        if args.dataset == "snli":
            print(f"  Premise: {c['text']['premise'][:120]}...")
            print(f"  Hypothesis: {c['text']['hypothesis'][:120]}...")
        else:
            print(f"  Q: {c['text']['question'][:100]}...")
            print(f"  Answer: {c['text']['answer']}")
        print(f"  Dropped ({len(c['dropped_tokens'])} shown): {c['dropped_tokens'][:15]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for c in error_cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(error_cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
