#!/usr/bin/env python3
"""
Train MrBERT on a real dataset (e.g. GLUE MRPC or IMDB).
Loss: L = L_CE + alpha * L_G (paper Eq.3); alpha = gate_weight or from PI controller.
Optional two-phase: Phase A = gate adaptation (first N steps with gate regularizer),
Phase B = task finetuning (same loss, typically lower gate_weight).
"""
import argparse
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

from mrbert import MrBertForSequenceClassification
from mrbert.pi_controller import PIController


def get_dataset_and_tokenizer(dataset_name: str, max_length: int):
    """Load dataset and tokenizer. Supports 'mrpc', 'imdb', 'sst2', 'snli'.
    Returns (train_ds, val_ds, tokenizer, num_labels)."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if dataset_name == "mrpc":
        ds = load_dataset("glue", "mrpc")
        def tokenize(ex):
            return tokenizer(
                ex["sentence1"],
                ex["sentence2"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
    elif dataset_name == "imdb":
        ds = load_dataset("imdb")
        def tokenize(ex):
            return tokenizer(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
    elif dataset_name == "sst2":
        ds = load_dataset("glue", "sst2")
        def tokenize(ex):
            return tokenizer(
                ex["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 2
    elif dataset_name == "snli":
        ds = load_dataset("snli")
        # SNLI has label -1 for some invalid rows; keep only 0, 1, 2
        for split in list(ds.keys()):
            ds[split] = ds[split].filter(lambda x: x["label"] >= 0)
        def tokenize(ex):
            return tokenizer(
                ex["premise"],
                ex["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        num_labels = 3  # entailment (0), neutral (1), contradiction (2)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use mrpc, imdb, sst2, or snli.")

    # Keep 'label' (and 'idx' if present)
    train_remove_cols = [c for c in ds["train"].column_names if c not in ("label", "idx")]
    train_ds = ds["train"].map(tokenize, batched=True, remove_columns=train_remove_cols)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if "validation" in ds:
        val_remove_cols = [c for c in ds["validation"].column_names if c not in ("label", "idx")]
        val_ds = ds["validation"].map(tokenize, batched=True, remove_columns=val_remove_cols)
    else:
        val_remove_cols = [c for c in ds["test"].column_names if c not in ("label", "idx")]
        val_ds = ds["test"].map(tokenize, batched=True, remove_columns=val_remove_cols)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, tokenizer, num_labels


def main():
    parser = argparse.ArgumentParser(description="Train MrBERT on a classification dataset")
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2", "snli"],
                        help="Dataset: mrpc, imdb, sst2, or snli")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gate_weight", type=float, default=1e-4, help="Gate regularizer weight (alpha)")
    parser.add_argument("--use_pi", action="store_true", help="Use PI controller for target deletion ratio")
    parser.add_argument("--target_deletion", type=float, default=0.5, help="Target deletion ratio (for PI)")
    parser.add_argument("--phase1_steps", type=int, default=0,
                        help="Phase A: first N steps with --phase1_gate_weight (gate adaptation); 0 = disabled")
    parser.add_argument("--phase1_gate_weight", type=float, default=1e-3,
                        help="Gate regularizer weight during phase 1 (gate adaptation)")
    parser.add_argument("--output_result", type=str, default=None,
                        help="Append one JSON line with dataset, gate_weight, use_pi, target_deletion, val_acc to this file")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Use only this many training samples (for quick GPU smoke test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.dataset}")

    train_ds, val_ds, tokenizer, num_labels = get_dataset_and_tokenizer(args.dataset, args.max_length)
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"Using first {len(train_ds)} train samples (--max_train_samples)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MrBertForSequenceClassification.from_bert_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        gate_layer_index=3,
        gate_k=-30.0,
        attn_implementation="eager",
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pi = PIController(target_ratio=args.target_deletion, kp=0.5, ki=1e-5) if args.use_pi else None
    gate_regularizer_weight = args.gate_weight

    # Tracking: deletion rate, gate stats, time (baseline has no gate)
    gate_k, threshold_ratio = -30.0, 0.5
    threshold = gate_k * threshold_ratio
    n_gate_batches = 0
    sum_del_rate = 0.0
    sum_gate_mean = 0.0
    sum_gate_var = 0.0
    final_avg_train_loss = None

    start_time = time.time()
    total_steps = len(train_loader) * args.epochs
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # Two-phase: Phase A = gate adaptation (first phase1_steps), Phase B = task finetuning
            alpha = args.phase1_gate_weight if (args.phase1_steps > 0 and step < args.phase1_steps) else gate_regularizer_weight
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                gate_regularizer_weight=alpha,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if pi is not None and out.get("gate") is not None:
                gate_regularizer_weight = pi.step(out["gate"], gate_k=gate_k)
            # Accumulate gate stats (only when gate is present, i.e. MrBERT)
            gate = out.get("gate")
            if gate is not None:
                with torch.no_grad():
                    del_rate = (gate < threshold).float().mean().item()
                    sum_del_rate += del_rate
                    sum_gate_mean += gate.mean().item()
                    sum_gate_var += gate.var().item()
                    n_gate_batches += 1
            if step == args.phase1_steps and args.phase1_steps > 0:
                print(f"  Phase A done at step {step}; switching to Phase B (gate_weight={gate_regularizer_weight:.2e})")
            if step % 100 == 0:
                print(f"  step {step}/{total_steps} loss: {loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)
        final_avg_train_loss = avg_train
        print(f"Epoch {epoch + 1}/{args.epochs} avg train loss: {avg_train:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gate_regularizer_weight=0.0,
            )
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Validation accuracy: {val_acc:.4f} ({correct}/{total})")

    duration_sec = round(time.time() - start_time, 1)
    actual_del_rate = round(sum_del_rate / n_gate_batches, 4) if n_gate_batches else None
    gate_mean = round(sum_gate_mean / n_gate_batches, 4) if n_gate_batches else None
    gate_std = round((sum_gate_var / n_gate_batches) ** 0.5, 4) if n_gate_batches else None
    alpha_final = round(gate_regularizer_weight, 6) if (pi is not None and n_gate_batches) else None
    if n_gate_batches:
        print(f"Actual deletion rate (train): {actual_del_rate:.2%}  |  gate mean: {gate_mean}  std: {gate_std}  |  alpha_final: {alpha_final}  |  time: {duration_sec}s")

    if args.output_result:
        import json
        from pathlib import Path
        Path(args.output_result).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": args.dataset,
            "gate_weight": args.gate_weight,
            "use_pi": args.use_pi,
            "target_deletion": args.target_deletion,
            "val_acc": round(val_acc, 4),
            "avg_train_loss": round(final_avg_train_loss, 4) if final_avg_train_loss is not None else None,
            "actual_deletion_rate": actual_del_rate,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "alpha_final": alpha_final,
            "duration_sec": duration_sec,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        with open(args.output_result, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Result appended to {args.output_result}")

    print("Done.")


if __name__ == "__main__":
    main()
