#!/usr/bin/env python3
"""
Train MrBERT on a real dataset (e.g. GLUE MRPC or IMDB).
Loss: L = L_CE + alpha * L_G (paper Eq.3); alpha = gate_weight or from PI controller.
Optional two-phase: Phase A = gate adaptation (first N steps with gate regularizer),
Phase B = task finetuning (same loss, typically lower gate_weight).
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

from mrbert import MrBertForSequenceClassification
from mrbert.pi_controller import PIController


def get_dataset_and_tokenizer(dataset_name: str, max_length: int):
    """Load dataset and tokenizer. Supports 'mrpc', 'imdb', or 'sst2'."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if dataset_name == "mrpc":
        # GLUE MRPC: sentence pair similarity, 2 classes
        ds = load_dataset("glue", "mrpc", trust_remote_code=True)
        def tokenize(ex):
            return tokenizer(
                ex["sentence1"],
                ex["sentence2"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        text_keys = None
    elif dataset_name == "imdb":
        # IMDB sentiment: 2 classes
        ds = load_dataset("imdb", trust_remote_code=True)
        def tokenize(ex):
            return tokenizer(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        text_keys = None
    elif dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", trust_remote_code=True)
        def tokenize(ex):
            return tokenizer(
                ex["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        text_keys = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use mrpc, imdb, or sst2.")

    train_ds = ds["train"].map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if "validation" in ds:
        val_ds = ds["validation"].map(tokenize, batched=True, remove_columns=ds["validation"].column_names)
    else:
        val_ds = ds["test"].map(tokenize, batched=True, remove_columns=ds["test"].column_names)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train MrBERT on a classification dataset")
    parser.add_argument("--dataset", type=str, default="mrpc", choices=["mrpc", "imdb", "sst2"],
                        help="Dataset: mrpc (GLUE), imdb, or sst2")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.dataset}")

    train_ds, val_ds, tokenizer = get_dataset_and_tokenizer(args.dataset, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    num_labels = 2  # mrpc, imdb, sst2 are all binary
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
                gate_regularizer_weight = pi.step(out["gate"], gate_k=-30.0)
            if step == args.phase1_steps and args.phase1_steps > 0:
                print(f"  Phase A done at step {step}; switching to Phase B (gate_weight={gate_regularizer_weight:.2e})")
            if step % 100 == 0:
                print(f"  step {step}/{total_steps} loss: {loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)
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
    print("Done.")


if __name__ == "__main__":
    main()
