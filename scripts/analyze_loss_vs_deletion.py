#!/usr/bin/env python3
"""
Analyze correlation between per-example loss and per-example deletion rate.
Answers: "When the model deletes more tokens on an example, does it have higher loss?"
If the model deletes wisely, we might see no strong positive correlation (or negative:
more deletion on "easy" examples). Strong positive correlation suggests over-deletion
hurts (model deletes more where it's already struggling).

Usage (from project root):
  python -m scripts.analyze_loss_vs_deletion --dataset mrpc
  python -m scripts.analyze_loss_vs_deletion --dataset tydiqa --max_length 256 --max_samples 2000 --output results/loss_vs_del_tydiqa.json
  python -m scripts.analyze_loss_vs_deletion --dataset sst2 --checkpoint path/to/model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow importing train_mrbert from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_mrbert import get_dataset_and_tokenizer
from mrbert import MrBertForSequenceClassification, MrBertForQuestionAnswering


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation (both 1D)."""
    x, y = x.float(), y.float()
    x = x - x.mean()
    y = y - y.mean()
    c = (x * y).sum()
    d = (x.pow(2).sum().clamp(min=1e-12) * y.pow(2).sum().clamp(min=1e-12)).sqrt()
    return (c / d).item()


def spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation (both 1D)."""
    x, y = x.float(), y.float()
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return pearson_r(rx, ry)


def main():
    parser = argparse.ArgumentParser(
        description="Correlation between per-example loss and per-example deletion rate"
    )
    parser.add_argument("--dataset", type=str, default="mrpc",
                        choices=["mrpc", "imdb", "sst2", "snli", "tydiqa"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max examples to evaluate (for speed)")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt); if not set, use pretrained BERT + random gate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/loss_vs_deletion_{dataset}.json)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_k, threshold_ratio = -30.0, 0.5
    threshold = gate_k * threshold_ratio

    print(f"Loading dataset: {args.dataset}")
    train_ds, val_ds, tokenizer, num_labels, task_type = get_dataset_and_tokenizer(
        args.dataset, args.max_length
    )
    is_qa = task_type == "qa"
    ds = val_ds if args.split == "validation" else train_ds
    n_total = len(ds)
    if n_total > args.max_samples:
        ds = ds.select(range(args.max_samples))
    print(f"Using {len(ds)} samples from {args.split}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    print(f"Building model ({'QA' if is_qa else 'classification'})")
    if is_qa:
        model = MrBertForQuestionAnswering.from_bert_pretrained(
            "bert-base-uncased",
            gate_layer_index=3,
            gate_k=gate_k,
            attn_implementation="eager",
        )
    else:
        model = MrBertForSequenceClassification.from_bert_pretrained(
            "bert-base-uncased",
            num_labels=num_labels or 2,
            gate_layer_index=3,
            gate_k=gate_k,
            attn_implementation="eager",
        )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    model = model.to(device)
    model.eval()

    all_loss = []
    all_del_rate = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if is_qa:
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    gate_regularizer_weight=0.0,
                )
                start_loss = F.cross_entropy(
                    out["start_logits"], start_positions, reduction="none"
                )
                end_loss = F.cross_entropy(
                    out["end_logits"], end_positions, reduction="none"
                )
                loss_per = ((start_loss + end_loss) / 2).cpu()
            else:
                labels = batch["label"].to(device)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    gate_regularizer_weight=0.0,
                )
                loss_per = F.cross_entropy(
                    out["logits"], labels, reduction="none"
                ).cpu()
            gate = out["gate"]
            if gate is None:
                print("No gate in output (baseline model?); skipping analysis.")
                return
            del_rate_per = (gate < threshold).float().mean(dim=1).cpu()
            all_loss.append(loss_per)
            all_del_rate.append(del_rate_per)

    loss_vec = torch.cat(all_loss, dim=0)
    del_vec = torch.cat(all_del_rate, dim=0)
    n = loss_vec.numel()
    pearson = pearson_r(loss_vec, del_vec)
    spearman = spearman_rho(loss_vec, del_vec)

    out_path = args.output or str(ROOT / "results" / f"loss_vs_deletion_{args.dataset}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Save summary + a subset of points for plotting (max 1000)
    cap = 1000
    indices = torch.randperm(n)[:cap] if n > cap else torch.arange(n)
    scatter = [
        {"loss": round(loss_vec[i].item(), 4), "deletion_rate": round(del_vec[i].item(), 4)}
        for i in indices
    ]
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "n_samples": n,
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "scatter_sample": scatter,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_path}")

    print("\n--- Loss vs deletion rate (per-example) ---")
    print(f"  Pearson r  = {pearson:.4f}")
    print(f"  Spearman ρ = {spearman:.4f}  (n = {n})")
    if pearson > 0.15:
        print("  Interpretation: Positive correlation — higher deletion tends to go with higher loss (model may be deleting important tokens on harder examples).")
    elif pearson < -0.15:
        print("  Interpretation: Negative correlation — higher deletion tends to go with lower loss (model may delete more on easier examples).")
    else:
        print("  Interpretation: Weak correlation — deletion rate and loss are largely independent at the example level.")
    print("Done.")


if __name__ == "__main__":
    main()
