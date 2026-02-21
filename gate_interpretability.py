#!/usr/bin/env python3
"""
Gate interpretability: show which tokens are kept vs deleted, and optional token-type stats.
Usage:
  python gate_interpretability.py
  python gate_interpretability.py --sentences "First sentence." "Second one."
  python gate_interpretability.py --stats   # print deletion rate by token type (punctuation / word / subword)
"""
import argparse
import re

import torch
from transformers import BertTokenizer

from mrbert import MrBertForSequenceClassification


def get_gate_and_tokens(model, tokenizer, sentences, device, max_length=128):
    """Run model in train mode (soft deletion), return gate scores and token lists."""
    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        out = model.mrbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_gate=True,
            use_soft_deletion=True,
        )
    if isinstance(out, tuple):
        _, gate = out
    else:
        gate = None
    gate = gate.cpu()  # (batch, seq_len)
    ids = input_ids.cpu()
    mask = attention_mask.cpu()
    return gate, ids, mask, tokenizer


def token_type(tok_str: str) -> str:
    """Classify as punctuation, subword (starts with ##), or word."""
    if not tok_str or tok_str in ["[CLS]", "[SEP]", "[PAD]"]:
        return "special"
    if tok_str.startswith("##"):
        return "subword"
    if re.match(r"^[\W_]+$", tok_str):
        return "punctuation"
    return "word"


def main():
    parser = argparse.ArgumentParser(description="Gate interpretability")
    parser.add_argument("--sentences", nargs="+", default=[
        "The cat sat on the mat.",
        "This movie was great and I loved it.",
    ], help="Sentences to analyze")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--threshold_ratio", type=float, default=0.5,
                        help="Token deleted if gate < gate_k * this (default 0.5)")
    parser.add_argument("--stats", action="store_true", help="Print deletion rate by token type")
    parser.add_argument("--num_labels", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MrBertForSequenceClassification.from_bert_pretrained(
        "bert-base-uncased",
        num_labels=args.num_labels,
        gate_layer_index=3,
        gate_k=-30.0,
        attn_implementation="eager",
    )
    model = model.to(device)

    gate_k = -30.0
    threshold = gate_k * args.threshold_ratio

    gate, input_ids, attention_mask, tokenizer = get_gate_and_tokens(
        model, tokenizer, args.sentences, device, args.max_length
    )

    batch_size = gate.shape[0]
    seq_len = gate.shape[1]

    if args.stats:
        by_type = {"word": [], "subword": [], "punctuation": [], "special": []}
        for b in range(batch_size):
            for i in range(seq_len):
                if attention_mask[b, i].item() == 0:
                    continue
                tok_id = input_ids[b, i].item()
                tok_str = tokenizer.decode([tok_id])
                ttype = token_type(tok_str)
                g = gate[b, i].item()
                deleted = 1 if g < threshold else 0
                by_type[ttype].append(deleted)
        print("Deletion rate by token type (1 = deleted):")
        for ttype, vals in by_type.items():
            if vals:
                rate = sum(vals) / len(vals)
                print(f"  {ttype}: {rate:.2%} ({sum(vals)}/{len(vals)})")
        return

    print("Per-token gate score (deleted if G < k/2 = -15):")
    print()
    for b in range(batch_size):
        toks = []
        for i in range(seq_len):
            if attention_mask[b, i].item() == 0:
                continue
            tok_id = input_ids[b, i].item()
            tok_str = tokenizer.decode([tok_id])
            g = gate[b, i].item()
            kept = "KEEP" if g >= threshold else "DEL"
            toks.append((tok_str, g, kept))
        print(f"Sentence {b + 1}:")
        for tok_str, g, kept in toks:
            print(f"  {tok_str!r}  G={g:.1f}  [{kept}]")
        print()


if __name__ == "__main__":
    main()
