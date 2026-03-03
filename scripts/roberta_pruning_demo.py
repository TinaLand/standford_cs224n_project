#!/usr/bin/env python3
"""
Minimal qualitative demo for MrRoBERTa.

Loads a RoBERTa backbone wrapped with the delete gate (MrRobertaModel),
runs a few example sentences, and prints which subword tokens are kept
vs deleted under hard deletion. This is intended for qualitative analysis
in the report, not for training.
"""

import torch
from transformers import RobertaTokenizerFast

from mrbert import MrRobertaModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "roberta-base"
    print(f"Loading MrRobertaModel from {model_name}...")
    model = MrRobertaModel.from_pretrained_roberta(model_name)
    model.to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    sentences = [
        "This movie is absolutely fantastic, I loved every minute of it.",
        "The plot was boring and the acting was terrible.",
    ]

    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_soft_deletion=False,
            return_gate=True,
        )

    gate = out.gate  # (batch, seq_len)
    keep_indices = out.keep_indices  # (batch, max_kept)
    kept_lengths = out.kept_lengths  # (batch,)

    threshold = model.gate_k * model.gate_threshold_ratio

    for b, sent in enumerate(sentences):
        print("\n==================================================")
        print(f"Sentence {b}: {sent}")
        ids = input_ids[b].cpu().tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        g = gate[b].cpu()
        keep_mask = (g > threshold)

        # Skip padding tokens in the printout for clarity.
        pad_id = tokenizer.pad_token_id

        print("Token-by-token view (KEEP / DEL):")
        for i, (tok, score, keep) in enumerate(zip(toks, g.tolist(), keep_mask.tolist())):
            if pad_id is not None and ids[i] == pad_id:
                continue
            status = "KEEP" if keep else "DEL "
            print(f"  {i:2d}: {tok:15s}  G={score:7.3f}  [{status}]")

        print("\nAfter hard deletion (indices in original sequence):")
        kl = int(kept_lengths[b].item())
        kept_idx = keep_indices[b, :kl].cpu().tolist()
        kept_tokens = [toks[i] for i in kept_idx]
        print(f"  kept_lengths = {kl}")
        print(f"  keep_indices = {kept_idx}")
        print(f"  kept tokens  = {kept_tokens}")


if __name__ == "__main__":
    main()

