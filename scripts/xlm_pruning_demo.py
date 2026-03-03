#!/usr/bin/env python3
"""
Minimal qualitative demo for MrXLMRobertaModel.

Loads an XLM-R backbone wrapped with the delete gate (MrXLMRobertaModel),
feeds a couple of toy sentences, and prints the sequence length before
and after hard deletion, along with gate statistics. Intended as a
lightweight architecture sanity check, not a full experiment.
"""

import torch
from transformers import XLMRobertaTokenizerFast

from mrbert import MrXLMRobertaModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "xlm-roberta-base"
    print(f"Loading MrXLMRobertaModel from {model_name}...")
    model = MrXLMRobertaModel.from_pretrained_xlm(model_name)
    model.to(device)
    model.eval()

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)

    sentences = [
        "This is a short English sentence for XLM-R.",
        "Voici une autre phrase pour tester le gate de suppression.",
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

    print("\n=== Shapes ===")
    print("input_ids:", tuple(input_ids.shape))
    print("last_hidden_state:", tuple(out.last_hidden_state.shape))
    print("kept_lengths:", kept_lengths.tolist())
    print("keep_indices shape:", tuple(keep_indices.shape))
    print("gate shape:", tuple(gate.shape))

    threshold = model.gate_k * model.gate_threshold_ratio
    del_rate = (gate < threshold).float().mean().item()
    print(f"\nEstimated deletion rate (hard threshold): {del_rate:.2%}")


if __name__ == "__main__":
    main()

