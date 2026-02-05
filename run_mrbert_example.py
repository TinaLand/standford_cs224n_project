#!/usr/bin/env python3
"""
Minimal example: load BERT as MrBERT, run forward with gate regularizer, and optional PI controller.
"""
import torch
from transformers import BertTokenizer

from mrbert import MrBertConfig, MrBertForSequenceClassification
from mrbert.pi_controller import PIController


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Build tiny batch
    texts = ["This is a short sentence.", "Another example for MrBERT."]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = torch.tensor([0, 1], device=device)

    # Load MrBERT from BERT (gate randomly initialized)
    # Use eager attention to avoid SDPA mask shape issues with our custom encoder path
    model = MrBertForSequenceClassification.from_bert_pretrained(
        "bert-base-uncased",
        num_labels=2,
        gate_layer_index=3,
        gate_k=-30.0,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.train()

    # Optional: PI controller for target deletion ratio 0.5
    pi = PIController(target_ratio=0.5, kp=0.5, ki=1e-5)
    gate_regularizer_weight = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for step in range(3):
        optimizer.zero_grad()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gate_regularizer_weight=gate_regularizer_weight,
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        if out.get("gate") is not None:
            gate_regularizer_weight = pi.step(out["gate"], gate_k=-30.0)
        print(f"Step {step + 1} loss: {loss.item():.4f}")

    # Eval: hard deletion (faster)
    model.eval()
    with torch.no_grad():
        result = model.mrbert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_gate=False,
            use_soft_deletion=False,
        )
        pooler = result.pooler_output if hasattr(result, "pooler_output") else result[1]
        logits = model.classifier(model.dropout(pooler))
    print("Eval logits shape:", logits.shape)
    print("Done.")


if __name__ == "__main__":
    main()
