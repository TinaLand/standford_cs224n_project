"""
Diagnostics and logging for MrBERT: parameters, tensor shapes, dropped-token analysis,
and loss/PI state. Use --log_level 1|2|3 in train_mrbert.py to enable.
"""
from __future__ import annotations

import re
from typing import Any

import torch
from torch.nn import Module


# Special token IDs in BERT vocab (bert-base-uncased)
CLS_ID = 101
SEP_ID = 102
PAD_ID = 0


def count_parameters(module: Module, name: str = "") -> tuple[int, int]:
    """Return (total_params, trainable_params) for module (and optionally its named submodule)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_parameter_summary(model: Module, label: str = "MrBERT") -> dict[str, int]:
    """
    Log total, trainable, and DeleteGate-only parameter counts.
    Returns dict with keys: total, trainable, gate_params.
    """
    total, trainable = count_parameters(model)
    gate_params = 0
    if hasattr(model, "mrbert") and hasattr(model.mrbert, "delete_gate"):
        gate_params, _ = count_parameters(model.mrbert.delete_gate)
    elif hasattr(model, "delete_gate"):
        gate_params, _ = count_parameters(model.delete_gate)
    out = {"total": total, "trainable": trainable, "gate_params": gate_params}
    print(f"[{label}] Parameters: total={total:,}  trainable={trainable:,}  DeleteGate={gate_params:,}")
    return out


def theoretical_compute_saved_pct(
    seq_before: int,
    seq_after: int,
    num_layers: int = 12,
    gate_layer_index: int = 3,
) -> float:
    """
    Approximate MACs saved by hard deletion (paper Appendix C).
    Self-attention cost is O(seq^2) per layer. Layers 0..gate_layer use seq_before;
    layers (gate_layer+1)..(num_layers-1) use seq_after.
    Returns fraction in [0, 1] (e.g. 0.425 for 42.5%).
    """
    if seq_before <= 0:
        return 0.0
    layers_before = gate_layer_index + 1
    layers_after = num_layers - layers_before
    total_before = num_layers * (seq_before ** 2)
    total_after = layers_before * (seq_before ** 2) + layers_after * (seq_after ** 2)
    if total_before <= 0:
        return 0.0
    saved = (total_before - total_after) / total_before
    return max(0.0, min(1.0, saved))


def log_shape_after_gate(
    batch_size: int,
    seq_before: int,
    seq_after: int,
    hidden_size: int,
    gate_layer_index: int,
    num_layers: int = 12,
) -> None:
    """Log tensor shape change at hard deletion and theoretical compute saved (paper Appendix C)."""
    print(
        f"  [Gate] After layer {gate_layer_index}: "
        f"[{batch_size}, {seq_before}, {hidden_size}] -> [{batch_size}, {seq_after}, {hidden_size}] "
        f"(kept {seq_after}/{seq_before} tokens)"
    )
    pct = theoretical_compute_saved_pct(seq_before, seq_after, num_layers, gate_layer_index)
    print(f"  [Gate] Theoretical compute saved (MACs, Appendix C): ~{pct * 100:.1f}%")


def _token_type(token_str: str, token_id: int) -> str:
    """Classify token: special, subword, punctuation, word."""
    if token_id in (CLS_ID, SEP_ID, PAD_ID):
        return "special"
    if token_str.startswith("##"):
        return "subword"
    if re.match(r"^[\W_]+$", token_str) or token_str in ("'", '"', ".", ",", "!", "?", "-", ";"):
        return "punctuation"
    return "word"


def analyze_dropped_tokens(
    tokenizer: Any,
    input_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    batch_index: int = 0,
) -> dict[str, Any]:
    """
    For one batch item, analyze which tokens were kept vs dropped and their types.
    input_ids: (batch, seq_len), keep_mask: (batch, seq_len) True = kept.
    Returns dict with: dropped_tokens (list of str), kept_tokens, counts by type (dropped/kept).
    """
    ids = input_ids[batch_index].tolist()
    mask = keep_mask[batch_index].tolist()
    dropped_tokens = []
    kept_tokens = []
    dropped_by_type = {"special": 0, "subword": 0, "punctuation": 0, "word": 0}
    kept_by_type = {"special": 0, "subword": 0, "punctuation": 0, "word": 0}
    for i, (tid, kept) in enumerate(zip(ids, mask)):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            s = f"[id={tid}]"
        ttype = _token_type(s, tid)
        if kept:
            kept_tokens.append(s)
            kept_by_type[ttype] += 1
        else:
            dropped_tokens.append(s)
            dropped_by_type[ttype] += 1
    return {
        "dropped_tokens": dropped_tokens,
        "kept_tokens": kept_tokens,
        "dropped_by_type": dropped_by_type,
        "kept_by_type": kept_by_type,
        "n_dropped": len(dropped_tokens),
        "n_kept": len(kept_tokens),
    }


def print_dropped_token_summary(
    tokenizer: Any,
    input_ids: torch.Tensor,
    gate: torch.Tensor,
    threshold: float,
    batch_index: int = 0,
    max_show: int = 20,
) -> dict[str, Any]:
    """
    Compute keep_mask from gate and threshold, then analyze and print dropped-token stats.
    Optionally print first max_show dropped token strings.
    """
    keep_mask = (gate > threshold).to(input_ids.device)
    stats = analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index)
    n_d, n_k = stats["n_dropped"], stats["n_kept"]
    total = n_d + n_k
    print(f"  [Dropped tokens] batch_idx={batch_index}  kept={n_k}  dropped={n_d}  (del={n_d/total*100:.1f}%)")
    print(f"    by type dropped: {stats['dropped_by_type']}  kept: {stats['kept_by_type']}")
    if stats["dropped_tokens"] and max_show > 0:
        show = [f"[{t}]" for t in stats["dropped_tokens"][:max_show]]
        print(f"    dropped tokens (highlighted): {' '.join(show)}")
    return stats


def aggregate_dropped_stats(stats_list: list[dict]) -> dict[str, float]:
    """Aggregate multiple analyze_dropped_tokens outputs into mean counts by type."""
    n = len(stats_list)
    if n == 0:
        return {}
    out = {"n_samples": n}
    for key in ("dropped_by_type", "kept_by_type"):
        agg = {"special": 0.0, "subword": 0.0, "punctuation": 0.0, "word": 0.0}
        for s in stats_list:
            for k, v in s.get(key, {}).items():
                agg[k] = agg.get(k, 0) + v
        for k in agg:
            agg[k] = round(agg[k] / n, 2)
        out[key] = agg
    total_dropped = sum(s["n_dropped"] for s in stats_list)
    total_kept = sum(s["n_kept"] for s in stats_list)
    out["avg_dropped"] = round(total_dropped / n, 2)
    out["avg_kept"] = round(total_kept / n, 2)
    return out


# ---------------------------------------------------------------------------
# Gate interpretability demo (merged from gate_interpretability.py)
# Run: python -m mrbert.diagnostics   or   python mrbert/diagnostics.py
# ---------------------------------------------------------------------------

def run_gate_interpretability_demo(
    sentences: list[str],
    max_length: int = 64,
    threshold_ratio: float = 0.5,
    stats_only: bool = False,
    num_labels: int = 2,
) -> None:
    """Run model on sentences and print per-token keep/delete or deletion-by-type stats."""
    from transformers import BertTokenizer
    from mrbert import MrBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MrBertForSequenceClassification.from_bert_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        gate_layer_index=3,
        gate_k=-30.0,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()

    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    gate_k = -30.0
    threshold = gate_k * threshold_ratio

    with torch.no_grad():
        out = model.mrbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_gate=True,
            use_soft_deletion=True,
        )
    gate = out[1] if isinstance(out, tuple) else None
    if gate is None:
        print("No gate in output.")
        return
    gate = gate.cpu()
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    keep_mask = (gate > threshold).to(torch.bool)

    batch_size, seq_len = gate.shape[0], gate.shape[1]

    if stats_only:
        by_type: dict[str, list[int]] = {"word": [], "subword": [], "punctuation": [], "special": []}
        for b in range(batch_size):
            stats = analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index=b)
            for ttype in ("word", "subword", "punctuation", "special"):
                dropped = stats["dropped_by_type"][ttype]
                kept = stats["kept_by_type"][ttype]
                total = dropped + kept
                if total > 0:
                    by_type[ttype].append(dropped / total)
        print("Deletion rate by token type:")
        for ttype, rates in by_type.items():
            if rates:
                print(f"  {ttype}: {sum(rates)/len(rates):.2%}")
        return

    print("Per-token gate score (deleted if G < k/2 = -15):")
    for b in range(batch_size):
        stats = analyze_dropped_tokens(tokenizer, input_ids, keep_mask, batch_index=b)
        toks_with_gate = []
        ids_b = input_ids[b].tolist()
        mask_b = attention_mask[b].tolist()
        gate_b = gate[b].tolist()
        for i in range(seq_len):
            if mask_b[i] == 0:
                continue
            s = tokenizer.decode([ids_b[i]])
            g = gate_b[i]
            kept = "KEEP" if g >= threshold else "DEL"
            toks_with_gate.append((s, g, kept))
        print(f"\nSentence {b + 1}:")
        for s, g, kept in toks_with_gate:
            print(f"  {s!r}  G={g:.1f}  [{kept}]")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gate interpretability: which tokens are kept vs deleted")
    p.add_argument("--sentences", nargs="+", default=["The cat sat on the mat.", "This movie was great and I loved it."])
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--threshold_ratio", type=float, default=0.5)
    p.add_argument("--stats", action="store_true", help="Print deletion rate by token type")
    p.add_argument("--num_labels", type=int, default=2)
    args = p.parse_args()
    run_gate_interpretability_demo(
        args.sentences,
        max_length=args.max_length,
        threshold_ratio=args.threshold_ratio,
        stats_only=args.stats,
        num_labels=args.num_labels,
    )
