#!/usr/bin/env python3
"""
Read results/train_results.jsonl and results/latency_results.json; update RESULTS.md with full comparison table.
Usage: python scripts/aggregate_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results" / "train_results.jsonl"
LATENCY_FILE = ROOT / "results" / "latency_results.json"
OUTPUT_FILE = ROOT / "RESULTS.md"


def _fmt(x, fmt=None):
    if x is None:
        return "—"
    if fmt is None and isinstance(x, float) and 0 <= x <= 1:
        return f"{x:.2%}"
    if fmt:
        return fmt % x
    return str(x)


def main():
    rows = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    known_mrbert = {"mrpc": 0.6838, "imdb": 0.8478}

    # Main table: Model | Dataset | Deletion (target) | Actual Del% | Val Acc | Avg Loss | Time(s) | Alpha
    lines = [
        "# Experiment Results",
        "",
        "Comparison: Baseline BERT (no gate) vs MrBERT (gate + PI, target deletion ~50%).",
        "",
        "| Model | Dataset | Deletion | Actual Del% | Val Acc | Avg Loss | Time(s) | Alpha |",
        "|-------|---------|----------|-------------|--------|----------|---------|-------|",
    ]

    seen = set()
    for r in rows:
        dataset = r["dataset"]
        gw = r["gate_weight"]
        use_pi = r.get("use_pi", False)
        acc = r["val_acc"]
        avg_loss = r.get("avg_train_loss")
        actual_del = r.get("actual_deletion_rate")
        duration = r.get("duration_sec")
        alpha = r.get("alpha_final")
        if gw == 0:
            model = "Baseline BERT"
            deletion = "0%"
        else:
            model = "MrBERT"
            deletion = "~50%" if use_pi else f"gw={gw}"
        key = (model.lower().split()[0], dataset)
        if key not in seen:
            seen.add(key)
            actual_str = _fmt(actual_del) if actual_del is not None else "—"
            loss_str = _fmt(avg_loss, "%.4f") if avg_loss is not None else "—"
            time_str = str(duration) if duration is not None else "—"
            alpha_str = _fmt(alpha, "%.2e") if alpha is not None else "—"
            lines.append(f"| {model} | {dataset.upper()} | {deletion} | {actual_str} | {acc:.2%} | {loss_str} | {time_str} | {alpha_str} |")

    for dataset, acc in known_mrbert.items():
        if ("mrbert", dataset) not in seen:
            seen.add(("mrbert", dataset))
            lines.append(f"| MrBERT | {dataset.upper()} | ~50% | — | {acc:.2%} | — | — | — |")

    for dataset in ["mrpc", "imdb"]:
        if ("baseline", dataset) not in seen:
            lines.append(f"| Baseline BERT | {dataset.upper()} | 0% | — | *(run baseline)* | — | — | — |")

    # Latency section: from latency_results.json if present
    lines.append("")
    lines.append("## Latency & sequence length (from latency_benchmark.py)")
    lines.append("")
    if LATENCY_FILE.exists():
        with open(LATENCY_FILE) as f:
            lat = json.load(f)
        lines.append("| Setting | Seq length | Avg time (ms) |")
        lines.append("|---------|------------|---------------|")
        lines.append(f"| Baseline BERT | {lat.get('seq_len_original', '—')} | {lat.get('baseline_ms', '—')} |")
        lines.append(f"| MrBERT (soft) | {lat.get('seq_len_soft', '—')} | — |")
        lines.append(f"| MrBERT (hard) | {lat.get('seq_len_hard', '—')} | {lat.get('mrbert_ms', '—')} |")
        sp = lat.get("speedup_pct")
        if sp is not None:
            lines.append("")
            lines.append(f"**Speedup:** {sp:.1f}% (positive = MrBERT faster).")
    else:
        lines.append("| Setting | Seq length (after gate) | Avg time (CPU) |")
        lines.append("|---------|-------------------------|----------------|")
        lines.append("| Soft deletion (train) | 512 | — |")
        lines.append("| Hard deletion (eval)  | 275 | ~1504 ms |")
        lines.append("| Baseline BERT         | 512 | ~1322 ms |")
    lines.append("")
    lines.append("Run: `python latency_benchmark.py --output_result results/latency_results.json` to refresh.")
    lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"Updated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
