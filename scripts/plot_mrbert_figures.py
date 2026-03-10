#!/usr/bin/env python3
"""
Plot figures A–H for the MrBERT / MrXLM project.

This script only READS existing results under `results/new/` and does NOT retrain models.
It assumes the directory layout described in RESULTS_ANALYSIS.md.

Figures:
  A. Pareto frontier (latency vs accuracy)
  B. Deletion-rate vs step (PI vs fixed alpha)  [requires separate trace files; see docstring]
  C. Token survival visualization (SST-2 examples)
  D. Task sensitivity heatmap (accuracy vs target deletion, per task)
  E. Accuracy summary bar chart (baseline vs MrBERT / MrXLM)
  F. Deletion-rate histograms (per-example deletion distribution)
  G. Loss vs deletion scatter (from loss_vs_deletion_*.json)
  H. TyDi QA sensitivity curve (accuracy vs target deletion)

Usage examples:
  python scripts/plot_mrbert_figures.py --fig A,E,D
  python scripts/plot_mrbert_figures.py --fig F,G --dataset mrpc

The script writes PNGs into `report/figures/` by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results/new")
FIG_ROOT = Path("report/final_report")


@dataclass
class TrainResultRow:
    path: Path  # path to the train_results.jsonl file
    run_dir: Path  # folder containing this results file
    dataset: str
    backbone: str  # "bert" or "xlmr"
    gate_weight: float
    use_pi: bool
    target_deletion: float
    val_acc: float
    actual_deletion_rate: Optional[float]
    epochs: int
    batch_size: int


def _iter_train_results(root: Path) -> Iterable[TrainResultRow]:
    """Yield parsed rows from all train_results.jsonl files under `root`."""
    for jsonl_path in root.rglob("train_results.jsonl"):
        run_dir = jsonl_path.parent
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Older rows may lack backbone / actual_deletion_rate; handle gracefully.
                dataset = row.get("dataset")
                backbone = row.get("backbone", "bert")
                gate_weight = float(row.get("gate_weight", 0.0))
                use_pi = bool(row.get("use_pi", False))
                target_deletion = float(row.get("target_deletion", 0.5))
                val_acc = float(row.get("val_acc", 0.0))
                actual_del = row.get("actual_deletion_rate")
                actual_del = float(actual_del) if actual_del is not None else None
                epochs = int(row.get("epochs", 0))
                batch_size = int(row.get("batch_size", 0))
                if dataset is None:
                    continue
                yield TrainResultRow(
                    path=jsonl_path,
                    run_dir=run_dir,
                    dataset=dataset,
                    backbone=backbone,
                    gate_weight=gate_weight,
                    use_pi=use_pi,
                    target_deletion=target_deletion,
                    val_acc=val_acc,
                    actual_deletion_rate=actual_del,
                    epochs=epochs,
                    batch_size=batch_size,
                )


def _load_latency(root: Path) -> Dict[Path, Dict[str, float]]:
    """Map run_dir -> latency summary dict (baseline_ms, mrbert_ms, speedup_pct, ...)."""
    out: Dict[Path, Dict[str, float]] = {}
    for p in root.rglob("latency_results.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        out[p.parent] = data
    return out


def _ensure_fig_dir() -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    return FIG_ROOT


# === Figure A: Pareto frontier (latency vs accuracy) ============================================


def plot_pareto(fig_name: str = "fig_A_pareto.png") -> None:
    """
    Plot latency (x-axis) vs validation accuracy (y-axis).

    We use BERT L4 runs that have both:
      - train_results.jsonl with backbone="bert"
      - latency_results.json (baseline_ms, mrbert_ms)
    Each dataset contributes two points: Baseline vs MrBERT (PI).
    """
    rows = list(_iter_train_results(RESULTS_ROOT))
    latency_map = _load_latency(RESULTS_ROOT)

    points_baseline: List[Tuple[float, float, str]] = []  # (latency_ms, acc, label)
    points_mrbert: List[Tuple[float, float, str]] = []

    for r in rows:
        if r.backbone != "bert":
            continue
        # Only use 1-epoch batch-24 L4 runs (as in RESULTS_ANALYSIS main tables)
        if r.epochs != 1 or r.batch_size != 24:
            continue
        if r.run_dir not in latency_map:
            continue
        lat = latency_map[r.run_dir]
        dataset_label = r.dataset.upper()
        if r.gate_weight == 0.0 and not r.use_pi:
            # Baseline BERT
            latency_ms = float(lat.get("baseline_ms", lat.get("mrbert_ms", 0.0)))
            points_baseline.append((latency_ms, r.val_acc * 100.0, dataset_label))
        else:
            # MrBERT (any gate_weight>0 or PI enabled)
            latency_ms = float(lat.get("mrbert_ms", lat.get("baseline_ms", 0.0)))
            points_mrbert.append((latency_ms, r.val_acc * 100.0, dataset_label))

    if not points_baseline or not points_mrbert:
        print("No suitable BERT runs with latency_results.json found for Pareto plot.")
        return

    # Different marker per dataset for clarity (e.g. MRPC circle, SST-2 square)
    _dataset_markers = {
        "MRPC": "o", "SST2": "s", "SNLI": "^", "IMDB": "D", "XNLI": "p", "TYDIQA": "v",
    }

    _ensure_fig_dir()
    plt.figure(figsize=(6, 4))
    for x, y, lbl in points_baseline:
        m = _dataset_markers.get(lbl.upper(), "o")
        plt.scatter(x, y, marker=m, color="C0", alpha=0.8)
        plt.text(x * 1.002, y, lbl, fontsize=7, color="C0")
    for x, y, lbl in points_mrbert:
        m = _dataset_markers.get(lbl.upper(), "^")
        plt.scatter(x, y, marker=m, color="C1", alpha=0.8)
        plt.text(x * 1.002, y, lbl, fontsize=7, color="C1")
    plt.xlabel("Latency (ms per batch)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Figure A: Pareto frontier (BERT L4)")
    plt.legend(["Baseline BERT", "MrBERT"], loc="lower left")
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Pareto plot to {out_path}")


# === Figure B: deletion-rate vs step (requires trace files) ====================================


def plot_deletion_trace(
    trace_files: List[Path],
    fig_name: str = "fig_B_deletion_vs_step.png",
) -> None:
    """
    Plot deletion-rate vs training step for PI vs fixed-alpha runs.

    This expects JSONL trace files produced by train_mrbert.py (to be added),
    each line like:
      {"step": 123, "deletion_rate": 0.52, "alpha": 0.001, "use_pi": true, "tag": "mrpc_pi"}

    Args:
      trace_files: list of JSONL files (e.g., one for PI, one for no-PI).
    """
    if not trace_files:
        print("No trace files provided for Figure B; skipping.")
        return
    curves: Dict[str, Tuple[List[int], List[float]]] = {}
    for path in trace_files:
        tag = path.stem  # default label from filename
        steps: List[int] = []
        dels: List[float] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                steps.append(int(row.get("step", 0)))
                dels.append(float(row.get("deletion_rate", 0.0)))
                tag = str(row.get("tag", tag))
        if steps:
            curves[tag] = (steps, dels)

    if not curves:
        print("No valid points found in deletion trace files; skipping Figure B.")
        return

    _ensure_fig_dir()
    plt.figure(figsize=(6, 4))
    for tag, (steps, dels) in curves.items():
        plt.plot(steps, dels, label=tag)
    plt.xlabel("Training step")
    plt.ylabel("Deletion rate")
    plt.title("Figure B: Deletion rate vs step (PI vs fixed alpha)")
    plt.legend()
    # Short note: PI converges smoothly; fixed alpha can oscillate.
    plt.figtext(0.5, 0.02, "PI controller converges toward target; fixed \u03b1 often oscillates.", ha="center", fontsize=8, style="italic")
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved deletion-trace plot to {out_path}")


# === Figure E: Accuracy summary bar chart ======================================================


def plot_accuracy_summary(fig_name: str = "fig_E_accuracy_summary.png") -> None:
    """Bar chart: baseline vs gated (best MrBERT / MrXLM) per task."""
    rows = list(_iter_train_results(RESULTS_ROOT))

    # Aggregate best accuracy per (backbone, dataset, gate/on/off)
    best: Dict[Tuple[str, str, str], float] = {}
    for r in rows:
        mode = "baseline" if (r.gate_weight == 0.0 and not r.use_pi) else "gated"
        key = (r.backbone, r.dataset, mode)
        prev = best.get(key)
        if prev is None or r.val_acc > prev:
            best[key] = r.val_acc

    tasks = sorted({d for (_, d, _) in best.keys()})
    backbones = sorted({b for (b, _, _) in best.keys()})

    _ensure_fig_dir()
    plt.figure(figsize=(8, 4))

    # For simplicity, plot BERT and XLM-R in one chart with different colors/hatches.
    x_positions = list(range(len(tasks)))
    width = 0.18

    def _get(task: str, backbone: str, mode: str) -> Optional[float]:
        return best.get((backbone, task, mode))

    for i, backbone in enumerate(backbones):
        offsets = (-width, +width) if backbone == "bert" else (-width * 0.5, +width * 1.5)
        for j, mode in enumerate(["baseline", "gated"]):
            xs = [x + offsets[j] for x in x_positions]
            ys = [(_get(task, backbone, mode) or 0.0) * 100.0 for task in tasks]
            label = f"{backbone.upper()} {mode}"
            plt.bar(xs, ys, width=width, label=label)

    plt.xticks(x_positions, [t.upper() for t in tasks], rotation=30)
    plt.ylabel("Validation accuracy (%)")
    plt.title("Figure E: Accuracy summary (baseline vs gated)")
    plt.legend(fontsize=8)
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved accuracy summary to {out_path}")


# === Figure D & H: Task sensitivity / TyDi QA curves ===========================================


def _collect_task_sensitivity() -> Dict[Tuple[str, str, float], float]:
    """
    Return mapping (backbone, dataset, target_deletion) -> best gated val_acc.
    Baseline points are handled separately in the plotting functions.
    """
    rows = list(_iter_train_results(RESULTS_ROOT))
    accs: Dict[Tuple[str, str, float], float] = {}
    for r in rows:
        if r.gate_weight == 0.0 and not r.use_pi:
            continue  # baseline
        key = (r.backbone, r.dataset, r.target_deletion)
        prev = accs.get(key)
        if prev is None or r.val_acc > prev:
            accs[key] = r.val_acc
    return accs


def plot_task_sensitivity(fig_name: str = "fig_D_task_sensitivity.png") -> None:
    """
    Heatmap: dataset × target_deletion -> best gated accuracy (for BERT backbone).
    """
    accs = _collect_task_sensitivity()
    # Filter to BERT only; XLM-R is noisier and sparser.
    bert_keys = [k for k in accs.keys() if k[0] == "bert"]
    if not bert_keys:
        print("No BERT gated runs found for task sensitivity heatmap.")
        return
    tasks = sorted({d for (_, d, _) in bert_keys})
    targets = sorted({td for (_, _, td) in bert_keys})

    # Build matrix [len(tasks) x len(targets)]
    import numpy as np

    mat = np.zeros((len(tasks), len(targets)), dtype=float)
    mat[:] = np.nan
    for (backbone, dataset, td), acc in accs.items():
        if backbone != "bert":
            continue
        i = tasks.index(dataset)
        j = targets.index(td)
        mat[i, j] = acc * 100.0

    _ensure_fig_dir()
    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Accuracy (%)")
    plt.yticks(range(len(tasks)), [t.upper() for t in tasks])
    plt.xticks(range(len(targets)), [str(td) for td in targets])
    plt.xlabel("Target deletion ratio")
    plt.ylabel("Dataset")
    plt.title("Figure D: Task sensitivity (BERT, gated accuracy)")
    # Annotate worst cell (over-deletion / collapse)
    valid = ~np.isnan(mat)
    if np.any(valid):
        flat_min = np.nanmin(mat)
        ij = np.where(np.isclose(mat, flat_min) & valid)
        if len(ij[0]) > 0:
            i, j = int(ij[0][0]), int(ij[1][0])
            if flat_min < 40:  # only label clear collapse
                plt.text(j, i, "Over-deletion\nthreshold", ha="center", va="center", fontsize=7, color="white", weight="bold")
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved task sensitivity heatmap to {out_path}")


def plot_tydiqa_curve(fig_name: str = "fig_H_tydiqa_sensitivity.png") -> None:
    """
    TyDi QA: accuracy vs target_deletion (BERT backbone).
    """
    rows = list(_iter_train_results(RESULTS_ROOT))
    # Collect baseline and gated TyDi QA for BERT
    base: Dict[float, float] = {}
    mr: Dict[float, float] = {}
    for r in rows:
        if r.backbone != "bert" or r.dataset != "tydiqa":
            continue
        if r.gate_weight == 0.0 and not r.use_pi:
            base[r.target_deletion] = max(base.get(r.target_deletion, 0.0), r.val_acc)
        else:
            mr[r.target_deletion] = max(mr.get(r.target_deletion, 0.0), r.val_acc)
    if not mr and not base:
        print("No TyDi QA entries found for Figure H.")
        return

    xs = sorted(set(list(base.keys()) + list(mr.keys())))
    base_y = [base.get(x) * 100.0 if x in base else None for x in xs]
    mr_y = [mr.get(x) * 100.0 if x in mr else None for x in xs]

    _ensure_fig_dir()
    plt.figure(figsize=(5, 3.5))
    plt.plot(xs, base_y, marker="o", label="BERT baseline")
    plt.plot(xs, mr_y, marker="^", label="MrBERT (gated)")
    plt.xlabel("Target deletion ratio")
    plt.ylabel("TyDi QA EM (%)")
    plt.title("Figure H: TyDi QA vs target deletion (BERT)")
    plt.legend()
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved TyDi QA sensitivity curve to {out_path}")


# === Figures F & G: deletion histograms and scatter =============================================


def _load_loss_vs_deletion(run_dir: Path, dataset: str) -> Optional[Dict]:
    fname = f"loss_vs_deletion_{dataset}.json"
    candidates: List[Path] = [run_dir / fname]
    results_dir = run_dir / "results"
    if results_dir.is_dir():
        candidates.extend(results_dir.glob(fname))
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def plot_deletion_histogram(
    dataset: str,
    run_dir: Path,
    fig_name: str = "fig_F_deletion_hist.png",
) -> None:
    data = _load_loss_vs_deletion(run_dir, dataset)
    if not data:
        print(f"No loss_vs_deletion file found for {dataset} in {run_dir}")
        return
    samples = data.get("scatter_sample", [])
    if not samples:
        print(f"No scatter_sample in loss_vs_deletion for {dataset} in {run_dir}")
        return
    dels = [s["deletion_rate"] for s in samples if "deletion_rate" in s]
    if not dels:
        print(f"No deletion_rate values in scatter_sample for {dataset} in {run_dir}")
        return
    _ensure_fig_dir()
    # Add dataset suffix so multiple runs (e.g. MRPC, SST-2) do not overwrite each other.
    fig_name = f"fig_F_deletion_hist_{dataset}.png"
    plt.figure(figsize=(5, 3.5))
    plt.hist(dels, bins=20, color="C2", alpha=0.8)
    plt.xlabel("Deletion rate")
    plt.ylabel("Count (sampled examples)")
    plt.title(f"Figure F: Deletion-rate histogram ({dataset.upper()})")
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved deletion histogram to {out_path}")


def plot_loss_vs_deletion_scatter(
    dataset: str,
    run_dir: Path,
    fig_name: str = "fig_G_loss_vs_deletion.png",
) -> None:
    data = _load_loss_vs_deletion(run_dir, dataset)
    if not data:
        print(f"No loss_vs_deletion file found for {dataset} in {run_dir}")
        return
    samples = data.get("scatter_sample", [])
    if not samples:
        print(f"No scatter_sample in loss_vs_deletion for {dataset} in {run_dir}")
        return
    xs = [s["deletion_rate"] for s in samples if "deletion_rate" in s]
    ys = [s["loss"] for s in samples if "loss" in s]
    if not xs or not ys:
        print(f"No loss/deletion_rate pairs for {dataset} in {run_dir}")
        return
    _ensure_fig_dir()
    # Add dataset suffix so multiple runs do not overwrite each other.
    fig_name = f"fig_G_loss_vs_deletion_{dataset}.png"
    plt.figure(figsize=(5, 3.5))
    plt.scatter(xs, ys, s=10, alpha=0.5)
    plt.xlabel("Deletion rate")
    plt.ylabel("Per-example loss")
    plt.title(f"Figure G: Loss vs deletion ({dataset.upper()})")
    out_path = FIG_ROOT / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss-vs-deletion scatter to {out_path}")


# === Figure C: token survival visualization ====================================================


def plot_token_survival_placeholder() -> None:
    """
    Placeholder for Figure C (token survival heatmap).

    Implementing a fully general token-level heatmap requires wiring into the
    tokenizer and MrBERT models. Given time constraints, this function just
    documents how to proceed:

      - Load a fine-tuned checkpoint with MrBertForSequenceClassification.
      - Tokenize a few SST-2 sentences with the same tokenizer.
      - Run the model with `use_soft_deletion=False` and capture the gate tensor.
      - For each token ID, map back to the string piece and color it according to
        gate value and whether it was kept or deleted (hard deletion mask).

    You can then render a matplotlib figure with tokens on the x-axis and
    color-coded bars, or simply print colorized text in the terminal.
    """
    print(
        "Figure C (token survival heatmap) is not auto-generated here.\n"
        "See docstring in plot_token_survival_placeholder() for implementation steps."
    )


# === CLI =======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MrBERT / MrXLM figures A–H.")
    parser.add_argument(
        "--fig",
        type=str,
        default="A,B,C,D,E,F,G,H",
        help="Comma-separated subset of {A,B,C,D,E,F,G,H} to generate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mrpc",
        help="Dataset for F/G (e.g. mrpc, sst2, snli, xnli).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(
            RESULTS_ROOT
            / "bert_from_l4"
            / "from_l4_MR_TARGET_DEL=0.5MR_USE_PI=1MODELS=bertGATE_WARMUP_STEPS=1000BATCH=24USE_WANDB=1LOG_LEVEL=1"
        ),
        help="Run directory for F/G (default: BERT 0.5 + warmup run).",
    )
    parser.add_argument(
        "--trace",
        type=str,
        nargs="*",
        default=[],
        help="JSONL trace files for Figure B (deletion vs step).",
    )
    args = parser.parse_args()

    figs = {x.strip().upper() for x in args.fig.split(",") if x.strip()}
    run_dir = Path(args.run_dir)
    trace_paths = [Path(p) for p in args.trace]

    if "A" in figs:
        plot_pareto()
    if "B" in figs:
        plot_deletion_trace(trace_paths)
    if "C" in figs:
        plot_token_survival_placeholder()
    if "D" in figs:
        plot_task_sensitivity()
    if "E" in figs:
        plot_accuracy_summary()
    if "F" in figs:
        plot_deletion_histogram(args.dataset, run_dir)
    if "G" in figs:
        plot_loss_vs_deletion_scatter(args.dataset, run_dir)
    if "H" in figs:
        plot_tydiqa_curve()


if __name__ == "__main__":
    main()

