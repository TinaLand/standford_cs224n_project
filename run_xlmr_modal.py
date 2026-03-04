#!/usr/bin/env python3
"""
Run XLM-R experiments on Modal with A100 GPU.
Only runs XLM-R / MrXLM (no BERT). Results are written to a Modal Volume.

Usage (from project root):
  modal run run_xlmr_modal.py

Same controls as BERT (run_experiments.sh):
  MR_TARGET_DEL=0.5, MR_USE_PI=1, EPOCHS=3, BATCH=8, LOG_LEVEL=1, GATE_WARMUP_STEPS=0

Example (from project root):
  modal run run_xlmr_modal.py
  modal run run_xlmr_modal.py --epochs 2 --batch 16 --log-level 1 --gate-warmup-steps 1000
"""
import os
import subprocess
import modal

# -----------------------------------------------------------------------------
# Build image: Python + deps + project code (exclude large/unneeded dirs)
# -----------------------------------------------------------------------------
project_ignore = [
    ".git",
    "results",
    "__pycache__",
    "*.pyc",
    "wandb",
    ".modal",
    "from_l4",
    "*.png",
    ".cursor",
    ".DS_Store",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=project_ignore,
    )
)

app = modal.App("cs224n-xlmr-a100", image=image)

# Persist results so you can download from Modal dashboard or via CLI
volume = modal.Volume.from_name("xlmr-results", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=86400,  # 24h max
    volumes={"/vol/results": volume},
)
def run_xlmr_experiments(
    epochs: int = 3,
    batch_size: int = 8,
    mr_target_del: float = 0.5,
    mr_use_pi: bool = True,
    log_level: int = 1,
    gate_warmup_steps: int = 0,
    use_wandb: bool = True,
    skip_snli: bool = False,
    skip_sst2: bool = False,
    skip_xnli: bool = False,
    skip_tydiqa: bool = False,
):
    """Run only XLM-R / MrXLM experiments (no BERT). Same env as BERT: MR_TARGET_DEL, MR_USE_PI, LOG_LEVEL, GATE_WARMUP_STEPS."""
    env = os.environ.copy()
    env["MODELS"] = "xlmr"
    env["EPOCHS"] = str(epochs)
    env["BATCH"] = str(batch_size)
    env["MR_TARGET_DEL"] = str(mr_target_del)
    env["MR_USE_PI"] = "1" if mr_use_pi else "0"
    env["LOG_LEVEL"] = str(log_level)
    env["GATE_WARMUP_STEPS"] = str(gate_warmup_steps)
    env["PYTHONPATH"] = "/workspace"
    # WandB: if use_wandb is True, enable --use_wandb for train_mrbert.py.
    # If WANDB_API_KEY is not present inside the container, default to offline mode
    # so the run does not block waiting for interactive login.
    if use_wandb:
        env["USE_WANDB"] = "1"
        if "WANDB_API_KEY" not in env and "WANDB_MODE" not in env:
            env["WANDB_MODE"] = "offline"
    else:
        env.pop("USE_WANDB", None)
        if "WANDB_MODE" not in env:
            env["WANDB_MODE"] = "disabled"
    if skip_snli:
        env["SKIP_SNLI"] = "1"
    if skip_sst2:
        env["SKIP_SST2"] = "1"
    if skip_xnli:
        env["SKIP_XNLI"] = "1"
    if skip_tydiqa:
        env["SKIP_TYDIQA"] = "1"

    os.chdir("/workspace")
    ret = subprocess.run(
        ["bash", "-c", "chmod +x run_experiments.sh && ./run_experiments.sh"],
        env=env,
    )

    # Copy results to Volume so they persist
    os.makedirs("/vol/results", exist_ok=True)
    if os.path.isdir("/workspace/results"):
        subprocess.run(
            ["cp", "-r", "/workspace/results/.", "/vol/results/"],
            check=False,
        )
    volume.commit()

    if ret.returncode != 0:
        raise RuntimeError(f"run_experiments.sh exited with {ret.returncode}")
    return "XLM-R experiments finished. Results saved to volume xlmr-results."


@app.local_entrypoint()
def main(
    epochs: int = 3,
    batch: int = 8,
    mr_target_del: float = 0.5,
    mr_use_pi: bool = True,
    log_level: int = 1,
    gate_warmup_steps: int = 0,
    use_wandb: bool = True,
    skip_snli: bool = False,
    skip_sst2: bool = False,
    skip_xnli: bool = False,
    skip_tydiqa: bool = False,
):
    """Invoke XLM-R on Modal (A100). Run from project root: modal run run_xlmr_modal.py"""
    print("Submitting XLM-R-only run on Modal (GPU=A100)...")
    run_xlmr_experiments.remote(
        epochs=epochs,
        batch_size=batch,
        mr_target_del=mr_target_del,
        mr_use_pi=mr_use_pi,
        log_level=log_level,
        gate_warmup_steps=gate_warmup_steps,
        use_wandb=use_wandb,
        skip_snli=skip_snli,
        skip_sst2=skip_sst2,
        skip_xnli=skip_xnli,
        skip_tydiqa=skip_tydiqa,
    )
    print("Run finished. View logs: modal dashboard")
    print("Download results: modal volume get xlmr-results ./results_from_modal")
