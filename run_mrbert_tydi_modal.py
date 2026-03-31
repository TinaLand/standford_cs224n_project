#!/usr/bin/env python3
"""
Run MrBERT TyDi QA on Modal A100.

Usage:
  modal run run_mrbert_tydi_modal.py
"""
import os
import subprocess
import modal


project_ignore = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".cursor",
    ".modal",
    ".DS_Store",
    "results",
    "wandb",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "transformers>=4.30.0", "datasets>=2.0.0", "wandb>=0.16.0")
    .add_local_dir(".", remote_path="/workspace", ignore=project_ignore)
)

app = modal.App("cs224n-mrbert-tydi-a100", image=image)
volume = modal.Volume.from_name("mrbert-tydi-results", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={"/vol/results": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def run_training() -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    # Force online logging for every run.
    env["WANDB_MODE"] = "online"

    os.chdir("/workspace")
    cmd = [
        "python",
        "train_mrbert.py",
        "--dataset",
        "tydiqa",
        "--backbone",
        "bert",
        "--epochs",
        "3",
        "--batch_size",
        "16",
        "--max_length",
        "256",
        "--lr",
        "2e-5",
        "--target_deletion",
        "0.3",
        "--gate_layer_index",
        "3",
        "--gate_threshold_ratio",
        "0.5",
        "--gate_warmup_steps",
        "1000",
        "--use_pi",
        "--controller_kp",
        "0.5",
        "--controller_ki",
        "1e-5",
        "--use_pre_deletion_blend",
        "--use_wandb",
        "--wandb_project",
        "mrbert-tydiqa",
        "--wandb_run_name",
        "tydiqa-with-blend-l3-30pct",
    ]
    ret = subprocess.run(cmd, env=env)
    if os.path.isdir("/workspace/results"):
        os.makedirs("/vol/results", exist_ok=True)
        subprocess.run(["cp", "-r", "/workspace/results/.", "/vol/results/"], check=False)
        volume.commit()
    if ret.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {ret.returncode}")
    return "Done"


@app.local_entrypoint()
def main():
    print("Submitting MrBERT TyDi run on Modal A100...")
    run_training.remote()
    print("Submitted. Check Modal dashboard logs for progress.")
