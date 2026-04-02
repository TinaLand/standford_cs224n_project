#!/usr/bin/env python3
"""
XLM-R SNLI baseline on Modal A100 (recommended: max_length=256, 5 epochs).

From repo root:
  modal run --detach run_xlmr_snli_modal.py

W&B defaults to project `alina` under entity `aronima7-stanford-university`
(same as TyDi post-fix runs). Requires Modal secret `wandb-api-key`.

Results copy to volume `xlmr-results` (same as run_xlmr_modal.py).
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

app = modal.App("cs224n-xlmr-snli-baseline-a100", image=image)
volume = modal.Volume.from_name("xlmr-results", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={"/vol/results": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def run_snli_baseline(
    epochs: int = 5,
    batch_size: int = 24,
    max_length: int = 256,
    lr: float = 2e-5,
    gate_layer_index: int = 3,
    gate_threshold_ratio: float = 0.5,
    log_level: int = 1,
    use_wandb: bool = True,
    wandb_project: str = "alina",
    wandb_entity: str = "aronima7-stanford-university",
    wandb_run_name: str = "xlmr-snli-baseline-m256-ep5-modal",
) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    env["WANDB_MODE"] = "online" if use_wandb else "disabled"

    os.chdir("/workspace")
    # Persist jsonl directly on the Modal Volume (survives after container exits).
    os.makedirs("/vol/results", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    result_jsonl = "/vol/results/train_results.jsonl"

    cmd = [
        "python",
        "train_mrbert.py",
        "--dataset",
        "snli",
        "--backbone",
        "xlmr",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
        "--lr",
        str(lr),
        "--gate_weight",
        "0.0",
        "--gate_layer_index",
        str(gate_layer_index),
        "--gate_threshold_ratio",
        str(gate_threshold_ratio),
        "--log_level",
        str(log_level),
        "--output_result",
        result_jsonl,
    ]
    if use_wandb:
        cmd.extend(
            [
                "--use_wandb",
                "--wandb_project",
                wandb_project,
                "--wandb_run_name",
                wandb_run_name,
                "--wandb_entity",
                wandb_entity,
            ]
        )

    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {ret.returncode}")
    # Copy any extra files under /workspace/results (e.g. future artifacts) into the volume.
    if os.path.isdir("/workspace/results"):
        subprocess.run(["cp", "-r", "/workspace/results/.", "/vol/results/"], check=False)
    volume.commit()
    return "XLM-R SNLI baseline finished. Results on volume xlmr-results (see /vol/results/train_results.jsonl)."


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 24,
    max_length: int = 256,
    lr: float = 2e-5,
    gate_layer_index: int = 3,
    gate_threshold_ratio: float = 0.5,
    log_level: int = 1,
    use_wandb: bool = True,
    wandb_project: str = "alina",
    wandb_entity: str = "aronima7-stanford-university",
    wandb_run_name: str = "xlmr-snli-baseline-m256-ep5-modal",
):
    print("Submitting XLM-R SNLI baseline on Modal A100...")
    run_snli_baseline.remote(
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        lr=lr,
        gate_layer_index=gate_layer_index,
        gate_threshold_ratio=gate_threshold_ratio,
        log_level=log_level,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
    )
    print("Submitted. Logs: modal app logs <app-id> --timestamps")
    print("Results: modal volume get xlmr-results ./results_from_modal")
