#!/usr/bin/env python3
"""
Run MrBERT TyDi QA on Modal A100 with configurable arguments.

Examples:
  # with blending
  modal run --detach run_mrbert_tydi_modal.py --use-pre-deletion-blend --wandb-run-name tydiqa-with-blend-l3-30pct

  # without blending
  modal run --detach run_mrbert_tydi_modal.py --no-use-pre-deletion-blend --wandb-run-name tydiqa-no-blend-l3-30pct
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
def run_training(
    dataset: str = "tydiqa",
    backbone: str = "bert",
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 2e-5,
    target_deletion: float = 0.3,
    gate_layer_index: int = 3,
    gate_threshold_ratio: float = 0.5,
    gate_warmup_steps: int = 1000,
    use_pi: bool = True,
    controller_kp: float = 0.5,
    controller_ki: float = 1e-5,
    use_pre_deletion_blend: bool = True,
    use_learnable_pre_deletion_blend: bool = False,
    pre_deletion_blend_init_scale: float | None = None,
    use_wandb: bool = True,
    wandb_project: str = "mrbert-tydiqa",
    wandb_entity: str | None = None,
    wandb_run_name: str = "tydiqa-with-blend-l3-30pct",
) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    if use_wandb:
        env["WANDB_MODE"] = "online"
    else:
        env["WANDB_MODE"] = "disabled"

    os.chdir("/workspace")
    cmd = [
        "python",
        "train_mrbert.py",
        "--dataset",
        dataset,
        "--backbone",
        backbone,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
        "--lr",
        str(lr),
        "--target_deletion",
        str(target_deletion),
        "--gate_layer_index",
        str(gate_layer_index),
        "--gate_threshold_ratio",
        str(gate_threshold_ratio),
        "--gate_warmup_steps",
        str(gate_warmup_steps),
        "--controller_kp",
        str(controller_kp),
        "--controller_ki",
        str(controller_ki),
    ]
    if use_pi:
        cmd.append("--use_pi")
    if use_pre_deletion_blend:
        cmd.append("--use_pre_deletion_blend")
    else:
        cmd.append("--no-use_pre_deletion_blend")
    if use_learnable_pre_deletion_blend:
        cmd.append("--use_learnable_pre_deletion_blend")
    else:
        cmd.append("--no-use_learnable_pre_deletion_blend")
    if pre_deletion_blend_init_scale is not None:
        cmd.extend(["--pre_deletion_blend_init_scale", str(pre_deletion_blend_init_scale)])
    if use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", wandb_project, "--wandb_run_name", wandb_run_name])
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])

    ret = subprocess.run(cmd, env=env)
    if os.path.isdir("/workspace/results"):
        os.makedirs("/vol/results", exist_ok=True)
        subprocess.run(["cp", "-r", "/workspace/results/.", "/vol/results/"], check=False)
        volume.commit()
    if ret.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {ret.returncode}")
    return "Done"


@app.local_entrypoint()
def main(
    dataset: str = "tydiqa",
    backbone: str = "bert",
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 2e-5,
    target_deletion: float = 0.3,
    gate_layer_index: int = 3,
    gate_threshold_ratio: float = 0.5,
    gate_warmup_steps: int = 1000,
    use_pi: bool = True,
    controller_kp: float = 0.5,
    controller_ki: float = 1e-5,
    use_pre_deletion_blend: bool = True,
    use_learnable_pre_deletion_blend: bool = False,
    pre_deletion_blend_init_scale: float | None = None,
    use_wandb: bool = True,
    wandb_project: str = "mrbert-tydiqa",
    wandb_entity: str | None = None,
    wandb_run_name: str = "tydiqa-with-blend-l3-30pct",
):
    print("Submitting MrBERT TyDi run on Modal A100...")
    run_training.remote(
        dataset=dataset,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        lr=lr,
        target_deletion=target_deletion,
        gate_layer_index=gate_layer_index,
        gate_threshold_ratio=gate_threshold_ratio,
        gate_warmup_steps=gate_warmup_steps,
        use_pi=use_pi,
        controller_kp=controller_kp,
        controller_ki=controller_ki,
        use_pre_deletion_blend=use_pre_deletion_blend,
        use_learnable_pre_deletion_blend=use_learnable_pre_deletion_blend,
        pre_deletion_blend_init_scale=pre_deletion_blend_init_scale,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
    )
    print("Submitted. Check Modal dashboard logs for progress.")
