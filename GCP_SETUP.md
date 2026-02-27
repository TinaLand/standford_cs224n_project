# Running MrBERT on GCP with GPU (Step-by-Step)

Yes — **GCP has GPU**. You create a VM with an NVIDIA GPU (e.g. T4) and run the same code there. Below is a full walkthrough.

---

## Step 1: Check gcloud on your Mac

You already have `gcloud` installed. In Terminal:

```bash
gcloud --version
gcloud config get-value project   # should show e.g. cs224n-ah
```

If no project is set:

```bash
gcloud auth login
gcloud config set project cs224n-ah
```

---

## Step 2: Turn on Compute Engine API (needed for VMs)

In Terminal:

```bash
gcloud services enable compute.googleapis.com
```

If it asks for a project, choose the one you use (e.g. `cs224n-ah`).

---

## Step 2b: (If you get “Quota GPUS_ALL_REGIONS exceeded, Limit: 0”) Request GPU quota

Your project has **0 GPU quota** by default. You must request an increase before creating a GPU VM:

1. Open [GCP Console → Quotas](https://console.cloud.google.com/iam-admin/quotas).
2. Set the project to `cs224n-ah` (or your project).
3. In the filter box, type **`GPUs`** or **`GPUS_ALL_REGIONS`**.
4. Find **Compute Engine API** → **GPUs (all regions)** (limit often shows as 0).
5. Check the box and click **Edit quotas** (or **Request increase**).
6. Request **1** (or 2) for “GPUs (all regions)”.
7. Submit; approval can take from minutes to 1–2 business days.

Until the quota is increased, **no zone will work** for GPU VMs. Alternatives:

- **Google Colab** (free): Upload your repo or clone from GitHub, Runtime → Change runtime type → T4 GPU, then run `!pip install -r requirements.txt` and your training script. No GCP project GPU quota needed.
- **CPU-only VM**: Create a VM **without** `--accelerator` (see “CPU-only VM” below) and run the same code; it will use CPU and take longer.

---

## Step 3: Create a GPU instance

Choose one of two options: **Vertex AI Workbench (Console, recommended)** or **gcloud CLI**.

---

### Option A: Vertex AI Workbench (Console, recommended)

1. In Google Cloud Console, use the **top search bar** to open **"Vertex AI Workbench"**.
2. Open the **Instances** tab and click **Create New**.
3. Configure as follows (**use region us-east1** to match quota):
   - **Region**: **us-east1** (if your GPU quota is there).
   - **Environment**: **PyTorch 2.x** (CS224N uses PyTorch).
   - **Machine type**: **n1-standard-4** (or similar).
   - **GPU type**: **NVIDIA T4**.
   - **GPU count**: **1**.
4. Click **Create** at the bottom and wait for the instance to be ready (about 1–2 minutes).

When ready, click **Open JupyterLab** in the instance list, then open **Terminal** in Jupyter. Upload or clone the project, then run `pip install -r requirements.txt` and `./run_experiments.sh` (or `QUICK=1 ./run_experiments.sh`).

---

### Option B: Create GPU VM with gcloud

Run the following on your **local Terminal** (GPU quota must be enabled):

**If you get `ZONE_RESOURCE_POOL_EXHAUSTED`**, try another zone (e.g. `us-east1-c`, `us-west1-b`):

```bash
export PROJECT_ID=$(gcloud config get-value project)
# Prefer us-east1 zone if your quota is there:
export ZONE=us-east1-c
# If that fails, try: us-east1-b, us-west1-b, europe-west1-b

gcloud compute instances create mrbert-gpu \
  --zone=us-east1-c \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --metadata="install-nvidia-driver=True"
```

# Set ZONE to the zone where the VM was created
export ZONE=us-east1-d
gcloud compute scp --recurse . mrbert-gpu:~/cs224n_project --zone=$ZONE

gcloud compute instances stop mrbert-gpu --zone=us-east1-d
gcloud compute ssh mrbert-gpu --zone=us-east1-d
gcloud compute instances start mrbert-gpu --zone=us-east1-d

- **GPU**: 1× `nvidia-tesla-t4`.
- **Image**: Deep Learning VM, CUDA 12.8, Ubuntu 22.04.
- **Note the ZONE** — you need it for `scp`, `ssh`, and delete (Steps 4, 5, 8).

---

### CPU-only VM (when you have no GPU quota)

If you don't have GPU quota yet, create a VM **without** GPU and run on CPU (slower):

```bash
export ZONE=us-central1-a
gcloud compute instances create mrbert-gpu \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB
```

After logging into the VM: `sudo apt-get update && sudo apt-get install -y python3-pip`, upload the project, `pip install -r requirements.txt`, then run `./run_experiments.sh` (you will see `Device: cpu`).

---

## Step 4: Upload your project to the VM

**If using Vertex AI Workbench**: Use the JupyterLab file browser to upload files or `git clone`; no need to scp from your machine. Then go to the project directory in the instance Terminal and run Step 6.

**If using a gcloud-created VM**: On your **Mac Terminal** (in the project directory), run the commands below. Set `ZONE` to the zone used in Step 3 (e.g. `us-east1-c`):

```bash
cd /Users/tianhuihuang/Desktop/cs224n_project
export ZONE=us-west1-b   # or whatever zone you used in Step 3

gcloud compute scp --recurse . mrbert-gpu:~/cs224n_project --zone=$ZONE
```

This copies the whole project into the VM at `~/cs224n_project`. It may ask for the VM’s SSH key the first time (choose “yes”).

---

## Step 5: SSH into the VM

```bash
gcloud compute ssh mrbert-gpu --zone=$ZONE
```

You are now **on the GPU machine**. The prompt will look like `yourname@mrbert-gpu ~ $`.

---

## Step 6: On the VM — prepare environment and run small data

Run these **on the VM** (after the SSH in Step 5):

```bash
cd ~/cs224n_project

# Use Python 3 and create a venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Quick GPU run: 200 MRPC samples, 1 epoch + latency benchmark
chmod +x run_experiments.sh
QUICK=1 ./run_experiments.sh
```

You should see:

- `Device: cuda`
- Training loss and then `Validation accuracy: 0.XX`
- Latency benchmark output and a **Speedup** line (positive = MrBERT faster on GPU).

---

## Step 7: (Optional) Copy results back to your Mac

Exit SSH (type `exit`), then on your **Mac**:

```bash
cd /Users/tianhuihuang/Desktop/cs224n_project
export ZONE=us-central1-a

gcloud compute scp mrbert-gpu:~/cs224n_project/results/train_results.jsonl ./results/ --zone=$ZONE
```

---

## Step 8: Delete the VM when you’re done (avoid extra cost)

On your **Mac**:

```bash
export ZONE=us-central1-a
gcloud compute instances delete mrbert-gpu --zone=$ZONE
```

Type `y` when it asks for confirmation.

---

## Summary: order of commands

| Step | Where   | Command / action |
|------|---------|-------------------|
| 1    | Mac     | `gcloud --version` and `gcloud config get-value project` |
| 2    | Mac     | `gcloud services enable compute.googleapis.com` |
| 3    | Mac     | Create VM with the long `gcloud compute instances create ...` block above |
| 4    | Mac     | `gcloud compute scp --recurse . mrbert-gpu:~/cs224n_project --zone=$ZONE` |
| 5    | Mac     | `gcloud compute ssh mrbert-gpu --zone=$ZONE` |
| 6    | **VM**  | `cd ~/cs224n_project`, `source .venv/bin/activate`, `pip install -r requirements.txt`, `QUICK=1 ./run_experiments.sh` |
| 7    | Mac     | (Optional) `gcloud compute scp mrbert-gpu:~/cs224n_project/results/...` |
| 8    | Mac     | `gcloud compute instances delete mrbert-gpu --zone=$ZONE` |

Yes — **GCP has GPU**; you use it by creating a GPU VM (Step 3) and running your code on it (Step 6).
