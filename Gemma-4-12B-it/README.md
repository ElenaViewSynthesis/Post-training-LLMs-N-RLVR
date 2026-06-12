# Post-training-LLMs-N-RLVR

## Gemma 4 12B-IT — Lambda Cloud GPU Guide

### Memory Footprint

| Component | VRAM |
|---|---|
| Model weights (12B × bf16) | ~24 GB |
| KV cache (1024 tokens, batch=1) | ~2–4 GB |
| Multimodal inputs (image/audio) | ~1–3 GB |
| **Total minimum** | **~28–31 GB** |

### Lambda GPU Instance Recommendations

| Instance | GPU | VRAM | Verdict |
|---|---|---|---|
| `gpu_1x_a100_sxm4` | 1× A100 80GB | 80 GB | **Recommended** — plenty of headroom for multimodal |
| `gpu_1x_h100_sxm5` | 1× H100 80GB | 80 GB | Best performance (faster bf16 throughput) |
| `gpu_2x_a100` | 2× A100 40GB | 80 GB total | Works with `device_map="auto"` |
| `gpu_1x_a100` | 1× A100 40GB | 40 GB | Tight but works; limit batch size and context length |
| `gpu_1x_a10` | 1× A10 | 24 GB | **Too small** — model weights alone hit 24 GB |

**Minimum viable:** 1× A100 40GB  
**Recommended:** 1× A100 80GB or 1× H100 80GB

### Launch Commands (SSH)

```bash
# Single A100 80GB
CUDA_VISIBLE_DEVICES=0 python 01_inference.py

# Multi-GPU (2× A100 40GB), device_map="auto" shards automatically
CUDA_VISIBLE_DEVICES=0,1 python 01_inference.py

# Pin to a specific GPU
GPU_ID=1 python 01_inference.py
```

### CUDA / Driver Requirements

- CUDA **12.1+** (Lambda standard images ship with 12.x)
- Driver **525+** for A100, **530+** for H100
- Python **3.10+**, PyTorch **2.3+**

The A100 80GB SXM4 instance covers inference, LoRA fine-tuning, DPO, and GRPO all on a single node.

---

### SSH Key Setup for Lambda Cloud

**1. Generate an SSH key pair** (skip if you already have one):
```bash
ssh-keygen -t ed25519 -C "lambda-gpu" -f ~/.ssh/lambda_gpu
```

**2. Add the public key to Lambda Cloud:**
- Go to [lambdalabs.com/cloud/ssh-keys](https://lambdalabs.com/cloud/ssh-keys)
- Click **Add SSH Key**, paste the contents of `~/.ssh/lambda_gpu.pub`
- Give it a name — this becomes your `LAMBDA_SSH_KEY_NAME` in `.env`

**3. Fill in `.env`:**
```bash
cp .env.example .env
```
```env
LAMBDA_API_KEY=your_lambda_api_key
LAMBDA_SSH_KEY_NAME=lambda-gpu          # name you gave the key on Lambda
LAMBDA_SSH_KEY_PATH=~/.ssh/lambda_gpu   # path to the private key on your machine
```

**4. Launch and connect:**
```bash
uv run lambda_gpu.py                        # launches default instance and SSHs in
uv run lambda_gpu.py --type gpu_1x_h100_sxm4  # specific instance type
```

---

### Deploying to the GPU Instance

**1. Sync the project from local WSL to the instance:**
```bash
rsync -av --exclude .venv --exclude .uv-cache --exclude .git \
  ~/project/ \
  ubuntu@<instance-ip>:~/project/
```

**2. On the Lambda instance — verify CUDA and install deps:**
```bash
cd ~/project
nvidia-smi
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**3. Authenticate:**
```bash
huggingface-cli login
wandb login
```

**4. Run a smoke test:**
```bash
CUDA_VISIBLE_DEVICES=0 python 01_inference.py
```

**5. For longer jobs, use tmux so training keeps running after disconnect:**
```bash
tmux new -s gemma
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python 01_inference.py
```

> **Important:** when done, terminate the instance to stop billing:
> ```bash
> uv run lambda_gpu.py terminate <instance-id>
> ```

---

### Troubleshooting — Project Folder Not Found on Instance

If `cd ~/Gemma-4-12B-it` fails with "No such file or directory", the folder hasn't been synced yet.

**From your local WSL terminal, run:**
```bash
rsync -av --exclude .venv --exclude .uv-cache --exclude .git \
  /mnt/c/Users/proxi/Documents/ccsyntheticdata/Gemma-4-12B-it/ \
  ubuntu@<instance-ip>:~/Gemma-4-12B-it/
```

**Then back in the SSH terminal:**
```bash
cd ~/Gemma-4-12B-it
ls
```

**If you already ran rsync but can't find the folder, check where it landed:**
```bash
ls ~
find ~ -maxdepth 2 -type d -name '*Gemma*'
```

**Then run the smoke test:**
```bash
CUDA_VISIBLE_DEVICES=0 python 01_inference.py
```

---

### SSH Connection to Lambda GPU — Established

```
ubuntu@129.146.110.174
Ubuntu 22.04.5 LTS — Linux 6.8.0-1046-nvidia x86_64
Lambda GPU Cloud
```

Connected via `lambda_gpu.py` using the configured `LAMBDA_SSH_KEY_PATH`. Instance is live and ready for model deployment.

![Lambda SSH Connection](assets/lambda-ssh-connection.png)
