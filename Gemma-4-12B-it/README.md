# Post-training-LLMs-N-RLVR

### Training Runtime Benchmark

Runtime comparison for fine-tuning `gemma-4-12b-it` on Lambda GPU instances (MITRE ATT&CK SFT, full run).

| GPU Instance | Hardware | VRAM | Runtime | Price | Est. Cost |
|---|---|---:|---:|---:|---:|
| GH200 | ARM64 + H100 (Hopper) | 96 GB unified | 57 min | — | ~$2.29 |
| A100 SXM4 | x86-64 (Ampere) | 40 GB | 1h 37m | $1.99/hr | ~$3.22 |

The GH200 was **40 min faster and ~$1 cheaper** on this workload. The unified 96 GB memory eliminates the VRAM pressure that forces gradient checkpointing and smaller micro-batches on the A100 40 GB, and Hopper-class bf16 throughput is meaningfully higher than Ampere.

**Note:** The GH200 runs Ubuntu on ARM64 (`aarch64`). The Nsight setup script (`scripts/setup_lambda_nsight.sh`) handles the ARM64-specific `libbpf.so.1` and `libssh` symbol issues automatically — see the [Profiling with Nsight Systems](#profiling-with-nsight-systems-and-nsight-compute) section below.

---

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
LAMBDA_INSTANCE_IP=132.226.76.207       # current running instance, for direct SSH
```

**4. Launch and connect:**
```bash
uv run lambda_gpu.py                        # launches default instance and SSHs in
uv run lambda_gpu.py --type gpu_1x_h100_sxm4  # specific instance type
```

**Connect to the current running instance:**
```bash
ssh ubuntu@132.226.76.207
ssh -i ~/.ssh/lambda_gpu ubuntu@132.226.76.207
uv run lambda_gpu.py connect 132.226.76.207
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

### Running MITRE ATT&CK Fine-tuning

**Run directly on the Lambda box:**
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

For long runs, use tmux so training keeps running if the SSH session drops.

**Start a tmux session and launch training:**
```bash
tmux new -s mitre-sft
cd ~/Gemma-4-12B-it
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

**Detach from tmux** (training continues in background):
```
Ctrl-b  then  d
```

**Reattach later:**
```bash
tmux attach -t mitre-sft
```

**With flash-attn (if installed):**
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py \
  --attn-implementation flash_attention_2
```

> Default run is QLoRA. Only use `--no-qlora` if you intentionally want full-precision LoRA.

---

### Profiling with Nsight Systems and Nsight Compute

**What changed in the codebase:**
- `06_mitre_sft.py` — `--nvtx` flag adds NVTX ranges around training steps; `--max-steps` limits the run to a short profiling window
- `mitre-attack.md` — added `nsys` and `ncu` profiling commands

**Sync the updated script to Lambda:**
```bash
scp /mnt/c/Users/proxi/Documents/ccsyntheticdata/Gemma-4-12B-it/06_mitre_sft.py \
  ubuntu@132.226.76.207:~/Gemma-4-12B-it/06_mitre_sft.py
```

**Install kernel tracing dependencies:**
```bash
sudo apt update
sudo apt install -y libbpf1
```

`libbpf1` is used by Linux tools to interact with eBPF/BPF programs in the kernel — required by profiling, tracing, and monitoring tools that rely on eBPF. It lets user-space programs load BPF bytecode into the kernel, create/read BPF maps, and attach probes.

**On Lambda, run the Nsight setup script:**
```bash
bash scripts/setup_lambda_nsight.sh
```

> Only run `source ~/nsys-libs/env.sh` after the script prints:
> ```
> [setup-lambda-nsight] Wrote /home/ubuntu/nsys-libs/env.sh
> ```

```bash
source ~/nsys-libs/env.sh
```

**Verify tools are available:**
```bash
which nsys
which ncu
```

**Run Nsight Systems (timeline trace):**
```bash
cd ~/Gemma-4-12B-it
source .venv/bin/activate
mkdir -p profiles/nsys

CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiles/nsys/mitre_sft_timeline \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

**Profiled smoke test (20 steps — run this first):**
```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiles/nsys/mitre_sft_smoke \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 20
```

> **Why 2.3 GB is a red flag:** That's just 20 steps. Full training would produce a trace tens of GBs large — completely unusable. The `--max-steps 20` limit was the right call.

**Run Nsight Compute (kernel-level metrics):**
```bash
mkdir -p profiles/ncu

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set basic \
  --launch-skip 20 \
  --launch-count 10 \
  --force-overwrite \
  -o profiles/ncu/mitre_sft_basic \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

**Copy reports back to local machine:**
```bash
scp -r ubuntu@132.226.76.207:~/Gemma-4-12B-it/profiles .
```

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

### Current Lambda GPU Endpoint

```
ubuntu@132.226.76.207
Ubuntu 22.04.5 LTS — Linux 6.8.0-1046-nvidia x86_64
Lambda GPU Cloud
```

Connect with `ssh ubuntu@132.226.76.207` if your default SSH identity is registered with Lambda. Otherwise use `ssh -i ~/.ssh/lambda_gpu ubuntu@132.226.76.207` or set `LAMBDA_SSH_KEY_PATH` before running `lambda_gpu.py connect`.

![Lambda SSH Connection](assets/lambda-ssh-connection.png)

