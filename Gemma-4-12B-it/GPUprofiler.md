# GPU Profiler — Gemma 4 12B on Lambda Cloud

```bash
watch -n 1 nvidia-smi
```

Monitor and profile GPU usage during inference and fine-tuning runs on the Lambda A100 instance.

**Check if Nsight tools are available on the instance:**
```bash
sudo apt update
apt-cache search nsight
```

**Install Nsight Systems:**
```bash
sudo apt install -y nsight-systems
```

**Run Nsight Systems:**
```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiles/nsys/mitre_sft_timeline \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

**Run Nsight Compute on a small kernel window:**
```bash
mkdir -p profiles/ncu

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --launch-count 10 \
  --force-overwrite \
  -o profiles/ncu/mitre_sft_basic \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

**Copy reports back to local machine:**
```bash
scp -r ubuntu@129.146.110.174:~/Gemma-4-12B-it/profiles .
```

> **Rule:** never profile the entire training run with PyTorch Profiler or Nsight. Profile only a short window after warmup — otherwise traces become huge and slow.

---

## Profiling stages

| Stage | Tool | Goal |
|---|---|---|
| 1 | `nvidia-smi` + training logs | Tokens/sec, loss/sec, VRAM, GPU utilisation |
| 2 | PyTorch Profiler | Slow operators, CPU/GPU split, memory spikes |
| 3 | Nsight Systems | GPU idle gaps, CPU stalls, CUDA memcpy, NCCL overhead |
| 4 | Nsight Compute | Inspect specific slow kernels |
| 5 | Tracy / OpenTelemetry | Profile the final SOC incident-analysis application |

### Stage 1 — Baseline

```bash
mkdir -p profiles logs

nvidia-smi dmon -s pucvmt > profiles/gpu_dmon.log &
python 06_mitre_sft.py
```

### Stage 2 — Nsight Systems (run this first)

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

### Stage 3 — Nsight Systems (short window only)

```bash
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas,nccl \
  --sample=cpu \
  --stats=true \
  --output=profiles/gemma4_mitre_nsys \
  python 06_mitre_sft.py --max_steps 30
```

---

---

## 1. Live GPU monitoring

### nvidia-smi (basic)

```bash
# snapshot
nvidia-smi

# refresh every 2 seconds
watch -n 2 nvidia-smi

# compact view — memory, utilization, temperature
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
           --format=csv -l 2
```

### nvitop (interactive, recommended)

```bash
pip install nvitop
nvitop
```

Shows per-process GPU memory, utilization, SM usage, and power draw in a live TUI.

---

## 2. Profile a training run with W&B

Make sure `WANDB_API_KEY` is set in your `.env`. The training script enables W&B automatically when it is.

### System metrics (automatic)

W&B logs GPU utilization, VRAM, temperature, and power draw out of the box during `trainer.train()`. No extra code needed — just log in:

```bash
wandb login   # paste your API key when prompted
```

### Custom GPU metrics

Log additional metrics mid-training by adding a callback:

```python
import wandb
from transformers import TrainerCallback

class GPUMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        import torch
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved  = torch.cuda.memory_reserved(i) / 1024**3
            wandb.log({
                f"gpu{i}/memory_allocated_gb": allocated,
                f"gpu{i}/memory_reserved_gb":  reserved,
            }, step=state.global_step)

trainer = SFTTrainer(
    ...
    callbacks=[GPUMetricsCallback()],
)
```

### PyTorch profiler → W&B

```python
from torch.profiler import profile, ProfilerActivity
import wandb

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    record_shapes=True,
    with_stack=True,
) as prof:
    trainer.train()

table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
print(table)
wandb.log({"profiler/cuda_ops": wandb.Html(f"<pre>{table}</pre>")})
```

---

## 3. Memory profiling

### Peak VRAM usage

```python
import torch

torch.cuda.reset_peak_memory_stats()
# ... run training step ...
peak = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"Peak allocated: {peak:.2f} GB  |  Reserved: {reserved:.2f} GB")
```

### Memory snapshot (PyTorch >= 2.1)

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run a few steps ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

Upload and visualize at: https://pytorch.org/memory_viz

---

## 4. Expected VRAM usage

| Setup | Approximate VRAM |
|---|---|
| Inference (bf16) | ~24 GB |
| QLoRA 4-bit (training) | ~28–32 GB |
| Full LoRA bf16 (training) | ~48 GB+ |
| Multi-GPU (2x A100) | splits evenly via `device_map="auto"` |

A100 SXM4 has **40 GB** VRAM — QLoRA is the safe default for this model.

---

## 5. Throughput benchmarking

Log tokens/sec during training by adding to `SFTConfig`:

```python
sft_config = SFTConfig(
    ...
    logging_steps=1,         # log every step
    include_tokens_per_second=True,
    include_num_input_tokens_seen=True,
)
```

Or measure manually:

```python
import time, torch

start = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=512)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

n_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
print(f"{n_tokens / elapsed:.1f} tokens/sec")
```

---

## 6. Quick profiling script

Run this on the Lambda instance to get an instant health report before starting a long training job:

```bash
python - <<'EOF'
import torch, subprocess, json

print("=== CUDA ===")
print(f"Available:    {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total = props.total_memory / 1024**3
    print(f"GPU {i}: {props.name}  {total:.1f} GB  SM {props.major}.{props.minor}")

print("\n=== nvidia-smi ===")
subprocess.run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
    "--format=csv,noheader"
])
EOF
```

---

## 7. W&B dashboard

All runs are logged to your W&B project (set `WANDB_PROJECT=gemma-4-12b` in `.env`).

Key panels to watch during training:

| Panel | What to look for |
|---|---|
| `train/loss` | Steady decrease — spikes may indicate bad batches |
| `train/tokens_per_second` | Throughput baseline for the A100 |
| `system/gpu.0.memoryAllocated` | Should stay below 38 GB on A100 40 GB with QLoRA |
| `system/gpu.0.gpu` | Utilization — aim for >85% during training steps |
| `system/gpu.0.temp` | Alert if consistently above 80°C |

Open your run at: https://wandb.ai/home
