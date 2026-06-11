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
