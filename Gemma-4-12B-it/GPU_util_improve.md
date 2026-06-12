# GPU Utilization — Observations and Improvement Notes

## Current Run Status

Estimated total runtime ~2h 45m. For Gemma 12B + QLoRA + 2048 token length + gradient checkpointing on a single GPU, that is expected.

### Live nvidia-smi snapshot (2026-06-12 02:39 UTC)

```
Fri Jun 12 02:39:03 2026
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:06:00.0 Off |                    0 |
| N/A   42C    P0            120W /  400W |   25331MiB /  40960MiB |     33%      Default |
+-----------------------------------------+------------------------+----------------------+

| Process                                              GPU Memory |
|   0   python3.12  ...4-12B-it/.venv/bin/python3.12    25194MiB |
```

- **VRAM:** 25.3 GB / 40 GB used — healthy headroom
- **GPU util:** 33% — gradient checkpointing and data loading cause idle gaps between backward passes
- **Temp:** 42°C — well within safe range
- **Power:** 120W / 400W — typical for QLoRA with checkpointing

If utilization is steady and VRAM is mostly used, let it run.

---

## Monitor GPU in a Second SSH Tab

```bash
watch -n 1 nvidia-smi
```

---

## Short Test Run (Optional)

Stop with `Ctrl+C`, then edit `06_mitre_sft.py` on the instance and reduce:

```python
num_train_epochs=1,
max_length=1024,
```

Re-launch:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

---

## Speed Improvements for Future Runs

| Improvement | How |
|---|---|
| FlashAttention | `pip install flash-attn --no-build-isolation` then `--attn-implementation flash_attention_2` |
| Re-enable packing | Set `packing=True` in `build_sft_config()` — only safe with FlashAttention |
| Upgrade GPU | Use H100 instead of A100 (2–3× faster on bf16) |
| W&B logging | Keep enabled only if you want run tracking — not the main bottleneck |

---

## GPU Underutilization Analysis

The job is running correctly, but the GPU is not fully loaded:

| Metric | Observed | Capacity | Utilization |
|---|---|---|---|
| GPU compute | 33% | 100% | underutilized |
| VRAM | 25.3 GB | 40 GB | 14.7 GB headroom |
| Power | 120W | 400W | headroom available |

This explains the slower throughput. The A100 can be pushed harder.

### For the current run

Let it finish — ~2h 45m is acceptable for a first full run.

### For the next run — increase batch size to improve utilization

Change in `06_mitre_sft.py`:

```python
per_device_train_batch_size=2,
gradient_accumulation_steps=4,
```

This keeps the same effective batch size of 8 but halves the number of accumulation cycles, reducing CPU/GPU idle gaps and improving utilization.

If it OOMs, revert to:

```python
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
```

### Even faster — FlashAttention

Install and run with FlashAttention to reduce memory overhead and unlock higher throughput:

```bash
pip install flash-attn --no-build-isolation
```

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py \
  --attn-implementation flash_attention_2
```

With FlashAttention, re-enabling `packing=True` is also safe and further improves GPU utilization by eliminating padding waste.
