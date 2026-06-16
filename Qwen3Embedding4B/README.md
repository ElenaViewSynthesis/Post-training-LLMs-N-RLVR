# Qwen3-Embedding-4B Fine-tuning

Fine-tuning `unsloth/Qwen3-Embedding-4B` on the ConvFinQA subset of `grasson/t2-ragbench` using Unsloth + LoRA.

## GPU Requirements

| Mode | Seq length | VRAM | Recommended instance |
|---|---|---|---|
| LoRA 16-bit (this config) | 4,096 | ~14.5 GB | A100 40GB |
| LoRA 16-bit | 8,192 | ~21–22 GB | A100 40GB |
| LoRA 16-bit | 16,384 | ~35 GB | A100 40GB (tight) |
| LoRA 16-bit | 32,768 (full context) | ~40–48 GB | A100 80GB |
| QLoRA 4-bit | 4,096 | ~12–16 GB | A10 24GB |
| QLoRA 4-bit | 32,768 (full context) | ~18–22 GB | A100 40GB |
| Full fine-tuning | 32,768 | 80 GB+ | H100 80GB |

> **Note:** The default `max_seq_length` in this config is **4,096** to fit an A100 40GB. To use the full 32k context window, switch to an **A100 80GB** instance and set `max_seq_length = 32768` in `MAX_SEQ_LENGTHS`.

### Context length scaling on A100 40GB

With Unsloth's gradient checkpointing, activation memory scales roughly **linearly** (not quadratically):

| Component | Memory |
|---|---|
| Fixed model weights | ~7.6 GB |
| Activations at seq_len=4,096 | ~6.9 GB → **~14.5 GB total** |
| Activations at seq_len=8,192 | ~13.8 GB → **~21–22 GB total** |
| Activations at seq_len=16,384 | ~27.6 GB → **~35 GB total** |

**Recommended next step:** set `max_seq_length = 8192` in `MAX_SEQ_LENGTHS` to reach ~20 GB — leaves ~18 GB headroom on the 40GB card. Beyond 16,384 is tight and risks OOM.

> **Note:** Use the **Lambda Stack** image when launching — not plain Ubuntu. It comes with CUDA, cuDNN, and NVIDIA drivers pre-installed.

## CUDA Setup

After SSH-ing into the instance, verify the GPU and fix the `libnvJitLink` path before running training:

```bash
# Verify GPU
nvidia-smi
lspci | grep -i nvidia

# Fix libnvJitLink path (required for bitsandbytes on CUDA 13.x)
export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

To make the path permanent:

```bash
echo 'export LD_LIBRARY_PATH=/path/to/Qwen3Embedding4B/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Setup

```bash
git clone https://github.com/ElenaViewSynthesis/Post-training-LLMs-N-RLVR.git
cd Post-training-LLMs-N-RLVR/Qwen3Embedding4B

# Copy .env from EmbeddingGemma300M or create a new one
cp ../EmbeddingGemma300M/.env .

bash install.sh
source .venv/bin/activate
python finetune.py
```

## .env

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=wandb_v1_...
WANDB_PROJECT=Qwen3Embedding4B
```

## Dataset

| Subset | Split | Samples used |
|---|---|---|
| ConvFinQA | train | 13,000 |
| ConvFinQA | dev | 2,000 |

ConvFinQA contains ~14,000 conversational financial QA pairs. Anchors include company name and report year for better contextual retrieval.

## Training Config

| Parameter | Value |
|---|---|
| Model | `unsloth/Qwen3-Embedding-4B` |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Epochs | 1 |
| Loss | `MultipleNegativesRankingLoss` |
| Sequence length | 32,768 |
| Precision | bfloat16 |

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Recall@5** | Is the relevant passage in the top 5? |
| **Recall@10** | Is the relevant passage in the top 10? |
| **MRR@10** | How early does the first relevant result appear? |
| **NDCG@10** | Ranking quality — penalises relevant results pushed lower |
