# Run 01 — Qwen3-Embedding-4B / ConvFinQA

## Config

| Parameter | Value |
|---|---|
| Model | `unsloth/Qwen3-Embedding-4B` |
| Dataset | `grasson/t2-ragbench` / ConvFinQA subset |
| Train examples | 1,458 |
| Eval examples | 2,000 |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 1 |
| Total steps | 183 |
| Batch size (per device) | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Trainable parameters | 33,030,144 / 4,054,804,480 (0.81%) |
| Sequence length | 4,096 |
| GPU | NVIDIA A100 SXM4 40GB |
| Precision | bfloat16 |

> **Note:** ConvFinQA yielded ~3,458 usable examples after loading (take(15000)). With test_size=2000 the train split reduced to 1,458.

## Memory

| Metric | Value |
|---|---|
| Memory before training | 13.965 GB / 39.494 GB |
| Peak reserved memory | 13.965 GB (35.36%) |
| Peak reserved memory for training (LoRA delta) | 6.34 GB (16.05%) |

## Training Loss

| Epoch | Loss | Grad norm | LR |
|---|---|---|---|
| 0.27 | 0.0 | 0.0 | 1.63e-4 |
| 0.55 | 0.0 | 0.0 | 1.02e-4 |
| 0.82 | 0.0 | 0.0 | 4.15e-5 |
| 1.00 | 0.0 (avg) | — | — |

> **Warning:** Loss and grad_norm are 0.0 throughout. With `per_device_batch_size=1`, `MultipleNegativesRankingLoss` has no in-batch negatives to contrast against — the loss becomes trivially zero. For a meaningful training signal, increase batch size (try 2–4) or use pre-mined hard negatives with `TripletLoss`. The post-training eval scores still improved over a random baseline, suggesting the pretrained model carries most of the signal.

## Runtime

| Metric | Value |
|---|---|
| Total training time | 1164.54 s (19.41 min) |
| Samples / second | 1.252 |
| Steps / second | 0.157 |

## Post-training Evaluation

| Metric | Score |
|---|---|
| Accuracy@1 | 0.1505 |
| Accuracy@3 | 0.343 |
| Accuracy@5 | 0.437 |
| Accuracy@10 | 0.5605 |
| Precision@5 | 0.0874 |
| Precision@10 | 0.0560 |
| Recall@5 | 0.437 |
| Recall@10 | 0.5605 |
| NDCG@10 | 0.3407 |
| MRR@10 | 0.2721 |
| MAP@100 | 0.2857 |

> Baseline evaluation (before training) was not recorded in this run.

## Output

- LoRA adapters: `output/lora-adapters/`
- Merged model: `output/merged-model/`
- PyTorch checkpoint: `output/merged-model/qwen3_embedding_4b_finetuned.pt`
- HuggingFace Hub: [borntobeignored/qwen3-embedding-4b_lora](https://huggingface.co/borntobeignored/qwen3-embedding-4b_lora)
- W&B run: [Qwen3Embedding4B project](https://wandb.ai/elenamylocuda-gemma/Qwen3Embedding4B)
