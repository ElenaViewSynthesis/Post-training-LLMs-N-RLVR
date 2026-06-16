# Run 01 — EmbeddingGemma-300M / FinQA

## Config

| Parameter | Value |
|---|---|
| Model | `unsloth/embeddinggemma-300m` |
| Dataset | `grasson/t2-ragbench` / FinQA subset |
| Train examples | 6,251 |
| Eval examples | 883 |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 1 |
| Total steps | 196 |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 8 |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Trainable parameters | 8,896,512 / 311,759,616 (2.85%) |
| GPU | NVIDIA A100 SXM4 40GB |
| Precision | bfloat16 |

## Memory

| Metric | Value |
|---|---|
| Memory before training | 8.732 GB / 39.494 GB |
| Peak reserved memory | 8.732 GB (22.11%) |
| Peak reserved memory for training (LoRA delta) | 7.58 GB (19.19%) |

## Training Loss

| Epoch | Loss | Grad norm | LR |
|---|---|---|---|
| 0.26 | 0.1109 | 4.648 | 1.67e-4 |
| 0.51 | 0.0218 | 0.145 | 1.10e-4 |
| 0.77 | 0.0112 | 0.020 | 5.34e-5 |
| 1.00 | 0.0397 (avg) | — | — |

## Runtime

| Metric | Value |
|---|---|
| Total training time | 1075.06 s (17.92 min) |
| Samples / second | 5.815 |
| Steps / second | 0.182 |

## Baseline Evaluation (before training)

| Metric | Score |
|---|---|
| Accuracy@1 | 0.0317 |
| Accuracy@5 | 0.1472 |
| Accuracy@10 | 0.2129 |
| Recall@5 | 0.1472 |
| Recall@10 | 0.2129 |
| NDCG@10 | 0.1117 |
| MRR@10 | 0.0806 |

## Output

- LoRA adapters: `output/lora-adapters/`
- Merged model: `output/merged-model/`
- PyTorch checkpoint: `output/merged-model/embeddinggemma_finetuned.pt`
- HuggingFace Hub: [borntobeignored/embeddinggemma_lora](https://huggingface.co/borntobeignored/embeddinggemma_lora)
- W&B run: [EmbeddingGemma300M project](https://wandb.ai/elenamylocuda-gemma/EmbeddingGemma300M)
