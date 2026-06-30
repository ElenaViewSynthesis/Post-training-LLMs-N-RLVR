# GPU Cost Estimates — 150K Sample Generation

## SageMaker Inference Endpoints (us-east-1)

| Instance | $/hr | Throughput | 150K × 6 sentences | Est. cost |
|---|---|---|---|---|
| ml.p5.48xlarge | ~$98 | very fast | ~3–4 hrs | **~$350** |
| ml.g5.12xlarge | ~$5.67 | moderate | ~20–25 hrs | **~$130** |
| ml.g5.2xlarge | ~$1.52 | slow | ~80 hrs | **~$120** |

> $100 budget covers less than 1 hour on ml.p5.48xlarge — not sufficient for 150K samples on any SageMaker instance.

## Per-Token API Providers (recommended for this budget)

| Provider | Model | $/M tokens | 150K samples cost |
|---|---|---|---|
| Together.ai | Llama-3.3-70B | ~$0.20 | **~$15–20** |
| Groq | Llama-3.3-70B | free tier | **$0** |
| Fireworks.ai | Gemma-3-27B | ~$0.22 | **~$15–20** |

> $100 on Together.ai or Fireworks covers 500K+ samples at 6 sentences each.
