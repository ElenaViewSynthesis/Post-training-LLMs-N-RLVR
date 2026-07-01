# SageMaker Deployment Notes — Gemma-4-31B

## Instance

**`ml.p4d.24xlarge`** — 8× A100 40GB = 320GB total VRAM, all 8 GPUs active via TGI 8-way tensor parallelism.
Gemma-4-31B is 62GB in FP16 → ~7.75GB per GPU for weights, leaving ~32GB per GPU for KV cache.

Previously `ml.p5.48xlarge` with `SM_NUM_GPUS=4` — only 4 of 8 H100s were used, leaving 4 GPUs idle.

---

## What was added to `deploy_sagemaker.py`

**Addition:** `volume_size=256` GB EBS
**Why:** Gemma-4-31B weights are 62GB — the instance disk needs room for the weights + swap + TGI cache

---

**Addition:** `model_data_download_timeout=3600`
**Why:** Default is 10 min; downloading 62GB from HF Hub takes 15–30 min alone

---

**Addition:** `container_startup_health_check_timeout=3600`
**Why:** TGI needs time to shard the model across 8 GPUs after download

---

**Addition:** `sagemaker.Session(boto_session=…)`
**Why:** Pins the region and makes the default S3 bucket deterministic — prints the exact `s3://<bucket>/gemma-4-31b` path so you know where artifacts land

---

## Before running the deploy script

You need the `ml.p4d.24xlarge` quota (may be 0 by default in your account).

1. Go to **AWS Service Quotas → SageMaker**
2. Search for `ml.p4d.24xlarge` endpoint quota
3. Click **Request increase to 1**

Once approved, run:

```bash
python deploy_sagemaker.py
```

The script takes ~20–30 min and prints the endpoint name to add to your `.env`:

```env
SAGEMAKER_ENDPOINT_NAME=gemma-4-31b-sft-pipeline
AWS_REGION=us-east-1
```
