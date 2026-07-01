# SageMaker Deployment Notes — Gemma-4-31B

## What was added to `deploy_sagemaker.py`

**Addition:** `volume_size=256` GB EBS
**Why:** Gemma-4-31B weights are 62GB — the instance disk needs room for the weights + swap + TGI cache

---

**Addition:** `model_data_download_timeout=3600`
**Why:** Default is 10 min; downloading 62GB from HF Hub takes 15–30 min alone

---

**Addition:** `container_startup_health_check_timeout=3600`
**Why:** TGI needs time to shard the model across 4 GPUs after download

---

**Addition:** `sagemaker.Session(boto_session=…)`
**Why:** Pins the region and makes the default S3 bucket deterministic — prints the exact `s3://<bucket>/gemma-4-31b` path so you know where artifacts land

---

## Before running the deploy script

You still need the `ml.p5.48xlarge` quota (currently 0 by default).

1. Go to **AWS Service Quotas → SageMaker**
2. Search `L-BC4DA661`
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
