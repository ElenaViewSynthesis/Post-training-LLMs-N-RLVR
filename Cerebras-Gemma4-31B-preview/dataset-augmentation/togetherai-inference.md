# Together.ai Inference — Running the Pipeline

Use this backend while the SageMaker `ml.p4d.24xlarge` quota is pending.
Cost: ~$15–20 for all 150K samples. No quota approval needed.

---

## Step 1 — Get a Together.ai API key

1. Sign up at `api.together.ai`
2. Go to **API Keys → Create key**
3. Copy the key

---

## Step 2 — Add to `.env`

```env
TOGETHER_API_KEY=your_key_here
TOGETHER_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

Leave `SAGEMAKER_ENDPOINT_NAME` blank — `gemma4_31b_agent.py` automatically falls through to Together.ai when it is not set.

---

## Step 3 — Run the pipeline in order

Activate your virtual environment first, then run each stage in sequence:

```bash
python extract_tasks.py       # pulls tasks from the 100K dataset
python plan_variants.py       # builds 225K variant plan (150K × 1.5 oversample)
python gemma4_31b_agent.py    # generates trajectories via Together.ai
python validate_n_dedup.py    # validates + deduplicates
python augment_150k_rows.py   # merges with original 100K → 250K Parquet
```

Each stage checkpoints its output to `~/pipeline/data/`. If a run stops midway, re-running the same script resumes from where it left off.

---

## Backend priority in `gemma4_31b_agent.py`

| Priority | Backend | Condition |
|---|---|---|
| 1 | SageMaker | `SAGEMAKER_ENDPOINT_NAME` is set in `.env` |
| 2 | Together.ai | `TOGETHER_API_KEY` is set, no SageMaker endpoint |
| 3 | Cerebras | Fallback — uses `zai-glm-4.7` (free tier) |

Once the SageMaker quota is approved, run `deploy_sagemaker.py` separately and add `SAGEMAKER_ENDPOINT_NAME` to `.env` for future runs.
