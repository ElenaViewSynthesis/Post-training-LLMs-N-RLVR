# Gemma-4-31B Reasoning Endpoint

Deploys `google/gemma-4-31b-it` as a SageMaker TGI inference endpoint with reasoning mode enabled.

## Endpoint Instance — `ml.p4d.24xlarge`

| Property | Value |
|---|---|
| GPUs | 8× NVIDIA A100 SXM4 40GB |
| Total GPU VRAM | 320GB (8 × 40GB) |
| GPU interconnect | NVLink 600 GB/s |
| vCPUs | 96 |
| System RAM | 1,152 GB |
| Local NVMe storage | 8 TB |
| Network bandwidth | 400 Gbps |
| Cost (us-east-1) | **~$32.77/hr** |
| Billing | Per second, billed per minute |

**Why this instance for Gemma-4-31B:**
Gemma-4-31B is 62GB in FP16. Sharded across 8 GPUs via TGI tensor parallelism, each A100 holds ~7.75GB of weights — leaving ~32GB per GPU free for KV cache, enabling long context at `MAX_TOTAL_TOKENS=65536`. All 8 GPUs are active with `SM_NUM_GPUS=8`, none sit idle.

**Cost estimate for a 1-hour session:**
```
Deploy + load model : ~25 min  →  ~$13.65
Testing             : ~35 min  →  ~$19.12
─────────────────────────────────────────
Total (1 hr)        :           ~$32.77
```

Delete the endpoint immediately after testing — idle instances cost the same as active ones.

---

## Prerequisites

- Python 3.10+
- AWS CLI installed
- AWS IAM user with: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `sagemaker-pass-role` inline policy
- SageMaker quota approved: `ml.p4d.24xlarge for endpoint usage` → request at AWS Service Quotas → SageMaker → search `ml.p4d.24xlarge`
- HuggingFace account with access to `google/gemma-4-31b-it` (gated model)

---

## 1. AWS Credentials

Configure your IAM access key:

```bash
aws configure
```

Enter:
- **AWS Access Key ID** — from IAM → Users → Security credentials
- **AWS Secret Access Key** — shown only once at creation
- **Default region** — `us-east-1`
- **Default output format** — `json`

Verify credentials are working:

```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "...",
    "Account": "149901539173",
    "Arn": "arn:aws:iam::149901539173:user/your-username"
}
```

If this returns your account ID, you're good to go.

---

## 2. Environment Variables

The `.env` file lives one level up at `Cerebras-Gemma4-31B-preview/.env` and is shared across both projects. Copy the template and fill in your credentials:

```bash
cp ../env.example ../.env
nano ../.env
```

Required values:

```env
HF_TOKEN=your_huggingface_token
AWS_REGION=us-east-1

# Set after deploy_sagemaker.py runs
SAGEMAKER_ENDPOINT_NAME=gemma-4-31-b-reasoning
```

---

## 3. Install Dependencies

```bash
pip install 'sagemaker<3.0.0' boto3 python-dotenv
```

---

## 4. Check Quota

Before deploying, confirm your `ml.p4d.24xlarge` quota:

```bash
python check_quotas.py
```

Must show at least 1 for `ml.p4d.24xlarge for endpoint usage`.

---

## 5. Deploy the Endpoint

```bash
python deploy_sagemaker.py
```

- Takes **20–30 minutes** to download the 62GB model and load across 8 GPUs
- Prints the endpoint name and S3 artifacts path when done
- Runs a smoke test + reasoning mode test automatically

Add the printed endpoint name to your `.env`:

```env
SAGEMAKER_ENDPOINT_NAME=gemma-4-31-b-reasoning
```

---

## 6. Test Reasoning Mode

```bash
python test_reasoning.py
```

Edit the `"inputs"` line to change the prompt. The model runs 6 reasoning passes before producing its final answer.

---

## 7. Terminate the Endpoint

SageMaker charges ~$32/hr while the endpoint is running — **delete it as soon as you're done**:

```bash
python -c "
import boto3
sm = boto3.client('sagemaker', region_name='us-east-1')
sm.delete_endpoint(EndpointName='gemma-4-31-b-reasoning')
print('Endpoint deleted')
"
```

Billing stops within minutes of deletion.

---

## Files

| File | Purpose |
|---|---|
| `deploy_sagemaker.py` | One-time endpoint deployment |
| `test_reasoning.py` | Test reasoning mode on live endpoint |
| `check_quotas.py` | Check ml.p4d.24xlarge quota status |
| `deploySage.md` | Deployment configuration notes |
| `Activate-Reasoning-depths.md` | Explanation of reasoning depth passes |
