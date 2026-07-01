"""
One-time deployment: Gemma-4-31B-it to a SageMaker TGI endpoint.

Infrastructure:
  Compute  — ml.p4d.24xlarge (8x A100 40GB = 320GB VRAM), all 8 GPUs active via TGI tensor parallelism
  Storage  — 256 GB EBS volume on the instance (holds 62GB model weights + overhead)
  Artifacts— S3: s3://<sagemaker-default-bucket>/gemma-4-31b/

Run once:
    python deploy_sagemaker.py

Then add the printed values to your .env:
    SAGEMAKER_ENDPOINT_NAME=gemma-4-31b-sft-pipeline
    AWS_REGION=us-east-1

After that, gemma4_31b_agent.py routes to SageMaker automatically.

Note: requires sagemaker<3.0.0
    pip install 'sagemaker<3.0.0'
"""
import json
import os
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

load_dotenv(Path(__file__).parent.parent / ".env")

# ── IAM + session ─────────────────────────────────────────────────────────────
role    = "arn:aws:iam::149901539173:role/Gemma-4-31b-deploy"
region  = os.getenv("AWS_REGION", "us-east-1")
sess    = sagemaker.Session(boto_session=boto3.Session(region_name=region))
bucket  = sess.default_bucket()                         # sagemaker-{region}-{account-id}
s3_base = f"s3://{bucket}/gemma-4-31b"

print(f"Region:  {region}")
print(f"Bucket:  {bucket}")
print(f"S3 path: {s3_base}")

# ── Model definition ──────────────────────────────────────────────────────────
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="2.4.1"),
    env={
        "HF_MODEL_ID":      "google/gemma-4-31b-it",
        "SM_NUM_GPUS":      "8",            # 8x A100 40GB = 320GB VRAM, all GPUs active
        "MAX_INPUT_LENGTH": "32768",
        "MAX_TOTAL_TOKENS": "65536",
        "HF_TOKEN":         os.environ.get("HF_TOKEN", ""),
        # Gemma 4 specific optimizations
        # "OPTION_ENABLE_REASONING_MODE": "true",
        # "OPTION_MULTIMODAL_SUPPORT":    "true",
    },
    role=role,
    sagemaker_session=sess,
)

# ── Deploy ────────────────────────────────────────────────────────────────────
# Timeouts are critical for 31B models:
#   model_data_download_timeout      — time to download 62GB from HF Hub
#   container_startup_health_check_timeout — time for TGI to load weights into GPU
# Both default to 10 min which is far too short; 1 hour is safe.
print("\nDeploying endpoint (this takes 15–30 minutes for a 31B model)...")
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.p4d.24xlarge",
    endpoint_name="gemma-4-31b-sft-pipeline",
    volume_size=256,                                    # GB EBS — model is 62GB
    model_data_download_timeout=3600,                  # 1 hr to download from HF
    container_startup_health_check_timeout=3600,       # 1 hr for TGI to load weights
)

print(f"\nEndpoint deployed: {predictor.endpoint_name}")
print(f"\nAdd to .env:")
print(f"  SAGEMAKER_ENDPOINT_NAME={predictor.endpoint_name}")
print(f"  AWS_REGION={region}")
print(f"\nS3 artifacts: {s3_base}")

# ── Smoke test ────────────────────────────────────────────────────────────────
print("\nRunning smoke test...")
test_payload = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an expert software/terminal agent. "
                "Output a JSON object with a single key 'conversations' "
                "whose value is a list of turns, each with 'role' and 'content'."
            ),
        },
        {
            "role": "user",
            "content": (
                "## Task\nList the files in /tmp and print the count.\n\n"
                "## Environment\nGeneric Ubuntu 22.04 container.\n\n"
                "Synthesize a complete, realistic agent trajectory that solves this task."
            ),
        },
    ],
    "temperature": 0.7,
    "max_tokens": 512,
}

response = predictor.predict(test_payload)
content  = response["choices"][0]["message"]["content"]
print("Smoke test response (first 300 chars):")
print(content[:300])
try:
    parsed = json.loads(content)
    assert isinstance(parsed.get("conversations"), list), "missing 'conversations' list"
    print(f"\nJSON valid — {len(parsed['conversations'])} turns generated")
    print("Endpoint is healthy and ready for gemma4_31b_agent.py")
except (json.JSONDecodeError, AssertionError) as e:
    print(f"\nWarning: response is not valid pipeline JSON ({e})")
    print("Check the system prompt or model output format before running the full pipeline.")
