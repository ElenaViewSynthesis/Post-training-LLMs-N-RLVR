"""
One-time deployment: Gemma-4-31B-it to a SageMaker TGI endpoint.

Run once:
    python deploy_sagemaker.py

Then add the printed endpoint name to your .env:
    SAGEMAKER_ENDPOINT_NAME=gemma-4-31b-sft-pipeline

After that, gemma4_31b_agent.py will route through SageMaker automatically
when SAGEMAKER_ENDPOINT_NAME is set in .env.

Instance: ml.p5.48xlarge — 8x H100 SXM 80GB (640 GB total VRAM), using 4 GPUs.
Gemma-4-31B-it at bfloat16 needs ~62 GB; fits comfortably with TGI tensor parallelism.

Note: requires sagemaker<3.0.0
    pip install 'sagemaker<3.0.0'
"""
import os
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

load_dotenv(Path(__file__).parent.parent / ".env")

role = "arn:aws:iam::149901539173:role/Gemma-4-31b-deploy"

huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="2.4.1"),
    env={
        "HF_MODEL_ID":       "google/gemma-4-31b-it",
        "SM_NUM_GPUS":       "4",           # 4x H100 80GB = 320GB VRAM
        "MAX_INPUT_LENGTH":  "32768",
        "MAX_TOTAL_TOKENS":  "65536",
        "HF_TOKEN":          os.environ.get("HF_TOKEN", ""),
    },
    role=role,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.p5.48xlarge",        # 8x H100 SXM 80GB
    endpoint_name="gemma-4-31b-sft-pipeline",
)

print(f"\nEndpoint deployed: {predictor.endpoint_name}")
print(f"Add to .env:\n  SAGEMAKER_ENDPOINT_NAME={predictor.endpoint_name}")
print(f"  AWS_REGION={boto3.session.Session().region_name}")

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

import json
response = predictor.predict(test_payload)
content  = response["choices"][0]["message"]["content"]
print("Smoke test response (first 300 chars):")
print(content[:300])
try:
    parsed = json.loads(content)
    assert isinstance(parsed.get("conversations"), list), "missing 'conversations' list"
    print(f"JSON valid — {len(parsed['conversations'])} turns generated")
    print("Endpoint is healthy and ready for gemma4_31b_agent.py")
except (json.JSONDecodeError, AssertionError) as e:
    print(f"Warning: response is not valid pipeline JSON ({e})")
    print("Check the system prompt or model output format before running the full pipeline.")
