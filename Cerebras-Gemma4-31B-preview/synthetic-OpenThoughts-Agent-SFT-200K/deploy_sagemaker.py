"""
One-time deployment: Gemma-4-31B-it to a SageMaker TGI endpoint.

Run once:
    python deploy_sagemaker.py

Then add the printed endpoint name to your .env:
    SAGEMAKER_ENDPOINT_NAME=gemma-4-31b-sft-pipeline

After that, gemma4_31b_agent.py will route through SageMaker automatically
when SAGEMAKER_ENDPOINT_NAME is set in .env.

Instance: ml.g6.12xlarge — 4x NVIDIA L4 (96 GB total VRAM).
Gemma-4-31B-it at bfloat16 needs ~62 GB; fits with TGI tensor parallelism.

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
        "SM_NUM_GPUS":       "4",
        "MAX_INPUT_LENGTH":  "4096",
        "MAX_TOTAL_TOKENS":  "8192",
        "HF_TOKEN":          os.environ.get("HF_TOKEN", ""),
    },
    role=role,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g6.12xlarge",
    endpoint_name="gemma-4-31b-sft-pipeline",
)

print(f"\nEndpoint deployed: {predictor.endpoint_name}")
print(f"Add to .env:\n  SAGEMAKER_ENDPOINT_NAME={predictor.endpoint_name}")
print(f"  AWS_REGION={boto3.session.Session().region_name}")
