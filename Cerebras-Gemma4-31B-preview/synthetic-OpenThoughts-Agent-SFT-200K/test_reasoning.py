"""
Test reasoning mode on the live SageMaker endpoint.

Requires SAGEMAKER_ENDPOINT_NAME and AWS_REGION in .env (set after deploy_sagemaker.py runs).

Run:
    python test_reasoning.py
"""
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
REGION        = os.getenv("AWS_REGION", "us-east-1")

sm = boto3.client("sagemaker-runtime", region_name=REGION)

payload = {
    "inputs": "Think step by step: explain gradient descent",
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "enable_reasoning": True,
        "reasoning_depth": 3,
    },
}

print(f"Endpoint : {ENDPOINT_NAME}")
print(f"Region   : {REGION}")
print(f"Prompt   : {payload['inputs']}\n")
print("─" * 60)

response = sm.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(response["Body"].read())
print(result)
