"""
Test reasoning mode and function calling on the live SageMaker endpoint.

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


def invoke(payload: dict, label: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"TEST: {label}")
    print(f"{'─' * 60}")
    response = sm.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    print(json.dumps(result, indent=2))


# ── Available functions Gemma can call ────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns current weather conditions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'New York'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Semantic search over an internal knowledge base. Returns top-k relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return. Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Executes a Python code snippet in a sandboxed environment and returns stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python 3 code to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds. Default 10.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


# ── Test 1: Reasoning mode ─────────────────────────────────────────────────────

reasoning_payload = {
    "inputs": "Think step by step: explain gradient descent",
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "enable_reasoning": True,
        "reasoning_depth": 6,
    },
}

# ── Test 2: Function calling ───────────────────────────────────────────────────

function_calling_payload = {
    "inputs": "What is the weather in Tokyo right now? Use the available tools.",
    "parameters": {
        "max_new_tokens": 500,
        "temperature": 0.1,
        "tools": TOOLS,
        "tool_choice": "auto",
    },
}

# ── Test 3: Function calling — code execution ──────────────────────────────────

code_exec_payload = {
    "inputs": (
        "I need to know the first 10 Fibonacci numbers. "
        "Use the run_python_code function to compute them."
    ),
    "parameters": {
        "max_new_tokens": 500,
        "temperature": 0.1,
        "tools": TOOLS,
        "tool_choice": "auto",
    },
}

# ── Test 4: Multimodal — image + text + reasoning ──────────────────────────────
#
# Replace IMAGE_URL with any publicly accessible image (JPEG/PNG/WebP).
# The endpoint must have OPTION_MULTIMODAL_SUPPORT=true (already set in deploy_sagemaker.py).
#
# TGI 2.4.1 multimodal format: pass inputs as a list of typed content blocks.
# Supported block types: "text", "image_url"
#
# Examples of suitable test images:
#   - A chart or diagram to test visual reasoning
#   - A photograph to test scene description
#   - A screenshot of code to test visual code understanding

IMAGE_URL = os.getenv("TEST_IMAGE_URL", "")  # set TEST_IMAGE_URL in .env or override here

multimodal_payload = {
    "inputs": [
        {
            "type": "image_url",
            "image_url": {"url": IMAGE_URL},
        },
        {
            "type": "text",
            "text": (
                "Think step by step: Describe everything you see in this image. "
                "Identify key objects, colors, spatial relationships, and any text present. "
                "Then explain what this image is likely communicating or used for."
            ),
        },
    ],
    "parameters": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "enable_reasoning": True,
        "reasoning_depth": 6,
    },
}


# ── Run all tests ──────────────────────────────────────────────────────────────

print(f"Endpoint : {ENDPOINT_NAME}")
print(f"Region   : {REGION}")
print(f"\nAvailable functions:")
for t in TOOLS:
    fn = t["function"]
    params = list(fn["parameters"]["properties"].keys())
    print(f"  • {fn['name']}({', '.join(params)}) — {fn['description']}")

invoke(reasoning_payload,        "Reasoning mode (depth 6)")
invoke(function_calling_payload, "Function calling — get_weather")
invoke(code_exec_payload,        "Function calling — run_python_code")

if IMAGE_URL:
    invoke(multimodal_payload, "Multimodal — image + text + reasoning (depth 6)")
else:
    print("\n[SKIPPED] Multimodal test — set TEST_IMAGE_URL in .env to enable")
