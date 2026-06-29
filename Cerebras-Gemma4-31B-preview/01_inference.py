"""
Cerebras Gemma4-31B-preview — Fast Inference via Cerebras Cloud API

Cerebras achieves 100k+ tokens/sec on their wafer-scale hardware, making
this the fastest hosted inference path for large models.

Run:
    python 01_inference.py
"""

import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("CEREBRAS_MODEL_ID", "cerebras/Gemma4-31B-preview")

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])


def generate(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    if stream:
        with client.chat.completions.stream(
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        ) as s:
            chunks = [chunk.choices[0].delta.content or "" for chunk in s]
        return "".join(chunks)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── Text ──────────────────────────────────────────────────────────────────────
text_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
print("=== Text ===")
print(generate(text_messages))

# ── Reasoning ─────────────────────────────────────────────────────────────────
reasoning_messages = [
    {"role": "user", "content": "Solve step-by-step: if 2x + 5 = 17, what is x?"},
]
print("\n=== Reasoning ===")
print(generate(reasoning_messages, temperature=0.1))

# ── Streaming ─────────────────────────────────────────────────────────────────
stream_messages = [
    {"role": "user", "content": "Explain transformer attention in three sentences."},
]
print("\n=== Streaming ===")
print(generate(stream_messages, stream=True))
