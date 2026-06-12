"""
Gemma 4 — W&B / Weave Monitoring
Tracks inference requests via Weave ops against the W&B inference endpoint.

Usage:
    python 06_wandb_monitor.py

Requires in .env:
    WANDB_API_KEY   — your W&B API key
    WANDB_PROJECT   — entity/project slug, e.g. elenamylocuda-gemma/gemma-4-12b-it
    MODEL_ID        — model served at the W&B inference endpoint
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import weave

load_dotenv()

WANDB_PROJECT = os.environ["WANDB_PROJECT"]
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-12B-it")

weave.init(WANDB_PROJECT)


@weave.op
def create_completion(message: str, system: str = "You are a helpful assistant.") -> str:
    client = OpenAI(
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.environ["WANDB_API_KEY"],
        default_headers={"project": WANDB_PROJECT},
    )
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = create_completion("Explain the difference between SFT and RLVR in 2 paragraphs.")
    print(result)
