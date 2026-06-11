"""
Gemma 4 — Basic Inference
Covers: text, multimodal (audio, image, video) inputs, reasoning toggle.

Cloud GPU (SSH) usage:
    Single GPU:  CUDA_VISIBLE_DEVICES=0 python 01_inference.py
    Multi-GPU:   CUDA_VISIBLE_DEVICES=0,1 python 01_inference.py
    Specific ID: set GPU_ID env var, e.g. GPU_ID=1 python 01_inference.py
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForMultimodalLM

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-12B-it")

# On cloud SSH nodes, pin to a specific GPU via GPU_ID or fall back to "auto"
# which distributes across all GPUs visible to the process.
_gpu = os.environ.get("GPU_ID")
DEVICE_MAP = f"cuda:{_gpu}" if _gpu else "auto"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # explicit bf16 — best for A100/H100
    device_map=DEVICE_MAP,
)


def generate(messages: list, max_new_tokens: int = 1024, thinking: bool = False) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(raw)


# ── Text ──────────────────────────────────────────────────────────────────────
text_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
print("=== Text ===")
print(generate(text_messages))

# ── Text + reasoning ──────────────────────────────────────────────────────────
reasoning_messages = [
    {"role": "user", "content": "Solve step-by-step: if 2x + 5 = 17, what is x?"},
]
print("\n=== Text + Reasoning ===")
print(generate(reasoning_messages, thinking=True))

# ── Audio (E2B / E4B / 12B only) ──────────────────────────────────────────────
audio_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Transcribe the following speech segment in its original language. "
                    "Only output the transcription, with no newlines. "
                    "Write numbers as digits."
                ),
            },
            {
                "type": "audio",
                "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/journal1.wav",
            },
        ],
    }
]
print("\n=== Audio ===")
print(generate(audio_messages, max_new_tokens=512))

# ── Image ─────────────────────────────────────────────────────────────────────
image_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]
print("\n=== Image ===")
print(generate(image_messages))
