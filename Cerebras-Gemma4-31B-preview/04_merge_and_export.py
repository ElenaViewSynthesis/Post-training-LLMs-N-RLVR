"""
Cerebras Gemma4-31B-preview — Merge LoRA adapters into the base model

Run after SFT or GRPO to produce a single stand-alone model checkpoint
that can be pushed to the Hub or served directly.

Run:
    python 04_merge_and_export.py
"""

import os
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

BASE_MODEL_ID  = os.getenv("HF_MODEL_ID", "cerebras/Gemma4-31B-preview")
ADAPTER_PATH   = os.getenv("ADAPTER_PATH", "./checkpoints/grpo-rlvr")
OUTPUT_PATH    = os.getenv("MERGED_OUTPUT", "./checkpoints/merged")
HF_REPO        = os.getenv("HF_REPO_ID", "")   # optional: push to Hub

print(f"Loading base model: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print(f"Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter weights into base model…")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

if HF_REPO:
    print(f"Pushing to Hub: {HF_REPO}")
    model.push_to_hub(HF_REPO, safe_serialization=True)
    tokenizer.push_to_hub(HF_REPO)

print("Done.")
