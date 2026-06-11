"""
Gemma 4 — Merge LoRA adapters into base model and export
Run after any of the finetuning scripts (02/03/04).

Dependencies:
    pip install transformers peft bitsandbytes accelerate

Steps:
  1. Load base model in bf16 (no quantisation — needed for clean merge)
  2. Load PEFT adapter weights on top
  3. Merge and unload adapters
  4. Save the full model for deployment / upload to HF Hub
"""

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForMultimodalLM

BASE_MODEL_ID = "google/gemma-4-12B-it"
ADAPTER_PATH = "./gemma4-sft-lora"   # or dpo / grpo output dir
MERGED_OUTPUT = "./gemma4-merged"

# Load base in full precision for a numerically clean merge
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",              # merge on CPU to avoid GPU OOM
)

# Attach adapter and merge
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()   # folds adapter weights into base tensors

# Save
model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
processor.save_pretrained(MERGED_OUTPUT)
print(f"Merged model saved to {MERGED_OUTPUT}")

# Optional: push to Hugging Face Hub
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(folder_path=MERGED_OUTPUT, repo_id="your-username/gemma4-finetuned")
