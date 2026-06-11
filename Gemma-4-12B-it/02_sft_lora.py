"""
Gemma 4 — Supervised Fine-Tuning (SFT) with LoRA / QLoRA
Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate

Run:
    accelerate launch 02_sft_lora.py
"""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-12B-it"
OUTPUT_DIR = "./gemma4-sft-lora"
USE_QLORA = True          # set False for full-precision LoRA

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

# ── Quantisation (QLoRA) ──────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
) if USE_QLORA else None

# ── Load processor & model ────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",   # remove if not available
)
model.enable_input_require_grads()             # required for gradient checkpointing + PEFT
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

# ── Toy dataset (replace with your own) ──────────────────────────────────────
raw_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is a Python list comprehension?"},
            {"role": "assistant", "content": "A list comprehension is a concise way to create lists: [expr for item in iterable if condition]."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do I reverse a string in Python?"},
            {"role": "assistant", "content": "Use slicing: reversed_s = s[::-1]"},
        ]
    },
]

def format_sample(sample):
    """Convert messages list to a single text string the trainer can consume."""
    return {
        "text": processor.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = Dataset.from_list(raw_data).map(format_sample)

# ── Training arguments ────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=processor,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
