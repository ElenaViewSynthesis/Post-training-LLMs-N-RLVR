"""
Cerebras Gemma4-31B-preview — Supervised Fine-Tuning (SFT) with QLoRA

QLoRA makes the 31B parameter model trainable on a single A100-80GB or
two A100-40GB GPUs. For multi-node runs use accelerate launch.

Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate

Run (single GPU):
    python 02_sft_lora.py

Run (multi-GPU):
    accelerate launch 02_sft_lora.py
"""

import os
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("HF_MODEL_ID", "cerebras/Gemma4-31B-preview")
OUTPUT_DIR = "./checkpoints/sft-lora"
USE_QLORA = True

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
) if USE_QLORA else None

# ── Load tokenizer & model ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.enable_input_require_grads()
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

# ── Dataset (replace with your own) ──────────────────────────────────────────
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
    return {
        "text": tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = Dataset.from_list(raw_data).map(format_sample)

# ── Training config ───────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
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
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT model saved to {OUTPUT_DIR}")
