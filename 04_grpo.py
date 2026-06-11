"""
Gemma 4 — GRPO (Group Relative Policy Optimisation) post-training
Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate

GRPO is the RL method used in DeepSeek-R1. It scores groups of sampled
completions relative to each other, which eliminates the need for a value
network (critic). It works especially well for verifiable reasoning tasks.

Run:
    accelerate launch 04_grpo.py
"""

import re
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-12B-it"
OUTPUT_DIR = "./gemma4-grpo"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.enable_input_require_grads()

# ── Dataset: simple arithmetic with verifiable answers ────────────────────────
math_problems = [
    {"prompt": "What is 15 + 27?",  "answer": "42"},
    {"prompt": "What is 8 × 9?",    "answer": "72"},
    {"prompt": "What is 144 / 12?", "answer": "12"},
    {"prompt": "What is 2^10?",     "answer": "1024"},
]

def format_prompt(sample):
    msgs = [
        {"role": "system", "content": "Answer the math question. End with 'Answer: <number>'."},
        {"role": "user", "content": sample["prompt"]},
    ]
    return {
        "prompt": processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        "answer": sample["answer"],
    }

dataset = Dataset.from_list(math_problems).map(format_prompt)

# ── Reward functions ──────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    """Pull the final 'Answer: X' from a completion."""
    match = re.search(r"Answer:\s*([0-9]+)", text)
    return match.group(1).strip() if match else ""

def correctness_reward(completions: list[str], answer: list[str], **_) -> list[float]:
    """+1.0 for a correct numerical answer, 0.0 otherwise."""
    return [1.0 if extract_answer(c) == a else 0.0 for c, a in zip(completions, answer)]

def format_reward(completions: list[str], **_) -> list[float]:
    """Small reward for following the 'Answer: N' format."""
    return [0.2 if re.search(r"Answer:\s*[0-9]+", c) else 0.0 for c in completions]

# ── GRPO training config ──────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    num_generations=4,          # completions sampled per prompt per step
    max_new_tokens=256,
    temperature=0.9,
    logging_steps=5,
    save_strategy="epoch",
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=lora_config,
    reward_funcs=[correctness_reward, format_reward],
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"GRPO model saved to {OUTPUT_DIR}")
