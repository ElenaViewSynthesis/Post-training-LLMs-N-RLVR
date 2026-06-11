"""
Gemma 4 — Direct Preference Optimisation (DPO) post-training
Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate

DPO trains the model to prefer chosen responses over rejected ones without
needing a separate reward model.

Run:
    accelerate launch 03_dpo.py
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-12B-it"   # use your SFT checkpoint if available
OUTPUT_DIR = "./gemma4-dpo"
BETA = 0.1                            # KL penalty coefficient

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

# Reference model stays frozen — DPOTrainer handles this automatically when
# you pass a PEFT config instead of an explicit ref_model.
ref_model = None   # DPOTrainer will clone the base model as frozen reference

# ── Preference dataset ────────────────────────────────────────────────────────
# Each row needs: prompt, chosen, rejected  (all as strings or message lists)
raw_prefs = [
    {
        "prompt": [{"role": "user", "content": "Explain gradient descent simply."}],
        "chosen": [{"role": "assistant", "content": "Gradient descent minimises a loss by taking small steps in the direction opposite to the gradient — like rolling a ball downhill to find the lowest point."}],
        "rejected": [{"role": "assistant", "content": "It is an optimisation algorithm used in machine learning."}],
    },
    {
        "prompt": [{"role": "user", "content": "What is overfitting?"}],
        "chosen": [{"role": "assistant", "content": "Overfitting is when a model memorises training data instead of learning general patterns, so it performs well on training data but poorly on unseen data."}],
        "rejected": [{"role": "assistant", "content": "Overfitting means the model is too complex."}],
    },
]

def format_pref(sample):
    def fmt(msgs):
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {
        "prompt": fmt(sample["prompt"]),
        "chosen": fmt(sample["chosen"]),
        "rejected": fmt(sample["rejected"]),
    }

dataset = Dataset.from_list(raw_prefs).map(format_pref)

# ── DPO training config ───────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    beta=BETA,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=5,
    save_strategy="epoch",
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=lora_config,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"DPO model saved to {OUTPUT_DIR}")
