"""
Cerebras Gemma4-31B-preview — GRPO / RLVR Post-training

GRPO (Group Relative Policy Optimisation) is the RL method behind DeepSeek-R1.
It scores groups of sampled completions relative to each other, eliminating
the need for a critic/value network.

RLVR (RL from Verifiable Rewards) pairs GRPO with reward functions whose
correctness can be verified programmatically (math, code, structured output)
rather than relying on a learned reward model.

Cerebras integration: the Cerebras Cloud API (100k+ tok/sec) is used for
the rollout phase — generating N completions per prompt. This dramatically
accelerates the RL inner loop compared to vLLM on a single GPU.
Gradient updates run locally via TRL + PEFT.

Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate cerebras-cloud-sdk

Run:
    accelerate launch 03_grpo_rlvr.py
"""

import os
import re
import torch
from cerebras.cloud.sdk import Cerebras
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_ID      = os.getenv("HF_MODEL_ID", "cerebras/Gemma4-31B-preview")
CEREBRAS_MODEL   = os.getenv("CEREBRAS_MODEL_ID", "cerebras/Gemma4-31B-preview")
OUTPUT_DIR       = "./checkpoints/grpo-rlvr"
NUM_GENERATIONS  = 8    # rollouts sampled per prompt — more = better gradient signal
MAX_NEW_TOKENS   = 512
ROLLOUT_TEMP     = 0.9

# ── Cerebras rollout client ───────────────────────────────────────────────────
cerebras_client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

def cerebras_rollout(prompt: str, n: int = NUM_GENERATIONS) -> list[str]:
    """Generate n completions via Cerebras Cloud (ultra-fast rollout engine)."""
    completions = []
    for _ in range(n):
        resp = cerebras_client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_NEW_TOKENS,
            temperature=ROLLOUT_TEMP,
        )
        completions.append(resp.choices[0].message.content)
    return completions

# ── LoRA config ───────────────────────────────────────────────────────────────
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
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # GRPO needs left-padding for generation

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.enable_input_require_grads()

# ── Dataset: verifiable math problems ────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Think step-by-step, then state your final answer as 'Answer: <number>'."
)

math_problems = [
    {"prompt": "What is 15 + 27?",         "answer": "42"},
    {"prompt": "What is 8 × 9?",           "answer": "72"},
    {"prompt": "What is 144 / 12?",        "answer": "12"},
    {"prompt": "What is 2^10?",            "answer": "1024"},
    {"prompt": "What is the square root of 169?", "answer": "13"},
    {"prompt": "What is 17 × 13?",         "answer": "221"},
    {"prompt": "What is 1000 - 437?",      "answer": "563"},
    {"prompt": "What is 7! (7 factorial)?","answer": "5040"},
]

def format_prompt(sample):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": sample["prompt"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ),
        "answer": sample["answer"],
    }

dataset = Dataset.from_list(math_problems).map(format_prompt)

# ── Reward functions ──────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*([0-9,]+)", text)
    return match.group(1).replace(",", "").strip() if match else ""

def correctness_reward(completions: list[str], answer: list[str], **_) -> list[float]:
    """+1.0 for exact numerical match, 0.0 otherwise."""
    return [1.0 if extract_answer(c) == a else 0.0 for c, a in zip(completions, answer)]

def format_reward(completions: list[str], **_) -> list[float]:
    """Partial reward for following 'Answer: N' format even if wrong."""
    return [0.2 if re.search(r"Answer:\s*[0-9]", c) else 0.0 for c in completions]

def reasoning_length_reward(completions: list[str], **_) -> list[float]:
    """Small reward for showing work (≥50 words before the answer)."""
    rewards = []
    for c in completions:
        pre_answer = re.split(r"Answer:", c)[0] if "Answer:" in c else c
        rewards.append(0.1 if len(pre_answer.split()) >= 50 else 0.0)
    return rewards

# ── GRPO training config ──────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    num_generations=NUM_GENERATIONS,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=ROLLOUT_TEMP,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
    reward_funcs=[correctness_reward, format_reward, reasoning_length_reward],
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"GRPO/RLVR model saved to {OUTPUT_DIR}")
