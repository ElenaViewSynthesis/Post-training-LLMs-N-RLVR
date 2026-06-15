"""
Full fine-tuning pipeline for embedding models using Unsloth + LoRA,
followed by vLLM serving of the merged model.

Models supported:
  google/embeddinggemma-300m
  Qwen/Qwen3-Embedding-0.6B
  sentence-transformers/all-MiniLM-L6-v2
  BAAI/bge-reranker-v2-m3
"""

import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from unsloth import FastSentenceTransformer

# ── Per-model sequence length limits ─────────────────────────────────────────
# Long documents: use the model's full context window.
# Reduce if you hit OOM — memory scales quadratically with sequence length.
MAX_SEQ_LENGTHS = {
    "google/embeddinggemma-300m":            8192,
    "Qwen/Qwen3-Embedding-0.6B":            32768,
    "sentence-transformers/all-MiniLM-L6-v2":  512,  # hard BERT cap
    "BAAI/bge-reranker-v2-m3":              8192,
}

# Batch size drops with longer sequences to keep VRAM stable.
# Rule of thumb: halve the batch size each time you double the seq length.
BATCH_SIZES = {
    "google/embeddinggemma-300m":             4,
    "Qwen/Qwen3-Embedding-0.6B":              2,
    "sentence-transformers/all-MiniLM-L6-v2": 32,
    "BAAI/bge-reranker-v2-m3":                4,
}

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_id: str        = "google/embeddinggemma-300m"
    dataset_id: str      = "sentence-transformers/natural-questions-hard-negatives"
    output_adapters: str = "output/lora-adapters"
    output_merged: str   = "output/merged-model"
    hub_repo: str        = ""           # set to push: "your-hf-username/embeddinggemma-finetuned"

    lora_r: int          = 16
    lora_alpha: int      = 32
    lora_dropout: float  = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    num_train_epochs: int        = 1
    gradient_accumulation_steps: int = 8   # effective batch = per_device_batch_size × 8
    learning_rate: float         = 2e-4
    warmup_ratio: float          = 0.1
    gradient_checkpointing: bool = True    # trades compute for VRAM; essential for long seqs
    fp16: bool                   = not torch.cuda.is_bf16_supported()
    bf16: bool                   = torch.cuda.is_bf16_supported()

    @property
    def max_seq_length(self) -> int:
        return MAX_SEQ_LENGTHS.get(self.model_id, 8192)

    @property
    def per_device_batch_size(self) -> int:
        return BATCH_SIZES.get(self.model_id, 4)

cfg = Config()

# ── 1. Load model + apply LoRA ────────────────────────────────────────────────

print(f"Loading {cfg.model_id} ...")
model = FastSentenceTransformer.from_pretrained(
    cfg.model_id,
    max_seq_length=cfg.max_seq_length,
)

model = FastSentenceTransformer.get_peft_model(
    model,
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=list(cfg.target_modules),
    bias="none",
)
model.print_trainable_parameters()

# ── 2. Dataset ────────────────────────────────────────────────────────────────

print(f"Loading dataset {cfg.dataset_id} ...")
dataset = load_dataset(cfg.dataset_id, split="train")

# Expected columns: anchor, positive, negative
# MultipleNegativesRankingLoss uses (anchor, positive) pairs;
# hard negatives are picked up automatically when a "negative" column exists.
train_dataset = dataset.select_columns(["anchor", "positive", "negative"])

# ── 3. Loss function ──────────────────────────────────────────────────────────

loss = MultipleNegativesRankingLoss(model)

# ── 4. Training arguments ─────────────────────────────────────────────────────

args = SentenceTransformerTrainingArguments(
    output_dir=cfg.output_adapters,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    warmup_ratio=cfg.warmup_ratio,
    fp16=cfg.fp16,
    bf16=cfg.bf16,
    gradient_checkpointing=cfg.gradient_checkpointing,
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
)

# ── 5. Train ──────────────────────────────────────────────────────────────────

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

print("Starting training ...")
trainer.train()

# ── 6. Save LoRA adapters ─────────────────────────────────────────────────────

print(f"Saving LoRA adapters to {cfg.output_adapters} ...")
model.save_pretrained(cfg.output_adapters)

# ── 7. Merge adapters into base model and save ────────────────────────────────

print(f"Merging and saving to {cfg.output_merged} ...")
model.save_pretrained_merged(cfg.output_merged)

# ── 8. (Optional) Push to Hugging Face Hub ───────────────────────────────────

if cfg.hub_repo:
    print(f"Pushing adapters to {cfg.hub_repo} ...")
    model.push_to_hub(cfg.hub_repo)
    model.push_to_hub_merged(f"{cfg.hub_repo}-merged")

# ── 9. vLLM inference on the merged model ────────────────────────────────────

print("Loading merged model into vLLM ...")
from vllm import LLM

llm = LLM(
    model=cfg.output_merged,
    task="embed",
    dtype="bfloat16" if cfg.bf16 else "float16",
    max_model_len=cfg.max_seq_length,
)

queries = [
    "What is the capital of France?",
    "How does photosynthesis work?",
]
documents = [
    "Paris is the capital and largest city of France.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
]

query_outputs    = llm.embed(queries)
document_outputs = llm.embed(documents)

query_vecs = torch.tensor([o.outputs.embedding for o in query_outputs])
doc_vecs   = torch.tensor([o.outputs.embedding for o in document_outputs])

similarity = torch.nn.functional.cosine_similarity(query_vecs, doc_vecs)
for q, d, s in zip(queries, documents, similarity):
    print(f"\nQuery:    {q}")
    print(f"Document: {d}")
    print(f"Score:    {s:.4f}")
