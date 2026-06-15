"""
Full fine-tuning pipeline for embedding models using Unsloth + LoRA,
followed by vLLM serving of the merged model.

Models supported:
  unsloth/embeddinggemma-300m
  unsloth/Qwen3-Embedding-0.6B
  unsloth/all-MiniLM-L6-v2
  unsloth/bge-reranker-v2-m3
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss
from unsloth import FastSentenceTransformer

# ── Per-model sequence length limits ─────────────────────────────────────────
# Long documents: use the model's full context window.
# Reduce if you hit OOM — memory scales quadratically with sequence length.
MAX_SEQ_LENGTHS = {
    "unsloth/embeddinggemma-300m":  8192,
    "unsloth/Qwen3-Embedding-0.6B": 32768,
    "unsloth/all-MiniLM-L6-v2":    512,    # hard BERT cap
    "unsloth/bge-reranker-v2-m3":  8192,
}

# Batch size drops with longer sequences to keep VRAM stable.
# Rule of thumb: halve the batch size each time you double the seq length.
BATCH_SIZES = {
    "unsloth/embeddinggemma-300m":  4,
    "unsloth/Qwen3-Embedding-0.6B": 2,
    "unsloth/all-MiniLM-L6-v2":    32,
    "unsloth/bge-reranker-v2-m3":  4,
}

# LoRA target modules differ by architecture:
#   decoder (Gemma, Qwen)   → attention + MLP projection layers
#   encoder (BERT, RoBERTa) → attention only, different naming
TARGET_MODULES = {
    "unsloth/embeddinggemma-300m":  ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "unsloth/Qwen3-Embedding-0.6B": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "unsloth/all-MiniLM-L6-v2":    ("query",  "key",    "value"),
    "unsloth/bge-reranker-v2-m3":  ("query",  "key",    "value",  "dense"),
}

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_id: str        = "unsloth/embeddinggemma-300m"
    dataset_id: str      = "grasson/t2-ragbench"
    output_adapters: str = "output/lora-adapters"
    output_merged: str   = "output/merged-model"
    hub_repo: str        = ""           # set to push: "your-hf-username/embeddinggemma-finetuned"

    lora_r: int               = 16        # 8, 16, 32, 64, 128
    lora_alpha: int           = 32        # 64
    lora_dropout: float       = 0.05
    use_rslora: bool          = False     # rank stabilized LoRA
    loftq_config              = None      # LoftQ quantization config
    task_type: str            = "FEATURE_EXTRACTION"
    use_gradient_checkpointing = "unsloth" # True or "unsloth" for very long context
    random_state: int         = 3407

    @property
    def target_modules(self) -> tuple:
        return TARGET_MODULES.get(self.model_id, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))

    num_train_epochs: int        = 1
    gradient_accumulation_steps: int = 8   # effective batch = per_device_batch_size × 8
    learning_rate: float         = 2e-4
    warmup_ratio: float          = 0.1
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
    full_finetuning = False, #
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

# ── Memory stats after model + LoRA setup ────────────────────────────────────
gpu_stats         = torch.cuda.get_device_properties(0)
start_gpu_memory  = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory        = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ── 2. Dataset ────────────────────────────────────────────────────────────────

print(f"Loading dataset {cfg.dataset_id} ...")

stream_train = list(
    load_dataset(cfg.dataset_id, split="train", streaming=True).take(10000)
)
stream_eval = list(
    load_dataset(cfg.dataset_id, split="validation", streaming=True).take(2000)
)

train_dataset = Dataset.from_generator(lambda: (yield from stream_train))
eval_dataset  = Dataset.from_generator(lambda: (yield from stream_eval))

# ── 2b. Map to (anchor, positive, negative) triplets ─────────────────────────

def create_embedding_examples(example):
    return {
        "anchor":   example["question"],
        "positive": example["context"],
        "negative": example["hard_negative"],  # adjust column name if needed
    }

train_dataset = train_dataset.map(
    create_embedding_examples,
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    create_embedding_examples,
    remove_columns=eval_dataset.column_names,
)

# ── 3. Evaluator ─────────────────────────────────────────────────────────────
# queries[i] is answered by corpus[i] — valid for RAG benchmarks with 1:1 alignment.
# train passages are appended as distractors to make retrieval harder.

queries      = dict(enumerate(eval_dataset["anchor"]))
corpus       = dict(enumerate(
    list(eval_dataset["positive"]) + train_dataset["positive"][:2000]
))
relevant_docs = {idx: [idx] for idx in queries}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    show_progress_bar=False,
    batch_size=64,
    recall_at_k=[5, 10],
    mrr_at_k=[10],
    ndcg_at_k=[10],
)

# Baseline score before training
autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
with torch.autocast(device_type="cuda", dtype=autocast_dtype):
    print("Baseline evaluator score:", evaluator(model))

if __name__ == "__main__":
    # ── 5. Loss function ──────────────────────────────────────────────────────────

    loss = TripletLoss(model)
    # loss = MultipleNegativesRankingLoss(model)  # alternative: in-batch negatives, no explicit negative column needed

    # ── 6. Training arguments ─────────────────────────────────────────────────────

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg.output_adapters,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.use_gradient_checkpointing,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────────

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # ── Memory stats before training ─────────────────────────────────────────────
    pre_train_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Memory reserved before training: {pre_train_memory} GB / {max_memory} GB.")

    print("Starting training ...")
    trainer_stats = trainer.train()

    # ── Final memory and time stats ───────────────────────────────────────────────
    used_memory            = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora   = round(used_memory - start_gpu_memory, 3)
    used_percentage        = round(used_memory / max_memory * 100, 3)
    lora_percentage        = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # ── Post-training evaluation ──────────────────────────────────────────────────
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float16):
        print("Post-training evaluator score:", evaluator(model))

    # ── Post-training inference example ──────────────────────────────────────────
    # Replace query and candidates with examples from your domain.
    query = "Patient presents with sharp chest pain that improves when leaning forward."

    candidates = [
        "Acute Pericarditis often involves pleuritic chest pain relieved by sitting up and leaning forward.",
        "Myocardial Infarction typically presents with crushing substernal pressure and radiation to the left arm.",
        "Pneumothorax is characterized by sudden onset shortness of breath and unilateral chest pain.",
        "Gastroesophageal Reflux Disease (GERD) causes burning retrosternal pain usually after meals.",
    ]

    query_emb      = model.encode(query, convert_to_tensor=True)
    candidate_embs = model.encode(candidates, convert_to_tensor=True)
    similarities   = model.similarity(query_emb, candidate_embs)

    ranking = similarities.argsort(descending=True)[0]

    print(f"\nQuery: {query}\n")
    for idx in ranking.tolist():
        score = similarities[0][idx].item()
        print(f"{score:.4f} | {candidates[idx]}")

    # ── 8. Save LoRA adapters only (16-bit) ──────────────────────────────────────
    # Saves adapter weights + tokenizer. Does NOT include base model weights.
    # Reload with: FastSentenceTransformer.from_pretrained(path, for_inference=True)

    print(f"Saving LoRA adapters to {cfg.output_adapters} ...")
    model.save_pretrained(cfg.output_adapters)
    model.tokenizer.save_pretrained(cfg.output_adapters)
    model.push_to_hub("borntobeignored/embeddinggemma_lora", token=HF_TOKEN)
    model.tokenizer.push_to_hub("borntobeignored/embeddinggemma_lora", token=HF_TOKEN)

    # ── 9. Merge adapters into base model and save ────────────────────────────────

    print(f"Merging and saving to {cfg.output_merged} ...")
    model.save_pretrained_merged(cfg.output_merged)

    # ── 10. (Optional) Push to Hugging Face Hub ──────────────────────────────────

    if cfg.hub_repo:
        print(f"Pushing adapters to {cfg.hub_repo} ...")
        model.push_to_hub(cfg.hub_repo, token=HF_TOKEN)
        model.push_to_hub_merged(f"{cfg.hub_repo}-merged", token=HF_TOKEN)

    # ── 11. vLLM inference on the merged model ────────────────────────────────────

    print("Loading merged model into vLLM ...")
    from vllm import LLM

    llm = LLM(
        model=cfg.output_merged,
        task="embed",
        dtype="bfloat16" if cfg.bf16 else "float16",
        max_model_len=cfg.max_seq_length,
    )

    test_queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
    ]
    test_documents = [
        "Paris is the capital and largest city of France.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]

    query_outputs    = llm.embed(test_queries)
    document_outputs = llm.embed(test_documents)

    query_vecs = torch.tensor([o.outputs.embedding for o in query_outputs])
    doc_vecs   = torch.tensor([o.outputs.embedding for o in document_outputs])

    similarity = torch.nn.functional.cosine_similarity(query_vecs, doc_vecs)
    for q, d, s in zip(test_queries, test_documents, similarity):
        print(f"\nQuery:    {q}")
        print(f"Document: {d}")
        print(f"Score:    {s:.4f}")
