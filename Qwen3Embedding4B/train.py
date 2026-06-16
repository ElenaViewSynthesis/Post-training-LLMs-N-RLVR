"""
Training script using MultipleNegativesRankingLoss with NO_DUPLICATES batch sampler.
Imports model, datasets, and evaluator from finetune.py (setup only, no training runs on import).

Run:
    python train.py
"""

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from unsloth import is_bf16_supported

from finetune import cfg, model, train_dataset, eval_dataset, evaluator

# ── Loss ──────────────────────────────────────────────────────────────────────
# Uses other positives in the same batch as negatives — no explicit negative column needed.
# NO_DUPLICATES sampler ensures duplicate anchors aren't used as false negatives.

loss = losses.MultipleNegativesRankingLoss(model)

# ── Trainer ───────────────────────────────────────────────────────────────────

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
    args=SentenceTransformerTrainingArguments(
        output_dir="output",
        max_steps=30,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        gradient_checkpointing=cfg.use_gradient_checkpointing,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={
            "anchor":   model.prompts.get("query",    ""),
            "positive": model.prompts.get("document", ""),
        },
        eval_strategy="steps",
        eval_steps=5,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    ),
)

# ── Run ───────────────────────────────────────────────────────────────────────

print("Starting training ...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"Saving LoRA adapters to {cfg.output_adapters} ...")
model.save_pretrained(cfg.output_adapters)

print(f"Merging and saving to {cfg.output_merged} ...")
model.save_pretrained_merged(cfg.output_merged)

if cfg.hub_repo:
    model.push_to_hub(cfg.hub_repo)
    model.push_to_hub_merged(f"{cfg.hub_repo}-merged")
