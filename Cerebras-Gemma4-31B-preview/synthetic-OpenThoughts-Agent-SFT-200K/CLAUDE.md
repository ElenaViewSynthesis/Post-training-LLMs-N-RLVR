# synthetic-OpenThoughts-Agent-SFT-200K

## Project overview

This project is a research-focused synthetic data generation framework designed to augment the **OpenThoughts-Agent-SFT-100K** dataset into approximately **200,000–250,000 high-quality supervision traces** for post-training large language models. Rather than creating entirely new tasks, the framework preserves the original task distribution while generating diverse conversation trajectories for the same underlying tasks.

The pipeline streams the dataset directly from Hugging Face using **Dask**, extracts task specifications from existing conversations, plans multiple augmentation strategies for each task, and uses **Gemma-4-31B Early Preview** served through **Cerebras Inference** to synthesize new multi-turn conversations with varied reasoning paths, conversation flow, clarification steps, and constraint handling.

## Architecture

The system is a scalable and reproducible data engineering pipeline optimized for high-throughput inference. It consists of modular stages:

| Stage | File | Purpose |
|---|---|---|
| 0-1 | `extract_tasks.py` | Dataset ingestion, schema inspection, task extraction |
| 2 | `plan_variants.py` | Augmentation planning per task |
| 3 | `gemini_agent_worker.py` | Async generation via Cerebras Inference |
| 4 | `poll_n_fetch.py` | Obsolete on the Gemini path; kept as marker |
| 5 | `validate_n_dedup.py` | Validation, dedup, quality filtering |
| 6 | `augment_150k_rows.py` | Merge with original 100K, export to Parquet |

## Teacher model: Gemma-4-31B via Cerebras Inference

Instead of relying on provider-specific agent runtimes or managed execution environments, the framework treats **Gemma-4-31B Early Preview** as a structured trajectory generator, producing complete conversations in a single inference request using carefully engineered prompts. Generation workers are fully asynchronous, resume-safe, checkpointed, and configurable through YAML files, enabling large-scale augmentation while efficiently utilizing the high-throughput capabilities of Cerebras Inference.

Set `CEREBRAS_API_KEY` in your `.env` file before running any generation stage. See `.env.example`.

## Augmentation strategy

- **Same tasks, new rollouts** — task distribution is preserved; only trajectories are regenerated.
- Diversity axes: `thinking_level` (low / medium / high) + instruction framing variants + inherent run-to-run nondeterminism.
- Sampling parameters (temperature, top_p, top_k) are left at model defaults — tuning them degrades reasoning quality on this model family.
- Target: ~150K new accepted rows on top of the original 100K → ~250K total.
- Oversample factor: 1.5× to account for Stage 5 rejection rate.

## Quality assurance

Every generated conversation passes through a multi-layer validation pipeline before being accepted:

1. **JSON schema validation** — output conforms to the expected structure.
2. **Structural verification** — turn count, trajectory completeness, no truncation.
3. **Semantic similarity / near-dup filter** — fingerprint-based dedup per task; upgrade to embedding similarity if collision rate is high.
4. **Duplicate detection** — global dedup across variant IDs.
5. **Safety and PII checks** — degenerate-output regex (refusals, "I cannot access a real environment" disclaimers).
6. **Reasoning quality assessment** — minimum content length, coherent tool-call/observation structure.
7. **Metadata enrichment** — `is_synthetic_augmentation`, `source_task_id`, `variant_id` added to every accepted row.

## Output

Accepted conversations are merged with the original corpus and exported as versioned Parquet datasets to S3, suitable for supervised fine-tuning. Configure the target bucket in `augment_150k_rows.py` (`S3_OUT`).

## Skills demonstrated

- Synthetic data generation for agentic reasoning models
- Large-scale LLM inference via Cerebras Inference API
- Asynchronous distributed systems (asyncio, resume-safe workers)
- Reproducible data pipelines (Dask, partitioned Parquet, S3)
- Dataset curation and automated quality evaluation
- Schema-agnostic ingestion (introspects real column names rather than hardcoding)
