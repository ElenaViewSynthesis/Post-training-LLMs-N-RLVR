# Design Notes — OpenThoughts-Agent-SFT 100K → 250K augmentation pipeline

## Problem statement

I will use this SFT dataset of 100K and trying to create a prompt where I can create a synthetic pipeline for creating new data rows based on the current samples that it has, in order to augment it to 250K traces instead. Using Dask to process it, with S3 storage.

```python
import dask.dataframe as dd
df = dd.read_parquet("hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet")
```

---

## Dataset research

OpenThoughts-Agent-SFT-100K is the top rung of an SFT data-scaling ladder from the OpenThoughts/OpenThinkerAgent release — a series of dataset checkpoints scaling from a few hundred up to 100K samples, sourced from **TaskTrove** and **AgentTrove**, generated via the **Terminus-2 harness** with a strong teacher model and vLLM sampling. That confirms the schema family (task instruction, environment/dockerfile, multi-turn trajectory, optional verifier).

The exact parquet schema is introspected at runtime rather than hardcoded — the pipeline asserts the schema it finds rather than assuming it.

---

## Design decisions

**Q1 — Main augmentation strategy for 100K → 250K?**
→ **Paraphrase/perturb existing traces (same tasks, new agent rollouts)**

**Q2 — What generates the new trajectories?**
→ **API-based teacher via batch calls**

---

## Architecture

Clean setup: same task distribution, new rollouts via an API teacher, batched. Core loop: read 100K rows with dask → extract the *task spec* (instruction + environment, not the trajectory) → re-run each task through a teacher with new sampling → validate/dedup → write out → repeat ~1.5× per row to reach 250K. Since this is API-based, the bottleneck is concurrency/rate limits, not GPU scheduling — an async-batch problem layered on dask partitioning.

```
S3 (raw 100K parquet, read-only)
   │
   ▼
Dask: read_parquet → inspect schema → partition
   │
   ▼
Stage 1 — Task extraction (strip to {task_id, instruction, environment, verifier})
   │
   ▼
Stage 2 — Perturbation plan (k variants/task; vary sampling + light framing)
   │
   ▼
Stage 3 — Batch submission (Anthropic Message Batches API)
   │
   ▼
Stage 4 — Poll + retrieve (≤24h)
   │
   ▼
Stage 5 — Validation + dedup (schema, structural, near-dup, verifier replay)
   │
   ▼
Stage 6 — Merge with original + write 250K parquet → S3
```

**Revised to Gemini path:** The design was subsequently updated to use Gemini 3.5 Flash with the Interactions API (real multi-turn sandboxed agent loop) instead of the Anthropic Batches API (single-shot simulated trajectories). With unlimited Gemini requests, cost is no longer the constraint — fidelity is. See `gemini_agent_worker.py` and `README.md` for the full rationale.

---

## S3 storage recommendations

- **Pin your own copy of the raw 100K** in S3 rather than re-pulling from `hf://` each run, so your 250K stays reproducible against a fixed base.
- **Bucket layout:** separate prefixes for raw, pipeline intermediates, and the final 250K.
- **Storage class:** S3 Standard for intermediates and the actively-trained final set; lifecycle-transition the final set to Standard-IA once a training run is stable. Avoid Glacier anywhere in the active pipeline (cold-retrieval latency stalls dask).
- **Partition file count matters more than bucket settings:** aim for 100–500MB parquet files per partition; too many small files tanks S3 list/get throughput. `repartition(partition_size="200MB")` before the final write.
- Use `s3fs` with credentials from env/instance role (don't hardcode keys). If on recent boto3/s3fs and seeing latency, try `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` / `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` — newer SDK checksum defaults are a known throughput footgun at high parallelism.
- Keep dask workers and the bucket **in the same region**.
- Scope a dedicated IAM policy to the pipeline prefixes; consider versioning/Object Lock on the final prefix only.
- At this scale storage cost is negligible next to generation cost — don't over-engineer the tier; get Stage 5 validation right first.

---

## Final file set

| Stage | File | Runs on | Notes |
|---|---|---|---|
| 0–1 | `extract_tasks.py` | dask | introspects real schema, splits task vs. trajectory columns |
| 2 | `plan_variants.py` | pandas | diversity via `thinking_level` + framing (not sampling params) |
| 3 | `gemini_agent_worker.py` | async API | real multi-turn agent loop on Gemini 3.5 Flash; streams trajectories as they finish |
| 4 | `poll_n_fetch.py` | — | obsolete on the Gemini path; worker is itself resume-safe |
| 5 | `validate_n_dedup.py` | dask | parse/structural checks, near-dup filter, optional verifier replay |
| 6 | `augment_150k_rows.py` | dask | reshapes to original schema, concatenates, writes 250K parquet to S3 |

---

## Pre-flight checklist before a full run

- [ ] Run `extract_tasks.py` standalone first and read the printed schema — column lists are introspected at runtime but confirm the task/trajectory split looks correct.
- [ ] Run a pilot on 500–1,000 tasks through stages 2–5 to calibrate the Stage 5 rejection rate and tune `OVERSAMPLE_FACTOR` accordingly.
- [ ] Confirm `CONCURRENCY` in `gemini_agent_worker.py` against your actual RPM ceiling — start conservative and ramp up.
- [ ] Verify your S3 bucket and dask workers are in the same AWS region.
- [ ] Set `CEREBRAS_API_KEY` and `GEMINI_API_KEY` in `.env` (see `.env.example`).
