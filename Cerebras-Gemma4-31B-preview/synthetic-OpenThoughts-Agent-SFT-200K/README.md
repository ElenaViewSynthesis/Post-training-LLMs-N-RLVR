# OpenThoughts-Agent-SFT 100K → 250K augmentation pipeline

## What this does

Takes the 100K-trace SFT dataset, treats each row as a **task** (instruction +
environment) decoupled from its **rollout** (the trajectory), and generates
~1.5x oversampled new rollouts for the same tasks via a real multi-turn agent
loop using **Gemma-4-31B Early Preview** served through **Cerebras Inference**,
with diversity from temperature variation + instruction framing. Validates,
dedups, and merges back with the original 100K to land at 250K total rows.

This is a "more rollouts of the same tasks" augmentation, not "new tasks" —
deliberate, since you want to scale trace count without shifting the task
distribution your model learns from.

## Teacher: Gemma-4-31B via Cerebras Inference

The pipeline uses **Gemma-4-31B Early Preview** served through the **Cerebras
Inference API** (`cerebras-cloud-sdk`, OpenAI-compatible). Cerebras hardware
delivers extremely high token throughput, making it well-suited for the
volume of multi-turn rollouts needed here.

Key design points:

1. **Temperature-based diversity.** Unlike Gemini 3.x, Gemma-4-31B on Cerebras
   has no restriction on sampling parameters. Temperature is a legitimate and
   effective diversity axis: low values (0.7) produce focused, deterministic
   rollouts; high values (1.0) produce more exploratory ones. Stage 2 plans a
   weighted distribution across [0.7, 0.85, 1.0]. See `plan_variants.py`.

2. **Manual conversation history.** Cerebras has no server-side conversation
   state, so the full message history is rebuilt and sent on every turn. At
   Cerebras throughput speeds this is not a bottleneck — the ceiling is sandbox
   I/O and RPM limits, not token generation.

3. **Function calling via OpenAI-compatible tools API.** `run_bash` and
   `edit_file` are defined as standard function-calling tools. The model issues
   tool calls, receives observations, and continues until completion or
   `MAX_TURNS` is reached. Wire `execute_tool()` in `gemini_agent_worker.py`
   to your real Docker sandbox.

**Throughput ceiling is `CONCURRENCY` + your sandbox capacity**, not token
generation speed. Start at 32 concurrent workers, watch for 429s, ramp up.
Set `CEREBRAS_API_KEY` in your `.env` (see `.env.example`).

## Pipeline stages

| Stage | File | Runs on | Notes |
|---|---|---|---|
| 0-1 | `01_extract_tasks.py` | dask | introspects real schema, splits task vs. trajectory columns |
| 2 | `02_plan_variants.py` | pandas | builds the diversity plan via temperature + framing |
| 3 | `gemini_agent_worker.py` | async API | real multi-turn agent loop via Cerebras Inference + Gemma-4-31B; streams trajectories as they finish |
| 4 | `poll_n_fetch.py` | — | obsolete on the Cerebras path (no batch to poll); worker is itself resume-safe |
| 5 | `05_validate_and_dedup.py` | dask | parse/structural checks, near-dup filter, (optional) verifier replay |
| 6 | `06_merge_and_write.py` | dask | reshapes to original schema, concatenates, writes 250K parquet to S3 |

Run in order. Stage 4 is poll-and-resume safe — re-run it until it reports
all batches retrieved.

## Before running at full scale: pilot first

Run stages 2-5 on a 500-1,000 task sample before committing the full 150K
request budget. You need this to calibrate:
- **Rejection rate** out of Stage 5 → tunes `OVERSAMPLE_FACTOR` in stage 2.
- **Degenerate-output rate** (model refusing to act as a terminal agent) →
  tunes the regex filter in `validate_n_dedup.py`.
- **Optimal temperature** — calibrate which temperature setting in [0.7, 0.85, 1.0]
  gives the best Stage 5 acceptance rate and tighten the weights in `plan_variants.py`
  before the full run.

## Why the real agent loop (and not a batch / simulated approach)

A batch API fundamentally can't run a multi-turn tool loop — each batched
request is one self-contained call with no environment in between — so batch
always implies simulated trajectories. The Cerebras async worker runs the real
loop: actually-executed tool calls and observations, matching how the source
dataset was generated via the Terminus-2 harness.

If you later want a self-hosted teacher (vLLM/SGLang infra), the loop in
`gemini_agent_worker.py` ports cleanly: the Cerebras client is already
OpenAI-compatible, so swapping to a local vLLM endpoint requires changing
only the `base_url` and `api_key`.

---

## S3 storage recommendations

For a dataset of this size and access pattern (write once per pipeline run,
read repeatedly by dask/training jobs, occasional reprocessing), here's what
I'd actually set up:

### Bucket structure
```
s3://your-org-sft-data/
  openthoughts-agent-sft-100k-raw/        # immutable copy of the original, pinned
  openthoughts-agent-sft-pipeline/
    tasks/                                 # Stage 1 output
    variant_plan/                          # Stage 2 output
    raw_results/                           # Stage 4 output (batch JSONL)
    validated/                             # Stage 5 output
  openthoughts-agent-sft-250k/             # Stage 6 final output, what training actually reads
```

Keep the **raw 100K pinned in your own bucket** rather than re-pulling from
`hf://` every run — HF repo content can update upstream, and you want your
250K to be reproducible against a fixed base. `aws s3 sync` or a single
`dd.read_parquet(...).to_parquet("s3://...")` once, then point everything
downstream at your own copy.

### Storage class
- **Intermediate stages** (`tasks/`, `variant_plan/`, `raw_results/`,
  `validated/`): **S3 Standard**. These get read/written repeatedly during
  pipeline iteration and re-runs; don't fight lifecycle transitions while
  you're still tuning Stage 5's rejection thresholds.
- **Final `250k/` dataset**: S3 Standard while actively training against
  it; transition to **S3 Standard-IA** via a lifecycle rule once a training
  run is checkpointed and stable, since you'll read it in full occasionally
  but not continuously.
- Skip Glacier tiers for anything in the active pipeline — retrieval latency
  will stall a dask job that hits a cold object.

### Performance for dask + parquet specifically
- **Partition file count matters more than bucket settings.** Aim for
  parquet files in the 100-500MB range per partition — too many small files
  (default dask chunking from a pandas-heavy stage can produce thousands of
  tiny files) tanks S3 list/get throughput. Use `repartition(partition_size="200MB")`
  before the final `to_parquet` in Stage 6 if your partition count looks
  too high after the merge.
- Use `s3fs` (dask's default S3 backend) with `anon=False` and let it pick up
  credentials from environment/instance role — avoid hardcoding keys in
  `storage_options`.
- Set `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` and
  `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` env vars if you're on a
  recent boto3/s3fs version and see unexpected latency — newer SDK defaults
  added checksums that cost throughput on highly parallel small-object
  workloads; this is a known footgun with dask+s3fs at scale.
- If your dask workers and S3 bucket aren't in the same AWS region, fix that
  first — cross-region S3 reads are the single biggest avoidable latency
  cost in pipelines like this.

### Access / IAM
- Scope a dedicated IAM role/policy to exactly these prefixes
  (`openthoughts-agent-sft-*`) rather than reusing a broad data-lake role —
  this is throwaway/regenerable training data, not something that needs
  org-wide read access by default.
- If multiple people on the team will re-run stages, consider S3 Object
  Lock or simple versioning on the `250k/` final prefix only, so a bad
  Stage 6 re-run doesn't silently overwrite the dataset a training job is
  mid-read on.

### Cost note
At this scale (100K→250K rows of agentic traces, likely tens of GB total),
S3 storage cost is negligible next to the Batches API generation cost —
don't over-engineer the storage tier; get Stage 5's validation right first,
since that's what determines whether you actually have a useful 250K
dataset versus 250K rows of mediocre trajectories.
