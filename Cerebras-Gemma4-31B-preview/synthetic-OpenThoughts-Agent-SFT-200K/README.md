# OpenThoughts-Agent-SFT 100K → 250K augmentation pipeline

## What this does

Takes the 100K-trace SFT dataset, treats each row as a **task** (instruction +
environment) decoupled from its **rollout** (the trajectory), and generates
~1.5x oversampled new rollouts for the same tasks via a real multi-turn agent
loop on Gemini 3.5 Flash, with diversity from `thinking_level` + instruction
framing. Validates, dedups, and merges back with the original 100K to land at
250K total rows.

This is a "more rollouts of the same tasks" augmentation, not "new tasks" —
deliberate, since you want to scale trace count without shifting the task
distribution your model learns from.

## Teacher: Gemini 3.5 Flash (unlimited requests)

Because request volume is free for you, the pipeline runs the **real multi-turn
sandboxed agent loop** rather than the single-shot *simulated* trajectories the
earlier Anthropic-Batches design used to save cost. Real loop = actually-executed
tool calls and observations, matching how the source dataset was generated.

Three Gemini-3.x-specific things baked into the design:

1. **No sampling-param perturbation.** Gemini 3.x is tuned for default
   temperature/top_p/top_k and explicitly recommends against changing them;
   doing so degrades reasoning. Rollout diversity instead comes from
   `thinking_level` (low/medium/high) + light instruction framing + inherent
   run-to-run nondeterminism. See Stage 2.

2. **Interactions API, not generateContent.** GA as of June 2026 and built for
   agent loops: `previous_interaction_id` keeps conversation state server-side
   (you don't resend history each turn), and `steps` gives you the full
   observable timeline to capture as your trajectory. Needs
   `google-genai >= 2.3.0`. The legacy `google-generativeai` package is
   deprecated — don't use it.

3. **Two execution backends** (choose in `03_gemini_agent_worker.py`):
   - `antigravity` — the managed Antigravity agent runs Bash/Python/Node in a
     Google-hosted sandbox. Zero container orchestration on your side. Best for
     the generic-Ubuntu / nl2bash-style tasks; weaker where a task depends on a
     specific Dockerfile you can't reproduce in Google's sandbox. Preview.
   - `custom_sandbox` — you spin each task's real Docker env, expose
     `run_bash`/`edit_file`/`run_tests` as function-calling tools, run the
     pytest verifier at the end. Most faithful (reproduces the original
     Terminus-2 method exactly); you own the sandbox lifecycle.

**"Unlimited requests" is not unlimited throughput.** Your real ceiling is now
RPM/concurrency and — for `custom_sandbox` — how many containers you can run at
once. `CONCURRENCY` in the worker is the tuning knob; start low, watch for 429s,
ramp gradually (Gemini penalizes sharp traffic spikes like most APIs).

A storage note specific to Gemini: interactions are **stored server-side by
default** (paid tier retains ~55 days). That's convenient for `previous_interaction_id`,
but if your task instructions are sensitive, set `store=false` — though that
disables `previous_interaction_id` and `background=true`, so you'd lose the
multi-turn state mechanism. For this open dataset it's a non-issue.

## Pipeline stages

| Stage | File | Runs on | Notes |
|---|---|---|---|
| 0-1 | `01_extract_tasks.py` | dask | introspects real schema, splits task vs. trajectory columns |
| 2 | `02_plan_variants.py` | pandas | builds the diversity plan via `thinking_level` + framing (NOT sampling params) |
| 3 | `03_gemini_agent_worker.py` | async API | real multi-turn agent loop on Gemini 3.5 Flash; streams trajectories as they finish |
| 4 | `04_OBSOLETE_*` | — | obsolete on the Gemini path (no batch to poll); worker is itself resume-safe |
| 5 | `05_validate_and_dedup.py` | dask | parse/structural checks, near-dup filter, (optional) verifier replay |
| 6 | `06_merge_and_write.py` | dask | reshapes to original schema, concatenates, writes 250K parquet to S3 |

Run in order. Stage 4 is poll-and-resume safe — re-run it until it reports
all batches retrieved.

## Before running at full scale: pilot first

Run stages 2-5 on a 500-1,000 task sample before committing the full 150K
request budget. You need this to calibrate:
- **Rejection rate** out of Stage 5 → tunes `OVERSAMPLE_FACTOR` in stage 2.
- **Degenerate-output rate** (model refusing to "pretend" to have a
  terminal) → tunes the regex filter or, better, swap to the true
  multi-turn approach noted in `03_submit_batches.py` if refusals are common
  for your task types.
- **Cost** — at Sonnet 4.6 batch pricing (50% off standard), ~200K requests
  with a few thousand input/output tokens each is a meaningful but
  budgetable spend; get the actual number from a pilot rather than
  estimating from list price alone, since real trajectories vary a lot in
  length.

## Why the real agent loop (and not a batch / simulated approach)

The earlier design used the Anthropic Batches API with single-shot *simulated*
trajectories purely to capture the 50% batch discount. Unlimited Gemini
requests removes that constraint, so the pipeline now runs the real loop. A
batch API (Anthropic's or Gemini's) fundamentally can't run a multi-turn tool
loop — each batched request is one self-contained call with no environment in
between — so batch always implies simulated trajectories. With cost off the
table, there's no reason to accept that fidelity hit.

If you later want a self-hosted teacher instead (you run vLLM/SGLang infra),
the loop in `03_gemini_agent_worker.py` ports cleanly: swap the Gemini client
for an OpenAI-compatible vLLM endpoint and keep the same function-calling
tool contract. Worth a $/trace and quality comparison if Gemini throughput
becomes the bottleneck.

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
