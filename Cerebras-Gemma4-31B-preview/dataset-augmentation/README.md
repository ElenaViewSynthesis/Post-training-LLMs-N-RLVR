# OpenThoughts-Agent-SFT 100K → 200K augmentation pipeline

## Objective

Expand **OpenThoughts-Agent-SFT-100K** to **200K traces** by augmenting the
`conversations` column using **Gemma-4-31B Early Preview** on **Cerebras
Inference**. The task distribution is preserved — the same 100K tasks get new
synthetic conversation trajectories, not new tasks.

## What this does

Takes the 100K-trace SFT dataset, treats each row as a **task** (instruction +
environment) decoupled from its **conversations** (the trajectory), generates
~1.5× oversampled new synthetic conversations for the same tasks, validates and
dedups, then merges back with the original 100K to land at 200K total rows.

## Architecture: structured trajectory synthesis, not agent execution

The original dataset was built with a live agent runtime (Terminus-2 harness):
a model executed real bash commands in a sandboxed environment. This pipeline
uses a different and more scalable approach — **structured trajectory synthesis**:

> Gemma-4-31B generates a complete synthetic conversation in a single inference
> call, conditioned on the original task and augmentation strategy. The model
> produces all turns — reasoning, tool calls, observations, and the final answer —
> without any live sandbox environment.

**Why this is correct for large-scale SFT augmentation:**
- Agent execution requires container orchestration and is 10–100× slower per sample.
- Trajectory synthesis runs at Cerebras throughput speeds, making 100K+ new samples practical.
- For SFT, the student model learns from the conversation pattern — well-structured
  synthetic trajectories are sufficient signal.
- Quality is enforced post-generation by the validation pipeline (Stage 5): schema
  validation, semantic similarity filtering, safety checks, and diversity scoring.

This demonstrates the distinction between **agent execution** (live environment,
real tool calls) and **trajectory generation** (structured synthesis conditioned
on task context) — the latter being the appropriate choice here.

## Teacher: Gemma-4-31B via Cerebras Inference

The pipeline uses **Gemma-4-31B Early Preview** via the **Cerebras Inference
API** (`cerebras-cloud-sdk`, OpenAI-compatible). Cerebras hardware delivers
extremely high token throughput, making it well-suited for synthesizing large
volumes of multi-turn conversations.

Key design points:

1. **Single inference call per variant.** One request generates a complete
   `conversations` list (all turns) as structured JSON. No multi-turn loop,
   no sandbox — just high-throughput structured generation.

2. **Temperature-based diversity.** Gemma-4-31B on Cerebras has no restriction
   on sampling parameters. Temperature varies across [0.7, 0.85, 1.0] with a
   weighted distribution biased toward 0.85. See `plan_variants.py`.

3. **Engineered synthesis prompt.** The system prompt instructs the model to
   produce realistic step-by-step reasoning before tool calls, plausible bash
   command sequences, realistic observations, and a conclusive final answer —
   matching the style of the original dataset.

**Throughput ceiling is `CONCURRENCY` + Cerebras RPM limits**, not token
generation speed. Start at 32 concurrent workers, watch for 429s, ramp up.
Set `CEREBRAS_API_KEY` in your `.env` (see `.env.example`).

## Setup

### 1. Install pip (WSL / Ubuntu — if not already present)

```bash
sudo apt update && sudo apt install -y python3-pip
```

### 2. Install pipeline dependencies

```bash
cd cerebras-gemma4-31b-preview/synthetic-openthoughts-agent-sft-200k
bash install.sh
```

`install.sh` creates a `.venv` virtual environment, installs all dependencies
into it, enables `HF_HUB_ENABLE_HF_TRANSFER=1`, and creates the local pipeline
data directories under `~/pipeline/data/`.

**Activate the venv before every session:**
```bash
source .venv/bin/activate
```

### 3. Configure credentials

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
CEREBRAS_API_KEY=your_cerebras_api_key_here
HF_TOKEN=your_huggingface_read_token_here
HF_HUB_ENABLE_HF_TRANSFER=1
CEREBRAS_MODEL_ID=cerebras/Gemma4-31B-preview
```

### 4. Verify the Cerebras backend (optional smoke test)

`test_cerebras_smoke.py` runs a few lightweight checks — API key auth, the
async SDK client under concurrency, and one real task run through the full
synthesis + parsing path — without touching the rest of the pipeline:

```bash
uv run python test_cerebras_smoke.py
```

This directory has its own `pyproject.toml` + `uv.lock`, scoped to just this
stage's dependencies (`cerebras-cloud-sdk`, `dask`, `boto3`, `python-dotenv`,
etc.). `uv run` reads that scoped `pyproject.toml` and materializes an
isolated `.venv` right here — not the heavy training stack from the parent
`Cerebras-Gemma4-31B-preview/pyproject.toml` (torch, transformers, trl...),
which is the whole reason it's scoped separately.

---

## Pipeline stages

| Stage | File | Runs on | Notes |
|---|---|---|---|
| 0-1 | `extract_tasks.py` | dask | introspects real schema, splits task vs. trajectory columns |
| 2 | `plan_variants.py` | pandas | builds the diversity plan via temperature + framing |
| 3 | `gemma4_31b_agent.py` | async API | structured trajectory synthesis via Cerebras Inference + Gemma-4-31B; streams results as they finish |
| 4 | `poll_n_fetch.py` | — | obsolete on the Cerebras path (no batch to poll); worker is itself resume-safe |
| 5 | `validate_n_dedup.py` | dask | parse/structural checks, near-dup filter, (optional) verifier replay |
| 6 | `augment_150k_rows.py` | dask | reshapes to original schema, concatenates, writes 250K parquet to HF bucket |

Run in order. Stage 4 is poll-and-resume safe — re-run it until it reports
all batches retrieved.

## Before running at full scale: pilot first

Run stages 2-5 on a 500-1,000 task sample before committing the full 150K
request budget. You need this to calibrate:
- **Rejection rate** out of Stage 5 → tunes `OVERSAMPLE_FACTOR` in stage 2.
- **Degenerate-output rate** (model refusing to act as a terminal agent) →
  tunes the regex filter in `validate_n_dedup.py`.
- **Optimal temperature** — calibrate which temperature in [0.7, 0.85, 1.0] gives
  the best Stage 5 acceptance rate and tighten the weights in `plan_variants.py` before the full run.
- **Synthesis prompt quality** — inspect a sample of generated `conversations` from the pilot
  to verify the model is producing realistic tool-call/observation sequences, not hallucinated
  or refusal outputs.

## Why the real agent loop (and not a batch / simulated approach)

A batch API fundamentally can't run a multi-turn tool loop — each batched
request is one self-contained call with no environment in between — so batch
always implies simulated trajectories. The Cerebras async worker runs the real
loop: actually-executed tool calls and observations, matching how the source
dataset was generated via the Terminus-2 harness.

If you later want a self-hosted teacher (vLLM/SGLang infra), the loop in
`gemma4_31b_agent.py` ports cleanly: the Cerebras client is already
OpenAI-compatible, so swapping to a local vLLM endpoint requires changing
only the `base_url` and `api_key`.

---

## Output: HuggingFace Bucket via `hf sync` (GCP EU CDN)

The final 250K dataset is synced to a public HuggingFace Bucket using the
`hf` CLI. `hf_transfer` (enabled by `HF_HUB_ENABLE_HF_TRANSFER=1`) uses
HuggingFace's prewarmed GCP EU CDN for maximum throughput.

**Bucket:** [`borntobeignored/OpenThoughts-Agents-SFT-250k`](https://huggingface.co/buckets/borntobeignored/OpenThoughts-Agents-SFT-250k)
(CLI path: `hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k`)

### CLI commands

```bash
# Install hf CLI (done automatically by install.sh)
uv tool install hf

# Upload local parquet shards to bucket
hf sync ./data hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k

# Download bucket to local folder
hf sync hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k ./local
```

### What Stage 6 does

1. Merges original 100K rows + ~150K validated synthetic rows
2. Writes sharded parquet files to `~/pipeline/data/upload/`
3. Runs `hf sync` to push all shards to the HF bucket
4. Prints the download command on completion

### Local intermediate data layout

```
~/pipeline/data/
  tasks/                              # Stage 1 — task table (parquet)
  variant_plan.parquet                # Stage 2 — 225K variant plan
  raw_results/
    trajectories.jsonl                # Stage 3 — raw synthetic conversations
  validated/
    validated_trajectories.parquet    # Stage 5 — filtered ~150K rows
  upload/                             # Stage 6 — sharded parquet, ready to sync
```

### Token permissions

HF token must have **write** access to the `borntobeignored` org bucket.
Generate one at `huggingface.co/settings/tokens`.
