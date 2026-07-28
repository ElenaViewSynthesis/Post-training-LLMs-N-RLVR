# Dataset augmentation resume runbook

This is the operational handoff for resuming the current source-row refinement
pipeline in a fresh terminal or Codex session.

The workflow is intentionally documented rather than packaged as one Bash
script. It contains paid provider calls, manual output inspection, production
generation, and remote publication gates that must remain separate operator
decisions.

## Verified implementation baseline

- Branch: `main`
- Implementation commit: `a653f5f0a6cee801ad5738818e510ff6def71152`
- Implementation subject: `docs(augmentation): archive oversampling runbook`
- Local suite: 89 tests, 87 passed, 2 intentionally skipped live Gemini tests
- Production Gemini requests executed after this implementation: none
- Production immutable source snapshot downloaded: no

The implementation through this baseline includes:

- Gemini credential and model preflight before source hashing.
- Exact concurrency-safe `--max-provider-requests` accounting.
- Immediate retry termination for fatal HTTP `400`, `401`, `403`, and `404`.
- A deterministic, isolated `refinement_pilot.py` workflow.
- Immutable local source snapshots that retain canonical remote file paths.
- Refinement and publication remote checksum read-back.
- A fresh-directory refinement restore drill.
- Retired legacy paid-generation entry points.
- Opt-in live Gemini integration tests.

## Files outside this workstream

Do not stage, delete, or modify the pre-existing untracked file:

```text
dataset-augmentation/gh
```

Preserve unrelated sibling-project changes under:

```text
../../EmbeddingGemma300M/
../../Gemma-4-12B-it/
../../Qwen3Embedding4B/
```

## Fresh-terminal bootstrap

```bash
cd /mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation

git switch main
git fetch origin main
git status -sb
git rev-list --left-right --count main...origin/main
git log -1 --oneline
git merge-base --is-ancestor a653f5f0a6cee801ad5738818e510ff6def71152 HEAD
gh auth status
```

The expected branch relationship is:

```text
0  0
```

The latest commit should describe this resume runbook, and the ancestor check
above should exit successfully. The implementation baseline remains:

```text
a653f5f docs(augmentation): archive oversampling runbook
```

Do not automatically pull, merge, reset, or rebase if the branch has diverged.
Inspect the local and remote histories first.

## Local regression suite

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache \
uv run python -m unittest discover -q
```

Expected result:

```text
Ran 89 tests
OK (skipped=2)
```

The skipped tests are the deliberately gated live Gemini checks.

## Paid verification and production sequence

Do not combine the following stages into one unattended command. Stop and
inspect the stated output at every gate.

### 1. Validate Gemini access without generation

Run this only after the Gemini key has been funded and placed in the configured
environment file:

```bash
uv run python stream_refinement_worker.py \
  --provider-preflight-only
```

This performs a model-access lookup. It does not generate a conversation and
does not scan or hash the source dataset.

### 2. Materialize the immutable local source

```bash
uv run python materialize_source_snapshot.py \
  --source 'hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet' \
  --snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot
```

Verify the sealed snapshot without downloading again:

```bash
uv run python materialize_source_snapshot.py \
  --snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --verify-only
```

The snapshot is promoted only after every Parquet checksum, its inventory, and
its completion marker pass. Canonical remote paths remain part of source-row
identity even though subsequent reads are local.

### 3. Make exactly one real provider request

Use a dedicated pilot directory that has never been used for another request
budget:

```bash
uv run python refinement_pilot.py \
  --source /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --sample-size 10 \
  --max-provider-requests 1 \
  --pilot-dir /mnt/c/Users/proxi/pipeline/data/pilots/one-request \
  --execute
```

Manually inspect:

```text
/mnt/c/Users/proxi/pipeline/data/pilots/one-request/pilot_report.json
```

Check the source/refined pair for task preservation, useful detail, valid turn
ordering, non-refusal completion, and retained paths, identifiers, commands,
URLs, quoted strings, and numbers.

### 4. Make exactly ten requests

Use a fresh pilot directory so the report describes exactly this invocation:

```bash
uv run python refinement_pilot.py \
  --source /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --sample-size 10 \
  --max-provider-requests 10 \
  --pilot-dir /mnt/c/Users/proxi/pipeline/data/pilots/ten-request \
  --execute
```

Review:

- Acceptance rate.
- Rejection codes and representative rejected output.
- Provider-error counts and HTTP codes.
- Duplicate-conversation collisions.
- Attempts per slot.
- Every reported source/refined sample pair.

### 5. Expand to 25–50 requests conditionally

Proceed only if the ten-request pilot has acceptable quality and no unexplained
provider or duplication behavior. Use another fresh directory and set both
`--sample-size` and `--max-provider-requests` to the chosen value.

Example for 25 requests:

```bash
uv run python refinement_pilot.py \
  --source /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --sample-size 25 \
  --max-provider-requests 25 \
  --pilot-dir /mnt/c/Users/proxi/pipeline/data/pilots/twenty-five-request \
  --execute
```

### 6. Repeat bounded-memory verification

Before the full paid run, inspect the available benchmark options and repeat the
10K, 100K, and 225K synthetic fixtures with peak-memory measurement:

```bash
uv run python benchmark_streaming_pipeline.py --help
```

Keep generated fixtures under `/tmp`; never commit them.

### 7. Start the isolated 150K production run

Proceed only after the pilot and scale gates pass:

```bash
uv run python stream_refinement_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --original-source 'hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet' \
  --source-snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --target-rows 150000
```

Do not set `--max-provider-requests` for the intended full run. The worker owns
one run-specific output directory and synchronizes only to:

```text
<hf-refinement-bucket>/runs/<run-instance-id>/
```

Check local status without initializing Gemini:

```bash
uv run python stream_refinement_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --status-only
```

Retry synchronization without source scanning or Gemini initialization:

```bash
uv run python stream_refinement_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --sync-only
```

### 8. Complete Stage 6 publication

Run the write-free preflight:

```bash
uv run python augment_150k_rows.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --dry-run \
  --no-sync
```

Create and inspect the local publication:

```bash
uv run python augment_150k_rows.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --no-sync
```

Publish only after local inspection:

```bash
uv run python augment_150k_rows.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data
```

Remote publication success includes automatic read-back of the manifest,
completion marker, exact file set, sizes, and Parquet SHA-256 checksums.

### 9. Perform the refinement restore drill

Read the run-instance UUID from:

```text
/mnt/c/Users/proxi/pipeline/data/refined/run_manifest.json
```

Then restore the corresponding remote prefix into a new directory:

```bash
uv run python restore_refinement_run.py \
  --remote-run 'hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/refined/runs/<run-instance-id>' \
  --output-dir '/mnt/c/Users/proxi/pipeline/data/restore-drill-<run-instance-id>'
```

The destination must not already exist.

## Opt-in live integration tests

Provider preflight only:

```bash
RUN_GEMINI_INTEGRATION=1 \
UV_CACHE_DIR=/tmp/codex-uv-cache \
uv run python -m unittest -v test_gemini_live.GeminiLiveTests.test_provider_preflight
```

Exactly one paid test request:

```bash
RUN_GEMINI_INTEGRATION=1 \
RUN_GEMINI_PAID_TESTS=1 \
UV_CACHE_DIR=/tmp/codex-uv-cache \
uv run python -m unittest -v test_gemini_live.GeminiLiveTests.test_exactly_one_real_refinement_request
```

The supported pilot remains the preferred real-source quality check because it
also emits an inspection report.

## Fresh Codex session handoff

Paste the following into a new session:

```text
Resume dataset-augmentation from the current origin/main. Confirm that commit
a653f5f0a6cee801ad5738818e510ff6def71152 is an ancestor; it is the verified
implementation baseline before this runbook was added.

Repository:
/mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation

Read RESUME_RUNBOOK.md and README.md before acting. Verify main against
origin/main and do not redo the completed implementation.

Completed: Gemini preflight, exact provider request budgets, isolated pilot
harness, immutable local source snapshots, remote refinement/publication
checksum verification, restore drill, retired legacy entry points, and local
tests. Baseline suite: 89 tests, 87 passed, 2 intentionally skipped live tests.

No live Gemini request or production source download was executed after the
implementation. The next gate is provider preflight after the key is funded,
then source snapshot materialization, exactly one request, manual inspection,
exactly ten requests, and conditional expansion to 25-50 before production.

Preserve the untracked dataset-augmentation/gh file and unrelated sibling
changes. Do not stage them.
```
