# Dataset augmentation: next steps and resume runbook

> **Architecture update:** The 225,000-candidate oversampling workflow described
> below has been superseded by `stream_refinement_worker.py`. The current path
> assigns exactly 150,000 deterministic source-row refinement slots, retries a
> rejected result for its original slot, and stores only the first accepted
> result. See `README.md` for current commands. The historical details below are
> retained as a record of the earlier design and must not be used to start a new
> paid run.

This document is the handoff point for a fresh Codex session. It records the
authoritative pipeline state, the remaining implementation work, the required
tests, and the safe order for resuming paid Gemini generation.

## Repository and operating rules

- Repository: `https://github.com/ElenaViewSynthesis/Post-training-LLMs-N-RLVR`
- Project directory:
  `/mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation`
- Work under WSL. Native Windows locking is not required unless it becomes an
  explicit future requirement.
- Commit and push only to remote `main`. Do not create a feature branch or pull
  request unless the user changes this instruction.
- Commit every changed file separately using a Conventional Commit subject and
  a detailed, multi-paragraph body that explains the change, rationale,
  validation, and reproducibility commands.
- Stage files by explicit path. Never use `git add -A` or include unrelated
  changes from sibling projects.
- Keep `.env` ignored. Never print, stage, or commit API keys or tokens.
- The Windows-mounted data directory is authoritative:
  `/mnt/c/Users/proxi/pipeline/data`.
- Synchronization before status is one-way from the authoritative local
  `raw_results` directory to the configured Hugging Face bucket. Never download
  over the authoritative local JSONL.

Known unrelated sibling-project modifications must remain untouched:

- `../../EmbeddingGemma300M/.gitignore`
- `../../Gemma-4-12B-it/.gitignore`
- `../../Gemma-4-12B-it/.python-version`
- `../../Gemma-4-12B-it/uv.lock`
- `../../Qwen3Embedding4B/.gitignore`

## Authoritative baseline

The following state was verified after synchronizing local raw results to the
Hugging Face bucket:

| Item | Verified value |
|---|---:|
| Planned variants | 225,000 |
| Unique variant IDs | 225,000 |
| Source tasks | 82,999 |
| Existing JSONL records | 511 |
| Unique successful variants | 169 |
| Remaining successful variants required | 224,831 |
| Malformed JSONL lines | 0 |
| Existing parse-failure attempts | 234 |
| Existing Cerebras quota-failure attempts | 107 |
| Existing invalid-Gemini-key attempts | 1 |

The local and remote `trajectories.jsonl` files were subsequently reported as
identical by a Hugging Face Bucket dry run, with zero pending writes or deletes.

Completion is defined by **225,000 unique variant IDs with at least one
`status: "ok"` record**. The JSONL will contain more than 225,000 physical lines
because failed attempts remain in the append-only history and successful
retries add new records.

The last known Gemini key is invalid. Do not start a paid generation run until
the key is replaced and the one-request gate below succeeds.

## Fresh-session bootstrap

Start every fresh Codex session by establishing repository and data state:

```bash
cd /mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation
git switch main
git fetch origin
git status --short --branch
git rev-list --left-right --count main...origin/main
gh auth status
```

The left/right count should be `0 0`. Do not merge or pull automatically if it
is not; inspect the divergence and the working tree first. The known sibling
changes listed above may still appear in `git status` and must be ignored.

Confirm the authoritative resume count without contacting Gemini:

```bash
uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --status-only
```

Expected baseline:

```text
Plan: 225,000 | successful: 169 | remaining: 224,831 | existing records: 511 | malformed lines: 0
```

## Implementation order

Complete the following work before beginning the full paid Gemini run. Each
workstream should be implemented, tested, committed file-by-file, and pushed to
remote `main` before proceeding to the next one.

### 1. Add deterministic Stage 5 target selection and accounting

This is the highest-priority correctness issue. The current fixture accepts all
169 successful trajectories. If that acceptance rate continues, passing all
225,000 successful variants directly to Stage 6 would create approximately
325,000 total rows instead of the intended 250,000.

Implement in `validate_n_dedup.py`:

- Load or otherwise validate against `variant_plan.parquet`.
- Require every accepted `variant_id` to exist in the plan.
- Collapse repeated successful records to one result per `variant_id` using a
  documented deterministic rule.
- Record counts for malformed lines, failed attempts, invalid schemas,
  structural failures, exact duplicates, near duplicates, unknown variant IDs,
  accepted candidates, and finally selected rows.
- Add `--target-rows`, defaulting to `150000`, and a deterministic `--seed`.
- Select exactly 150,000 accepted trajectories when at least that many survive.
- Preserve task balance. A suitable deterministic policy is stable per-task
  ranking followed by round-robin selection across tasks until the target is
  reached.
- Fail clearly without publishing output when fewer than the requested target
  survive.
- Write a `validation_manifest.json` beside the validated Parquet output. It
  should contain input paths or fingerprints, thresholds, seed, counts,
  selected-row count, timestamp, and source Git commit.
- Keep provider-specific turn normalization deterministic and retain the stable
  `{role, content}` Parquet schema.

Acceptance criteria:

- More than 150,000 valid candidates always yields exactly 150,000 selected
  rows.
- Identical input, seed, and thresholds produce identical selected variant IDs.
- Task allocation differs by at most one row where candidate availability
  permits balanced allocation.
- Fewer than 150,000 valid candidates exits nonzero and does not replace the
  last valid output.

Required unit tests:

- Over-target deterministic selection.
- Exactly-target selection.
- Under-target failure.
- Balanced allocation across uneven task groups.
- Determinism across input ordering and Dask partitions.
- Duplicate successful records for the same variant.
- Failure followed by success for the same variant.
- Unknown variant IDs.
- Malformed JSON, invalid roles, missing conversations, and empty turns.
- Provider-specific tool-call normalization.
- Manifest counts and reproducibility metadata.

### 2. Make Stage 5 safe at 225,000-row scale

The current implementation calls `parsed.compute()` and constructs one pandas
DataFrame containing every accepted nested conversation. That can consume
several gigabytes when the raw-results file reaches full size.

Implement a bounded-memory, two-pass pipeline:

1. Stream or partition raw JSONL input, normalize and structurally validate
   records, and write candidate Parquet shards plus a compact candidate index.
2. Deduplicate and perform deterministic balanced selection using the compact
   index.
3. Materialize only selected conversations into the final validated dataset.
4. Write to a temporary run directory and promote the completed result only
   after all invariants pass.

Required scale tests:

- Generate non-production fixtures with 10,000, 100,000, and 225,000 records.
- Record wall time, peak RSS, input size, candidate size, and final Parquet size.
- Verify selected IDs and manifest counts across repeated runs.
- Verify the final Parquet can be read in full and by individual row groups.
- Verify interrupted or failed validation never replaces the previous valid
  output.

Keep large benchmark fixtures under `/tmp`; never commit them.

### 3. Make Stage 6 path-configurable and safe to dry-run

`augment_150k_rows.py` still hard-codes `~/pipeline/data`, which is not the
authoritative WSL location.

Implement:

- `PIPELINE_DATA_DIR` and `--data-dir` support consistent with Stages 3 and 5.
- Explicit `--original-source`, `--validated`, `--tasks`, `--upload-dir`, and
  `--hf-bucket` overrides.
- `--dry-run` for validation and planning without writing final shards.
- `--no-sync` for writing and inspecting local shards without remote mutation.
- `--expected-new-rows`, defaulting to `150000`.
- A hard assertion that the final row count equals the original row count plus
  the expected synthetic count.
- Hard failures for missing task lookups, duplicate synthetic `run_id` values,
  null required fields, invalid conversations, or schema mismatches.
- Schema comparison against a real sample of the original dataset.
- Output staging in a new temporary directory so stale local shards cannot be
  included accidentally.
- An explicit remote-stale-file policy. `sync_bucket` does not delete remote
  extras by default; prefer a versioned destination prefix or require a
  separately confirmed delete option rather than silently deleting data.
- A final dataset manifest with counts, schema, input manifests, shard names,
  file sizes, and checksums.

Required Stage 6 tests:

- Argument and environment path precedence.
- `--dry-run` performs no local or remote writes.
- `--no-sync` never calls `sync_bucket`.
- Mocked synchronization receives the exact staged directory and destination.
- Missing task IDs fail instead of producing mostly-null rows.
- Duplicate IDs fail.
- Conversation schema matches the original dataset schema.
- Final row count is exact.
- Every generated shard is readable and shard names are deterministic.
- Stale files in a prior output directory cannot leak into the new run.

### 4. Strengthen Stage 5 quality gates

The current quality gate checks turn count, overall text length, refusal
patterns, and exact normalized-content hashes. The documentation currently
describes stronger behavior than the implementation provides.

Implement and calibrate on a pilot before enforcing globally:

- Require a non-empty final assistant response.
- Validate plausible tool-call and tool-result ordering.
- Detect repeated-turn loops.
- Add per-task near-duplicate detection instead of relying only on exact hashes.
  Because each task has only a few variants, pairwise normalized token or
  n-gram similarity is feasible without an all-pairs global comparison.
- Report acceptance and rejection rates by provider, model, task group, and
  failure reason where metadata is available.
- Manually inspect a deterministic sample of accepted and rejected records.
- Keep verifier replay as a later, separately sandboxed workstream. It is still
  a TODO and must not be described as implemented until it exists.

Required quality tests:

- Missing or non-assistant final turn.
- Empty final response.
- Tool result without a preceding tool call.
- Tool call without a following result when a result is required.
- Repeated assistant/tool loops.
- Exact duplicates and threshold-boundary near duplicates.
- Deterministic manual-review sample selection.

### 5. Expand Gemini worker resilience tests

Add mocked tests around `gemini_trajectory_worker.py` without making paid API
calls:

- Existing failure followed by a successful retry.
- Duplicate successful records.
- Malformed and truncated existing JSONL lines.
- Valid structured Gemini response parsing.
- Empty, invalid, and over-limit conversations.
- HTTP 400, 401, 403, and 404 fatal circuit-breaking.
- HTTP 408, 429, and 5xx retry/exhaustion behavior.
- Fully failed batch shutdown.
- Partial batch success.
- Periodic and final Hugging Face synchronization.
- Nonfatal synchronization failure during generation.
- Fatal synchronization failure before status.
- Two workers contending for the same output lock.
- Interruption followed by a resume scan with no lost successful IDs.
- Status-only mode never imports or initializes the Gemini client.

## Test ladder

### Fast tests after every implementation change

```bash
UV_CACHE_DIR=/tmp/dataset-augmentation-uv-cache \
  uv run --offline --no-project --with 'dask[dataframe]' \
  --with pandas --with pyarrow --with python-dotenv \
  python -m unittest -v test_pipeline_path_configuration.py

UV_CACHE_DIR=/tmp/dataset-augmentation-uv-cache \
  uv run --offline --no-project --with ruff \
  ruff check gemini_trajectory_worker.py validate_n_dedup.py \
  test_pipeline_path_configuration.py augment_150k_rows.py

UV_CACHE_DIR=/tmp/dataset-augmentation-uv-cache uv lock --offline --check
bash -n install.sh
```

Add new focused test modules as the Stage 5 and Stage 6 work grows rather than
placing every test in one file.

### Real authoritative-data Stage 5 regression

Always write regression output to `/tmp` until final validation is approved:

```bash
uv run python validate_n_dedup.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --output /tmp/validated_trajectories.parquet
```

The current small-data baseline is 169 accepted trajectories, zero exact
duplicates, and a uniform conversation schema containing only `system`, `user`,
`assistant`, and `tool` roles with exactly `role` and `content` fields.

### End-to-end local integration fixture

Create a temporary fixture containing a small task parquet, variant plan, and
mixed success/failure JSONL. Exercise:

1. Resume-status calculation.
2. Stage 5 normalization, rejection accounting, deduplication, and selection.
3. Stage 6 reshape and manifest generation with `--no-sync`.
4. Parquet reload and schema assertions.

Mock Gemini and Hugging Face calls. The test must not require provider keys or
network access.

## Valid-key gate and paid pilot

Only begin this section after replacing `GEMINI_API_KEY` in the ignored `.env`.
Do not print the key while checking configuration.

### 1. Reconfirm and synchronize status without Gemini

```bash
uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --status-only \
  --sync-to-hf-before-status
```

The command must complete synchronization before printing counts. It must not
make a Gemini request.

### 2. Make exactly one live request without synchronization

```bash
uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --limit 1 \
  --no-sync
```

Inspect only the newly appended record. Confirm:

- `status` is `ok`.
- `variant_id` belongs to the plan and was previously unfinished.
- The configured model is recorded.
- `conversations` contains 2-40 turns.
- Every turn has an allowed role and string content.
- The final assistant response is complete and non-empty.
- Tool calls and observations are plausible.
- The output is not visibly truncated.

If successful from the current baseline, status should become 170 successful
and 224,830 remaining. Run Stage 5 against a temporary output and ensure the
new record survives before continuing.

### 3. Run a controlled 10-50 request pilot

Begin conservatively:

```bash
GEMINI_CONCURRENCY=1 GEMINI_BATCH_SIZE=10 \
  uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --limit 10 \
  --no-sync
```

Measure and record:

- API success and error rates.
- Authentication, quota, retry, and timeout behavior.
- Output truncation and structured-parse failure rates.
- Stage 5 acceptance and rejection reasons.
- Average output length, latency, and token usage when available.
- Qualitative tool-call realism from a deterministic review sample.

Increase to 50 only after the 10-request run is clean. Adjust concurrency,
batch size, output token limit, or validation thresholds only from observed
pilot evidence.

After inspecting the pilot, synchronize and recount:

```bash
uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --status-only \
  --sync-to-hf-before-status
```

### 4. Resume the full plan

Run without `--limit` only after all gates pass:

```bash
uv run python gemini_trajectory_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data
```

The worker may stop on quota exhaustion or a fully failed batch. Preserve the
append-only JSONL, diagnose the stop reason, and rerun when the provider is
available. Never delete failed records; resume logic ignores them and skips
only variant IDs that already have a successful record.

Periodically run status-only mode. The generation phase is complete only when
the status reports:

```text
Plan: 225,000 | successful: 225,000 | remaining: 0
```

Perform one final authoritative local-to-HF synchronization before final Stage
5 validation.

## Final validation and publication

1. Run Stage 5 against all raw results using the approved deterministic
   thresholds and target selection.
2. Confirm the manifest reports exactly 150,000 selected synthetic rows.
3. Run Stage 6 with `--dry-run --no-sync` after those options are implemented.
4. Confirm exactly 250,000 total rows, unique synthetic IDs, compatible schema,
   readable shards, manifest checksums, and no missing task mappings.
5. Inspect a deterministic sample of original and synthetic rows.
6. Synchronize the versioned final output to the Hugging Face bucket only after
   explicit approval.
7. Run a post-sync dry run and require zero pending writes or deletes.

## Definition of done

- 225,000 planned variant IDs have at least one successful trajectory.
- Raw local results and the raw-results HF bucket prefix are identical.
- Stage 5 produces exactly 150,000 deterministically selected, validated new
  trajectories and a complete reproducibility manifest.
- Stage 5 completes within measured memory and runtime limits at full scale.
- Stage 6 produces exactly 250,000 total rows with the original-compatible
  schema, unique identifiers, readable shards, and a complete dataset manifest.
- The final versioned bucket prefix matches local output with no pending sync
  operations.
- Tests, commands, thresholds, counts, source commit, and known limitations are
  documented.
- Every source change has its own detailed commit and is pushed directly to
  remote `main` without including unrelated sibling modifications or `.env`.

## Suggested prompt for a fresh Codex session

Copy the following into a new session:

> Read `NEXT_STEPS.md` completely and inspect the current code and Git state.
> Continue from the first unfinished implementation step. Work only under WSL,
> treat `/mnt/c/Users/proxi/pipeline/data` as authoritative, never expose `.env`
> values, and do not make a Gemini request until the valid-key one-request gate.
> Commit and push only to remote `main`. Commit every changed file separately
> with a Conventional Commit subject and a detailed multi-paragraph body covering
> rationale, behavior, tests, and reproducibility. Do not stage the known sibling
> project modifications.
