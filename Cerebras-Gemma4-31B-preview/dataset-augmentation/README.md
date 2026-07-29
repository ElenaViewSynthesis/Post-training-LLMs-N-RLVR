# OpenThoughts-Agent-SFT source-row refinement pipeline

## Objective

Preserve every row from `OpenThoughts-Agent-SFT-100K` and add exactly 150,000
refined conversations. The configured source currently contains 94,334
physical rows, so the expected publication is 244,334 rows. Stage 6 derives
that total from the source instead of trusting the dataset's `100K` label.

The current pipeline does not oversample and does not create multiple accepted
candidates for a refinement slot. It assigns:

- one refinement to every source row; and
- one additional refinement to a deterministic 55,666-row subset.

The subset is chosen by a stable hash of a physical source-row identity derived
from the sorted Parquet file path, row index, lineage fields, and a canonical
conversation fingerprint. Both `run_id` and `trial_name` repeat in the real
data, so neither is safe as a row key. They and the original task remain
lineage metadata. A rejected generation is retried for the same slot until a
result passes validation. The first accepted result completes that slot
permanently.

## Architecture

```text
source Parquet row groups
        │
        ├── deterministic refinement slots (150K exact)
        │
        ├── bounded-concurrency Gemini requests
        │       ├── reject → retry same slot
        │       └── accept → atomic Parquet shard
        │
        └── Stage 6 preflight
                ├── verify all source relationships and schemas
                ├── stream every original + 150K refined rows
                ├── verify the derived physical Parquet row count
                └── promote and optionally sync a versioned publication
```

The model refines an existing source conversation in a single structured-output
request. Tool calls and observations remain synthetic; this pipeline does not
execute them in a live sandbox.

## Safety properties

- API calls are concurrent but bounded by `--concurrency`.
- Gemini credentials and model visibility are checked before a new/incomplete
  run hashes or scans its source. `--max-provider-requests` is a concurrency-safe
  ceiling on generation calls, including rejected responses and retries; SDK
  transport retries are disabled so the ceiling remains exact.
- Only one worker can own a refinement output directory.
- Every paid request is assigned to a deterministic source row and slot.
- A remote source URI is resolved to a full immutable dataset commit before
  slot assignment. Source manifest v2 records the sorted Parquet paths, byte
  sizes, full per-file SHA-256 values, aggregate content digest, source-identity
  digest, and schema digest.
- Complete source-file hashes are checked on every resume before paid requests,
  during Stage 6 preflight, and again after originals are
  streamed but before publication promotion.
- Generated conversations must remain nested `{role, content}` records, contain
  2–40 non-empty turns, and end with an assistant turn.
- The deterministic `quality-v1` gate rejects task drift, refusal-only output,
  missing participants, orphaned tool blocks, and degenerate/repeated content.
- Normalized conversation fingerprints are unique across every accepted slot;
  collisions retry the losing assigned slot instead of creating duplicates.
- Accepted rows use an explicit Arrow schema and atomic immutable shards.
- Every output directory has an atomic `run_manifest.json` with a random UUID,
  frozen source/configuration digests, target, assignment version, model, and
  validation-policy version.
- Refinement backups are sealed by a checksum inventory and complete marker,
  then synchronized only to `runs/<run-instance-id>/`; a fresh output directory
  cannot inherit shards from an earlier run.
- Every refinement and publication upload is read back from its remote prefix;
  UUID/manifest markers, complete file sets, byte sizes, and SHA-256 values must
  match before synchronization is reported as successful.
- Resume state is derived from accepted Parquet shards, not optimistic counters.
- Every enabled worker run synchronizes in a finalization path, including fatal
  and incomplete batches; a completed rerun synchronizes without opening Gemini.
- Duplicate or unknown synthetic IDs stop the run.
- Stage 6 dry-run performs no local or remote writes.
- Stage 6 always writes to a fresh staging directory.
- A publication is promoted only after schema, checksum, and physical row-count
  validation succeeds.
- Remote publication uses a versioned destination and a failed final sync exits
  nonzero.

## Setup

```bash
bash install.sh
source .venv/bin/activate
cp .env.example .env
```

Required configuration:

```dotenv
GEMINI_API_KEY=...
HF_TOKEN=...
PIPELINE_DATA_DIR=/mnt/c/Users/proxi/pipeline/data
```

Useful optional settings:

```dotenv
GEMINI_MODEL_ID=gemini-3.6-flash
GEMINI_CONCURRENCY=4
GEMINI_MAX_OUTPUT_TOKENS=4096
REFINEMENT_REQUEST_BATCH_SIZE=32
REFINEMENT_MAX_ATTEMPTS_PER_RUN=3
```

The data directory defaults to `~/pipeline/data`. In the current WSL setup,
`/mnt/c/Users/proxi/pipeline/data` is authoritative.

## Stage 3–5: streaming refinement

Before requests begin, the source is read for the compact physical-row →
(`trial_name`, `run_id`, `task`) identity map, hashed byte-for-byte against the
manifest, and read again for complete-row validation. Requests then run in
bounded batches. Only the active request batch and one accepted output batch
are held in memory.

For a new or incomplete run, the lightweight provider/model preflight occurs
before those source scans. A completed run still skips Gemini entirely so it can
retry synchronization offline. Each real generation call receives a monotonic
`provider_request_number` in `attempts.jsonl` when a hard request budget is in
use.

### Retry and failure classification

Every attempt is recorded in `attempts.jsonl` with a status that determines
whether its assigned slot may be retried:

- `rejected` means the provider returned a response, but local validation did
  not accept it. Validation failures include malformed JSON, a response that is
  not an object, a missing or invalid `conversations` list, invalid roles or
  non-string/empty content, fewer than 2 or more than 40 turns, a conversation
  that does not end with an assistant turn, an unchanged copy of the source
  conversation, or source/slot lineage that no longer matches. The worker
  retries the same deterministic slot up to `--max-attempts-per-run`; a rejected
  response never becomes an accepted row or a replacement candidate.
- Schema-valid candidates also pass the deterministic `quality-v1` policy. Its
  long-task rewritten-user token floor is 5%, and its assistant/tool execution
  floor is 15%. These conservative thresholds were rounded down from the
  observed minima across 173 locally available successful trajectories (5.09%
  and 15.82%, respectively). Complete loss of paths, identifiers, commands,
  URLs, quoted values, and numbers is also rejected when lexical task retention
  is weak.
- Exact normalized conversation collisions are recorded as `rejected` with
  `duplicate_conversation`. Existing accepted rows always win; within a
  concurrent batch, the lexicographically smallest `synthetic_id` wins. Losing
  slots are retried concurrently within their remaining per-run attempt budget.
- `provider_error` means the request failed at the API/client boundary rather
  than failing conversation validation. Non-fatal provider errors may retry the
  same slot within its configured attempt limit.
- Provider errors with HTTP status `400`, `401`, `403`, or `404` are fatal.
  These normally indicate an invalid request, credentials, permissions, model,
  or endpoint configuration, so the worker stops retrying that slot immediately
  instead of consuming additional paid attempts. After the concurrent batch is
  persisted, the overall run performs its enabled final synchronization and
  returns exit code `2`. A final-synchronization failure is instead surfaced as
  exit code `1` by the CLI boundary.

An attempt becomes `accepted` only after the generated conversation and its
source relationship pass every validation check and its computed
`refined_conversation_fingerprint` is globally unique. The first accepted
result permanently completes that slot.

Check status without initializing Gemini:

```bash
uv run python stream_refinement_worker.py --status-only
```

Check credentials and model access without reading the source or generating:

```bash
uv run python stream_refinement_worker.py --provider-preflight-only
```

Retry synchronization without reading the source or initializing Gemini:

```bash
uv run python stream_refinement_worker.py --sync-only
```

Make exactly one provider request without remote synchronization:

```bash
uv run python stream_refinement_worker.py --max-provider-requests 1 --no-sync
```

`--limit` limits assigned slots and is not a request budget because a slot may
retry. Use `--max-provider-requests` for exact paid-call pilots. Reaching that
budget is an intentional resumable stop and does not mark unfilled slots as
failed.

Resume every incomplete slot:

```bash
uv run python stream_refinement_worker.py
```

Important options:

```text
--original-source
--output-dir
--target-rows
--model
--concurrency
--request-batch-size
--max-attempts-per-run
--max-provider-requests
--limit
--status-only
--sync-only
--provider-preflight-only
--source-snapshot-dir
--no-sync
```

Local refinement state:

```text
<data-dir>/refined/
  accepted/
    accepted-<content-id>.parquet
  attempts.jsonl
  progress.json
  source_manifest.json
  run_manifest.json
  checksum_inventory.json
  complete.json
```

`attempts.jsonl` is an audit log. Successful resume state comes from validated
accepted Parquet shards. A crash after an API response but before shard
promotion may cause the assigned slot to be requested again, but it cannot
create two stored successes for that slot.

`source_manifest.json` version 2 binds the run to its requested and resolved
source, exact physical-row/conversation identity digest, Arrow schema digest,
and the complete bytes of every sorted Parquet file. `run_manifest.json` binds
those source digests to the random run instance, exact target, assignment
algorithm, model/generation settings, and `quality-v1`. The checksum inventory
and marker cover both manifests, progress/audit files, and every accepted
shard. Resume, `--sync-only`, and Stage 6 reject missing, changed, or unexpected
state instead of repairing or silently migrating it.

Refinement backup destinations are isolated:

```text
<hf-refinement-bucket>/runs/<run-instance-id>/
```

Deleting and recreating a local output directory creates a different UUID and
therefore a different remote prefix. `--sync-only` loads the existing manifest
and reuses its original prefix.

### Immutable local source snapshot

Materialize the pinned source once before the production run:

```bash
uv run python materialize_source_snapshot.py \
  --source "$ORIGINAL_DATASET_SOURCE" \
  --snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot
```

Verify it later without downloading:

```bash
uv run python materialize_source_snapshot.py \
  --snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --verify-only
```

The snapshot is built in a sibling staging directory and promoted only after
all Parquet files, the inventory, and the completion marker verify. Its manifest
retains each pinned remote path, so source-row IDs are identical whether the
pipeline reads the remote source or the local copy. Generation uses it with:

```bash
uv run python stream_refinement_worker.py \
  --source-snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot
```

### Codex production refinement and tmux operation

`codex_refinement_worker.py` is the ChatGPT-authenticated alternative refinement
worker. It retains deterministic source-slot ownership in Python and uses one
bounded `codex exec` subprocess as the specialist for each micro-batch. The
worker, rather than tmux or Codex, owns row identity, validation, retries,
accepted-shard promotion, progress, and checksum sealing.

Check the saved ChatGPT login without scanning the dataset or making a model
call:

```bash
.venv/bin/python codex_refinement_worker.py --preflight-only
```

The worker creates a fresh temporary directory and a batch-specific JSON Schema
for every call. Its native subprocess is equivalent to the command below. The
trailing `-` is required because the generated batch prompt is supplied on
standard input. Do not reuse a prior `/tmp/codex-refinement-call-*` schema: its
`synthetic_id` enum is valid for exactly one micro-batch and the worker removes
the temporary directory after the call.

```bash
codex exec \
  --json \
  --ephemeral \
  --ignore-user-config \
  --sandbox read-only \
  --skip-git-repo-check \
  --color never \
  --model gpt-5.6-sol \
  --output-schema /tmp/codex-refinement-call-XXXXXXXX/output-schema.json \
  -
```

For an unattended production run, use the checked wrapper. It preserves the
worker's immutable configuration, gives each invocation a large explicit call
budget, and relaunches only when that budget ends successfully before the row
target. A nonzero worker exit stops the wrapper so configuration, authentication,
or state-integrity failures do not become an infinite restart loop.

```bash
export PIPELINE_DATA_DIR=/mnt/c/Users/proxi/pipeline/data
export CODEX_REFINEMENT_MODEL=gpt-5.6-sol
export CODEX_REFINEMENT_OUTPUT_DIR="$PIPELINE_DATA_DIR/codex-refined"
export CODEX_REFINEMENT_BATCH_SIZE=4
export CODEX_REFINEMENT_CONCURRENCY=1
export CODEX_MAX_AGENT_CALLS_PER_INVOCATION=100000

./run_codex_refinement_loop.sh
```

Run the same resumable loop in tmux and keep a durable operator log:

```bash
augmentation_dir="$(pwd)"
production_session=codex-refinement-production
production_log="$PIPELINE_DATA_DIR/codex-refined-production.log"

tmux new-session -d -s "$production_session" \
  "cd '$augmentation_dir' && \
  exec env PYTHONUNBUFFERED=1 ./run_codex_refinement_loop.sh \
  >> '$production_log' 2>&1"
```

Inspect the session, recent output, native Codex child, and durable row count:

```bash
tmux has-session -t "$production_session"
tmux capture-pane -p -t "$production_session" -S -80
tail -n 80 "$production_log"
ps -C codex -o pid=,ppid=,etimes=,stat=,args=
sed -n '/completed_rows\|remaining_rows\|target_rows/p' \
  "$CODEX_REFINEMENT_OUTPUT_DIR/progress.json"
```

If the tmux server or host exits, rerun the same `tmux new-session` command.
The worker validates the sealed manifests and accepted Parquet shards before
selecting the next incomplete source slot. It never infers completion from the
log or from an in-flight `codex exec` process.

Synchronize accepted rows independently every time another 10-row threshold is
sealed:

```bash
checkpoint_session=codex-refinement-hf-checkpoints
checkpoint_log="$PIPELINE_DATA_DIR/codex-refined-hf-checkpoints.log"

tmux new-session -d -s "$checkpoint_session" \
  "cd '$augmentation_dir' && \
  exec env PYTHONUNBUFFERED=1 .venv/bin/python codex_hf_checkpoint_sidecar.py \
  --output-dir '$CODEX_REFINEMENT_OUTPUT_DIR' --checkpoint-rows 10 \
  >> '$checkpoint_log' 2>&1"
```

The sidecar never locks, changes, or restarts the production worker. It uploads
only newly sealed immutable Parquet shards, verifies their remote SHA-256 values,
commits a chained checkpoint marker, and then updates `latest.json`. Its local
cursor is stored beside the production output in
`codex-refined-hf-checkpoints/state.json`; remote checkpoints are isolated at:

```text
<hf-refinement-bucket>/checkpoints/runs/<run-instance-id>/
```

This checkpoint namespace is a recovery stream for accepted synthetic rows. It
is separate from the final Stage 6 publication and never deletes remote files.

### Supported pilot harness

`refinement_pilot.py` deterministically selects 10–50 rows into its own local
manifest-v2 fixture. Preparation is offline by default:

```bash
uv run python refinement_pilot.py --sample-size 10
```

Real provider access requires the explicit `--execute` flag. The harness always
passes `--no-sync`, refuses the production refinement/publication directories,
and writes `pilot_report.json` with acceptance, rejection, provider-error,
duplicate, attempt-per-slot, and source/refinement sample metrics:

```bash
uv run python refinement_pilot.py \
  --sample-size 10 \
  --max-provider-requests 10 \
  --pilot-dir /mnt/c/Users/proxi/pipeline/data/pilots/ten-request \
  --execute
```

Rebuild a report without provider access with `--report-only`.

### Remote restore drill

After a refinement sync, restore its isolated run into a new directory and
perform the normal offline status validation:

```bash
uv run python restore_refinement_run.py \
  --remote-run "${HF_REFINEMENT_BUCKET}/runs/<run-instance-id>" \
  --output-dir /mnt/c/Users/proxi/pipeline/data/restore-drill-<run-instance-id>
```

The drill verifies the remote copy before download, verifies the restored local
inventory, checks the prefix UUID, and runs `--status-only`. It refuses an
existing destination.

## Stage 6: exact publication

Validate the complete local result without writing:

```bash
uv run python augment_150k_rows.py --dry-run --no-sync
```

Create and inspect a local publication:

```bash
uv run python augment_150k_rows.py --no-sync
```

Publish after local inspection:

```bash
uv run python augment_150k_rows.py
```

Important options:

```text
--original-source
--refined-dir
--upload-dir
--hf-bucket
--expected-new-rows
--expected-total-rows
--rows-per-shard
--dry-run
--no-sync
```

Local publication layout:

```text
<data-dir>/upload/
  current.json
  .work/                         # never synchronized
  publications/
    publication-<content-id>/
      publication_manifest.json
      publication_complete.json
      data/
        train-00000-of-00050.parquet
        ...
```

Remote publication is versioned:

```text
hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/
  publications/publication-<content-id>/
```

Old local or remote shards are never mixed into a new publication. Upload
success requires a read-back of `publication_manifest.json`,
`publication_complete.json`, and every declared Parquet checksum.
`--expected-total-rows` is an optional extra assertion; when omitted, the total
is derived as the physical source count plus `--expected-new-rows`.

Stage 6 requires current accepted-shard metadata. It recomputes normalized
conversation fingerprints, enforces global uniqueness, verifies every row's
`refinement_validation_policy` against the run manifest, and rejects legacy
accepted schemas explicitly. Publication IDs, manifests, and `current.json`
include the run UUID and full source-content digest.

## Verification

Run the local suite:

```bash
.venv/bin/python -m unittest discover -v
```

Before the full paid run, after replacing the Gemini key:

1. Run `stream_refinement_worker.py --provider-preflight-only`.
2. Prepare an isolated pilot, then run it with
   `--max-provider-requests 1 --execute`.
3. Inspect the source/refinement pair in `pilot_report.json`.
4. Use a fresh pilot directory for exactly 10 calls.
5. Expand to 25–50 calls only if the ten-call report warrants it.
6. Review rejection, duplication, provider-error, and attempts-per-slot metrics.
7. Materialize and verify the immutable ten-file source snapshot.
8. Run all local tests and the 10K, 100K, and 225K bounded-memory fixtures.
9. Start the isolated 150K run against `--source-snapshot-dir` only after those
   gates pass.
10. Complete Stage 6 `--dry-run`, local `--no-sync`, remote publication, remote
    checksum read-back, and a fresh-directory refinement restore drill.

The live checks are intentionally skipped in the default test suite. After the
key is available, enable provider preflight with
`RUN_GEMINI_INTEGRATION=1`; enable the single paid integration request only by
also setting `RUN_GEMINI_PAID_TESTS=1`.

Large generated fixtures belong under `/tmp` and must not be committed.

### Verified synthetic scale results

Measured under WSL on 2026-07-27 with 512-byte source-conversation fixture
payloads. Peak RSS covers fixture creation, preflight, and publication in one
process.

| Accepted refinements | Source rows | Published rows | Peak RSS | Wall time |
|---:|---:|---:|---:|---:|
| 10,000 | 6,667 | 16,667 | 187 MiB | 19.7 s |
| 100,000 | 66,667 | 166,667 | 326 MiB | 41.4 s |
| 225,000 | 150,000 | 375,000 | 390 MiB | 67.2 s |

These fixtures include global fingerprint indexing and full fingerprint
recomputation during Stage 6 preflight, complete source-file hashing, run
manifest validation, checksum-inventory sealing, and post-stream source
verification. They validate bounded control-state memory and exact row
accounting; they do not predict Gemini latency or the final compressed size of
real model outputs.

## Legacy pipeline

The following files describe the superseded 225K oversampling path and remain
only for historical compatibility and comparison:

- `extract_tasks.py`
- `plan_variants.py`
- `gemma4_31b_agent.py`
- `gemini_trajectory_worker.py`
- `validate_n_dedup.py`
- `run_pilot.py` (executable migration stub; use `refinement_pilot.py`)

`gemma4_31b_agent.py`, `gemini_trajectory_worker.py`, and `run_pilot.py` exit
with code `2` before client initialization when executed. Their importable
historical helpers and existing raw JSONL data remain untouched for audit and
compatibility purposes.
