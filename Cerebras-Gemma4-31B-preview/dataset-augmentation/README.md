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
from the sorted Parquet file path and row index. Both `run_id` and `trial_name`
repeat in the real data, so neither is safe as a row key. They and the original
task remain lineage metadata. A rejected generation is retried for the same
slot until a result passes validation. The first accepted result completes that
slot permanently.

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
- Only one worker can own a refinement output directory.
- Every paid request is assigned to a deterministic source row and slot.
- All source identities and conversations are checked before paid requests.
- Generated conversations must remain nested `{role, content}` records, contain
  2–40 non-empty turns, and end with an assistant turn.
- Accepted rows use an explicit Arrow schema and atomic immutable shards.
- Resume state is derived from accepted Parquet shards, not optimistic counters.
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

The source dataset is read twice before requests begin: first for the compact
physical-row → (`trial_name`, `run_id`, `task`) identity map, then for
complete source-row validation. Requests then run in bounded batches. Only the
active request batch and one accepted output batch are held in memory.

Check status without initializing Gemini:

```bash
uv run python stream_refinement_worker.py --status-only
```

Make one request without remote synchronization:

```bash
uv run python stream_refinement_worker.py --limit 1 --no-sync
```

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
--limit
--status-only
--no-sync
```

Local refinement state:

```text
<data-dir>/refined/
  accepted/
    accepted-<content-id>.parquet
  attempts.jsonl
  progress.json
```

`attempts.jsonl` is an audit log. Successful resume state comes from validated
accepted Parquet shards. A crash after an API response but before shard
promotion may cause the assigned slot to be requested again, but it cannot
create two stored successes for that slot.

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
      data/
        train-00000-of-00050.parquet
        ...
```

Remote publication is versioned:

```text
hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/
  publications/publication-<content-id>/
```

Old local or remote shards are never mixed into a new publication.
`--expected-total-rows` is an optional extra assertion; when omitted, the total
is derived as the physical source count plus `--expected-new-rows`.

## Verification

Run the local suite:

```bash
.venv/bin/python -m unittest discover -v
```

Before the full paid run:

1. Run all local tests.
2. Run a generated 10K fixture through assignment, accepted-shard resume, and
   Stage 6 publication.
3. Repeat at 100K and 225K while recording wall time and peak RSS.
4. Run one Gemini request with `--limit 1 --no-sync`.
5. Inspect its source and refined conversations manually.
6. Run a controlled 10–50 request pilot.
7. Resume the full 150K slot assignment only after the pilot passes.

Large generated fixtures belong under `/tmp` and must not be committed.

### Verified synthetic scale results

Measured under WSL on 2026-07-26 with 512-byte source-conversation fixture
payloads. Peak RSS covers fixture creation, preflight, and publication in one
process.

| Accepted refinements | Source rows | Published rows | Peak RSS | Wall time |
|---:|---:|---:|---:|---:|
| 10,000 | 6,667 | 16,667 | 174 MiB | 5.3 s |
| 100,000 | 66,667 | 166,667 | 269 MiB | 9.2 s |
| 225,000 | 150,000 | 375,000 | 327 MiB | 17.6 s |

These fixtures validate bounded control-state memory and exact row accounting;
they do not predict Gemini latency or the final compressed size of real model
outputs.

## Legacy pipeline

The following files describe the superseded 225K oversampling path and remain
only for historical compatibility and comparison:

- `extract_tasks.py`
- `plan_variants.py`
- `gemma4_31b_agent.py`
- `gemini_trajectory_worker.py`
- `validate_n_dedup.py`
- `run_pilot.py`

Do not start a new full run with those entry points. Existing raw JSONL data is
left untouched so the migration remains reversible.
