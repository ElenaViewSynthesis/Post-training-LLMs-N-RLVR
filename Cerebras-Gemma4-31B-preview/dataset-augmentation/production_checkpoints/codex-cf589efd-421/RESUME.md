# Codex production checkpoint: 421 accepted rows

This repository checkpoint freezes the durable production state for run
`cf589efd-de37-4a48-bd77-5239b423936c` after a graceful stop on 2026-07-29.
The worker had accepted 421 of 150,000 synthetic rows, leaving 149,579.

## Included proof

- `accepted_rows.jsonl` contains all 421 complete synthetic records, one
  canonical JSON object per line, sorted by `run_id`.
- `checkpoint_manifest.json` records the JSONL byte size and SHA-256, every
  source Parquet shard checksum, the run/source identity, row counters, and the
  validation results used during export.
- `sealed_state/` preserves the exact progress, run manifest, source manifest,
  audit log, checksum inventory, and completion marker from the stopped worker.

The checkpoint exporter fully verified the worker's checksum inventory, scanned
all accepted Parquet rows, recomputed normalized conversation fingerprints,
and required 421 unique synthetic IDs and 421 unique refined conversations.

This Git artifact is evidence and a reviewable data snapshot. It is not a
replacement for `/mnt/c/Users/proxi/pipeline/data/codex-refined`: resuming the
immutable worker requires that original directory and its 106 accepted Parquet
shards to remain intact.

## Verify from a fresh terminal

From `Cerebras-Gemma4-31B-preview/dataset-augmentation`:

```bash
checkpoint_dir=production_checkpoints/codex-cf589efd-421

python -c 'import hashlib,json,pathlib; p=pathlib.Path("production_checkpoints/codex-cf589efd-421"); m=json.loads((p/"checkpoint_manifest.json").read_text()); b=(p/m["accepted_rows"]["path"]).read_bytes(); rows=sum(1 for line in b.splitlines() if line); digest=hashlib.sha256(b).hexdigest(); assert rows==m["accepted_rows"]["rows"]==421; assert digest==m["accepted_rows"]["sha256"]; print({"rows":rows,"sha256":digest,"run_instance_id":m["run_instance_id"]})'
```

Expected JSONL SHA-256:

```text
a05ff334a83364087a9cb3a1f99cdc647bb07a7b346d88415c2403db616763f4
```

Validate the original resumable state without making a model call:

```bash
.venv/bin/python codex_refinement_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --source-snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --output-dir /mnt/c/Users/proxi/pipeline/data/codex-refined \
  --target-rows 150000 \
  --model gpt-5.6-sol \
  --batch-size 4 \
  --concurrency 1 \
  --max-agent-calls 37499 \
  --max-attempts-per-run 3 \
  --timeout-seconds 600
```

## Resume production in tmux

Confirm the ChatGPT login first:

```bash
.venv/bin/python codex_refinement_worker.py --preflight-only
```

Then launch the checked long-run wrapper. It validates and resumes the existing
immutable run; it does not create a new output directory or replace the first
421 rows.

```bash
augmentation_dir="$(pwd)"
production_session=codex-refinement-prod-cf589efd
production_log=/mnt/c/Users/proxi/pipeline/data/codex-refined-production.log

export PIPELINE_DATA_DIR=/mnt/c/Users/proxi/pipeline/data
export CODEX_REFINEMENT_MODEL=gpt-5.6-sol
export CODEX_REFINEMENT_OUTPUT_DIR="$PIPELINE_DATA_DIR/codex-refined"
export CODEX_REFINEMENT_TARGET_ROWS=150000
export CODEX_REFINEMENT_BATCH_SIZE=4
export CODEX_REFINEMENT_CONCURRENCY=1
export CODEX_MAX_AGENT_CALLS_PER_INVOCATION=37499
export CODEX_MAX_ATTEMPTS_PER_RUN=3
export CODEX_REFINEMENT_TIMEOUT_SECONDS=600

tmux new-session -d -s "$production_session" \
  "cd '$augmentation_dir' && \
  exec env PYTHONUNBUFFERED=1 ./run_codex_refinement_loop.sh \
  >> '$production_log' 2>&1"
```

Monitor the durable count:

```bash
tmux capture-pane -p -t "$production_session" -S -80
sed -n '/completed_rows\|remaining_rows\|target_rows/p' \
  "$CODEX_REFINEMENT_OUTPUT_DIR/progress.json"
```

To restart the independent 10-row Hugging Face checkpoint sidecar as well:

```bash
checkpoint_session=codex-refinement-hf-cf589efd
checkpoint_log="$PIPELINE_DATA_DIR/codex-refined-hf-checkpoints.log"

tmux new-session -d -s "$checkpoint_session" \
  "cd '$augmentation_dir' && \
  exec env PYTHONUNBUFFERED=1 .venv/bin/python codex_hf_checkpoint_sidecar.py \
  --output-dir '$CODEX_REFINEMENT_OUTPUT_DIR' \
  --state-dir '$PIPELINE_DATA_DIR/codex-refined-hf-checkpoints' \
  --checkpoint-rows 10 >> '$checkpoint_log' 2>&1"
```
