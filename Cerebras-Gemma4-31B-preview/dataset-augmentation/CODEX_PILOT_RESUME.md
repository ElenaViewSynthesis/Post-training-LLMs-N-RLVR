# Codex trajectory-augmentation resume handoff

Last updated: 2026-07-29 (Europe/London)

This file is the handoff for continuing the Codex trajectory-augmentation work
in a fresh terminal session. It records the exact repository state, completed
pilots, measured results, authorization boundaries, and next incomplete gate.

## Start here in the next session

Repository:

```text
/mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation
```

Before making changes:

1. Read this file completely.
2. Read `RESUME_RUNBOOK.md` completely.
3. Read the relevant Codex sections of `README.md`.
4. Run the read-only Git checks below.
5. Preserve all unrelated sibling-project changes.
6. Do not make another model call, run production, or synchronize data without
   fresh explicit authorization.

Read-only Git bootstrap:

```bash
cd /mnt/c/Users/proxi/Documents/codex4/Post-training-LLMs-N-RLVR/Cerebras-Gemma4-31B-preview/dataset-augmentation

git status --short --branch
git rev-parse HEAD
git rev-parse '@{upstream}'
git rev-list --left-right --count HEAD...'@{upstream}'
git diff --check -- codex_refinement_worker.py test_codex_refinement_worker.py CODEX_PILOT_RESUME.md
git diff -- codex_refinement_worker.py test_codex_refinement_worker.py
```

## Current Git state

Current branch:

```text
agent/codex-quality-pilot
```

Current pushed v2 implementation baseline:

```text
c5d775b
feat(codex): preserve terminal feedback and deduplicate batches
```

The preceding committed baseline is:

```text
6ff8cfbd154cc766414ee79aa604eda6eb45d4df
feat(codex): harden trajectory validation and batching
```

The branch tracks `origin/agent/codex-quality-pilot`. Commit `c5d775b` was
pushed successfully before this handoff was updated. This handoff is committed
as a subsequent documentation-only commit, so use the bootstrap commands to
read the exact current HEAD and confirm `0/0` divergence.

No uncommitted changes are expected in these Codex-scoped paths after the
handoff commit:

```text
codex_refinement_worker.py
test_codex_refinement_worker.py
CODEX_PILOT_RESUME.md
```

Do not stage or modify these unrelated user-owned paths:

```text
../../EmbeddingGemma300M/.gitignore
../../Gemma-4-12B-it/.gitignore
../../Gemma-4-12B-it/.python-version
../../Gemma-4-12B-it/uv.lock
../../Qwen3Embedding4B/.gitignore
gh
```

Never use `git add -A` in this mixed worktree. For future Codex-scoped changes,
stage only the explicitly intended paths, for example:

```bash
git add -- codex_refinement_worker.py test_codex_refinement_worker.py CODEX_PILOT_RESUME.md
```

## Authorization boundary

Completed and authorized:

- Materialize and verify the immutable production source snapshot.
- Make the Codex-only prompt and validation changes.
- Run two separate one-call Codex pilots.
- Use four refinement slots per Codex invocation.

Not currently authorized:

- Another Codex, Gemini, Cerebras, or other model request.
- A production augmentation run.
- Synchronization or publication of any data.
- Re-downloading or replacing the sealed source snapshot.

The Codex worker uses saved ChatGPT/Codex CLI authentication. It removes API-key
and token environment variables before spawning `codex exec`; no OpenAI API key
is required. Calls still consume the account's Codex/ChatGPT usage allowance.

## Immutable production source snapshot

Snapshot directory:

```text
/mnt/c/Users/proxi/pipeline/data/source-snapshot
```

Verified identity:

```text
file_count: 10
physical_rows: 94,334
snapshot_size: approximately 1.7 GiB
snapshot_instance_id: 3c4ade2e-c654-48f8-8423-3502c30128e2
source_content_sha256: 060d3b9e43aa6540a8c70759a8260658cecb7d6cb1022955215c2d72b6151db7
resolved_revision: 45fb28fcc38d352133cb28a1c8a43a2f14fea97b
```

Resolved source:

```text
hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K@45fb28fcc38d352133cb28a1c8a43a2f14fea97b/data/train-*-of-*.parquet
```

The snapshot has already passed an independent `--verify-only` check. Do not
materialize it again. A future read-only verification, if genuinely needed, is:

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run python materialize_source_snapshot.py \
  --snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --verify-only
```

## Production assignment semantics

`--batch-size 4` means four **refinement slots** per Codex invocation, not four
rows total and not necessarily four distinct source rows.

For the configured 150,000-row synthetic target:

- All 94,334 physical source rows receive one synthetic refinement.
- A deterministic subset of 55,666 source rows receives a second refinement.
- Total synthetic trajectories: 150,000.
- Combined original plus synthetic publication, if later authorized: 244,334
  rows.
- Minimum calls at four slots per call, before retries: 37,500.

The pilot batch has four slots but three distinct source rows because the first
source row has refinement indexes 0 and 1.

## Committed Codex quality work

Commit `6ff8cfb` contains:

- Default Codex batch size changed from 2 to 4 slots.
- A leaner, outcome-oriented prompt with explicit success criteria.
- Codex-only `codex-quality-v1` checks for:
  - added deliverables;
  - adjacent duplicate participant roles;
  - removal of source tool evidence;
  - vague completion claims without concrete evidence.
- Combined shared `quality-v1` and Codex-only validation in the Codex worker.
- Codex validation policy metadata in attempts and progress state.
- Five focused unit tests.

Gemini and Cerebras behavior was not modified.

## Committed v2 changes

Commit `c5d775b` addresses the measured first-pilot failure with two surgical
changes:

1. Legacy terminal-feedback envelope preservation
   - The prompt now explains that terminal outputs, warnings, grader prompts,
     and current-terminal-state updates are often encoded as `user` turns.
   - It requires retaining concrete evidence in `user -> assistant` pairs.
   - It prohibits relabeling those legacy source user turns as tool turns.

2. In-request source deduplication
   - The prompt payload now has a `sources` array containing one copy of each
     unique source row.
   - A separate `requests` array contains all four synthetic IDs and references
     the appropriate `source_record_id`.
   - Multiple refinement requests for the same source still require distinct,
     source-supported outputs.
   - Inconsistent content for one `source_record_id` fails locally before a
     model call.

Two regression tests were added for the legacy envelope and source
deduplication. The exact v2 pilot payload contained four requests, three unique
sources, and 191,007 UTF-8 prompt bytes. The implementation and tests are pushed
to `origin/agent/codex-quality-pilot`.

This change follows the GPT-5.6 guidance to make small prompt changes tied to a
measured failure and validate on representative traces:

https://developers.openai.com/api/docs/guides/model-guidance?model=gpt-5.6#prompting-best-practices

## Offline validation

Latest result:

```text
Ran 96 tests
OK (skipped=2)
```

The two skipped tests are explicitly opt-in Gemini live tests. No provider calls
occurred during offline validation.

Commands:

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run python -m py_compile \
  codex_refinement_worker.py test_codex_refinement_worker.py

UV_CACHE_DIR=/tmp/codex-uv-cache uv run python -m unittest -v \
  test_codex_refinement_worker.py

UV_CACHE_DIR=/tmp/codex-uv-cache uv run python -m unittest discover -v
```

## Pilot v1: committed prompt without source deduplication

Local-only sealed state:

```text
/mnt/c/Users/proxi/pipeline/data/codex-pilots/four-row-20260729-gpt56sol-quality-v1
```

Configuration:

```text
model: gpt-5.6-sol
calls: exactly 1
batch: 4 refinement slots / 3 distinct source rows
concurrency: 1
retries: 0
synchronization: disabled
```

Results:

```text
schema-valid candidates: 4/4
accepted: 0/4
input_tokens: 96,611
output_tokens: 5,005
reasoning_output_tokens: 101
cached_input_tokens: 0
```

All four passed the new Codex-only checks. All four failed shared `task_drift`
validation because source user text retention was only 1.7%-4.4%. Two also
failed assistant/tool retention at 14.0% and 11.3% against a 15% threshold.

Root cause: these source trajectories alternate `user` and `assistant`, using
later `user` turns for large terminal outputs. Shared `quality-v1` treats every
source user turn as task text, not only the initial request.

## Pilot v2: legacy envelope plus deduplicated sources

Local-only sealed state:

```text
/mnt/c/Users/proxi/pipeline/data/codex-pilots/four-slot-20260729-gpt56sol-legacy-dedupe-v2
```

Configuration:

```text
model: gpt-5.6-sol
calls: exactly 1
batch: 4 refinement slots / 3 distinct source rows
concurrency: 1
retries: 0
synchronization: disabled
interaction_id: 019fab3c-8161-7520-b972-d199111c4d57
```

Results:

```text
schema-valid candidates: 4/4
accepted: 0/4
input_tokens: 73,793
output_tokens: 7,860
reasoning_output_tokens: 81
cached_input_tokens: 0
```

Comparison with v1:

| Metric | V1 | V2 | Change |
| --- | ---: | ---: | ---: |
| Input tokens | 96,611 | 73,793 | -22,818 (-23.6%) |
| Output tokens | 5,005 | 7,860 | +2,855 (+57.0%) |
| Reasoning tokens | 101 | 81 | -20 (-19.8%) |
| Source-user retention failures | 4 | 0 | fixed |
| Accepted trajectories | 0/4 | 0/4 | still blocked |

The v2 envelope prompt fixed the source-user retention failure for all four
candidates. The remaining failures are solely shared `quality-v1`
assistant/tool task-token retention:

```text
refined-9ab418dcb26aaf88f9e8ee32: 0.111
refined-1b6fe6a5eb76d4c2ded0bf57: 0.101
refined-acf23b00ba60408cc2bd3911: 0.133
refined-ccde6b68eb46eeeff98f987e: 0.108
required threshold: 0.150
```

Both pilot directories passed refinement-state checksum-inventory verification.
Neither pilot synchronized or published anything.

## Important observability limitation

The worker records fingerprints, usage, status, and rejection details but does
not persist rejected candidate conversation bodies. Each Codex call uses
`--ephemeral`, so the rejected v1 and v2 candidates cannot be reconstructed.

Opt-in retention of rejected bodies was previously recommended but was not part
of the user's v2 request and has not been implemented. If pair-level audit is
required for the next pilot, design and test an explicitly opt-in local audit
artifact before spending another call.

## Current completed state

- Immutable source snapshot: complete and verified.
- Codex worker isolation and ChatGPT authentication path: complete.
- Four-slot micro-batching: complete.
- Codex-only quality gate: complete.
- Legacy terminal-feedback prompt guidance: complete.
- Repeated-source prompt deduplication: complete and measured at 23.6% lower
  input tokens on the same pilot batch.
- V2 offline regression suite: complete.
- V2 one-call pilot: complete.
- Production readiness: **not complete** because 0/4 trajectories were
  accepted.

## Next incomplete gate

The next measured failure is assistant-side evidence retention. A minimal next
prompt adjustment should require each assistant response following a legacy
terminal-feedback user turn to explicitly interpret and carry forward concrete
source-supported task evidence: relevant paths, commands, symbols, failures,
test counts, and findings. The aim is complete evidence-bearing trajectories,
not mechanical keyword repetition.

Recommended sequence:

1. Add one surgical assistant-evidence success criterion to the Codex prompt.
2. Add a focused prompt regression test.
3. Run targeted tests and the complete offline suite.
4. Report the exact diff and test results.
5. Obtain explicit authorization for exactly one additional Codex call.
6. Use a fresh isolated v3 pilot directory, `--limit 4`,
   `--max-agent-calls 1`, and `--max-attempts-per-run 1`.
7. Compare input/output usage and all rejection codes with v1 and v2.
8. Keep the production gate closed unless the representative pilot passes.

Do not weaken or remove the shared validator merely to make the pilot pass. Do
not modify Gemini or Cerebras behavior. Any validator adaptation for the legacy
envelope requires a separate, explicit design decision and tests demonstrating
that validation remains meaningful.

## Template for a future authorized v3 pilot

Use only after the user explicitly authorizes another real call. Choose a fresh
output directory that does not already exist.

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run python codex_refinement_worker.py \
  --data-dir /mnt/c/Users/proxi/pipeline/data \
  --source-snapshot-dir /mnt/c/Users/proxi/pipeline/data/source-snapshot \
  --output-dir /mnt/c/Users/proxi/pipeline/data/codex-pilots/four-slot-v3-CHOOSE-A-FRESH-NAME \
  --target-rows 150000 \
  --model gpt-5.6-sol \
  --batch-size 4 \
  --concurrency 1 \
  --limit 4 \
  --max-agent-calls 1 \
  --max-attempts-per-run 1 \
  --timeout-seconds 600 \
  --execute
```

The full source scan over the WSL-mounted Windows volume is slow and silent for
several minutes before the output directory appears. Absence of the output
directory means the Codex call has not started and the one-call budget has not
been consumed.
