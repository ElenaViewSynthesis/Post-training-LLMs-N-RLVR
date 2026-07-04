"""
Stage 0-1: Load the 100K SFT dataset with dask, assert its actual schema,
then write a task table — every real column plus a derived `instruction`.

Actual schema (confirmed from dataset — these are the ONLY real columns;
there is no flat `instruction`/`prompt`/`task_description` column):
    agent           string   — agent name (e.g. "terminus-2")
    conversations   object   — the multi-turn trajectory (to be regenerated)
    date            string   — ISO timestamp
    episode         string   — episode identifier
    model           string   — teacher model used in original generation
    model_provider  string   — e.g. "hosted_vllm"
    result          string   — pass/fail outcome
    run_id          string   — UUID per row
    task            string   — task name/identifier (dedup key)
    trace_source    string   — e.g. "main"
    trial_name      string   — unique trial identifier per row

The real task instruction is NOT a separate column — it's embedded inside
conversations[0] (role="user"), which is the Terminus-2 harness's fixed
template followed by "Task Description:\n<the actual task>\n\nCurrent
terminal state:\n...". extract_instruction() below pulls that substring out
before the `conversations` column is dropped, so downstream synthesis
(gemma4_31b_agent.build_synthesis_prompt) has real task content to work
from instead of an empty task section. Confirmed consistent across a
sample of 8 real rows.

Augmentation strategy: task table = all real columns except `conversations`,
plus the derived `instruction` column.
"""
import os
import json
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

SRC = "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet"

TRAJECTORY_COL = "conversations"
TASK_ID_COL    = "task"          # dedup key for unique tasks
TASKS_OUT      = Path("~/pipeline/data/tasks/").expanduser()

REAL_COLUMNS = [
    "agent", "conversations", "date", "episode", "model", "model_provider",
    "result", "run_id", "task", "trace_source", "trial_name",
]

TASK_DESC_MARKER     = "Task Description:\n"
TERMINAL_STATE_MARKER = "\n\nCurrent terminal state:"


def extract_instruction(conversations) -> str:
    """Pull the real task instruction out of conversations[0]'s harness
    template. Returns "" if the expected markers aren't present (e.g. a
    differently-formatted row) rather than raising, so callers can detect
    and log the miss instead of the pipeline crashing on one bad row."""
    if conversations is None or len(conversations) == 0:
        return ""
    first = conversations[0]
    if first.get("role") != "user":
        return ""
    content = first.get("content") or ""
    start = content.find(TASK_DESC_MARKER)
    if start == -1:
        return ""
    start += len(TASK_DESC_MARKER)
    end = content.find(TERMINAL_STATE_MARKER, start)
    return (content[start:end] if end != -1 else content[start:]).strip()


def inspect_schema(df: dd.DataFrame) -> None:
    print("=== dtypes ===")
    print(df.dtypes)
    print("\n=== sample row (first partition, first record) ===")
    sample = df.head(1)
    for col in sample.columns:
        val = sample.iloc[0][col]
        preview = json.dumps(val, default=str)[:300] if not isinstance(val, str) else val[:300]
        print(f"--- {col} ---\n{preview}\n")


def build_task_table(df: dd.DataFrame) -> dd.DataFrame:
    """Task table = every column except conversations, plus `instruction`
    extracted from conversations[0] (see extract_instruction — the real
    task text lives inside the trajectory, not in a flat column).
    Deduplicated on `task` to get one row per unique task.
    Column order: `instruction` is placed right after `agent` (the first
    column) since it's the most important field to see at a glance."""
    task_cols = [c for c in df.columns if c != TRAJECTORY_COL]
    agent_idx = task_cols.index("agent")
    ordered_cols = task_cols[:agent_idx + 1] + ["instruction"] + task_cols[agent_idx + 1:]
    print(f"Task columns:      {ordered_cols}")
    print(f"Trajectory column: {TRAJECTORY_COL}")

    df = df.assign(
        instruction=df[TRAJECTORY_COL].apply(extract_instruction, meta=("instruction", "object"))
    )
    tasks = df[ordered_cols].drop_duplicates(subset=[TASK_ID_COL])
    return tasks


if __name__ == "__main__":
    df = dd.read_parquet(SRC)
    print(f"Partitions: {df.npartitions}, approx rows: {len(df)}")

    inspect_schema(df)
    tasks = build_task_table(df)

    # Compute to pandas before writing: dask's multi-partition parquet writer fails
    # to infer a pyarrow schema for the `instruction` column (object dtype from
    # .apply()) on some partitions. The deduped task table (~83K rows) fits in
    # memory comfortably, so write it as a single file via pandas/pyarrow instead.
    tasks_pdf = tasks.compute()
    print(f"\nUnique tasks: {len(tasks_pdf)}")

    TASKS_OUT.mkdir(parents=True, exist_ok=True)
    tasks_pdf.to_parquet(TASKS_OUT / "tasks.parquet", index=False)
    print(f"Wrote task table -> {TASKS_OUT}")
