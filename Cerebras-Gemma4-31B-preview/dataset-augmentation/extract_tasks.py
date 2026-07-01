"""
Stage 0-1: Load the 100K SFT dataset with dask, assert its actual schema,
then write a task table — every column except `conversations`.

Actual schema (confirmed from dataset):
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

Augmentation strategy: task table = all columns EXCEPT `conversations`.
We carry the full task context forward into generation and regenerate
`conversations` only.
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
    """Task table = every column except conversations.
    Deduplicated on `task` to get one row per unique task."""
    task_cols = [c for c in df.columns if c != TRAJECTORY_COL]
    print(f"Task columns:      {task_cols}")
    print(f"Trajectory column: {TRAJECTORY_COL}")
    tasks = df[task_cols].drop_duplicates(subset=[TASK_ID_COL])
    return tasks


if __name__ == "__main__":
    df = dd.read_parquet(SRC)
    print(f"Partitions: {df.npartitions}, approx rows: {len(df)}")

    inspect_schema(df)
    tasks = build_task_table(df)

    n_tasks = len(tasks)
    print(f"\nUnique tasks: {n_tasks}")

    TASKS_OUT.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(str(TASKS_OUT), write_index=False)
    print(f"Wrote task table -> {TASKS_OUT}")
