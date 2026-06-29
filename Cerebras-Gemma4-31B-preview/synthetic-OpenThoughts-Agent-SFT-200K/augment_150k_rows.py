"""
Stage 6: Merge original 100K rows with accepted new synthetic rows, reshape
them into the original schema (preserving the `conversations` column structure),
and sync the final ~250K dataset to HuggingFace Bucket via `hf sync`.

Upload path:
    local sharded parquet → hf sync → hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k

Requires the `hf` CLI to be installed:
    uv tool install hf

Sync commands:
    Upload:   hf sync ./data hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k
    Download: hf sync hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k ./local

Objective: OpenThoughts-Agent-SFT-100K → 250K by augmenting `conversations`.
"""
import os
import subprocess
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ORIGINAL_SRC  = "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet"
VALIDATED_PATH = Path("~/pipeline/data/validated/validated_trajectories.parquet").expanduser()
TASKS_PATH     = Path("~/pipeline/data/tasks/").expanduser()
UPLOAD_DIR     = Path("~/pipeline/data/upload").expanduser()

HF_BUCKET = "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k"
ROWS_PER_SHARD = 5_000


def reshape_to_original_schema(validated: pd.DataFrame, tasks: pd.DataFrame, original_cols: list) -> pd.DataFrame:
    task_lookup = tasks.set_index("task_id")
    rows = []
    for _, rec in validated.iterrows():
        task = task_lookup.loc[rec["task_id"]]
        row = {col: task.get(col) for col in original_cols if col in task.index}
        row["conversations"] = rec["conversations"]  # synthetic conversations from Gemma-4-31B
        row["id"] = rec["variant_id"]
        row["is_synthetic_augmentation"] = True
        row["source_task_id"] = rec["task_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # ── Load & merge ──────────────────────────────────────────────────────────
    original = dd.read_parquet(ORIGINAL_SRC)
    original_cols = list(original.columns)

    validated = pd.read_parquet(VALIDATED_PATH)
    tasks = dd.read_parquet(str(TASKS_PATH)).compute()

    new_rows = reshape_to_original_schema(validated, tasks, original_cols)
    new_ddf = dd.from_pandas(new_rows, npartitions=max(1, len(new_rows) // ROWS_PER_SHARD))

    # Align columns — fill missing original columns with None
    for col in original_cols:
        if col not in new_ddf.columns:
            new_ddf[col] = None
    new_ddf = new_ddf[original.columns.tolist() + ["is_synthetic_augmentation", "source_task_id"]]
    original = original.assign(is_synthetic_augmentation=False, source_task_id=None)

    combined = dd.concat([original, new_ddf])
    print(f"Combined row count: {len(combined):,}")

    # ── Write sharded parquet to upload dir ───────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing sharded parquet to {UPLOAD_DIR} ...")
    combined.to_parquet(
        str(UPLOAD_DIR),
        write_index=False,
        engine="pyarrow",
        name_function=lambda i: f"train-{i:05d}-of-{combined.npartitions:05d}.parquet",
    )

    # ── Sync to HuggingFace Bucket via hf CLI ─────────────────────────────────
    print(f"Syncing to {HF_BUCKET} ...")
    subprocess.run(
        ["hf", "sync", str(UPLOAD_DIR), HF_BUCKET],
        check=True,
    )

    print(f"Done. Bucket: {HF_BUCKET}")
    print(f"Download with: hf sync {HF_BUCKET} ./local")


if __name__ == "__main__":
    main()
