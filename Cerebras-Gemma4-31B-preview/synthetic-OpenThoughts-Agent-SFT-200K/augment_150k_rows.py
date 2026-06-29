"""
Stage 6: Merge original 100K rows with accepted new synthetic rows, reshape
them into the original schema (preserving the `conversations` column structure),
and push the final ~250K dataset to HuggingFace Hub.

Upload path: local sharded parquet → HuggingFace Hub via huggingface_hub.
hf_transfer (C extension) is used for fast GCP EU CDN-backed uploads when
HF_HUB_ENABLE_HF_TRANSFER=1 is set in your environment.

Objective: OpenThoughts-Agent-SFT-100K → 250K by augmenting `conversations`.
"""
import os
import shutil
import tempfile
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()

ORIGINAL_SRC = "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet"
VALIDATED_PATH = Path("~/pipeline/data/validated/validated_trajectories.parquet").expanduser()
TASKS_PATH = Path("~/pipeline/data/tasks/").expanduser()

# Target HuggingFace dataset repo — set in .env or override here
HF_REPO_ID = os.environ["HF_DATASET_REPO_ID"]   # e.g. "your-username/OpenThoughts-Agent-SFT-250K"
HF_TOKEN   = os.environ["HF_TOKEN"]

# Shard size for parquet files written to HF Hub (100MB target per file)
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
    # ── Authenticate ──────────────────────────────────────────────────────────
    login(token=HF_TOKEN)
    api = HfApi()

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

    # ── Write sharded parquet to a temp dir, then upload to HF Hub ────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    data_dir = tmp_dir / "data"
    data_dir.mkdir()

    print(f"Writing sharded parquet to {data_dir} ...")
    combined.to_parquet(
        str(data_dir),
        write_index=False,
        engine="pyarrow",
        name_function=lambda i: f"train-{i:05d}-of-{combined.npartitions:05d}.parquet",
    )

    # ── Create HF dataset repo if it doesn't exist ────────────────────────────
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=True,
    )

    # ── Upload — hf_transfer uses GCP EU CDN when HF_HUB_ENABLE_HF_TRANSFER=1 ─
    print(f"Uploading to HuggingFace Hub: {HF_REPO_ID} ...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_in_repo="data",
        token=HF_TOKEN,
        commit_message="Add augmented 250K dataset (Gemma-4-31B synthetic conversations)",
    )

    shutil.rmtree(tmp_dir)
    print(f"Done. Dataset live at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
