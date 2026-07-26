"""
Stage 6: Merge original 100K rows with accepted new synthetic rows, reshape
them into the original schema (preserving the `conversations` column structure),
and sync the final ~250K dataset with ``huggingface_hub.sync_bucket``.

Upload path:
    local sharded parquet → sync_bucket → hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k

Requires ``huggingface_hub`` with Bucket support (installed by this project).
The equivalent manual CLI commands are:

Sync commands:
    Upload:   hf buckets sync ./data hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k
    Download: hf buckets sync hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k ./local

Objective: OpenThoughts-Agent-SFT-100K → 250K total (100K original + 150K synthetic) by augmenting `conversations`.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dask
import dask.dataframe as dd
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import sync_bucket

load_dotenv()

ORIGINAL_SRC  = "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet"
VALIDATED_PATH = Path("~/pipeline/data/validated/validated_trajectories.parquet").expanduser()
TASKS_PATH     = Path("~/pipeline/data/tasks/").expanduser()
UPLOAD_DIR     = Path("~/pipeline/data/upload").expanduser()

HF_BUCKET = "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k"
ROWS_PER_SHARD = 5_000
EXPECTED_NEW_ROWS = 150_000
EXPECTED_TOTAL_ROWS = 250_000


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    """Return the nested role/content representation used by the source parquet.

    PyArrow-backed pandas reads may expose a parquet list as a numpy array, while
    freshly generated values are usually Python lists.  Normalize both forms
    before handing the frame to Dask, and fail instead of silently stringifying
    an unexpected value.
    """
    if isinstance(value, list):
        turns = value
    elif isinstance(value, tuple):
        turns = list(value)
    elif hasattr(value, "tolist"):
        turns = value.tolist()
    else:
        raise TypeError(
            "conversations must be a list-like sequence of role/content mappings"
        )

    if not isinstance(turns, list):
        raise TypeError("conversations did not normalize to a list")

    normalized = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise TypeError(f"conversation turn {index} is not a mapping")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError(
                f"conversation turn {index} must contain string role and content"
            )
        # Match the source struct's field order as well as its field names.
        normalized.append({"content": content, "role": role})
    return normalized


def read_original_arrow_schema(source: str) -> pa.Schema:
    """Read the exact Arrow schema from the first parquet file in ``source``."""
    filesystem, path_pattern = fsspec.core.url_to_fs(source)
    matches = sorted(filesystem.glob(path_pattern))
    if not matches:
        raise FileNotFoundError(f"no source parquet files matched {source}")

    with filesystem.open(matches[0], "rb") as stream:
        schema = pq.read_schema(stream).remove_metadata()
    if "conversations" not in schema.names:
        raise ValueError("source parquet schema has no conversations field")
    return schema


def build_output_arrow_schema(original_schema: pa.Schema) -> pa.Schema:
    """Extend the source schema with the two augmentation metadata columns."""
    if "task" not in original_schema.names:
        raise ValueError("source parquet schema has no task field")
    if "is_synthetic_augmentation" in original_schema.names:
        raise ValueError("source schema already has is_synthetic_augmentation")
    if "source_task_id" in original_schema.names:
        raise ValueError("source schema already has source_task_id")

    return pa.schema(
        [
            *original_schema,
            pa.field("is_synthetic_augmentation", pa.bool_(), nullable=False),
            pa.field(
                "source_task_id", original_schema.field("task").type, nullable=True
            ),
        ]
    )


def dataframe_from_synthetic_rows(rows: pd.DataFrame) -> dd.DataFrame:
    """Partition synthetic rows without Dask converting nested objects to text."""
    npartitions = max(1, len(rows) // ROWS_PER_SHARD)
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(rows, npartitions=npartitions)


def assert_output_arrow_schema(output_dir: Path, expected: pa.Schema) -> int:
    """Reload every generated shard and require the declared output schema."""
    shards = sorted(output_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no output parquet shards found in {output_dir}")

    for shard in shards:
        actual = pq.read_schema(shard).remove_metadata()
        if not actual.equals(expected, check_metadata=False):
            raise TypeError(
                f"output schema mismatch in {shard}:\n"
                f"expected:\n{expected}\nactual:\n{actual}"
            )
    return len(shards)


def require_exact_row_counts(
    original_count: int,
    synthetic_count: int,
    *,
    expected_new_rows: int = EXPECTED_NEW_ROWS,
    expected_total_rows: int = EXPECTED_TOTAL_ROWS,
) -> int:
    """Require the Stage 6 inputs to produce the exact publication target."""
    if synthetic_count != expected_new_rows:
        raise ValueError(
            "validated synthetic row count must be exactly "
            f"{expected_new_rows:,}; found {synthetic_count:,}"
        )

    total_count = original_count + synthetic_count
    if total_count != expected_total_rows:
        raise ValueError(
            f"final row count must be exactly {expected_total_rows:,}; "
            f"found {original_count:,} original + {synthetic_count:,} synthetic "
            f"= {total_count:,}"
        )
    return total_count


def reshape_to_original_schema(validated: pd.DataFrame, tasks: pd.DataFrame, original_cols: list) -> pd.DataFrame:
    id_col = next((c for c in ("task_id", "id", "task", "run_id") if c in tasks.columns), tasks.columns[0])
    task_lookup = {row[id_col]: row.to_dict() for _, row in tasks.iterrows()}
    rows = []
    for _, rec in validated.iterrows():
        task = task_lookup.get(rec["task_id"], {})
        row = {col: task.get(col) for col in original_cols}
        row["conversations"] = normalize_conversations(rec["conversations"])
        row["run_id"] = rec["variant_id"]
        row["trial_name"] = rec["variant_id"]
        row["is_synthetic_augmentation"] = True
        row["source_task_id"] = rec["task_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # ── Load & merge ──────────────────────────────────────────────────────────
    original_arrow_schema = read_original_arrow_schema(ORIGINAL_SRC)
    output_arrow_schema = build_output_arrow_schema(original_arrow_schema)
    original = dd.read_parquet(ORIGINAL_SRC)
    original_cols = list(original.columns)
    if original_cols != original_arrow_schema.names:
        raise ValueError(
            "Dask columns do not match the source Arrow schema: "
            f"{original_cols} != {original_arrow_schema.names}"
        )

    validated = pd.read_parquet(VALIDATED_PATH)
    expected_combined_count = require_exact_row_counts(len(original), len(validated))
    tasks = dd.read_parquet(str(TASKS_PATH)).compute()

    new_rows = reshape_to_original_schema(validated, tasks, original_cols)
    new_ddf = dataframe_from_synthetic_rows(new_rows)

    # Align columns — fill missing original columns with None
    for col in original_cols:
        if col not in new_ddf.columns:
            new_ddf[col] = None
    new_ddf = new_ddf[output_arrow_schema.names]
    original = original.assign(is_synthetic_augmentation=False, source_task_id=None)

    combined = dd.concat([original, new_ddf])[output_arrow_schema.names]
    combined_count = len(combined)
    if combined_count != expected_combined_count:
        raise RuntimeError(
            "Dask concatenation changed the expected row count: "
            f"{combined_count:,} != {expected_combined_count:,}"
        )
    print(f"Combined row count: {combined_count:,}")

    # ── Write sharded parquet to upload dir ───────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing sharded parquet to {UPLOAD_DIR} ...")
    combined.to_parquet(
        str(UPLOAD_DIR),
        write_index=False,
        engine="pyarrow",
        schema=output_arrow_schema,
        name_function=lambda i: f"train-{i:05d}-of-{combined.npartitions:05d}.parquet",
    )
    shard_count = assert_output_arrow_schema(UPLOAD_DIR, output_arrow_schema)
    print(f"Verified Arrow schema across {shard_count} parquet shards")

    # ── Sync to HuggingFace Bucket ────────────────────────────────────────────
    print(f"Syncing to {HF_BUCKET} ...")
    sync_bucket(str(UPLOAD_DIR), HF_BUCKET)

    print(f"Done. Bucket: {HF_BUCKET}")
    print(f"Download with: hf buckets sync {HF_BUCKET} ./local")


if __name__ == "__main__":
    main()
