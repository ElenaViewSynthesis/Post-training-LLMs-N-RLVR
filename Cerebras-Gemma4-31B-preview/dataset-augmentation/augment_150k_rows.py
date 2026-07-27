"""Stage 6: stream original and accepted refined rows into an exact publication.

The current path consumes immutable accepted shards produced by
``stream_refinement_worker.py``.  It validates every refinement against its
source row, writes into a fresh staging directory, verifies physical Parquet
row counts and checksums, atomically promotes the completed local publication,
and optionally synchronizes a versioned destination.

Requires ``huggingface_hub`` with Bucket support (installed by this project).
The equivalent manual CLI commands are:

Sync commands:
    Upload:   hf buckets sync ./data hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k
    Download: hf buckets sync hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k ./local

Objective: preserve every source row and add exactly 150K refinements.
"""
import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import Any

import dask
import dask.dataframe as dd
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import sync_bucket
from publication_pipeline import preflight_publication, write_local_publication
from refinement_pipeline import load_source_manifest

load_dotenv()

DEFAULT_DATA_DIR = Path("~/pipeline/data")
ORIGINAL_SRC = "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/data/train-*-of-*.parquet"
VALIDATED_PATH = Path("~/pipeline/data/validated/validated_trajectories.parquet").expanduser()
TASKS_PATH     = Path("~/pipeline/data/tasks/").expanduser()
UPLOAD_DIR     = Path("~/pipeline/data/upload").expanduser()

HF_BUCKET = "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k"
ROWS_PER_SHARD = 5_000
EXPECTED_NEW_ROWS = 150_000
EXPECTED_TOTAL_ROWS = 250_000  # Legacy helper default; the streaming CLI derives it.


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data_dir = Path(
        os.getenv("PIPELINE_DATA_DIR", str(DEFAULT_DATA_DIR))
    ).expanduser()
    parser = argparse.ArgumentParser(
        description="Build an exact, versioned Stage 6 Parquet publication."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--original-source",
        default=os.getenv("ORIGINAL_DATASET_SOURCE", ORIGINAL_SRC),
    )
    parser.add_argument(
        "--refined-dir",
        type=Path,
        help="Accepted refinement shards (default: <data-dir>/refined/accepted)",
    )
    parser.add_argument(
        "--upload-dir",
        type=Path,
        help="Local publication root (default: <data-dir>/upload)",
    )
    parser.add_argument(
        "--hf-bucket",
        default=os.getenv("HF_PUBLICATION_BUCKET", HF_BUCKET),
    )
    parser.add_argument(
        "--expected-new-rows",
        type=_positive_int,
        default=EXPECTED_NEW_ROWS,
    )
    parser.add_argument(
        "--expected-total-rows",
        type=_positive_int,
        default=(
            int(os.environ["EXPECTED_TOTAL_ROWS"])
            if os.getenv("EXPECTED_TOTAL_ROWS")
            else None
        ),
        help=(
            "Optional fixed publication total. By default this is derived as "
            "the physical source row count plus --expected-new-rows."
        ),
    )
    parser.add_argument(
        "--rows-per-shard",
        type=_positive_int,
        default=ROWS_PER_SHARD,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser()
    args.refined_dir = (
        args.refined_dir or args.data_dir / "refined" / "accepted"
    ).expanduser()
    args.upload_dir = (args.upload_dir or args.data_dir / "upload").expanduser()
    return args


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
    if tasks[id_col].duplicated().any():
        raise ValueError(f"task table contains duplicate {id_col} values")
    if validated["variant_id"].duplicated().any():
        raise ValueError("validated data contains duplicate variant_id values")
    task_lookup = {row[id_col]: row.to_dict() for _, row in tasks.iterrows()}
    missing_tasks = sorted(set(validated["task_id"]).difference(task_lookup))
    if missing_tasks:
        raise ValueError(
            f"validated data contains {len(missing_tasks)} missing task lookups; "
            f"sample={missing_tasks[:3]}"
        )
    rows = []
    for _, rec in validated.iterrows():
        task = task_lookup[rec["task_id"]]
        row = {col: task.get(col) for col in original_cols}
        row["conversations"] = normalize_conversations(rec["conversations"])
        row["run_id"] = rec["variant_id"]
        row["trial_name"] = rec["variant_id"]
        row["is_synthetic_augmentation"] = True
        row["source_task_id"] = rec["task_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    source_manifest_path = args.refined_dir.parent / "source_manifest.json"
    source_manifest = load_source_manifest(source_manifest_path)
    if args.original_source not in {
        source_manifest.requested_source,
        source_manifest.resolved_source,
    }:
        raise ValueError(
            "--original-source conflicts with the refinement source manifest: "
            f"{args.original_source!r}"
        )
    preflight = preflight_publication(
        source_manifest.resolved_source,
        args.refined_dir,
        expected_new_rows=args.expected_new_rows,
        expected_total_rows=args.expected_total_rows,
        expected_source_identity_sha256=(
            source_manifest.source_identity_sha256
        ),
        expected_source_schema_sha256=source_manifest.source_schema_sha256,
    )
    print(
        f"Preflight: original={preflight.original_rows:,} "
        f"synthetic={preflight.synthetic_rows:,} total={preflight.total_rows:,} "
        f"publication={preflight.publication_id}"
    )
    if args.dry_run:
        print("Dry run complete; no local or remote files were written.")
        return 0

    publication_dir, created = write_local_publication(
        preflight,
        args.upload_dir,
        rows_per_shard=args.rows_per_shard,
    )
    action = "Created" if created else "Reused"
    print(f"{action} verified local publication: {publication_dir}")
    if args.no_sync:
        print("Remote synchronization disabled by --no-sync.")
        return 0

    destination = (
        f"{args.hf_bucket.rstrip('/')}/publications/{preflight.publication_id}"
    )
    print(f"Synchronizing versioned publication to {destination} ...")
    sync_bucket(str(publication_dir), destination)
    print(f"Publication synchronization complete: {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    load_dotenv(here / ".env", override=False)
    load_dotenv(here.parent / ".env", override=False)
    try:
        return run(parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - CLI boundary must fail synchronization
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
