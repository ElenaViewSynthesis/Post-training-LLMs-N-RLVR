"""Generate non-production scale fixtures and measure the streaming pipeline.

Examples:
    uv run python benchmark_streaming_pipeline.py --accepted-rows 10000
    uv run python benchmark_streaming_pipeline.py --accepted-rows 100000
    uv run python benchmark_streaming_pipeline.py --accepted-rows 225000

The fixture contains no API output or production data. Temporary files are
removed unless ``--keep`` or ``--work-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from publication_pipeline import preflight_publication, write_local_publication
from refinement_pipeline import (
    build_augmented_schema,
    build_refined_row,
    choose_secondary_source_ids,
    ensure_run_manifest,
    ensure_source_manifest,
    iter_source_batches_with_coordinates,
    load_source_identities,
    load_source_manifest,
    refinement_slots_for_source,
    source_identity_for_row,
    write_accepted_shard,
    write_refinement_state_inventory,
)


CONVERSATIONS_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("content", pa.string()),
            pa.field("role", pa.string()),
        ]
    )
)
SOURCE_SCHEMA = pa.schema(
    [
        pa.field("conversations", CONVERSATIONS_TYPE),
        pa.field("model", pa.string()),
        pa.field("model_provider", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("task", pa.string()),
        pa.field("trial_name", pa.string()),
    ]
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-rows", type=_positive_int, default=10_000)
    parser.add_argument("--source-rows", type=_positive_int)
    parser.add_argument("--content-bytes", type=_positive_int, default=2_048)
    parser.add_argument("--fixture-shard-rows", type=_positive_int, default=5_000)
    parser.add_argument("--publication-shard-rows", type=_positive_int, default=5_000)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def peak_rss_bytes() -> int:
    # Linux reports KiB. macOS reports bytes, but this project runs under WSL.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1_024


def fixture_row(index: int, content_bytes: int) -> dict:
    padding = "x" * max(1, content_bytes - 64)
    source_id = f"source-{index:08d}"
    return {
        "conversations": [
            {"content": f"Inspect task {index}: {padding}", "role": "user"},
            {"content": f"Task {index} inspected", "role": "assistant"},
        ],
        "model": "fixture-source",
        "model_provider": "fixture",
        "run_id": source_id,
        "task": f"task-{index:08d}",
        "trial_name": source_id,
    }


def write_source_fixture(
    source_dir: Path,
    *,
    source_rows: int,
    content_bytes: int,
    shard_rows: int,
) -> None:
    source_dir.mkdir(parents=True)
    shard_count = math.ceil(source_rows / shard_rows)
    for shard_index, start in enumerate(range(0, source_rows, shard_rows)):
        stop = min(start + shard_rows, source_rows)
        rows = [fixture_row(index, content_bytes) for index in range(start, stop)]
        path = source_dir / (
            f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA), path)


def write_refinement_fixture(
    source_dir: Path,
    accepted_dir: Path,
    *,
    target_rows: int,
    shard_rows: int,
) -> None:
    original_schema, identities = load_source_identities(str(source_dir))
    source_manifest = ensure_source_manifest(
        accepted_dir.parent / "source_manifest.json",
        requested_source=str(source_dir),
        resolved_source=str(source_dir),
        schema=original_schema,
        identities=identities,
    )
    ensure_run_manifest(
        accepted_dir.parent / "run_manifest.json",
        source_manifest=source_manifest,
        target_rows=target_rows,
        model="fixture-refiner",
        generation_config={
            "concurrency": 1,
            "request_batch_size": shard_rows,
            "max_output_tokens": 1,
            "max_attempts_per_run": 1,
            "timeout_seconds": 1,
        },
    )
    secondary = choose_secondary_source_ids(list(identities), target_rows)
    output_schema = build_augmented_schema(original_schema)
    buffer: list[dict] = []
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        str(source_dir)
    ):
        for index, source_row in enumerate(batch.to_pylist(), start=row_offset):
            identity = source_identity_for_row(source_file, index, source_row)
            for slot in refinement_slots_for_source(identity, secondary):
                buffer.append(
                    build_refined_row(
                        source_row,
                        slot,
                        [
                            {
                                "role": "user",
                                "content": (
                                    f"Refine {source_row['task']} using approach "
                                    f"{slot.refinement_index}"
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    f"Refined {source_row['task']} successfully "
                                    f"using approach {slot.refinement_index}"
                                ),
                            },
                        ],
                        original_schema,
                        model="fixture-refiner",
                        provider="fixture",
                    )
                )
                if len(buffer) == shard_rows:
                    write_accepted_shard(accepted_dir, buffer, output_schema)
                    buffer.clear()
    if buffer:
        write_accepted_shard(accepted_dir, buffer, output_schema)
    write_refinement_state_inventory(accepted_dir.parent)


def execute(root: Path, args: argparse.Namespace) -> dict:
    source_rows = args.source_rows or math.ceil(args.accepted_rows / 1.5)
    if not source_rows <= args.accepted_rows <= 2 * source_rows:
        raise ValueError(
            "accepted rows must be between source rows and twice source rows"
        )
    source_dir = root / "source"
    accepted_dir = root / "refined" / "accepted"
    upload_dir = root / "upload"
    timings: dict[str, float] = {}

    started = time.perf_counter()
    write_source_fixture(
        source_dir,
        source_rows=source_rows,
        content_bytes=args.content_bytes,
        shard_rows=args.fixture_shard_rows,
    )
    timings["generate_source_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    write_refinement_fixture(
        source_dir,
        accepted_dir,
        target_rows=args.accepted_rows,
        shard_rows=args.fixture_shard_rows,
    )
    timings["generate_refinements_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    source_manifest = load_source_manifest(
        accepted_dir.parent / "source_manifest.json"
    )
    preflight = preflight_publication(
        str(source_dir),
        accepted_dir,
        expected_new_rows=args.accepted_rows,
        expected_total_rows=source_rows + args.accepted_rows,
        expected_source_identity_sha256=(
            source_manifest.source_identity_sha256
        ),
        expected_source_schema_sha256=source_manifest.source_schema_sha256,
        expected_source_content_sha256=source_manifest.source_content_sha256,
    )
    timings["preflight_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    publication_dir, _ = write_local_publication(
        preflight,
        upload_dir,
        rows_per_shard=args.publication_shard_rows,
    )
    timings["publication_seconds"] = time.perf_counter() - started

    return {
        "source_rows": source_rows,
        "accepted_rows": args.accepted_rows,
        "published_rows": source_rows + args.accepted_rows,
        "content_bytes_per_source": args.content_bytes,
        "source_bytes": directory_bytes(source_dir),
        "accepted_bytes": directory_bytes(accepted_dir),
        "publication_bytes": directory_bytes(publication_dir),
        "peak_rss_bytes": peak_rss_bytes(),
        "publication_id": preflight.publication_id,
        "timings": timings,
    }


def main() -> int:
    args = parse_args()
    if args.work_dir is not None:
        root = args.work_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        report = execute(root, args)
        (root / "benchmark_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="refinement-scale-kept-"))
        report = execute(root, args)
        (root / "benchmark_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Kept fixture at {root}")
        return 0

    temporary = tempfile.TemporaryDirectory(prefix="refinement-scale-")
    root = Path(temporary.name)
    report = execute(root, args)
    print(json.dumps(report, indent=2, sort_keys=True))
    temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
