"""Prepare, execute, and report isolated pilots for the streaming worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from refinement_pipeline import (
    ensure_source_manifest,
    iter_source_batches_with_coordinates,
    load_source_identities,
    pin_source_revision,
    source_identity_for_row,
)
from stream_refinement_worker import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORIGINAL_SOURCE,
    async_main as run_streaming_worker,
    load_environment,
    parse_args as parse_worker_args,
)


PILOT_MARKER_VERSION = 1
PILOT_REPORT_VERSION = 1


def _pilot_size(value: str) -> int:
    parsed = int(value)
    if not 10 <= parsed <= 50:
        raise argparse.ArgumentTypeError("must be between 10 and 50")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_dir = Path(os.getenv("PIPELINE_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    parser = argparse.ArgumentParser(
        description="Run an isolated 10-50 request pilot through the production worker."
    )
    parser.add_argument("--source", default=DEFAULT_ORIGINAL_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--sample-size", type=_pilot_size, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL_ID", "gemini-3.6-flash"))
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--request-batch-size", type=_positive_int, default=10)
    parser.add_argument("--max-attempts-per-run", type=_positive_int, default=3)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=4096)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=180)
    parser.add_argument(
        "--max-provider-requests",
        type=_positive_int,
        help="Exact generation-call cap (default: sample size).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to real Gemini calls after preparing the isolated fixture.",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from an existing pilot without provider access.",
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser()
    args.pilot_dir = (
        args.pilot_dir
        or args.data_dir / "pilots" / f"pilot-{args.seed}-{args.sample_size}"
    ).expanduser()
    args.max_provider_requests = args.max_provider_requests or args.sample_size
    if args.request_batch_size < args.concurrency:
        parser.error("--request-batch-size must be at least --concurrency")
    return args


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_isolated_pilot_directory(data_dir: Path, pilot_dir: Path) -> None:
    pilot = pilot_dir.resolve()
    forbidden = {
        (data_dir / "refined").resolve(),
        (data_dir / "upload").resolve(),
        (data_dir / "source-snapshot").resolve(),
    }
    if pilot in forbidden:
        raise ValueError(f"pilot directory overlaps production state: {pilot_dir}")


def _selection_key(seed: int, source_record_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"pilot-v1\0{seed}\0{source_record_id}".encode()).digest()
    return digest, source_record_id


def prepare_pilot_snapshot(
    source: str,
    pilot_dir: Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Select a deterministic bounded-memory fixture and write manifest v2."""
    marker_path = pilot_dir / "pilot_marker.json"
    fixture_dir = pilot_dir / "source"
    output_dir = pilot_dir / "refined"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("version") != PILOT_MARKER_VERSION
            or marker.get("requested_source") != source
            or marker.get("sample_size") != sample_size
            or marker.get("seed") != seed
        ):
            raise ValueError("existing pilot marker does not match this request")
        schema, identities = load_source_identities(str(fixture_dir))
        ensure_source_manifest(
            output_dir / "source_manifest.json",
            requested_source=str(fixture_dir),
            resolved_source=str(fixture_dir),
            schema=schema,
            identities=identities,
        )
        return marker
    if pilot_dir.exists() and any(pilot_dir.iterdir()):
        raise ValueError(
            "pilot directory is non-empty but has no matching pilot marker"
        )

    resolved_source = pin_source_revision(source)
    selected: list[tuple[tuple[bytes, str], dict[str, Any]]] = []
    source_schema: pa.Schema | None = None
    rows_seen = 0
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        resolved_source,
        batch_size=256,
    ):
        if source_schema is None:
            source_schema = batch.schema.remove_metadata()
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            rows_seen += 1
            identity = source_identity_for_row(source_file, index, row)
            candidate = (_selection_key(seed, identity.source_record_id), row)
            if len(selected) < sample_size:
                selected.append(candidate)
                selected.sort(key=lambda item: item[0], reverse=True)
            elif candidate[0] < selected[0][0]:
                selected[0] = candidate
                selected.sort(key=lambda item: item[0], reverse=True)
    if source_schema is None or rows_seen < sample_size:
        raise ValueError(
            f"source contains {rows_seen} rows; pilot requires {sample_size}"
        )
    selected.sort(key=lambda item: item[0])

    pilot_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir()
    output_dir.mkdir()
    destination = fixture_dir / "pilot-00000.parquet"
    handle = tempfile.NamedTemporaryFile(
        prefix=".pilot-", suffix=".parquet.tmp", dir=fixture_dir, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        pq.write_table(
            pa.Table.from_pylist([row for _, row in selected], schema=source_schema),
            temporary,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    schema, identities = load_source_identities(str(fixture_dir))
    source_manifest = ensure_source_manifest(
        output_dir / "source_manifest.json",
        requested_source=str(fixture_dir),
        resolved_source=str(fixture_dir),
        schema=schema,
        identities=identities,
    )
    marker = {
        "version": PILOT_MARKER_VERSION,
        "requested_source": source,
        "resolved_source": resolved_source,
        "sample_size": sample_size,
        "seed": seed,
        "rows_scanned": rows_seen,
        "fixture_source": str(fixture_dir),
        "source_manifest_version": source_manifest.version,
        "fixture_content_sha256": source_manifest.source_content_sha256,
    }
    _atomic_json(marker_path, marker)
    return marker


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed pilot attempt record at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"pilot attempt record {line_number} is not an object"
                )
            records.append(value)
    return records


def build_pilot_report(pilot_dir: Path, *, sample_pairs: int = 5) -> dict[str, Any]:
    marker = json.loads((pilot_dir / "pilot_marker.json").read_text(encoding="utf-8"))
    output_dir = pilot_dir / "refined"
    attempts = _read_jsonl(output_dir / "attempts.jsonl")
    accepted_rows: list[dict[str, Any]] = []
    for shard in sorted((output_dir / "accepted").glob("accepted-*.parquet")):
        accepted_rows.extend(pq.read_table(shard).to_pylist())

    fixture_source = str(pilot_dir / "source")
    source_by_id: dict[str, dict[str, Any]] = {}
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        fixture_source
    ):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            identity = source_identity_for_row(source_file, index, row)
            source_by_id[identity.source_record_id] = row

    rejection_codes: Counter[str] = Counter()
    provider_errors: Counter[str] = Counter()
    attempts_per_slot: Counter[str] = Counter()
    for attempt in attempts:
        synthetic_id = attempt.get("synthetic_id")
        if isinstance(synthetic_id, str):
            attempts_per_slot[synthetic_id] += 1
        rejection_codes.update(
            code
            for code in attempt.get("rejection_codes", [])
            if isinstance(code, str)
        )
        if attempt.get("status") == "provider_error":
            provider_errors[str(attempt.get("error_code") or "unknown")] += 1

    pairs = []
    for row in sorted(accepted_rows, key=lambda item: str(item.get("run_id"))):
        source_record_id = row.get("source_record_id")
        source_row = source_by_id.get(str(source_record_id))
        if source_row is None:
            raise ValueError(
                f"accepted pilot row has unknown source record: {source_record_id!r}"
            )
        pairs.append(
            {
                "synthetic_id": row.get("run_id"),
                "source_record_id": source_record_id,
                "source_conversations": source_row.get("conversations"),
                "refined_conversations": row.get("conversations"),
            }
        )
        if len(pairs) >= sample_pairs:
            break

    target = int(marker["sample_size"])
    report = {
        "version": PILOT_REPORT_VERSION,
        "pilot": marker,
        "target_slots": target,
        "accepted_slots": len(accepted_rows),
        "acceptance_rate": len(accepted_rows) / target,
        "provider_requests": len(attempts),
        "rejection_codes": dict(sorted(rejection_codes.items())),
        "provider_errors": dict(sorted(provider_errors.items())),
        "duplicate_collisions": rejection_codes["duplicate_conversation"],
        "attempts_per_slot": dict(sorted(attempts_per_slot.items())),
        "sample_pairs": pairs,
    }
    _atomic_json(pilot_dir / "pilot_report.json", report)
    return report


def _worker_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--original-source",
        str(args.pilot_dir / "source"),
        "--output-dir",
        str(args.pilot_dir / "refined"),
        "--target-rows",
        str(args.sample_size),
        "--model",
        args.model,
        "--concurrency",
        str(args.concurrency),
        "--request-batch-size",
        str(args.request_batch_size),
        "--max-attempts-per-run",
        str(args.max_attempts_per_run),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-provider-requests",
        str(args.max_provider_requests),
        "--no-sync",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _assert_isolated_pilot_directory(args.data_dir, args.pilot_dir)
        if not args.report_only:
            prepare_pilot_snapshot(
                args.source,
                args.pilot_dir,
                sample_size=args.sample_size,
                seed=args.seed,
            )
        if args.execute:
            load_environment()
            worker_exit = asyncio.run(
                run_streaming_worker(parse_worker_args(_worker_argv(args)))
            )
        else:
            worker_exit = 0
        report = build_pilot_report(args.pilot_dir)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute and not args.report_only:
        print(
            "Pilot prepared without provider calls. Re-run with --execute to opt in.",
            file=sys.stderr,
        )
    return worker_exit


if __name__ == "__main__":
    raise SystemExit(main())
