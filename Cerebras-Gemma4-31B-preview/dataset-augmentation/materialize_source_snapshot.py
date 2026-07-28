"""Create or verify the immutable local source used by generation/publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refinement_pipeline import (
    load_local_source_snapshot,
    materialize_local_source_snapshot,
)
from stream_refinement_worker import DEFAULT_ORIGINAL_SOURCE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, checksum, and seal a local Parquet source snapshot."
    )
    parser.add_argument("--source", default=DEFAULT_ORIGINAL_SOURCE)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing snapshot without downloading anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_only:
            snapshot = load_local_source_snapshot(args.snapshot_dir.expanduser())
        else:
            snapshot = materialize_local_source_snapshot(
                args.source,
                args.snapshot_dir.expanduser(),
            )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "snapshot_dir": str(args.snapshot_dir.expanduser()),
                "snapshot_instance_id": snapshot.snapshot_instance_id,
                "resolved_source": snapshot.resolved_source,
                "source_content_sha256": snapshot.source_content_sha256,
                "file_count": len(snapshot.files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
