"""Restore an isolated remote refinement run and perform an offline status drill."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from huggingface_hub import sync_bucket

from refinement_pipeline import (
    load_run_manifest,
    load_source_manifest,
    verify_refinement_state_inventory,
    verify_remote_refinement_state,
)
from stream_refinement_worker import async_main, parse_args as parse_worker_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a remote run into a fresh directory and run --status-only."
    )
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _status_worker_args(output_dir: Path) -> argparse.Namespace:
    source_manifest = load_source_manifest(output_dir / "source_manifest.json")
    run_manifest = load_run_manifest(output_dir / "run_manifest.json")
    config = run_manifest.generation_config
    required = {
        "concurrency",
        "request_batch_size",
        "max_output_tokens",
        "max_attempts_per_run",
        "timeout_seconds",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(
            f"restored run manifest lacks generation settings: {sorted(missing)}"
        )
    return parse_worker_args(
        [
            "--original-source",
            source_manifest.requested_source,
            "--output-dir",
            str(output_dir),
            "--target-rows",
            str(run_manifest.target_rows),
            "--model",
            run_manifest.model,
            "--concurrency",
            str(config["concurrency"]),
            "--request-batch-size",
            str(config["request_batch_size"]),
            "--max-output-tokens",
            str(config["max_output_tokens"]),
            "--max-attempts-per-run",
            str(config["max_attempts_per_run"]),
            "--timeout-seconds",
            str(config["timeout_seconds"]),
            "--status-only",
            "--no-sync",
        ]
    )


def restore_refinement_run(remote_run: str, output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        raise FileExistsError(
            f"restore destination must not already exist: {output_dir}"
        )
    verify_remote_refinement_state(remote_run)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sync_bucket(remote_run, str(output_dir))
    verify_refinement_state_inventory(output_dir)
    run_manifest = load_run_manifest(output_dir / "run_manifest.json")
    expected_suffix = f"/runs/{run_manifest.run_instance_id}"
    if not remote_run.rstrip("/").endswith(expected_suffix):
        raise ValueError(
            "remote run prefix does not match the restored run-instance UUID"
        )
    status_code = asyncio.run(async_main(_status_worker_args(output_dir)))
    if status_code != 0:
        raise RuntimeError(f"restored --status-only returned {status_code}")
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    return {
        "remote_run": remote_run.rstrip("/"),
        "output_dir": str(output_dir),
        "run_instance_id": run_manifest.run_instance_id,
        "accepted_rows": progress.get("completed_rows"),
        "target_rows": run_manifest.target_rows,
        "status": "verified",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = restore_refinement_run(args.remote_run, args.output_dir)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
