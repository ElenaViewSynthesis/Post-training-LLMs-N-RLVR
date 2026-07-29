"""Incrementally checkpoint sealed Codex refinement rows to an HF bucket.

The production worker remains synchronization-free.  This sidecar observes its
checksum-sealed state and publishes only newly accepted immutable Parquet
shards.  A checkpoint is committed after every configured accepted-row
threshold (10 by default), and local state advances only after remote read-back
verification succeeds.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq

from refinement_pipeline import verify_refinement_state_inventory


DEFAULT_HF_BUCKET = (
    "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/refined"
)
SIDECAR_STATE_VERSION = 1
REMOTE_CHECKPOINT_VERSION = 1


class CheckpointNotReady(RuntimeError):
    """The worker is between atomic state-sealing steps; retry shortly."""


@dataclass(frozen=True)
class SealedSnapshot:
    run_instance_id: str
    source_content_sha256: str
    completed_rows: int
    target_rows: int
    shards: dict[str, dict[str, Any]]
    accepted_inventory_sha256: str
    root_manifests: dict[str, dict[str, Any]]


SyncDirectory = Callable[[Path, str], None]
VerifyRemoteFiles = Callable[[str, Mapping[str, Mapping[str, Any]]], None]


def load_environment() -> None:
    from dotenv import load_dotenv

    here = Path(__file__).resolve().parent
    load_dotenv(here / ".env", override=False)
    load_dotenv(here.parent / ".env", override=False)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_dir = Path(os.getenv("PIPELINE_DATA_DIR", "~/pipeline/data")).expanduser()
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint sealed Codex accepted rows to an isolated Hugging Face "
            "bucket namespace."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir / "codex-refined",
        help="Live Codex refinement-state directory.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Sidecar-only durable state (default: <output-dir>-hf-checkpoints).",
    )
    parser.add_argument(
        "--hf-bucket",
        default=os.getenv("HF_REFINEMENT_BUCKET", DEFAULT_HF_BUCKET),
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=_positive_int,
        default=10,
    )
    parser.add_argument("--poll-seconds", type=_positive_int, default=15)
    parser.add_argument("--retry-seconds", type=_positive_int, default=60)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Attempt one checkpoint and exit instead of monitoring continuously.",
    )
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.state_dir = (
        args.state_dir.expanduser().resolve()
        if args.state_dir is not None
        else args.output_dir.with_name(f"{args.output_dir.name}-hf-checkpoints")
    )
    if not isinstance(args.hf_bucket, str) or not args.hf_bucket.strip():
        parser.error("--hf-bucket must be a non-empty hf:// bucket path")
    if not args.hf_bucket.startswith("hf://buckets/"):
        parser.error("--hf-bucket must start with 'hf://buckets/'")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value


def _accepted_inventory_sha256(shards: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(shards):
        entry = shards[relative]
        digest.update(
            (
                f"{relative}\0{entry['size']}\0{entry['sha256']}\0"
                f"{entry['rows']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _verified_entry_bytes(
    output_dir: Path,
    relative: str,
    entries: Mapping[str, Mapping[str, Any]],
) -> bytes:
    entry = entries.get(relative)
    if entry is None:
        raise CheckpointNotReady(f"sealed state does not include {relative}")
    data = (output_dir / relative).read_bytes()
    if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise CheckpointNotReady(f"{relative} changed while reading sealed state")
    return data


def load_sealed_snapshot(
    output_dir: Path,
    known_shards: Mapping[str, Mapping[str, Any]],
) -> SealedSnapshot:
    """Read one checksum-consistent worker snapshot without locking the worker."""
    complete_before = _sha256(output_dir / "complete.json")
    try:
        entries = verify_refinement_state_inventory(output_dir, verify_hashes=False)
    except (OSError, ValueError) as exc:
        raise CheckpointNotReady(str(exc)) from exc

    progress = json.loads(
        _verified_entry_bytes(output_dir, "progress.json", entries)
    )
    run_manifest = json.loads(
        _verified_entry_bytes(output_dir, "run_manifest.json", entries)
    )
    _verified_entry_bytes(output_dir, "source_manifest.json", entries)
    if not isinstance(progress, dict) or not isinstance(run_manifest, dict):
        raise CheckpointNotReady("progress or run manifest is malformed")

    run_instance_id = run_manifest.get("run_instance_id")
    completed_rows = progress.get("completed_rows")
    target_rows = progress.get("target_rows")
    if (
        not isinstance(run_instance_id, str)
        or progress.get("run_instance_id") != run_instance_id
        or not isinstance(completed_rows, int)
        or completed_rows < 0
        or not isinstance(target_rows, int)
        or target_rows <= 0
        or completed_rows > target_rows
    ):
        raise CheckpointNotReady("progress does not match the immutable run")

    current_entries = {
        relative: entry
        for relative, entry in entries.items()
        if relative.startswith("accepted/")
        and relative.endswith(".parquet")
    }
    missing_known = sorted(set(known_shards).difference(current_entries))
    if missing_known:
        raise ValueError(
            "local accepted state rolled back behind its synchronized checkpoint: "
            f"{missing_known[:3]}"
        )

    shards: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for relative in sorted(current_entries):
        entry = current_entries[relative]
        known = known_shards.get(relative)
        if known is not None:
            if (
                known.get("size") != entry["size"]
                or known.get("sha256") != entry["sha256"]
                or not isinstance(known.get("rows"), int)
                or known["rows"] <= 0
            ):
                raise ValueError(f"accepted shard changed after synchronization: {relative}")
            row_count = known["rows"]
        else:
            path = output_dir / relative
            if _sha256(path) != entry["sha256"]:
                raise CheckpointNotReady(
                    f"accepted shard changed while reading sealed state: {relative}"
                )
            row_count = pq.ParquetFile(path).metadata.num_rows
            if row_count <= 0:
                raise ValueError(f"accepted shard is empty: {relative}")
        shards[relative] = {
            "size": entry["size"],
            "sha256": entry["sha256"],
            "rows": row_count,
        }
        total_rows += row_count

    if total_rows != completed_rows:
        raise CheckpointNotReady(
            "sealed progress count does not match accepted Parquet rows: "
            f"{completed_rows} != {total_rows}"
        )
    if complete_before != _sha256(output_dir / "complete.json"):
        raise CheckpointNotReady("worker committed a newer snapshot while reading")

    source_content_sha256 = run_manifest.get("source_content_sha256")
    if not isinstance(source_content_sha256, str):
        raise ValueError("run manifest has no source content digest")
    root_manifests = {
        relative: {
            "size": entries[relative]["size"],
            "sha256": entries[relative]["sha256"],
        }
        for relative in ("source_manifest.json", "run_manifest.json")
    }
    return SealedSnapshot(
        run_instance_id=run_instance_id,
        source_content_sha256=source_content_sha256,
        completed_rows=completed_rows,
        target_rows=target_rows,
        shards=shards,
        accepted_inventory_sha256=_accepted_inventory_sha256(shards),
        root_manifests=root_manifests,
    )


def remote_checkpoint_destination(bucket: str, run_instance_id: str) -> str:
    return (
        f"{bucket.rstrip('/')}/checkpoints/runs/{run_instance_id}"
    )


def _empty_state(
    snapshot: SealedSnapshot,
    remote_run: str,
    checkpoint_rows: int,
) -> dict[str, Any]:
    return {
        "version": SIDECAR_STATE_VERSION,
        "run_instance_id": snapshot.run_instance_id,
        "source_content_sha256": snapshot.source_content_sha256,
        "remote_run": remote_run,
        "checkpoint_rows": checkpoint_rows,
        "last_checkpoint_threshold": 0,
        "synced_completed_rows": 0,
        "last_checkpoint": None,
        "accepted_inventory_sha256": None,
        "shards": {},
    }


def load_sidecar_state(
    path: Path,
    snapshot: SealedSnapshot,
    remote_run: str,
    checkpoint_rows: int,
) -> dict[str, Any]:
    if not path.exists():
        return _empty_state(snapshot, remote_run, checkpoint_rows)
    state = _read_json(path)
    expected = {
        "version": SIDECAR_STATE_VERSION,
        "run_instance_id": snapshot.run_instance_id,
        "source_content_sha256": snapshot.source_content_sha256,
        "remote_run": remote_run,
        "checkpoint_rows": checkpoint_rows,
    }
    conflicts = [key for key, value in expected.items() if state.get(key) != value]
    if conflicts:
        raise ValueError(
            "sidecar state conflicts with the active immutable run: "
            + ", ".join(conflicts)
        )
    if (
        not isinstance(state.get("last_checkpoint_threshold"), int)
        or state["last_checkpoint_threshold"] < 0
        or not isinstance(state.get("synced_completed_rows"), int)
        or state["synced_completed_rows"] < 0
        or not isinstance(state.get("shards"), dict)
    ):
        raise ValueError("sidecar state has invalid counters or shard inventory")
    shards = state["shards"]
    for relative, entry in shards.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith("accepted/accepted-")
            or not relative.endswith(".parquet")
            or not isinstance(entry, dict)
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or not isinstance(entry.get("rows"), int)
            or entry["rows"] <= 0
        ):
            raise ValueError(f"sidecar state has an invalid shard entry: {relative!r}")
    if shards:
        if (
            sum(entry["rows"] for entry in shards.values())
            != state["synced_completed_rows"]
            or state.get("accepted_inventory_sha256")
            != _accepted_inventory_sha256(shards)
            or not isinstance(state.get("last_checkpoint"), str)
            or not state["last_checkpoint"]
            or state["last_checkpoint_threshold"] <= 0
            or state["last_checkpoint_threshold"] % checkpoint_rows
            or not (
                state["last_checkpoint_threshold"]
                <= state["synced_completed_rows"]
                < state["last_checkpoint_threshold"] + checkpoint_rows
            )
        ):
            raise ValueError("sidecar state checkpoint metadata is inconsistent")
    elif any(
        (
            state["last_checkpoint_threshold"] != 0,
            state["synced_completed_rows"] != 0,
            state.get("last_checkpoint") is not None,
            state.get("accepted_inventory_sha256") is not None,
        )
    ):
        raise ValueError("empty sidecar state has non-empty checkpoint metadata")
    return state


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _default_sync_directory(source: Path, destination: str) -> None:
    from huggingface_hub import sync_bucket

    sync_bucket(str(source), destination, quiet=True)


def _default_verify_remote_files(
    remote_run: str,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    import fsspec

    last_error: Exception | None = None
    for attempt in range(12):
        try:
            # A fresh filesystem instance avoids retaining a directory-listing
            # cache while the bucket makes a newly uploaded object visible.
            filesystem, root = fsspec.core.url_to_fs(
                remote_run, skip_instance_cache=True
            )
            root = root.rstrip("/")
            for relative in sorted(expected):
                wanted = expected[relative]
                digest = hashlib.sha256()
                size = 0
                with filesystem.open(f"{root}/{relative}", "rb") as stream:
                    while chunk := stream.read(1_048_576):
                        size += len(chunk)
                        digest.update(chunk)
                if size != wanted["size"] or digest.hexdigest() != wanted["sha256"]:
                    raise ValueError(
                        f"remote checkpoint checksum mismatch: {relative}"
                    )
            return
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 11:
                time.sleep(5)
    assert last_error is not None
    raise last_error


def _stage_json(
    root: Path,
    relative: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / relative
    _atomic_write_json(path, value)
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def checkpoint_once(
    output_dir: Path,
    state_dir: Path,
    bucket: str,
    checkpoint_rows: int,
    *,
    sync_directory: SyncDirectory = _default_sync_directory,
    verify_remote_files: VerifyRemoteFiles = _default_verify_remote_files,
) -> dict[str, Any]:
    """Publish at most one cumulative checkpoint from the latest sealed state."""
    state_path = state_dir / "state.json"
    raw_state = _read_json(state_path) if state_path.exists() else {}
    known_shards = raw_state.get("shards", {})
    if not isinstance(known_shards, dict):
        raise ValueError("sidecar state has an invalid shard inventory")
    snapshot = load_sealed_snapshot(output_dir, known_shards)
    remote_run = remote_checkpoint_destination(bucket, snapshot.run_instance_id)
    state = load_sidecar_state(state_path, snapshot, remote_run, checkpoint_rows)

    threshold = (snapshot.completed_rows // checkpoint_rows) * checkpoint_rows
    if threshold <= state["last_checkpoint_threshold"]:
        return {
            "status": "waiting",
            "completed_rows": snapshot.completed_rows,
            "next_checkpoint": state["last_checkpoint_threshold"] + checkpoint_rows,
            "remote_run": remote_run,
        }

    new_shards = {
        relative: entry
        for relative, entry in snapshot.shards.items()
        if relative not in state["shards"]
    }
    if not new_shards:
        raise ValueError("checkpoint threshold advanced without new accepted shards")

    marker_name = (
        f"checkpoints/checkpoint-{threshold:012d}-"
        f"{snapshot.accepted_inventory_sha256[:12]}.json"
    )
    marker = {
        "version": REMOTE_CHECKPOINT_VERSION,
        "run_instance_id": snapshot.run_instance_id,
        "source_content_sha256": snapshot.source_content_sha256,
        "checkpoint_interval_rows": checkpoint_rows,
        "checkpoint_threshold": threshold,
        "completed_rows": snapshot.completed_rows,
        "target_rows": snapshot.target_rows,
        "accepted_shard_count": len(snapshot.shards),
        "accepted_inventory_sha256": snapshot.accepted_inventory_sha256,
        "previous_checkpoint": state["last_checkpoint"],
        "new_accepted_shards": [
            {"path": relative, **new_shards[relative]}
            for relative in sorted(new_shards)
        ],
        "root_manifests": snapshot.root_manifests,
    }

    state_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-data-", dir=state_dir) as raw:
        stage = Path(raw)
        expected: dict[str, dict[str, Any]] = {}
        for relative, entry in new_shards.items():
            _link_or_copy(output_dir / relative, stage / relative)
            expected[relative] = {"size": entry["size"], "sha256": entry["sha256"]}
        if not state["shards"]:
            for relative, entry in snapshot.root_manifests.items():
                _link_or_copy(output_dir / relative, stage / relative)
                expected[relative] = entry
        sync_directory(stage, remote_run)
        verify_remote_files(remote_run, expected)

    with tempfile.TemporaryDirectory(prefix=".stage-marker-", dir=state_dir) as raw:
        stage = Path(raw)
        expected = {marker_name: _stage_json(stage, marker_name, marker)}
        sync_directory(stage, remote_run)
        verify_remote_files(remote_run, expected)

    latest = {
        "version": REMOTE_CHECKPOINT_VERSION,
        "run_instance_id": snapshot.run_instance_id,
        "checkpoint": marker_name,
        "checkpoint_sha256": expected[marker_name]["sha256"],
        "checkpoint_threshold": threshold,
        "completed_rows": snapshot.completed_rows,
        "accepted_inventory_sha256": snapshot.accepted_inventory_sha256,
    }
    with tempfile.TemporaryDirectory(prefix=".stage-latest-", dir=state_dir) as raw:
        stage = Path(raw)
        latest_expected = {"latest.json": _stage_json(stage, "latest.json", latest)}
        sync_directory(stage, remote_run)
        verify_remote_files(remote_run, latest_expected)

    next_state = {
        "version": SIDECAR_STATE_VERSION,
        "run_instance_id": snapshot.run_instance_id,
        "source_content_sha256": snapshot.source_content_sha256,
        "remote_run": remote_run,
        "checkpoint_rows": checkpoint_rows,
        "last_checkpoint_threshold": threshold,
        "synced_completed_rows": snapshot.completed_rows,
        "last_checkpoint": marker_name,
        "accepted_inventory_sha256": snapshot.accepted_inventory_sha256,
        "shards": snapshot.shards,
    }
    _atomic_write_json(state_path, next_state)
    return {
        "status": "synchronized",
        "completed_rows": snapshot.completed_rows,
        "checkpoint_threshold": threshold,
        "new_shards": len(new_shards),
        "remote_run": remote_run,
        "checkpoint": marker_name,
    }


def acquire_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    stream = (state_dir / "sidecar.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError("another Hugging Face checkpoint sidecar is running") from exc
    return stream


def cleanup_stale_stages(state_dir: Path) -> None:
    """Remove sidecar-owned staging trees left by an interrupted process."""
    for path in state_dir.glob(".stage-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)
    lock = acquire_lock(args.state_dir)
    try:
        cleanup_stale_stages(args.state_dir)
        while True:
            try:
                result = checkpoint_once(
                    args.output_dir,
                    args.state_dir,
                    args.hf_bucket,
                    args.checkpoint_rows,
                )
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
            except CheckpointNotReady as exc:
                if args.once:
                    raise
                print(
                    json.dumps({"status": "snapshot_retry", "error": str(exc)}),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.poll_seconds)
            except Exception as exc:  # noqa: BLE001 - durable monitor boundary
                if args.once:
                    raise
                print(
                    json.dumps(
                        {"status": "synchronization_retry", "error": str(exc)}
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.retry_seconds)
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
