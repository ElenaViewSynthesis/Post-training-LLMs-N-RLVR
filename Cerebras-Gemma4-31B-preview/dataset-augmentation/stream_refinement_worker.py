"""Generate exactly one accepted Gemini refinement for each assigned source slot.

This replaces the 225K oversampling path.  Every source row receives one
deterministic refinement slot and a stable subset receives a second slot so the
accepted synthetic total is exact.  Requests run concurrently in bounded
batches, while accepted rows are written through one atomic Parquet-shard
writer.  Failed generations are retried for the same slot and never become
additional successful candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import itertools
import json
import os
import sys
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from refinement_pipeline import (
    ASSIGNMENT_ALGORITHM_VERSION,
    RefinementSlot,
    REFINEMENT_VALIDATION_POLICY_VERSION,
    SourceIdentity,
    build_augmented_schema,
    build_refined_row,
    choose_secondary_source_ids,
    conversation_fingerprint,
    ensure_source_manifest,
    ensure_run_manifest,
    iter_source_batches_with_coordinates,
    load_source_identities,
    load_run_manifest,
    normalize_conversations,
    refinement_slots_for_source,
    remote_run_destination,
    resolve_source_for_run,
    scan_accepted_shards,
    source_conversation_fingerprint,
    source_identity_for_row,
    validate_refinement_quality,
    verify_refinement_state_inventory,
    write_accepted_shard,
    write_refinement_state_inventory,
)


DEFAULT_DATA_DIR = Path("~/pipeline/data")
DEFAULT_ORIGINAL_SOURCE = (
    "hf://datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/"
    "data/train-*-of-*.parquet"
)
DEFAULT_HF_BUCKET = (
    "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/refined"
)

SYSTEM_PROMPT = """You are an expert software-agent dataset editor.
Refine the supplied conversation while preserving its original task and outcome.

Requirements:
1. Preserve the task semantics and terminal-agent setting.
2. Improve clarity, concision, reasoning flow, and tool-call/result consistency.
3. Do not introduce a different task or claim unsupported external execution.
4. Keep 2-40 role/content turns and end with an assistant completion summary.
5. Return only data matching the requested JSON schema."""

TRAJECTORY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "minItems": 2,
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant", "tool"],
                    },
                    "content": {"type": "string"},
                },
                "required": ["role", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["conversations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Settings:
    model: str
    concurrency: int
    request_batch_size: int
    max_output_tokens: int
    max_attempts_per_run: int
    timeout_ms: int
    sync_every_shards: int
    hf_bucket: str
    sync_enabled: bool


@dataclass
class SlotResult:
    slot: RefinementSlot
    accepted_row: dict[str, Any] | None
    attempts: list[dict[str, Any]]


class RefinementSyncError(RuntimeError):
    """Raised when the durable refinement backup cannot be synchronized."""


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
    default_data_dir = Path(
        os.getenv("PIPELINE_DATA_DIR", str(DEFAULT_DATA_DIR))
    ).expanduser()
    parser = argparse.ArgumentParser(
        description="Stream source rows into exactly assigned Gemini refinements."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--original-source",
        default=os.getenv("ORIGINAL_DATASET_SOURCE", DEFAULT_ORIGINAL_SOURCE),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Refinement state directory (default: <data-dir>/refined)",
    )
    parser.add_argument(
        "--target-rows",
        type=_positive_int,
        default=int(os.getenv("REFINEMENT_TARGET_ROWS", "150000")),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL_ID", "gemini-3.6-flash"),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=int(os.getenv("GEMINI_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--request-batch-size",
        type=_positive_int,
        default=int(os.getenv("REFINEMENT_REQUEST_BATCH_SIZE", "32")),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
    )
    parser.add_argument(
        "--max-attempts-per-run",
        type=_positive_int,
        default=int(os.getenv("REFINEMENT_MAX_ATTEMPTS_PER_RUN", "3")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--sync-every-shards",
        type=_positive_int,
        default=int(os.getenv("REFINEMENT_SYNC_EVERY_SHARDS", "10")),
    )
    parser.add_argument(
        "--hf-bucket",
        default=os.getenv("HF_REFINEMENT_BUCKET", DEFAULT_HF_BUCKET),
    )
    parser.add_argument("--limit", type=_positive_int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status-only", action="store_true")
    mode.add_argument(
        "--sync-only",
        action="store_true",
        help="Synchronize existing local refinement state without reading the source or initializing Gemini.",
    )
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser()
    args.output_dir = (args.output_dir or args.data_dir / "refined").expanduser()
    if args.request_batch_size < args.concurrency:
        parser.error("--request-batch-size must be at least --concurrency")
    if args.sync_only and args.no_sync:
        parser.error("--sync-only cannot be combined with --no-sync")
    if args.sync_only and args.limit is not None:
        parser.error("--sync-only cannot be combined with --limit")
    return args


def acquire_run_lock(output_dir: Path) -> TextIO:
    digest = hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"stream-refinement-{digest}.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(f"another refinement worker holds {lock_path}") from exc
    return lock_stream


def build_refinement_prompt(
    source_row: dict[str, Any], slot: RefinementSlot
) -> str:
    source_conversation = normalize_conversations(
        source_row["conversations"],
        require_nonempty_content=False,
        require_final_assistant=False,
        min_turns=1,
        max_turns=None,
    )
    serialized = json.dumps(
        source_conversation,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Task identifier: {slot.source_task_id}\n"
        f"Refinement index: {slot.refinement_index}\n\n"
        "Original conversation JSON:\n"
        f"{serialized}\n\n"
        "Return a refined conversation for the same task."
    )


def attempt_seed(synthetic_id: str, attempt_number: int) -> int:
    digest = hashlib.sha256(
        f"{synthetic_id}\0attempt\0{attempt_number}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _safe_error(exc: Exception, api_key: str) -> tuple[str, int | None]:
    candidates = [
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ]
    code = None
    for candidate in candidates:
        try:
            if candidate is not None:
                code = int(candidate)
                break
        except (TypeError, ValueError):
            continue
    message = " ".join(str(exc).split())[:500]
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return f"{type(exc).__name__}: {message}", code


def load_attempt_counts(path: Path) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    malformed = 0
    if not path.exists():
        return counts, malformed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            synthetic_id = record.get("synthetic_id")
            if not isinstance(synthetic_id, str) or not synthetic_id:
                malformed += 1
                continue
            counts[synthetic_id] += 1
    return counts, malformed


def append_attempts(path: Path, attempts: list[dict[str, Any]]) -> None:
    if not attempts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for attempt in attempts:
            stream.write(json.dumps(attempt, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _attempt_record(
    slot: RefinementSlot,
    attempt_number: int,
    *,
    status: str,
    error: str | None = None,
    error_code: int | None = None,
    fingerprint: str | None = None,
    interaction_id: str | None = None,
    rejection_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "synthetic_id": slot.synthetic_id,
        "source_record_id": slot.source_record_id,
        "source_file": slot.source_file,
        "source_row_index": slot.source_row_index,
        "source_trial_name": slot.source_trial_name,
        "source_conversation_fingerprint": (
            slot.source_conversation_fingerprint
        ),
        "source_run_id": slot.source_run_id,
        "source_task_id": slot.source_task_id,
        "refinement_index": slot.refinement_index,
        "attempt_number": attempt_number,
        "status": status,
        "error": error,
        "error_code": error_code,
        "conversation_fingerprint": fingerprint,
        "interaction_id": interaction_id,
        "rejection_codes": rejection_codes or [],
        "validation_policy": REFINEMENT_VALIDATION_POLICY_VERSION,
    }


async def refine_slot(
    client: Any,
    source_row: dict[str, Any],
    slot: RefinementSlot,
    original_schema: Any,
    settings: Settings,
    api_key: str,
    semaphore: asyncio.Semaphore,
    starting_attempt: int,
    attempt_budget: int | None = None,
) -> SlotResult:
    prompt = build_refinement_prompt(source_row, slot)
    attempts: list[dict[str, Any]] = []
    fatal_codes = {400, 401, 403, 404}

    async with semaphore:
        max_attempts = (
            settings.max_attempts_per_run
            if attempt_budget is None
            else attempt_budget
        )
        for offset in range(max_attempts):
            attempt_number = starting_attempt + offset + 1
            try:
                interaction = await client.interactions.create(
                    model=settings.model,
                    input=prompt,
                    system_instruction=SYSTEM_PROMPT,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": TRAJECTORY_JSON_SCHEMA,
                    },
                    generation_config={
                        "max_output_tokens": settings.max_output_tokens,
                        "thinking_level": "medium",
                        "seed": attempt_seed(slot.synthetic_id, attempt_number),
                    },
                    store=False,
                )
                interaction_status = getattr(interaction, "status", None)
                interaction_status = getattr(
                    interaction_status, "value", interaction_status
                )
                if interaction_status not in (None, "completed"):
                    raise ValueError(
                        f"interaction finished with status {interaction_status!r}"
                    )
                raw = getattr(interaction, "output_text", None) or ""
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise TypeError("Gemini response must be a JSON object")
                conversations = normalize_conversations(
                    parsed.get("conversations"),
                    require_final_assistant=False,
                )
                quality_issues = validate_refinement_quality(
                    source_row["conversations"],
                    conversations,
                )
                fingerprint = (
                    source_conversation_fingerprint(conversations)
                    if quality_issues
                    else conversation_fingerprint(conversations)
                )
                if quality_issues:
                    attempts.append(
                        _attempt_record(
                            slot,
                            attempt_number,
                            status="rejected",
                            error="; ".join(
                                f"{issue.code}: {issue.detail}"
                                for issue in quality_issues
                            ),
                            fingerprint=fingerprint,
                            interaction_id=getattr(interaction, "id", None),
                            rejection_codes=[
                                issue.code for issue in quality_issues
                            ],
                        )
                    )
                    continue
                refined_row = build_refined_row(
                    source_row,
                    slot,
                    conversations,
                    original_schema,
                    model=settings.model,
                    provider="gemini",
                )
                attempts.append(
                    _attempt_record(
                        slot,
                        attempt_number,
                        status="accepted",
                        fingerprint=fingerprint,
                        interaction_id=getattr(interaction, "id", None),
                    )
                )
                return SlotResult(slot, refined_row, attempts)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error, code = _safe_error(exc, api_key)
                attempts.append(
                    _attempt_record(
                        slot,
                        attempt_number,
                        status="rejected",
                        error=error,
                        error_code=code,
                        rejection_codes=[
                            "invalid_json"
                            if isinstance(exc, json.JSONDecodeError)
                            else "invalid_conversation"
                        ],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - provider error boundary
                error, code = _safe_error(exc, api_key)
                attempts.append(
                    _attempt_record(
                        slot,
                        attempt_number,
                        status="provider_error",
                        error=error,
                        error_code=code,
                    )
                )
                if code in fatal_codes:
                    break
    return SlotResult(slot, None, attempts)


def resolve_batch_conversation_collisions(
    results: list[SlotResult],
    existing_fingerprints: set[str],
) -> tuple[set[str], set[str]]:
    """Deterministically reject exact duplicates before shard promotion.

    Existing accepted rows always win. Within a new concurrent batch, sorting
    by synthetic ID makes the winner independent of request completion order.
    """
    claimed = set(existing_fingerprints)
    accepted_fingerprints: set[str] = set()
    duplicate_ids: set[str] = set()
    candidates = sorted(
        (result for result in results if result.accepted_row is not None),
        key=lambda result: result.slot.synthetic_id,
    )
    for result in candidates:
        assert result.accepted_row is not None
        fingerprint = conversation_fingerprint(
            result.accepted_row.get("conversations")
        )
        stored_fingerprint = result.accepted_row.get(
            "refined_conversation_fingerprint"
        )
        if stored_fingerprint != fingerprint:
            raise ValueError(
                "candidate conversation fingerprint does not match its row: "
                f"{result.slot.synthetic_id!r}"
            )
        if fingerprint not in claimed:
            claimed.add(fingerprint)
            accepted_fingerprints.add(fingerprint)
            continue

        if not result.attempts or result.attempts[-1].get("status") != "accepted":
            raise ValueError(
                "accepted candidate has no matching accepted attempt record: "
                f"{result.slot.synthetic_id!r}"
            )
        attempt = result.attempts[-1]
        attempt["status"] = "rejected"
        attempt["error"] = (
            "duplicate_conversation: normalized conversation fingerprint "
            f"{fingerprint} is already accepted"
        )
        attempt["error_code"] = None
        attempt["conversation_fingerprint"] = fingerprint
        attempt["rejection_codes"] = ["duplicate_conversation"]
        result.accepted_row = None
        duplicate_ids.add(result.slot.synthetic_id)
    return accepted_fingerprints, duplicate_ids


def preflight_source_rows(
    source: str,
    identities: dict[str, SourceIdentity],
) -> int:
    """Validate every source lookup and conversation before any paid request."""
    seen: set[str] = set()
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        source
    ):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            observed = source_identity_for_row(source_file, index, row)
            identity = identities.get(observed.source_record_id)
            if identity != observed:
                raise ValueError(
                    "source identity changed during preflight: "
                    f"{observed.source_record_id!r}"
                )
            normalize_conversations(
                row.get("conversations"),
                require_nonempty_content=False,
                require_final_assistant=False,
                min_turns=1,
                max_turns=None,
            )
            seen.add(observed.source_record_id)
    if seen != identities.keys():
        raise ValueError("source rows changed between identity and full preflight scans")
    return len(seen)


def iter_pending_slots(
    source: str,
    identities: dict[str, SourceIdentity],
    secondary_source_ids: frozenset[str],
    completed: set[str],
):
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        source
    ):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            identity = source_identity_for_row(source_file, index, row)
            if identities.get(identity.source_record_id) != identity:
                raise ValueError(
                    "source identity changed after preflight: "
                    f"{identity.source_record_id!r}"
                )
            for slot in refinement_slots_for_source(identity, secondary_source_ids):
                if slot.synthetic_id not in completed:
                    yield row, slot


def expected_synthetic_ids(
    identities: dict[str, SourceIdentity], secondary_source_ids: frozenset[str]
) -> set[str]:
    return {
        slot.synthetic_id
        for identity in identities.values()
        for slot in refinement_slots_for_source(identity, secondary_source_ids)
    }


def sync_output(
    output_dir: Path,
    bucket: str,
    *,
    verify_hashes: bool = False,
) -> None:
    from huggingface_hub import sync_bucket

    try:
        run_manifest = load_run_manifest(output_dir / "run_manifest.json")
        verify_refinement_state_inventory(
            output_dir,
            verify_hashes=verify_hashes,
        )
        destination = remote_run_destination(bucket, run_manifest)
        sync_bucket(str(output_dir), destination)
    except Exception as exc:
        raise RefinementSyncError(
            f"failed to sync {output_dir} to the isolated run under {bucket}: {exc}"
        ) from exc


def create_gemini_client(api_key: str, settings: Settings) -> Any:
    """Create the provider client behind a small testable boundary."""
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options={
            "api_version": "v1beta",
            "timeout": settings.timeout_ms,
            "retry_options": {
                "attempts": 5,
                "initial_delay": 1.0,
                "max_delay": 32.0,
                "exp_base": 2.0,
                "jitter": 1.0,
                "http_status_codes": [408, 429, 500, 502, 503, 504],
            },
        },
    )


def write_progress_manifest(
    output_dir: Path,
    *,
    source: str,
    source_rows: int,
    target_rows: int,
    completed_rows: int,
    model: str,
    run_instance_id: str,
    source_content_sha256: str,
) -> None:
    manifest = {
        "source": source,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "completed_rows": completed_rows,
        "remaining_rows": target_rows - completed_rows,
        "model": model,
        "run_instance_id": run_instance_id,
        "source_content_sha256": source_content_sha256,
        "assignment": ASSIGNMENT_ALGORITHM_VERSION,
        "validation_policy": REFINEMENT_VALIDATION_POLICY_VERSION,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "progress.json"
    temporary = output_dir / ".progress.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def generation_config_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "concurrency": args.concurrency,
        "request_batch_size": args.request_batch_size,
        "max_output_tokens": args.max_output_tokens,
        "max_attempts_per_run": args.max_attempts_per_run,
        "timeout_seconds": args.timeout_seconds,
    }


async def async_main(args: argparse.Namespace) -> int:
    if args.sync_only:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(
                f"refinement output directory does not exist: {args.output_dir}"
            )
        lock_stream = acquire_run_lock(args.output_dir)
        try:
            sync_output(args.output_dir, args.hf_bucket, verify_hashes=True)
            print("Refinement-state synchronization complete.")
            return 0
        finally:
            lock_stream.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = acquire_run_lock(args.output_dir)
    try:
        source_manifest_path = args.output_dir / "source_manifest.json"
        resolved_source = resolve_source_for_run(
            source_manifest_path, args.original_source
        )
        original_schema, identities = load_source_identities(resolved_source)
        source_manifest = ensure_source_manifest(
            source_manifest_path,
            requested_source=args.original_source,
            resolved_source=resolved_source,
            schema=original_schema,
            identities=identities,
        )
        run_manifest = ensure_run_manifest(
            args.output_dir / "run_manifest.json",
            source_manifest=source_manifest,
            target_rows=args.target_rows,
            model=args.model,
            generation_config=generation_config_from_args(args),
        )
        inventory_path = args.output_dir / "checksum_inventory.json"
        complete_path = args.output_dir / "complete.json"
        if inventory_path.exists() != complete_path.exists():
            raise ValueError(
                "refinement state has an incomplete checksum inventory/complete marker"
            )
        if inventory_path.exists():
            inventory_entries = verify_refinement_state_inventory(args.output_dir)
        else:
            preexisting_state = (
                (args.output_dir / "progress.json").exists()
                or (args.output_dir / "attempts.jsonl").exists()
                or any((args.output_dir / "accepted").glob("accepted-*.parquet"))
            )
            if preexisting_state:
                raise ValueError(
                    "existing refinement state has no checksum inventory; "
                    "legacy or partially restored state cannot be resumed"
                )
            inventory_entries = write_refinement_state_inventory(args.output_dir)
        secondary_source_ids = choose_secondary_source_ids(
            list(identities), args.target_rows
        )
        augmented_schema = build_augmented_schema(original_schema)
        accepted_dir = args.output_dir / "accepted"
        completed, accepted_rows, accepted_fingerprints = scan_accepted_shards(
            accepted_dir,
            augmented_schema,
            expected_validation_policy=run_manifest.validation_policy_version,
        )
        expected_ids = expected_synthetic_ids(identities, secondary_source_ids)
        unexpected = completed.difference(expected_ids)
        if unexpected:
            sample = sorted(unexpected)[:3]
            raise ValueError(f"accepted shards contain unknown synthetic IDs: {sample}")
        if accepted_rows != len(completed):
            raise ValueError("accepted row count does not match unique completed IDs")

        attempts_path = args.output_dir / "attempts.jsonl"
        attempt_counts, malformed_attempts = load_attempt_counts(attempts_path)
        remaining = args.target_rows - len(completed)
        print(
            f"Source: {len(identities):,} | target: {args.target_rows:,} | "
            f"accepted: {len(completed):,} | remaining: {remaining:,} | "
            f"malformed attempt records: {malformed_attempts:,}"
        )
        write_progress_manifest(
            args.output_dir,
            source=resolved_source,
            source_rows=len(identities),
            target_rows=args.target_rows,
            completed_rows=len(completed),
            model=args.model,
            run_instance_id=run_manifest.run_instance_id,
            source_content_sha256=source_manifest.source_content_sha256,
        )
        inventory_entries = write_refinement_state_inventory(
            args.output_dir,
            previous_entries=inventory_entries,
            changed_paths={"progress.json"},
        )
        if args.status_only:
            return 0
        if remaining == 0:
            if not args.no_sync:
                sync_output(args.output_dir, args.hf_bucket)
                print("Completed refinement state synchronized.")
            else:
                print("Refinement is complete; synchronization disabled by --no-sync.")
            return 0

        checked_rows = preflight_source_rows(resolved_source, identities)
        if checked_rows != len(identities):
            raise ValueError("source preflight row count changed")

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        settings = Settings(
            model=args.model,
            concurrency=args.concurrency,
            request_batch_size=args.request_batch_size,
            max_output_tokens=args.max_output_tokens,
            max_attempts_per_run=args.max_attempts_per_run,
            timeout_ms=args.timeout_seconds * 1_000,
            sync_every_shards=args.sync_every_shards,
            hf_bucket=args.hf_bucket,
            sync_enabled=not args.no_sync,
        )
        sync_client = create_gemini_client(api_key, settings)
        client = sync_client.aio
        semaphore = asyncio.Semaphore(settings.concurrency)
        pending = iter_pending_slots(
            resolved_source,
            identities,
            secondary_source_ids,
            completed,
        )
        if args.limit is not None:
            pending = itertools.islice(pending, args.limit)
        retry_queue: deque[tuple[dict[str, Any], RefinementSlot]] = deque()
        attempts_used_this_run: Counter[str] = Counter()

        processed = accepted_this_run = failed_this_run = shard_count = 0
        forced_exit_code = 0
        try:
            while True:
                request_batch: list[tuple[dict[str, Any], RefinementSlot]] = []
                while retry_queue and len(request_batch) < settings.request_batch_size:
                    source_row, slot = retry_queue.popleft()
                    if (
                        attempts_used_this_run[slot.synthetic_id]
                        < settings.max_attempts_per_run
                    ):
                        request_batch.append((source_row, slot))
                request_batch.extend(
                    itertools.islice(
                        pending,
                        settings.request_batch_size - len(request_batch),
                    )
                )
                if not request_batch:
                    break
                results = await asyncio.gather(
                    *(
                        refine_slot(
                            client,
                            source_row,
                            slot,
                            original_schema,
                            settings,
                            api_key,
                            semaphore,
                            attempt_counts[slot.synthetic_id],
                            attempt_budget=(
                                settings.max_attempts_per_run
                                - attempts_used_this_run[slot.synthetic_id]
                            ),
                        )
                        for source_row, slot in request_batch
                    )
                )
                new_fingerprints, duplicate_ids = (
                    resolve_batch_conversation_collisions(
                        results,
                        accepted_fingerprints,
                    )
                )
                accepted_rows_batch = [
                    result.accepted_row
                    for result in results
                    if result.accepted_row is not None
                ]
                if accepted_rows_batch:
                    batch_ids = {str(row["run_id"]) for row in accepted_rows_batch}
                    overlap = completed.intersection(batch_ids)
                    if overlap:
                        raise ValueError(
                            "request batch attempted to store an already accepted slot: "
                            f"{sorted(overlap)[:3]}"
                        )
                    accepted_shard = write_accepted_shard(
                        accepted_dir,
                        accepted_rows_batch,
                        augmented_schema,
                    )
                    shard_count += 1
                    completed.update(batch_ids)
                    accepted_fingerprints.update(new_fingerprints)
                else:
                    accepted_shard = None
                all_attempts = [
                    attempt for result in results for attempt in result.attempts
                ]
                append_attempts(attempts_path, all_attempts)
                for attempt in all_attempts:
                    attempt_counts[attempt["synthetic_id"]] += 1
                    attempts_used_this_run[attempt["synthetic_id"]] += 1

                request_by_id = {
                    slot.synthetic_id: (source_row, slot)
                    for source_row, slot in request_batch
                }
                for synthetic_id in sorted(duplicate_ids):
                    if (
                        attempts_used_this_run[synthetic_id]
                        < settings.max_attempts_per_run
                    ):
                        retry_queue.append(request_by_id[synthetic_id])

                batch_accepted = len(accepted_rows_batch)
                batch_failed = len(results) - batch_accepted
                processed += len(results)
                accepted_this_run += batch_accepted
                failed_this_run += batch_failed
                print(
                    f"Processed {processed:,}: accepted={batch_accepted} "
                    f"incomplete={batch_failed} | run accepted={accepted_this_run:,} "
                    f"incomplete={failed_this_run:,}"
                )
                write_progress_manifest(
                    args.output_dir,
                    source=resolved_source,
                    source_rows=len(identities),
                    target_rows=args.target_rows,
                    completed_rows=len(completed),
                    model=args.model,
                    run_instance_id=run_manifest.run_instance_id,
                    source_content_sha256=source_manifest.source_content_sha256,
                )
                changed_state_paths = {"progress.json", "attempts.jsonl"}
                if accepted_shard is not None:
                    changed_state_paths.add(
                        accepted_shard.relative_to(args.output_dir).as_posix()
                    )
                inventory_entries = write_refinement_state_inventory(
                    args.output_dir,
                    previous_entries=inventory_entries,
                    changed_paths=changed_state_paths,
                )

                fatal_codes = {400, 401, 403, 404}
                if any(
                    attempt.get("error_code") in fatal_codes
                    for attempt in all_attempts
                ):
                    print("Stopping after a fatal provider/configuration response.")
                    forced_exit_code = 2
                    break
                if batch_accepted == 0:
                    if retry_queue:
                        print(
                            "Retrying duplicate conversations for their assigned "
                            "slots within the remaining attempt budget."
                        )
                        continue
                    print("Stopping because the entire request batch remained incomplete.")
                    forced_exit_code = 2
                    break
                if (
                    settings.sync_enabled
                    and shard_count % settings.sync_every_shards == 0
                ):
                    sync_output(args.output_dir, settings.hf_bucket)
                    print(f"Synchronized after {shard_count} new shards.")
        finally:
            try:
                await client.aclose()
            finally:
                if settings.sync_enabled:
                    sync_output(args.output_dir, settings.hf_bucket)
                    print("Final refinement-state synchronization complete.")

        if forced_exit_code:
            return forced_exit_code

        final_remaining = args.target_rows - len(completed)
        if final_remaining and (args.limit is None or failed_this_run):
            print(
                f"Run finished with {final_remaining:,} incomplete slots; rerun to retry."
            )
            return 2
        return 0
    finally:
        lock_stream.close()


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted. Accepted Parquet shards are durable; rerun to resume.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
