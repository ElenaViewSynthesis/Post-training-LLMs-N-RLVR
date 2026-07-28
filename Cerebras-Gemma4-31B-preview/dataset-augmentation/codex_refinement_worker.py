"""Create resumable conversation refinements with the authenticated Codex CLI.

This is deliberately separate from the Gemini and Cerebras workers.  It never
loads their clients, never reads an API key, never downloads a source, and has
no synchronization code.  Every generation subprocess uses the existing
ChatGPT-backed ``codex login`` session with an ephemeral, read-only agent.

The default mode is a write-free local snapshot scan.  Real Codex calls require
``--execute`` and are capped by ``--max-agent-calls`` (one call may contain a
small micro-batch of source slots).
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from refinement_pipeline import (
    ASSIGNMENT_ALGORITHM_VERSION,
    REFINEMENT_VALIDATION_POLICY_VERSION,
    RefinementSlot,
    SourceIdentity,
    build_augmented_schema,
    build_refined_row,
    choose_secondary_source_ids,
    conversation_fingerprint,
    ensure_run_manifest,
    ensure_source_manifest,
    iter_source_batches_with_coordinates,
    load_local_source_snapshot,
    load_run_manifest,
    load_source_identities,
    normalize_conversations,
    refinement_slots_for_source,
    scan_accepted_shards,
    source_conversation_fingerprint,
    source_identity_for_row,
    validate_refinement_quality,
    verify_refinement_state_inventory,
    write_accepted_shard,
    write_refinement_state_inventory,
)


DEFAULT_DATA_DIR = Path("~/pipeline/data")
DEFAULT_TARGET_ROWS = 150_000
DEFAULT_BATCH_SIZE = 2
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_AGENT_CALLS = 1
MAX_BATCH_SIZE = 8
CODEX_PROVIDER = "openai-codex-cli-chatgpt"

SYSTEM_INSTRUCTIONS = """You are an expert software-agent dataset editor.
This is a text transformation task only. Do not inspect files, run commands,
browse, call tools, or access the network.

For every supplied item, create one distinct, high-quality software-agent
trajectory for the same task and outcome. Preserve paths, identifiers,
commands, URLs, quoted strings, numbers, and other task-critical details.
Improve clarity, reasoning flow, useful intermediate detail, and consistency
between assistant actions and tool observations. Do not claim real execution
beyond what the source conversation supports. Keep 2-40 non-empty role/content
turns, include user and assistant participation, keep tool observations paired
with assistant actions, and end with an assistant completion summary.

Return exactly one result for every supplied synthetic_id and only return data
that matches the requested JSON schema."""

CONVERSATION_SCHEMA: dict[str, Any] = {
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
            "content": {"type": "string", "minLength": 1},
        },
        "required": ["role", "content"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class PendingItem:
    source_row: dict[str, Any]
    slot: RefinementSlot


@dataclass(frozen=True)
class CodexRuntime:
    executable: str
    version: str
    environment: dict[str, str]


@dataclass(frozen=True)
class CodexResponse:
    payload: dict[str, Any]
    thread_id: str | None
    usage: dict[str, int]


@dataclass
class Candidate:
    item: PendingItem
    row: dict[str, Any]
    attempt: dict[str, Any]


@dataclass
class BatchResult:
    candidates: list[Candidate]
    attempts: list[dict[str, Any]]
    incomplete: list[PendingItem]


class CodexInvocationError(RuntimeError):
    """A bounded Codex subprocess did not produce a usable final response."""


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
        description=(
            "Refine a sealed local source snapshot with authenticated Codex CLI "
            "micro-batches; never use Gemini/Cerebras or synchronize data."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--source-snapshot-dir",
        type=Path,
        help="Existing sealed local snapshot (default: <data-dir>/source-snapshot).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Isolated local state (default: <data-dir>/codex-refined).",
    )
    parser.add_argument(
        "--target-rows",
        type=_positive_int,
        default=DEFAULT_TARGET_ROWS,
        help="Exact number of synthetic rows assigned across the source.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_REFINEMENT_MODEL"),
        help=(
            "Explicit Codex model ID. Required with --execute so one immutable "
            "run cannot silently mix changing CLI defaults."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Source slots per Codex call (default: {DEFAULT_BATCH_SIZE}, "
            f"max: {MAX_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help="Concurrent Codex CLI subprocesses (start with 1).",
    )
    parser.add_argument(
        "--max-agent-calls",
        type=_positive_int,
        default=DEFAULT_MAX_AGENT_CALLS,
        help=(
            "Hard cap on Codex subprocess calls in this invocation, including "
            "retries (default: 1)."
        ),
    )
    parser.add_argument(
        "--max-attempts-per-run",
        type=_positive_int,
        default=3,
        help="Maximum attempts for one slot during this invocation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=600,
        help="Wall-clock timeout for each Codex micro-batch.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Limit newly selected slots; retries do not consume this limit.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to real Codex calls and durable local accepted shards.",
    )
    mode.add_argument(
        "--status-only",
        action="store_true",
        help="Validate and report existing local Codex refinement state.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check Codex CLI and saved ChatGPT login without scanning the source.",
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.source_snapshot_dir = (
        args.source_snapshot_dir or args.data_dir / "source-snapshot"
    ).expanduser().resolve()
    args.output_dir = (
        args.output_dir or args.data_dir / "codex-refined"
    ).expanduser().resolve()
    if args.batch_size > MAX_BATCH_SIZE:
        parser.error(f"--batch-size cannot exceed {MAX_BATCH_SIZE}")
    if args.execute and not (isinstance(args.model, str) and args.model.strip()):
        parser.error("--model is required with --execute")
    if isinstance(args.model, str):
        args.model = args.model.strip()
    return args


def _assert_isolated_paths(args: argparse.Namespace) -> None:
    protected = {
        args.data_dir,
        args.data_dir / "refined",
        args.data_dir / "upload",
        args.source_snapshot_dir,
    }
    if args.output_dir in protected:
        raise ValueError(
            "Codex output must be isolated from data-dir, Gemini refinement, "
            "publication, and source-snapshot directories"
        )


def _codex_environment() -> dict[str, str]:
    """Retain CLI account auth paths while removing inherited secret variables."""
    secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    environment: dict[str, str] = {}
    for name in os.environ:
        if not any(marker in name.upper() for marker in secret_markers):
            environment[name] = os.environ[name]
    # These are the two credentials Codex exec recognizes explicitly.  Keep the
    # removal obvious even if future marker rules above are changed.
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_API_KEY", None)
    return environment


def inspect_codex_runtime() -> CodexRuntime:
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("codex CLI is not installed or is not on PATH")
    environment = _codex_environment()
    status = subprocess.run(
        [executable, "login", "status"],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        check=False,
    )
    status_text = f"{status.stdout}\n{status.stderr}".strip()
    if status.returncode != 0 or "logged in using chatgpt" not in status_text.casefold():
        raise RuntimeError(
            "Codex must be logged in with ChatGPT account auth; run `codex login` "
            "and do not use CODEX_API_KEY/OPENAI_API_KEY for this worker"
        )
    version_result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        check=False,
    )
    if version_result.returncode != 0:
        raise RuntimeError("could not read the installed Codex CLI version")
    version = " ".join(version_result.stdout.split())
    if not version:
        raise RuntimeError("Codex CLI returned an empty version")
    return CodexRuntime(executable, version, environment)


def acquire_run_lock(output_dir: Path) -> TextIO:
    digest = hashlib.sha256(str(output_dir).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"codex-refinement-{digest}.lock"
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"another Codex worker holds {lock_path}") from exc
    return stream


def build_output_schema(items: Sequence[PendingItem]) -> dict[str, Any]:
    synthetic_ids = [item.slot.synthetic_id for item in items]
    if not synthetic_ids or len(synthetic_ids) != len(set(synthetic_ids)):
        raise ValueError("Codex micro-batch IDs must be non-empty and unique")
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": len(items),
                "maxItems": len(items),
                "items": {
                    "type": "object",
                    "properties": {
                        "synthetic_id": {
                            "type": "string",
                            "enum": synthetic_ids,
                        },
                        "conversations": CONVERSATION_SCHEMA,
                    },
                    "required": ["synthetic_id", "conversations"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def build_batch_prompt(items: Sequence[PendingItem]) -> str:
    inputs: list[dict[str, Any]] = []
    for item in items:
        conversations = normalize_conversations(
            item.source_row.get("conversations"),
            require_nonempty_content=False,
            require_final_assistant=False,
            min_turns=1,
            max_turns=None,
        )
        inputs.append(
            {
                "synthetic_id": item.slot.synthetic_id,
                "refinement_index": item.slot.refinement_index,
                "task": item.slot.source_task_id,
                "source_conversations": conversations,
            }
        )
    serialized = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    return f"{SYSTEM_INSTRUCTIONS}\n\nInput items JSON:\n{serialized}"


def _parse_codex_events(stdout: str) -> CodexResponse:
    final_message: str | None = None
    thread_id: str | None = None
    usage: dict[str, int] = {}
    errors: list[str] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexInvocationError(
                f"Codex JSONL event {line_number} is malformed"
            ) from exc
        if not isinstance(event, dict):
            raise CodexInvocationError(
                f"Codex JSONL event {line_number} is not an object"
            )
        event_type = event.get("type")
        if event_type == "thread.started" and isinstance(
            event.get("thread_id"), str
        ):
            thread_id = event["thread_id"]
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final_message = text
        elif event_type == "turn.completed":
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): value
                    for key, value in raw_usage.items()
                    if isinstance(value, int) and value >= 0
                }
        elif event_type in {"error", "turn.failed"}:
            message = event.get("message") or event.get("error") or event_type
            errors.append(" ".join(str(message).split())[:300])
    if final_message is None:
        detail = f": {'; '.join(errors)}" if errors else ""
        raise CodexInvocationError(f"Codex returned no final agent message{detail}")
    try:
        payload = json.loads(final_message)
    except json.JSONDecodeError as exc:
        raise CodexInvocationError("Codex final message is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CodexInvocationError("Codex final JSON is not an object")
    return CodexResponse(payload, thread_id, usage)


async def invoke_codex(
    runtime: CodexRuntime,
    *,
    model: str,
    items: Sequence[PendingItem],
    timeout_seconds: int,
) -> CodexResponse:
    schema = build_output_schema(items)
    prompt = build_batch_prompt(items)
    with tempfile.TemporaryDirectory(prefix="codex-refinement-call-") as temp_name:
        temp_dir = Path(temp_name)
        schema_path = temp_dir / "output-schema.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        command = [
            runtime.executable,
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--model",
            model,
            "--output-schema",
            str(schema_path),
            "-",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=temp_dir,
            env=runtime.environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise CodexInvocationError(
                f"Codex call exceeded {timeout_seconds} seconds"
            ) from exc
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            detail = " ".join(stderr.split())[-500:]
            raise CodexInvocationError(
                f"Codex exited {process.returncode}: {detail or 'no diagnostic'}"
            )
        return _parse_codex_events(stdout)


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:500]}"


def _attempt_record(
    item: PendingItem,
    attempt_number: int,
    agent_call_number: int,
    *,
    status: str,
    runtime: CodexRuntime,
    thread_id: str | None = None,
    usage: Mapping[str, int] | None = None,
    error: str | None = None,
    fingerprint: str | None = None,
    rejection_codes: Sequence[str] = (),
) -> dict[str, Any]:
    slot = item.slot
    return {
        "synthetic_id": slot.synthetic_id,
        "source_record_id": slot.source_record_id,
        "source_file": slot.source_file,
        "source_row_index": slot.source_row_index,
        "source_trial_name": slot.source_trial_name,
        "source_conversation_fingerprint": slot.source_conversation_fingerprint,
        "source_run_id": slot.source_run_id,
        "source_task_id": slot.source_task_id,
        "refinement_index": slot.refinement_index,
        "attempt_number": attempt_number,
        "status": status,
        "error": error,
        "error_code": None,
        "conversation_fingerprint": fingerprint,
        "interaction_id": thread_id,
        "agent_call_number": agent_call_number,
        "provider_request_number": None,
        "codex_cli_version": runtime.version,
        "usage": dict(usage or {}),
        "rejection_codes": list(rejection_codes),
        "validation_policy": REFINEMENT_VALIDATION_POLICY_VERSION,
    }


def load_attempt_state(path: Path) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    malformed = 0
    max_call_number = 0
    if not path.exists():
        return counts, malformed, max_call_number
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
            call_number = record.get("agent_call_number")
            if isinstance(call_number, int) and call_number > max_call_number:
                max_call_number = call_number
    return counts, malformed, max_call_number


def append_attempts(path: Path, attempts: Sequence[Mapping[str, Any]]) -> None:
    if not attempts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for attempt in attempts:
            stream.write(
                json.dumps(attempt, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


async def process_batch(
    runtime: CodexRuntime,
    *,
    model: str,
    items: Sequence[PendingItem],
    attempt_numbers: Mapping[str, int],
    agent_call_number: int,
    timeout_seconds: int,
    original_schema: Any,
) -> BatchResult:
    try:
        response = await invoke_codex(
            runtime,
            model=model,
            items=items,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - Codex subprocess boundary
        attempts = [
            _attempt_record(
                item,
                attempt_numbers[item.slot.synthetic_id],
                agent_call_number,
                status="agent_error",
                runtime=runtime,
                error=_safe_error(exc),
                rejection_codes=("codex_invocation_error",),
            )
            for item in items
        ]
        return BatchResult([], attempts, list(items))

    try:
        raw_results = response.payload.get("results")
        if not isinstance(raw_results, list):
            raise TypeError("results is not a list")
        by_id: dict[str, dict[str, Any]] = {}
        for result in raw_results:
            if not isinstance(result, dict):
                raise TypeError("result is not an object")
            synthetic_id = result.get("synthetic_id")
            if not isinstance(synthetic_id, str) or synthetic_id in by_id:
                raise ValueError("result synthetic IDs are invalid or duplicated")
            by_id[synthetic_id] = result
        expected_ids = {item.slot.synthetic_id for item in items}
        if set(by_id) != expected_ids:
            raise ValueError("result synthetic IDs do not exactly match the batch")
    except (TypeError, ValueError) as exc:
        attempts = [
            _attempt_record(
                item,
                attempt_numbers[item.slot.synthetic_id],
                agent_call_number,
                status="rejected",
                runtime=runtime,
                thread_id=response.thread_id,
                usage=response.usage,
                error=_safe_error(exc),
                rejection_codes=("invalid_batch_output",),
            )
            for item in items
        ]
        return BatchResult([], attempts, list(items))

    candidates: list[Candidate] = []
    attempts: list[dict[str, Any]] = []
    incomplete: list[PendingItem] = []
    for item in items:
        attempt_number = attempt_numbers[item.slot.synthetic_id]
        try:
            conversations = normalize_conversations(
                by_id[item.slot.synthetic_id].get("conversations"),
                require_final_assistant=False,
            )
            issues = validate_refinement_quality(
                item.source_row.get("conversations"), conversations
            )
            if issues:
                fingerprint = source_conversation_fingerprint(conversations)
                attempt = _attempt_record(
                    item,
                    attempt_number,
                    agent_call_number,
                    status="rejected",
                    runtime=runtime,
                    thread_id=response.thread_id,
                    usage=response.usage,
                    error="; ".join(
                        f"{issue.code}: {issue.detail}" for issue in issues
                    ),
                    fingerprint=fingerprint,
                    rejection_codes=tuple(issue.code for issue in issues),
                )
                attempts.append(attempt)
                incomplete.append(item)
                continue
            fingerprint = conversation_fingerprint(conversations)
            row = build_refined_row(
                item.source_row,
                item.slot,
                conversations,
                original_schema,
                model=model,
                provider=CODEX_PROVIDER,
            )
            attempt = _attempt_record(
                item,
                attempt_number,
                agent_call_number,
                status="accepted",
                runtime=runtime,
                thread_id=response.thread_id,
                usage=response.usage,
                fingerprint=fingerprint,
            )
            attempts.append(attempt)
            candidates.append(Candidate(item, row, attempt))
        except (TypeError, ValueError) as exc:
            attempts.append(
                _attempt_record(
                    item,
                    attempt_number,
                    agent_call_number,
                    status="rejected",
                    runtime=runtime,
                    thread_id=response.thread_id,
                    usage=response.usage,
                    error=_safe_error(exc),
                    rejection_codes=("invalid_conversation",),
                )
            )
            incomplete.append(item)
    return BatchResult(candidates, attempts, incomplete)


def resolve_candidate_collisions(
    candidates: Sequence[Candidate], existing_fingerprints: set[str]
) -> tuple[list[Candidate], list[PendingItem], set[str]]:
    claimed = set(existing_fingerprints)
    accepted: list[Candidate] = []
    incomplete: list[PendingItem] = []
    new_fingerprints: set[str] = set()
    for candidate in sorted(candidates, key=lambda value: value.item.slot.synthetic_id):
        fingerprint = conversation_fingerprint(candidate.row.get("conversations"))
        if candidate.row.get("refined_conversation_fingerprint") != fingerprint:
            raise ValueError(
                "candidate fingerprint metadata changed before shard promotion"
            )
        if fingerprint not in claimed:
            claimed.add(fingerprint)
            new_fingerprints.add(fingerprint)
            accepted.append(candidate)
            continue
        candidate.attempt["status"] = "rejected"
        candidate.attempt["error"] = (
            "duplicate_conversation: normalized conversation fingerprint is "
            "already accepted"
        )
        candidate.attempt["rejection_codes"] = ["duplicate_conversation"]
        incomplete.append(candidate.item)
    return accepted, incomplete, new_fingerprints


def preflight_source_rows(
    source: str, identities: Mapping[str, SourceIdentity]
) -> int:
    seen: set[str] = set()
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(source):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            observed = source_identity_for_row(source_file, index, row)
            if identities.get(observed.source_record_id) != observed:
                raise ValueError(
                    f"source identity changed during preflight: {observed.source_record_id}"
                )
            normalize_conversations(
                row.get("conversations"),
                require_nonempty_content=False,
                require_final_assistant=False,
                min_turns=1,
                max_turns=None,
            )
            seen.add(observed.source_record_id)
    if seen != set(identities):
        raise ValueError("source rows changed between identity and preflight scans")
    return len(seen)


def iter_pending_slots(
    source: str,
    identities: Mapping[str, SourceIdentity],
    secondary_source_ids: frozenset[str],
    completed: set[str],
) -> Iterator[PendingItem]:
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(source):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            identity = source_identity_for_row(source_file, index, row)
            if identities.get(identity.source_record_id) != identity:
                raise ValueError(
                    f"source identity changed after preflight: {identity.source_record_id}"
                )
            for slot in refinement_slots_for_source(identity, secondary_source_ids):
                if slot.synthetic_id not in completed:
                    yield PendingItem(row, slot)


def expected_synthetic_ids(
    identities: Mapping[str, SourceIdentity],
    secondary_source_ids: frozenset[str],
) -> set[str]:
    return {
        slot.synthetic_id
        for identity in identities.values()
        for slot in refinement_slots_for_source(identity, secondary_source_ids)
    }


def write_progress(
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
    value = {
        "source": source,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "completed_rows": completed_rows,
        "remaining_rows": target_rows - completed_rows,
        "model": model,
        "provider": CODEX_PROVIDER,
        "run_instance_id": run_instance_id,
        "source_content_sha256": source_content_sha256,
        "assignment": ASSIGNMENT_ALGORITHM_VERSION,
        "validation_policy": REFINEMENT_VALIDATION_POLICY_VERSION,
        "synchronization": "disabled",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".progress.json.tmp"
    destination = output_dir / "progress.json"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _generation_config(args: argparse.Namespace) -> dict[str, int]:
    return {
        "agent_batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "max_attempts_per_run": args.max_attempts_per_run,
        "timeout_seconds": args.timeout_seconds,
    }


def _load_source(args: argparse.Namespace) -> tuple[str, Any, dict[str, SourceIdentity], Any]:
    if not args.source_snapshot_dir.is_dir():
        raise FileNotFoundError(
            "sealed source snapshot does not exist; this worker refuses remote "
            f"sources and will not download it: {args.source_snapshot_dir}"
        )
    snapshot = load_local_source_snapshot(args.source_snapshot_dir)
    source = str(args.source_snapshot_dir)
    original_schema, identities = load_source_identities(source)
    secondary_source_ids = choose_secondary_source_ids(
        list(identities), args.target_rows
    )
    return source, original_schema, identities, (snapshot, secondary_source_ids)


def report_plan_or_status(args: argparse.Namespace) -> int:
    source, original_schema, identities, extra = _load_source(args)
    snapshot, secondary_source_ids = extra
    if not args.output_dir.exists():
        if args.status_only:
            print(f"No Codex refinement state exists at {args.output_dir}.")
        print(
            f"Verified local snapshot: {len(identities):,} source rows | "
            f"assigned target: {args.target_rows:,} | pending: {args.target_rows:,}"
        )
        print("No Codex calls or local state writes were made.")
        return 0
    if not args.output_dir.is_dir():
        raise ValueError(f"Codex output path is not a directory: {args.output_dir}")
    run_manifest = load_run_manifest(args.output_dir / "run_manifest.json")
    if run_manifest.target_rows != args.target_rows:
        raise ValueError("--target-rows conflicts with existing Codex run state")
    verify_refinement_state_inventory(args.output_dir)
    augmented_schema = build_augmented_schema(original_schema)
    completed, rows, _ = scan_accepted_shards(
        args.output_dir / "accepted",
        augmented_schema,
        expected_validation_policy=run_manifest.validation_policy_version,
    )
    expected = expected_synthetic_ids(identities, secondary_source_ids)
    unknown = completed.difference(expected)
    if unknown:
        raise ValueError(f"Codex state contains unknown IDs: {sorted(unknown)[:3]}")
    if rows != len(completed):
        raise ValueError("accepted Codex row count does not match unique IDs")
    if snapshot.source_content_sha256 != run_manifest.source_content_sha256:
        raise ValueError("Codex run source digest does not match the sealed snapshot")
    print(
        f"Codex state: accepted={len(completed):,} | "
        f"remaining={args.target_rows - len(completed):,} | model={run_manifest.model}"
    )
    print("State is local-only; this worker has no synchronization operation.")
    return 0


async def execute(args: argparse.Namespace, runtime: CodexRuntime) -> int:
    _assert_isolated_paths(args)
    source, original_schema, identities, extra = _load_source(args)
    snapshot, secondary_source_ids = extra
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(args.output_dir)
    try:
        source_manifest = ensure_source_manifest(
            args.output_dir / "source_manifest.json",
            requested_source=snapshot.requested_source,
            resolved_source=source,
            schema=original_schema,
            identities=identities,
        )
        manifest_model = f"codex-cli-chatgpt:{args.model}"
        run_manifest = ensure_run_manifest(
            args.output_dir / "run_manifest.json",
            source_manifest=source_manifest,
            target_rows=args.target_rows,
            model=manifest_model,
            generation_config=_generation_config(args),
        )
        inventory_path = args.output_dir / "checksum_inventory.json"
        complete_path = args.output_dir / "complete.json"
        if inventory_path.exists() != complete_path.exists():
            raise ValueError("Codex state has an incomplete inventory marker pair")
        if inventory_path.exists():
            inventory = verify_refinement_state_inventory(args.output_dir)
        else:
            preexisting = (
                (args.output_dir / "progress.json").exists()
                or (args.output_dir / "attempts.jsonl").exists()
                or any((args.output_dir / "accepted").glob("accepted-*.parquet"))
            )
            if preexisting:
                raise ValueError("unsealed pre-existing Codex state cannot be resumed")
            inventory = write_refinement_state_inventory(args.output_dir)

        augmented_schema = build_augmented_schema(original_schema)
        accepted_dir = args.output_dir / "accepted"
        completed, accepted_rows, fingerprints = scan_accepted_shards(
            accepted_dir,
            augmented_schema,
            expected_validation_policy=run_manifest.validation_policy_version,
        )
        expected = expected_synthetic_ids(identities, secondary_source_ids)
        unknown = completed.difference(expected)
        if unknown:
            raise ValueError(
                f"accepted Codex shards contain unknown IDs: {sorted(unknown)[:3]}"
            )
        if accepted_rows != len(completed):
            raise ValueError("accepted row count does not match unique completed IDs")

        attempts_path = args.output_dir / "attempts.jsonl"
        attempt_counts, malformed, previous_max_call = load_attempt_state(
            attempts_path
        )
        print(
            f"Source: {len(identities):,} | target: {args.target_rows:,} | "
            f"accepted: {len(completed):,} | "
            f"remaining: {args.target_rows - len(completed):,} | "
            f"malformed attempts: {malformed:,}"
        )
        write_progress(
            args.output_dir,
            source=source,
            source_rows=len(identities),
            target_rows=args.target_rows,
            completed_rows=len(completed),
            model=manifest_model,
            run_instance_id=run_manifest.run_instance_id,
            source_content_sha256=source_manifest.source_content_sha256,
        )
        inventory = write_refinement_state_inventory(
            args.output_dir,
            previous_entries=inventory,
            changed_paths={"progress.json"},
        )
        if len(completed) == args.target_rows:
            print("Codex refinement is already complete; no agent call was made.")
            return 0

        checked = preflight_source_rows(source, identities)
        if checked != len(identities):
            raise ValueError("source preflight row count changed")

        pending: Iterator[PendingItem] = iter_pending_slots(
            source, identities, secondary_source_ids, completed
        )
        if args.limit is not None:
            pending = itertools.islice(pending, args.limit)
        retry_queue: deque[PendingItem] = deque()
        attempts_this_run: Counter[str] = Counter()
        calls_used = 0
        accepted_this_run = 0

        def next_item() -> PendingItem | None:
            while retry_queue:
                candidate = retry_queue.popleft()
                if attempts_this_run[candidate.slot.synthetic_id] < args.max_attempts_per_run:
                    return candidate
            return next(pending, None)

        while calls_used < args.max_agent_calls:
            scheduled: list[tuple[int, list[PendingItem], dict[str, int]]] = []
            available_calls = min(
                args.concurrency, args.max_agent_calls - calls_used
            )
            for _ in range(available_calls):
                batch: list[PendingItem] = []
                while len(batch) < args.batch_size:
                    item = next_item()
                    if item is None:
                        break
                    batch.append(item)
                if not batch:
                    break
                call_number = previous_max_call + calls_used + len(scheduled) + 1
                attempt_numbers = {
                    item.slot.synthetic_id: (
                        attempt_counts[item.slot.synthetic_id]
                        + 1
                    )
                    for item in batch
                }
                scheduled.append((call_number, batch, attempt_numbers))
            if not scheduled:
                break
            calls_used += len(scheduled)
            results = await asyncio.gather(
                *(
                    process_batch(
                        runtime,
                        model=args.model,
                        items=batch,
                        attempt_numbers=attempt_numbers,
                        agent_call_number=call_number,
                        timeout_seconds=args.timeout_seconds,
                        original_schema=original_schema,
                    )
                    for call_number, batch, attempt_numbers in scheduled
                )
            )
            attempts = [
                attempt for result in results for attempt in result.attempts
            ]
            candidates = [
                candidate for result in results for candidate in result.candidates
            ]
            accepted_candidates, collision_retries, new_fingerprints = (
                resolve_candidate_collisions(candidates, fingerprints)
            )
            accepted_rows_batch = [candidate.row for candidate in accepted_candidates]
            accepted_shard: Path | None = None
            if accepted_rows_batch:
                accepted_ids = {str(row["run_id"]) for row in accepted_rows_batch}
                overlap = completed.intersection(accepted_ids)
                if overlap:
                    raise ValueError(
                        f"attempted to store completed IDs: {sorted(overlap)[:3]}"
                    )
                accepted_shard = write_accepted_shard(
                    accepted_dir, accepted_rows_batch, augmented_schema
                )
                completed.update(accepted_ids)
                fingerprints.update(new_fingerprints)
                accepted_this_run += len(accepted_ids)

            append_attempts(attempts_path, attempts)
            for attempt in attempts:
                synthetic_id = str(attempt["synthetic_id"])
                attempt_counts[synthetic_id] += 1
                attempts_this_run[synthetic_id] += 1
            incomplete = [
                item for result in results for item in result.incomplete
            ]
            incomplete.extend(collision_retries)
            for item in incomplete:
                if attempts_this_run[item.slot.synthetic_id] < args.max_attempts_per_run:
                    retry_queue.append(item)

            write_progress(
                args.output_dir,
                source=source,
                source_rows=len(identities),
                target_rows=args.target_rows,
                completed_rows=len(completed),
                model=manifest_model,
                run_instance_id=run_manifest.run_instance_id,
                source_content_sha256=source_manifest.source_content_sha256,
            )
            changed = {"attempts.jsonl", "progress.json"}
            if accepted_shard is not None:
                changed.add(accepted_shard.relative_to(args.output_dir).as_posix())
            inventory = write_refinement_state_inventory(
                args.output_dir,
                previous_entries=inventory,
                changed_paths=changed,
            )
            print(
                f"Codex calls: {calls_used}/{args.max_agent_calls} | "
                f"run accepted: {accepted_this_run:,} | "
                f"total accepted: {len(completed):,} | "
                f"retry queued: {len(retry_queue):,}"
            )

        if calls_used >= args.max_agent_calls:
            print(
                f"Stopped at the exact Codex call budget of {args.max_agent_calls}."
            )
        elif retry_queue:
            print("Stopped with retryable slots still queued; rerun to continue.")
        else:
            print("Selected pending slots are complete or exhausted for this run.")
        print("All output remains local; synchronization is not implemented here.")
        return 0
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _assert_isolated_paths(args)
        if args.preflight_only:
            runtime = inspect_codex_runtime()
            print(f"Codex ChatGPT authentication is ready ({runtime.version}).")
            print("No source scan or generation call was made.")
            return 0
        if not args.execute:
            return report_plan_or_status(args)
        runtime = inspect_codex_runtime()
        print(f"Codex ChatGPT authentication is ready ({runtime.version}).")
        return asyncio.run(execute(args, runtime))
    except KeyboardInterrupt:
        print("Interrupted. Durable accepted shards can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
