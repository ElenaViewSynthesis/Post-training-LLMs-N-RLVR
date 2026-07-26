"""Retired 225K candidate-generation worker.

The importable helpers remain temporarily for compatibility tests and audit
history, but executing this file exits before credentials or clients are
loaded. Use ``stream_refinement_worker.py`` for assigned-slot refinement.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

if __name__ == "__main__":
    print(
        "This 225K candidate-generation entry point is retired. "
        "Use stream_refinement_worker.py for assigned-slot refinement.",
        file=sys.stderr,
    )
    raise SystemExit(2)

DEFAULT_DATA_DIR = Path("~/pipeline/data")
DEFAULT_HF_BUCKET = (
    "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/raw_results"
)

SYSTEM_PROMPT = """You are an expert software/terminal agent and dataset curator.
Synthesize a realistic, high-quality agent trajectory for the given task.

The trajectory must:
1. Put a brief reasoning step before each tool call ("I'll start by...")
2. Include realistic bash commands (run_bash) or file edits (edit_file) in assistant turns
3. Follow each tool call with a plausible tool observation
4. End with an assistant summary confirming the task is complete
5. Match the OpenThoughts-Agent-SFT style: concise, technical, and step-by-step

The tool calls and observations are a synthetic training trajectory; do not claim that
you cannot access a terminal. Respond only with data matching the requested schema."""

TRAJECTORY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "description": "Ordered assistant trajectory turns, ending in a completion summary.",
            "minItems": 2,
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Who produced this trajectory turn.",
                        "enum": ["system", "user", "assistant", "tool"],
                    },
                    "content": {
                        "type": "string",
                        "description": "Reasoning, a tool call, its observation, or the final summary.",
                    },
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
    thinking_level: str
    concurrency: int
    batch_size: int
    max_output_tokens: int
    timeout_ms: int
    sync_every_batches: int
    hf_bucket: str
    sync_enabled: bool


class HFSyncError(RuntimeError):
    """Raised when authoritative pipeline state cannot be synchronized."""


def load_environment() -> None:
    """Load repository-local credentials, with the parent file as fallback.

    Older pipeline files load only ``../.env`` while the README tells users to
    create ``./.env``.  Supporting both locations avoids that mismatch.  An
    already-exported environment variable always wins.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - exercised during setup failures
        raise RuntimeError(
            "python-dotenv is not installed; run `uv sync` or install requirements.txt"
        ) from exc

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
        description="Resume unfinished trajectory variants with the Gemini API."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Pipeline data root used by path defaults (default: %(default)s)",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="Variant plan parquet (default: <data-dir>/variant_plan.parquet)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        help="Task parquet path (default: <data-dir>/tasks)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result JSONL (default: <data-dir>/raw_results/trajectories.jsonl)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL_ID", "gemini-3.6-flash"),
        help="Gemini model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("medium", "high"),
        default=os.getenv("GEMINI_THINKING_LEVEL", "medium"),
        help="Gemini 3.6 thinking level (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=int(os.getenv("GEMINI_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=int(os.getenv("GEMINI_BATCH_SIZE", "60")),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--sync-every-batches",
        type=_positive_int,
        default=int(os.getenv("GEMINI_SYNC_EVERY_BATCHES", "10")),
    )
    parser.add_argument(
        "--hf-bucket",
        default=os.getenv("HF_RAW_RESULTS_BUCKET", DEFAULT_HF_BUCKET),
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Process at most this many unfinished variants (recommended: 1 first).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not synchronize raw results to HF during or after this invocation.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print resume counts without requiring a Gemini key or making requests.",
    )
    parser.add_argument(
        "--sync-to-hf-before-status",
        action="store_true",
        help=(
            "Upload the authoritative local raw-results directory to --hf-bucket "
            "before calculating --status-only counts."
        ),
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser()
    args.plan = (args.plan or args.data_dir / "variant_plan.parquet").expanduser()
    args.tasks = (args.tasks or args.data_dir / "tasks").expanduser()
    args.output = (
        args.output or args.data_dir / "raw_results" / "trajectories.jsonl"
    ).expanduser()
    if args.sync_to_hf_before_status and not args.status_only:
        parser.error("--sync-to-hf-before-status requires --status-only")
    if args.sync_to_hf_before_status and args.no_sync:
        parser.error("--sync-to-hf-before-status cannot be combined with --no-sync")
    return args


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _first_text(task: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = task.get(name)
        if not _is_missing(value) and str(value).strip():
            return str(value).strip()
    return ""


def build_synthesis_prompt(task: dict[str, Any], framing: Any) -> str:
    instruction = _first_text(task, ("instruction", "prompt", "task_description"))
    environment = _first_text(task, ("environment", "dockerfile", "setup_script"))
    framing_text = "" if _is_missing(framing) else str(framing).strip()
    framed_instruction = (
        f"{framing_text}\n\n{instruction}" if framing_text else instruction
    )
    return (
        f"## Task\n{framed_instruction}\n\n"
        f"## Environment\n{environment or 'Generic Ubuntu 22.04 container.'}\n\n"
        "Synthesize a complete, realistic agent trajectory that solves this task."
    )


def variant_seed(variant_id: str) -> int:
    """Give every planned variant a stable, distinct best-effort Gemini seed."""
    digest = hashlib.sha256(variant_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def scan_completed(output_path: Path) -> tuple[set[str], int, int]:
    """Return successful IDs, valid record count, and malformed line count."""
    completed: set[str] = set()
    records = malformed = 0
    if not output_path.exists():
        return completed, records, malformed

    with output_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            records += 1
            if record.get("status") == "ok" and record.get("variant_id"):
                completed.add(str(record["variant_id"]))
    return completed, records, malformed


def _task_id_column(columns: Any) -> str:
    for candidate in ("task_id", "id", "task", "run_id"):
        if candidate in columns:
            return candidate
    if len(columns) == 0:
        raise ValueError("task parquet has no columns")
    return str(columns[0])


def load_work(plan_path: Path, tasks_path: Path, output_path: Path, limit: int | None):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised during setup failures
        raise RuntimeError(
            "pandas/pyarrow are not installed; run `uv sync` or install requirements.txt"
        ) from exc

    if not plan_path.exists():
        raise FileNotFoundError(f"variant plan not found: {plan_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"task parquet not found: {tasks_path}")

    plan = pd.read_parquet(plan_path)
    required = {"variant_id", "task_id", "temperature", "instruction_framing"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"variant plan is missing columns: {sorted(missing)}")
    if plan["variant_id"].duplicated().any():
        raise ValueError("variant plan contains duplicate variant_id values")

    completed, existing_records, malformed = scan_completed(output_path)
    unfinished = plan.loc[~plan["variant_id"].astype(str).isin(completed)].copy()
    if limit is not None:
        unfinished = unfinished.iloc[:limit].copy()

    tasks = pd.read_parquet(tasks_path)
    id_column = _task_id_column(tasks.columns)
    task_lookup = {str(row[id_column]): row.to_dict() for _, row in tasks.iterrows()}
    return plan, unfinished, task_lookup, completed, existing_records, malformed


def _usage_dict(interaction: Any) -> dict[str, int]:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {}
    fields = (
        "total_input_tokens",
        "total_output_tokens",
        "total_thought_tokens",
        "total_tokens",
        "total_tool_use_tokens",
        "total_cached_tokens",
    )
    return {
        field: int(value)
        for field in fields
        if (value := getattr(usage, field, None)) is not None
    }


def validate_conversations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("'conversations' must be a list")
    if not 2 <= len(value) <= 40:
        raise ValueError("'conversations' must contain between 2 and 40 turns")
    allowed_roles = {"system", "user", "assistant", "tool"}
    for index, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise TypeError(f"conversation turn {index} is not an object")
        if turn.get("role") not in allowed_roles:
            raise ValueError(f"conversation turn {index} has an invalid role")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError(f"conversation turn {index} has empty content")
    return value


def _safe_error(exc: Exception, api_key: str) -> tuple[str, int | None]:
    code = getattr(exc, "code", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    message = " ".join(str(exc).split())[:500]
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    prefix = f"error:{type(exc).__name__}"
    if code is not None:
        prefix += f":{code}"
    return f"{prefix}:{message}", code


def _base_record(variant: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "variant_id": str(variant["variant_id"]),
        "task_id": str(variant["task_id"]),
        "temperature": float(variant["temperature"]),
        "temperature_applied": False,
        "provider": "gemini",
        "model": settings.model,
        "thinking_level": settings.thinking_level,
    }


async def synthesize_trajectory(
    client: Any,
    variant: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    base = _base_record(variant, settings)
    variant_id = base["variant_id"]
    prompt = build_synthesis_prompt(task, variant.get("instruction_framing"))

    async with semaphore:
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
                    "thinking_level": settings.thinking_level,
                    "seed": variant_seed(variant_id),
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
            conversations = validate_conversations(parsed.get("conversations"))

            return {
                **base,
                "status": "ok",
                "conversations": conversations,
                "raw": None,
                "interaction_id": getattr(interaction, "id", None),
                "usage": _usage_dict(interaction),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            status, code = _safe_error(exc, api_key)
            return {
                **base,
                "status": f"parse_{status}",
                "error_code": code,
                "conversations": None,
                "raw": locals().get("raw", "")[:20_000] or None,
            }
        # Provider releases expose several unrelated error subclasses, all of
        # which must become resumable JSONL failures instead of killing tasks.
        except Exception as exc:  # noqa: BLE001
            status, code = _safe_error(exc, api_key)
            return {
                **base,
                "status": status,
                "error_code": code,
                "conversations": None,
                "raw": None,
            }


def append_record(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


async def run_batch(
    client: Any,
    variants: list[dict[str, Any]],
    task_lookup: dict[str, dict[str, Any]],
    settings: Settings,
    api_key: str,
    semaphore: asyncio.Semaphore,
    stream: TextIO,
) -> list[dict[str, Any]]:
    requests = []
    records: list[dict[str, Any]] = []
    for variant in variants:
        task_id = str(variant["task_id"])
        task = task_lookup.get(task_id)
        if task is None:
            record = {
                **_base_record(variant, settings),
                "status": "error:MissingTask",
                "error_code": None,
                "conversations": None,
                "raw": None,
            }
            append_record(stream, record)
            records.append(record)
            continue
        requests.append(
            synthesize_trajectory(client, variant, task, settings, api_key, semaphore)
        )

    for future in asyncio.as_completed(requests):
        record = await future
        records.append(record)
        append_record(stream, record)
    os.fsync(stream.fileno())
    return records


def should_stop(records: list[dict[str, Any]]) -> tuple[bool, str]:
    """Stop safely on auth/config failures or a completely failed batch."""
    if not records:
        return True, "the batch produced no request records"

    fatal_codes = {400, 401, 403, 404}
    for record in records:
        if record.get("error_code") in fatal_codes:
            return True, f"Gemini returned fatal HTTP {record['error_code']}"

    ok_count = sum(record.get("status") == "ok" for record in records)
    if ok_count == 0:
        return True, "the entire batch failed after SDK retries"
    return False, ""


def sync_to_hf(output_path: Path, bucket: str) -> None:
    try:
        from huggingface_hub import sync_bucket
    except ImportError as exc:  # pragma: no cover - exercised during setup failures
        raise HFSyncError(
            "huggingface_hub with Bucket support is not installed; run `uv sync`"
        ) from exc

    try:
        sync_bucket(str(output_path.parent), bucket)
    except Exception as exc:
        raise HFSyncError(
            f"failed to sync {output_path.parent} to {bucket}: {exc}"
        ) from exc


def acquire_run_lock(output_path: Path) -> TextIO:
    output_digest = hashlib.sha256(str(output_path.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"gemini-trajectory-{output_digest}.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(
            f"another trajectory worker holds the run lock: {lock_path}"
        ) from exc
    return lock_stream


async def async_main(args: argparse.Namespace) -> int:
    status_lock = None
    if args.sync_to_hf_before_status:
        if not args.output.exists():
            raise FileNotFoundError(
                f"authoritative local result file not found: {args.output}"
            )
        status_lock = acquire_run_lock(args.output)

    try:
        if args.sync_to_hf_before_status:
            print(
                f"Syncing authoritative local results from {args.output.parent} "
                f"to {args.hf_bucket} ..."
            )
            sync_to_hf(args.output, args.hf_bucket)
            print("Raw-result sync complete; calculating resume status.")
        (
            plan,
            unfinished,
            task_lookup,
            completed,
            existing_records,
            malformed,
        ) = load_work(args.plan, args.tasks, args.output, args.limit)
    finally:
        if status_lock is not None:
            status_lock.close()

    total_remaining = len(plan) - len(completed)
    print(
        f"Plan: {len(plan):,} | successful: {len(completed):,} | "
        f"remaining: {total_remaining:,} | existing records: {existing_records:,} | "
        f"malformed lines: {malformed:,}"
    )
    if args.limit is not None:
        print(f"This invocation is limited to {len(unfinished):,} variants.")
    if args.status_only or unfinished.empty:
        return 0

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set; export it or add it to ./.env or ../.env"
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed; run `uv sync` or install requirements.txt"
        ) from exc

    settings = Settings(
        model=args.model,
        thinking_level=args.thinking_level,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        timeout_ms=args.timeout_seconds * 1000,
        sync_every_batches=args.sync_every_batches,
        hf_bucket=args.hf_bucket,
        sync_enabled=not args.no_sync,
    )
    print(
        f"Backend: Gemini model={settings.model!r} concurrency={settings.concurrency} "
        f"batch_size={settings.batch_size} thinking_level={settings.thinking_level} "
        f"max_output_tokens={settings.max_output_tokens}"
    )

    # The official SDK retries 408, 429, and 5xx responses with jittered
    # exponential backoff.  Explicit values keep behavior stable across SDK
    # releases; the worker-level circuit breaker handles persistent failures.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = acquire_run_lock(args.output)
    try:
        sync_client = genai.Client(
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
        client = sync_client.aio
        semaphore = asyncio.Semaphore(settings.concurrency)
        total_batches = math.ceil(len(unfinished) / settings.batch_size)
        total_ok = total_errors = 0
        try:
            with args.output.open("a", encoding="utf-8") as output_stream:
                for batch_number, start in enumerate(
                    range(0, len(unfinished), settings.batch_size), start=1
                ):
                    frame = unfinished.iloc[start : start + settings.batch_size]
                    variants = frame.to_dict(orient="records")
                    records = await run_batch(
                        client,
                        variants,
                        task_lookup,
                        settings,
                        api_key,
                        semaphore,
                        output_stream,
                    )
                    ok_count = sum(record.get("status") == "ok" for record in records)
                    error_count = len(records) - ok_count
                    total_ok += ok_count
                    total_errors += error_count
                    print(
                        f"Batch {batch_number}/{total_batches}: ok={ok_count} "
                        f"errors={error_count} | run total ok={total_ok} "
                        f"errors={total_errors}"
                    )

                    stop, reason = should_stop(records)
                    if stop:
                        print(
                            f"Stopping safely: {reason}. "
                            "Failed variants remain resumable."
                        )
                        return 2

                    if (
                        settings.sync_enabled
                        and batch_number % settings.sync_every_batches == 0
                    ):
                        try:
                            sync_to_hf(args.output, settings.hf_bucket)
                            print(f"Synced results after batch {batch_number}.")
                        except HFSyncError as exc:
                            print(f"[HF sync warning] {exc}", file=sys.stderr)

            if settings.sync_enabled:
                try:
                    sync_to_hf(args.output, settings.hf_bucket)
                    print("Final result sync complete.")
                except HFSyncError as exc:
                    print(f"[HF sync warning] {exc}", file=sys.stderr)
            return 0
        finally:
            await client.aclose()
    finally:
        lock_stream.close()


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted. Completed JSONL records are safe; rerun to resume.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints a concise error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
