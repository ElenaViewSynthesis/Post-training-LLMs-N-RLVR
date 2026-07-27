"""Bounded-memory primitives for source-row conversation refinement.

The refinement pipeline assigns exactly one slot to every source row and a
second slot to a deterministic subset when the requested target exceeds the
source row count.  A slot stops after its first accepted conversation; rejected
attempts are retried for that same slot instead of creating extra candidates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
REQUIRED_SOURCE_FIELDS = {
    "conversations",
    "run_id",
    "task",
    "trial_name",
}
AUGMENTATION_FIELDS = {
    "is_synthetic_augmentation",
    "source_record_id",
    "source_file",
    "source_row_index",
    "source_trial_name",
    "source_conversation_fingerprint",
    "refined_conversation_fingerprint",
    "source_task_id",
    "source_run_id",
    "refinement_index",
}


@dataclass(frozen=True)
class SourceIdentity:
    """Stable source-row identity plus original lineage fields."""

    source_record_id: str
    source_file: str
    source_row_index: int
    source_trial_name: str
    source_conversation_fingerprint: str
    source_run_id: str
    source_task_id: str


@dataclass(frozen=True)
class RefinementSlot:
    """One deterministic refinement assigned to a source dataset row."""

    source_record_id: str
    source_file: str
    source_row_index: int
    source_trial_name: str
    source_conversation_fingerprint: str
    source_run_id: str
    source_task_id: str
    refinement_index: int
    synthetic_id: str


@dataclass(frozen=True)
class RefinementQualityIssue:
    """One stable, machine-readable reason a candidate cannot be accepted."""

    code: str
    detail: str


REFINEMENT_VALIDATION_POLICY_VERSION = "quality-v1"

# Calibrated against 173 locally available successful trajectories. Their
# lowest rewritten-user source-token retention was 5.09%, and the lowest
# assistant/tool retention was 15.82%. The rounded-down floors remain
# conservative while exact anchor loss is considered separately.
MIN_LONG_TASK_TOKEN_RETENTION = 0.05
MIN_EXECUTION_TASK_TOKEN_RETENTION = 0.15
WEAK_TASK_RETENTION_FOR_ANCHOR_LOSS = 0.25

_TASK_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/#:@~-]*")
_URL_PATTERN = re.compile(r"https?://[^\s`\"'<>]+", re.IGNORECASE)
_PATH_PATTERN = re.compile(
    r"(?<!\w)(?:~?/|\./|\.\./)[^\s`\"'<>(),;]+"
)
_BACKTICK_PATTERN = re.compile(r"`([^`\n]+)`")
_QUOTED_PATTERN = re.compile(r'''(?<![A-Za-z])["']([^"'\n]{2,120})["']''')
_CLI_FLAG_PATTERN = re.compile(r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9-]*")
_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)*(?!\w)")
_REFUSAL_ONLY_PATTERN = re.compile(
    r"\b(?:i(?:['’]m| am) sorry|i apologize|"
    r"i (?:cannot|can't|won't|am unable to) "
    r"(?:help|assist|comply|fulfill)|"
    r"(?:cannot|can't|unable to) "
    r"(?:complete|continue|proceed|perform|fulfill)|"
    r"request cannot be fulfilled|as an ai|something went wrong|no solution)\b",
    re.IGNORECASE,
)
_TASK_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "can",
        "could",
        "does",
        "for",
        "from",
        "have",
        "how",
        "into",
        "its",
        "may",
        "not",
        "please",
        "should",
        "that",
        "the",
        "their",
        "then",
        "this",
        "use",
        "using",
        "want",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def _list_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise TypeError("conversations must be a list-like sequence")


def normalize_conversations(
    value: Any,
    *,
    require_nonempty_content: bool = True,
    require_final_assistant: bool = True,
) -> list[dict[str, str]]:
    """Normalize a conversation without allowing Dask-style stringification."""
    turns = _list_like(value)
    if not 2 <= len(turns) <= 40:
        raise ValueError("conversations must contain between 2 and 40 turns")

    normalized: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise TypeError(f"conversation turn {index} is not a mapping")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"conversation turn {index} has invalid role {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"conversation turn {index} content is not a string")
        if require_nonempty_content and not content.strip():
            raise ValueError(f"conversation turn {index} has empty content")
        normalized.append({"content": content, "role": role})

    if require_final_assistant and normalized[-1]["role"] != "assistant":
        raise ValueError("conversation must end with an assistant turn")
    return normalized


def canonical_conversation_json(conversations: Any) -> str:
    """Return a stable representation used for retry and audit fingerprints."""
    normalized = normalize_conversations(conversations)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def conversation_fingerprint(conversations: Any) -> str:
    return hashlib.sha256(canonical_conversation_json(conversations).encode()).hexdigest()


def source_conversation_fingerprint(conversations: Any) -> str:
    """Fingerprint source turns while permitting source-specific empty/final roles."""
    normalized = normalize_conversations(
        conversations,
        require_nonempty_content=False,
        require_final_assistant=False,
    )
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _normalized_content_key(content: str) -> str:
    return " ".join(content.casefold().split())


def _task_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TASK_TOKEN_PATTERN.finditer(text):
        token = match.group(0).casefold().strip(".,:;")
        if len(token) >= 3 and token not in _TASK_STOPWORDS:
            tokens.add(token)
    return tokens


def _task_token_sequence(text: str) -> list[str]:
    return [
        match.group(0).casefold().strip(".,:;")
        for match in _TASK_TOKEN_PATTERN.finditer(text)
        if len(match.group(0).strip(".,:;")) >= 2
    ]


def _task_anchors(text: str) -> set[str]:
    """Extract exact, high-signal values whose complete loss suggests drift."""
    anchors: set[str] = set()
    for pattern in (
        _URL_PATTERN,
        _PATH_PATTERN,
        _CLI_FLAG_PATTERN,
        _NUMBER_PATTERN,
    ):
        anchors.update(
            match.group(0).casefold().rstrip(".,:;")
            for match in pattern.finditer(text)
        )

    quoted_values = [
        match.group(1) for match in _BACKTICK_PATTERN.finditer(text)
    ]
    quoted_values.extend(
        match.group(1) for match in _QUOTED_PATTERN.finditer(text)
    )
    for value in quoted_values:
        normalized = _normalized_content_key(value).rstrip(".,:;")
        if 2 <= len(normalized) <= 120:
            anchors.add(normalized)

    for match in _TASK_TOKEN_PATTERN.finditer(text):
        raw = match.group(0).strip(".,:;")
        if (
            any(character in raw for character in "_./:@+-")
            or any(character.isdigit() for character in raw)
            or (any(character.isupper() for character in raw[1:]) and len(raw) >= 4)
        ):
            anchors.add(raw.casefold())
    return {anchor for anchor in anchors if anchor}


def validate_refinement_quality(
    source_conversations: Any,
    candidate_conversations: Any,
) -> tuple[RefinementQualityIssue, ...]:
    """Apply deterministic, side-effect-free acceptance checks.

    Structural schema validation remains the responsibility of
    :func:`normalize_conversations`. This validator covers semantic-preservation
    proxies and trajectory-quality invariants that the JSON schema cannot
    express.
    """
    source = normalize_conversations(
        source_conversations,
        require_nonempty_content=False,
        require_final_assistant=False,
    )
    candidate = normalize_conversations(
        candidate_conversations,
        require_final_assistant=False,
    )
    issues: dict[str, RefinementQualityIssue] = {}

    def reject(code: str, detail: str) -> None:
        existing = issues.get(code)
        if existing is None:
            issues[code] = RefinementQualityIssue(code, detail)
        elif detail not in existing.detail:
            issues[code] = RefinementQualityIssue(
                code,
                f"{existing.detail}; {detail}",
            )

    roles = [turn["role"] for turn in candidate]
    if "user" not in roles or "assistant" not in roles:
        reject(
            "missing_participant",
            "candidate must contain at least one user and one assistant turn",
        )
    if candidate[-1]["role"] != "assistant":
        reject(
            "invalid_turn_order",
            "candidate must end with an assistant interpretation or summary",
        )

    for index, role in enumerate(roles):
        if role != "tool":
            continue
        previous = index - 1
        while previous >= 0 and roles[previous] == "tool":
            previous -= 1
        following = index + 1
        while following < len(roles) and roles[following] == "tool":
            following += 1
        if previous < 0 or roles[previous] != "assistant":
            reject(
                "orphan_tool",
                f"tool block beginning near turn {index} has no assistant invocation",
            )
        if following >= len(roles) or roles[following] != "assistant":
            reject(
                "orphan_tool",
                f"tool block ending near turn {index} has no assistant interpretation",
            )

    final_assistant = next(
        (
            turn["content"]
            for turn in reversed(candidate)
            if turn["role"] == "assistant"
        ),
        "",
    )
    if (
        len(_task_token_sequence(final_assistant)) <= 80
        and _REFUSAL_ONLY_PATTERN.search(final_assistant)
    ):
        reject(
            "refusal_only",
            "candidate ends in a refusal or generic failure",
        )

    content_keys = [
        _normalized_content_key(turn["content"]) for turn in candidate
    ]
    repeated = Counter(content_keys)
    if any(
        left == right
        for left, right in zip(content_keys, content_keys[1:], strict=False)
    ):
        reject("degenerate_content", "candidate repeats adjacent turn content")
    if repeated and max(repeated.values()) >= 3:
        reject("degenerate_content", "candidate repeats the same content three times")
    if len(content_keys) >= 6 and len(repeated) / len(content_keys) < 0.5:
        reject("degenerate_content", "candidate has insufficient turn diversity")

    participant_text = "\n".join(
        turn["content"]
        for turn in candidate
        if turn["role"] in {"user", "assistant"}
    )
    participant_tokens = _task_token_sequence(participant_text)
    if len(participant_tokens) < 4 or len(participant_text.strip()) < 24:
        reject("degenerate_content", "candidate contains too little task information")
    if (
        len(participant_tokens) >= 8
        and len(set(participant_tokens)) / len(participant_tokens) < 0.25
    ):
        reject("degenerate_content", "candidate content is excessively repetitive")

    source_task = "\n".join(
        turn["content"] for turn in source if turn["role"] == "user"
    )
    candidate_task = "\n".join(
        turn["content"] for turn in candidate if turn["role"] == "user"
    )
    candidate_execution = "\n".join(
        turn["content"]
        for turn in candidate
        if turn["role"] in {"assistant", "tool"}
    )
    source_tokens = _task_tokens(source_task)
    candidate_tokens = _task_tokens(candidate_task)
    overlap_count = len(source_tokens.intersection(candidate_tokens))
    retention = overlap_count / len(source_tokens) if source_tokens else 1.0
    if source_tokens and (
        overlap_count == 0
        or (
            len(source_tokens) >= 20
            and retention < MIN_LONG_TASK_TOKEN_RETENTION
        )
    ):
        reject(
            "task_drift",
            f"source-task token retention {retention:.3f} is below policy",
        )

    execution_tokens = _task_tokens(candidate_execution)
    execution_retention = (
        len(source_tokens.intersection(execution_tokens)) / len(source_tokens)
        if source_tokens
        else 1.0
    )
    if (
        len(source_tokens) >= 5
        and execution_retention < MIN_EXECUTION_TASK_TOKEN_RETENTION
    ):
        reject(
            "task_drift",
            "assistant/tool task-token retention "
            f"{execution_retention:.3f} is below policy",
        )

    source_anchors = _task_anchors(source_task)
    candidate_anchors = _task_anchors(
        f"{candidate_task}\n{candidate_execution}"
    )
    retained_anchors = source_anchors.intersection(candidate_anchors)
    if (
        source_anchors
        and not retained_anchors
        and (
            len(source_anchors) >= 2
            or retention < WEAK_TASK_RETENTION_FOR_ANCHOR_LOSS
        )
    ):
        reject(
            "task_drift",
            "candidate loses every required high-signal task anchor",
        )

    if candidate == source:
        reject(
            "unchanged_conversation",
            "refined conversation is identical to the source conversation",
        )
    return tuple(issues.values())


def stable_synthetic_id(source_record_id: str, refinement_index: int) -> str:
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError("source record ID must be a non-empty string")
    if refinement_index not in (0, 1):
        raise ValueError("refinement_index must be 0 or 1")
    digest = hashlib.sha256(
        f"refinement\0{source_record_id}\0{refinement_index}".encode()
    ).hexdigest()[:24]
    return f"refined-{digest}"


def stable_source_record_id(
    source_file: str,
    source_row_index: int,
    source_trial_name: str,
    source_run_id: str,
    source_task_id: str,
    conversation_sha256: str,
) -> str:
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("source file must be a non-empty string")
    if not isinstance(source_row_index, int) or source_row_index < 0:
        raise ValueError("source row index must be a non-negative integer")
    lineage = (
        source_trial_name,
        source_run_id,
        source_task_id,
        conversation_sha256,
    )
    if any(not isinstance(value, str) or not value for value in lineage):
        raise ValueError("source lineage and conversation fingerprint must be strings")
    digest = hashlib.sha256(
        (
            f"source-row\0{source_file}\0{source_row_index}\0"
            f"{source_trial_name}\0{source_run_id}\0{source_task_id}\0"
            f"{conversation_sha256}"
        ).encode()
    ).hexdigest()[:24]
    return f"source-{digest}"


def _update_source_identity_digest(
    digest: Any, identity: SourceIdentity
) -> None:
    digest.update(
        (
            f"{identity.source_record_id}\0{identity.source_file}\0"
            f"{identity.source_row_index}\0{identity.source_trial_name}\0"
            f"{identity.source_run_id}\0{identity.source_task_id}\0"
            f"{identity.source_conversation_fingerprint}\n"
        ).encode()
    )


def source_identity_digest(identities: Iterable[SourceIdentity]) -> str:
    """Hash source identities in deterministic physical iteration order."""
    digest = hashlib.sha256()
    for identity in identities:
        _update_source_identity_digest(digest, identity)
    return digest.hexdigest()


def source_schema_digest(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def pin_source_revision(source: str) -> str:
    """Resolve an HF dataset URI to an immutable full commit revision."""
    if not source.startswith("hf://datasets/"):
        return source

    from huggingface_hub import HfApi, parse_hf_uri

    uri = parse_hf_uri(source)
    info = HfApi().dataset_info(uri.id, revision=uri.revision)
    revision = getattr(info, "sha", None)
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in string.hexdigits for character in revision)
    ):
        raise ValueError(f"could not resolve an immutable revision for {source}")
    suffix = f"/{uri.path_in_repo}" if uri.path_in_repo else ""
    return f"hf://datasets/{uri.id}@{revision}{suffix}"


@dataclass(frozen=True)
class SourceManifest:
    version: int
    requested_source: str
    resolved_source: str
    source_rows: int
    source_identity_sha256: str
    source_schema_sha256: str


def load_source_manifest(path: Path) -> SourceManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"source manifest is not an object: {path}")
    try:
        manifest = SourceManifest(
            version=value["version"],
            requested_source=value["requested_source"],
            resolved_source=value["resolved_source"],
            source_rows=value["source_rows"],
            source_identity_sha256=value["source_identity_sha256"],
            source_schema_sha256=value["source_schema_sha256"],
        )
    except KeyError as exc:
        raise ValueError(f"source manifest is missing {exc.args[0]!r}: {path}") from exc
    if (
        manifest.version != 1
        or not isinstance(manifest.requested_source, str)
        or not manifest.requested_source
        or not isinstance(manifest.resolved_source, str)
        or not manifest.resolved_source
        or not isinstance(manifest.source_rows, int)
        or manifest.source_rows <= 0
        or not isinstance(manifest.source_identity_sha256, str)
        or len(manifest.source_identity_sha256) != 64
        or any(
            character not in string.hexdigits
            for character in manifest.source_identity_sha256
        )
        or not isinstance(manifest.source_schema_sha256, str)
        or len(manifest.source_schema_sha256) != 64
        or any(
            character not in string.hexdigits
            for character in manifest.source_schema_sha256
        )
    ):
        raise ValueError(f"source manifest contains invalid values: {path}")
    return manifest


def resolve_source_for_run(manifest_path: Path, requested_source: str) -> str:
    """Reuse a run's pinned source, or resolve a new immutable source."""
    if manifest_path.exists():
        manifest = load_source_manifest(manifest_path)
        if requested_source not in {
            manifest.requested_source,
            manifest.resolved_source,
        }:
            raise ValueError(
                "requested source conflicts with the existing source manifest: "
                f"{requested_source!r}"
            )
        return manifest.resolved_source
    return pin_source_revision(requested_source)


def ensure_source_manifest(
    path: Path,
    *,
    requested_source: str,
    resolved_source: str,
    schema: pa.Schema,
    identities: Mapping[str, SourceIdentity],
) -> SourceManifest:
    manifest = SourceManifest(
        version=1,
        requested_source=requested_source,
        resolved_source=resolved_source,
        source_rows=len(identities),
        source_identity_sha256=source_identity_digest(identities.values()),
        source_schema_sha256=source_schema_digest(schema),
    )
    if path.exists():
        existing = load_source_manifest(path)
        if existing != manifest:
            raise ValueError("source data does not match the immutable run manifest")
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return manifest


def choose_secondary_source_ids(
    source_record_ids: Sequence[str], target_rows: int
) -> frozenset[str]:
    """Choose the exact source rows that receive a second refinement."""
    source_ids = list(source_record_ids)
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("every source record ID must be a non-empty string")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source record IDs must be unique")
    if not len(source_ids) <= target_rows <= 2 * len(source_ids):
        raise ValueError(
            "target rows must be between the source row count and twice that count"
        )

    additional = target_rows - len(source_ids)
    ranked = sorted(
        source_ids,
        key=lambda source_id: (
            hashlib.sha256(f"secondary\0{source_id}".encode()).digest(),
            source_id,
        ),
    )
    return frozenset(ranked[:additional])


def refinement_slots_for_source(
    identity: SourceIdentity,
    secondary_source_ids: frozenset[str],
) -> tuple[RefinementSlot, ...]:
    if not isinstance(identity.source_task_id, str) or not identity.source_task_id:
        raise ValueError("source task must be a non-empty string")
    indices = (
        (0, 1) if identity.source_record_id in secondary_source_ids else (0,)
    )
    return tuple(
        RefinementSlot(
            source_record_id=identity.source_record_id,
            source_file=identity.source_file,
            source_row_index=identity.source_row_index,
            source_trial_name=identity.source_trial_name,
            source_conversation_fingerprint=(
                identity.source_conversation_fingerprint
            ),
            source_run_id=identity.source_run_id,
            source_task_id=identity.source_task_id,
            refinement_index=index,
            synthetic_id=stable_synthetic_id(identity.source_record_id, index),
        )
        for index in indices
    )


def resolve_parquet_source(source: str) -> tuple[Any, list[str]]:
    filesystem, path_pattern = fsspec.core.url_to_fs(source)
    matches = sorted(filesystem.glob(path_pattern))
    parquet_files = [path for path in matches if str(path).endswith(".parquet")]
    if not parquet_files and filesystem.isdir(path_pattern):
        parquet_files = sorted(filesystem.glob(f"{path_pattern.rstrip('/')}/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files matched {source}")
    return filesystem, parquet_files


def read_source_schema(source: str) -> pa.Schema:
    filesystem, files = resolve_parquet_source(source)
    expected: pa.Schema | None = None
    for path in files:
        with filesystem.open(path, "rb") as stream:
            actual = pq.read_schema(stream).remove_metadata()
        if expected is None:
            expected = actual
        elif not actual.equals(expected, check_metadata=False):
            raise TypeError(f"source parquet schema mismatch in {path}")
    assert expected is not None
    missing = REQUIRED_SOURCE_FIELDS.difference(expected.names)
    if missing:
        raise ValueError(f"source schema is missing fields: {sorted(missing)}")
    return expected


def iter_source_batches(
    source: str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 1_024,
) -> Iterator[pa.RecordBatch]:
    filesystem, files = resolve_parquet_source(source)
    for path in files:
        with filesystem.open(path, "rb") as stream:
            parquet_file = pq.ParquetFile(stream)
            yield from parquet_file.iter_batches(
                batch_size=batch_size,
                columns=list(columns) if columns is not None else None,
            )


def iter_source_batches_with_coordinates(
    source: str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 1_024,
) -> Iterator[tuple[str, int, pa.RecordBatch]]:
    """Yield each batch with its source file and first physical row index."""
    filesystem, files = resolve_parquet_source(source)
    for path in files:
        row_offset = 0
        with filesystem.open(path, "rb") as stream:
            parquet_file = pq.ParquetFile(stream)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=list(columns) if columns is not None else None,
            ):
                yield str(path), row_offset, batch
                row_offset += batch.num_rows


def source_identity_for_row(
    source_file: str,
    source_row_index: int,
    row: Mapping[str, Any],
) -> SourceIdentity:
    source_trial_name = row.get("trial_name")
    source_run_id = row.get("run_id")
    source_task_id = row.get("task")
    conversation_sha256 = source_conversation_fingerprint(row.get("conversations"))
    if not isinstance(source_trial_name, str) or not source_trial_name:
        raise ValueError("source contains a missing or non-string trial_name")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("source contains a missing or non-string run_id")
    if not isinstance(source_task_id, str) or not source_task_id:
        raise ValueError(
            f"source row {source_run_id!r} has a missing or non-string task"
        )
    return SourceIdentity(
        source_record_id=stable_source_record_id(
            source_file,
            source_row_index,
            source_trial_name,
            source_run_id,
            source_task_id,
            conversation_sha256,
        ),
        source_file=source_file,
        source_row_index=source_row_index,
        source_trial_name=source_trial_name,
        source_conversation_fingerprint=conversation_sha256,
        source_run_id=source_run_id,
        source_task_id=source_task_id,
    )


def load_source_identities(
    source: str,
) -> tuple[pa.Schema, dict[str, SourceIdentity]]:
    """Load compact physical-row-keyed source relationships into memory.

    The real dataset contains repeated ``run_id`` and ``trial_name`` values.
    Sorted source file path plus physical row index is therefore the authoritative
    row identity. Logical identifiers remain lineage metadata.
    """
    schema = read_source_schema(source)
    identities: dict[str, SourceIdentity] = {}
    for source_file, row_offset, batch in iter_source_batches_with_coordinates(
        source, columns=("conversations", "trial_name", "run_id", "task")
    ):
        for index, row in enumerate(batch.to_pylist(), start=row_offset):
            identity = source_identity_for_row(source_file, index, row)
            if identity.source_record_id in identities:
                raise ValueError(
                    f"duplicate physical source identity: {identity.source_record_id}"
                )
            identities[identity.source_record_id] = identity
    return schema, identities


def build_augmented_schema(original_schema: pa.Schema) -> pa.Schema:
    overlap = AUGMENTATION_FIELDS.intersection(original_schema.names)
    if overlap:
        raise ValueError(
            f"source schema already contains augmentation fields: {sorted(overlap)}"
        )
    return pa.schema(
        [
            *original_schema,
            pa.field("is_synthetic_augmentation", pa.bool_(), nullable=False),
            pa.field(
                "source_record_id",
                pa.string(),
                nullable=True,
            ),
            pa.field("source_file", pa.string(), nullable=True),
            pa.field("source_row_index", pa.int64(), nullable=True),
            pa.field(
                "source_trial_name",
                original_schema.field("trial_name").type,
                nullable=True,
            ),
            pa.field("source_conversation_fingerprint", pa.string(), nullable=True),
            pa.field("refined_conversation_fingerprint", pa.string(), nullable=True),
            pa.field(
                "source_task_id",
                original_schema.field("task").type,
                nullable=True,
            ),
            pa.field(
                "source_run_id",
                original_schema.field("run_id").type,
                nullable=True,
            ),
            pa.field("refinement_index", pa.int8(), nullable=True),
        ]
    )


def build_refined_row(
    source_row: Mapping[str, Any],
    slot: RefinementSlot,
    conversations: Any,
    original_schema: pa.Schema,
    *,
    model: str,
    provider: str,
) -> dict[str, Any]:
    if source_row.get("trial_name") != slot.source_trial_name:
        raise ValueError("refinement slot does not match the source row trial_name")
    if source_row.get("run_id") != slot.source_run_id:
        raise ValueError("refinement slot does not match the source row run_id")
    if source_row.get("task") != slot.source_task_id:
        raise ValueError("refinement slot does not match the source row task")
    if (
        source_conversation_fingerprint(source_row.get("conversations"))
        != slot.source_conversation_fingerprint
    ):
        raise ValueError("refinement slot does not match the source conversation")

    normalized_conversations = normalize_conversations(conversations)
    row = {name: source_row.get(name) for name in original_schema.names}
    row["conversations"] = normalized_conversations
    row["run_id"] = slot.synthetic_id
    row["trial_name"] = slot.synthetic_id
    if "model" in row:
        row["model"] = model
    if "model_provider" in row:
        row["model_provider"] = provider
    row["is_synthetic_augmentation"] = True
    row["source_record_id"] = slot.source_record_id
    row["source_file"] = slot.source_file
    row["source_row_index"] = slot.source_row_index
    row["source_trial_name"] = slot.source_trial_name
    row["source_conversation_fingerprint"] = (
        slot.source_conversation_fingerprint
    )
    row["refined_conversation_fingerprint"] = conversation_fingerprint(
        normalized_conversations
    )
    row["source_task_id"] = slot.source_task_id
    row["source_run_id"] = slot.source_run_id
    row["refinement_index"] = slot.refinement_index
    return row


def scan_accepted_shards(
    accepted_dir: Path, expected_schema: pa.Schema
) -> tuple[set[str], int, set[str]]:
    completed: set[str] = set()
    fingerprints: set[str] = set()
    row_count = 0
    if not accepted_dir.exists():
        return completed, row_count, fingerprints

    for shard in sorted(accepted_dir.glob("accepted-*.parquet")):
        actual = pq.read_schema(shard).remove_metadata()
        if not actual.equals(expected_schema, check_metadata=False):
            raise TypeError(f"accepted shard schema mismatch: {shard}")
        parquet_file = pq.ParquetFile(shard)
        for batch in parquet_file.iter_batches(
            batch_size=1_024,
            columns=[
                "run_id",
                "conversations",
                "refined_conversation_fingerprint",
            ],
        ):
            for row in batch.to_pylist():
                synthetic_id = row.get("run_id")
                if not isinstance(synthetic_id, str) or not synthetic_id:
                    raise ValueError(
                        f"accepted shard contains invalid run_id: {shard}"
                    )
                if synthetic_id in completed:
                    raise ValueError(
                        f"duplicate accepted synthetic ID: {synthetic_id}"
                    )
                stored_fingerprint = row.get("refined_conversation_fingerprint")
                actual_fingerprint = conversation_fingerprint(
                    row.get("conversations")
                )
                if stored_fingerprint != actual_fingerprint:
                    raise ValueError(
                        "accepted conversation fingerprint mismatch for "
                        f"{synthetic_id!r}"
                    )
                if actual_fingerprint in fingerprints:
                    raise ValueError(
                        "duplicate accepted conversation fingerprint: "
                        f"{actual_fingerprint}"
                    )
                completed.add(synthetic_id)
                fingerprints.add(actual_fingerprint)
            row_count += batch.num_rows
    return completed, row_count, fingerprints


def write_accepted_shard(
    accepted_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> Path:
    """Atomically add one immutable accepted-result shard."""
    if not rows:
        raise ValueError("cannot write an empty accepted shard")
    accepted_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=lambda row: str(row["run_id"]))
    ids = [str(row["run_id"]) for row in ordered_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("accepted shard contains duplicate synthetic IDs")
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:20]
    destination = accepted_dir / f"accepted-{digest}.parquet"

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    if destination.exists():
        existing = pq.read_table(destination, columns=["run_id"])
        if existing.column("run_id").to_pylist() != ids:
            raise ValueError(f"accepted shard name collision: {destination}")
        return destination

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=accepted_dir,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        pq.write_table(table, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        actual = pq.read_schema(temporary_path).remove_metadata()
        if not actual.equals(schema, check_metadata=False):
            raise TypeError("temporary accepted shard has an unexpected schema")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
