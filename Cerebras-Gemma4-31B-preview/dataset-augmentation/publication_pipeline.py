"""Streaming validation and atomic local publication for Stage 6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from refinement_pipeline import (
    REFINEMENT_VALIDATION_POLICY_VERSION,
    RunManifest,
    SourceFileManifest,
    SourceIdentity,
    build_augmented_schema,
    choose_secondary_source_ids,
    conversation_fingerprint,
    iter_source_batches_with_coordinates,
    load_run_manifest,
    load_source_identities,
    load_source_manifest,
    normalize_conversations,
    refinement_slots_for_source,
    scan_accepted_shards,
    snapshot_source_files,
    source_content_digest,
    source_identity_digest,
    source_identity_for_row,
    source_schema_digest,
    stable_synthetic_id,
    verify_refinement_state_inventory,
    verify_source_manifest_content,
)


@dataclass(frozen=True)
class PublicationPreflight:
    original_source: str
    accepted_dir: Path
    original_schema: pa.Schema
    output_schema: pa.Schema
    original_rows: int
    synthetic_rows: int
    total_rows: int
    accepted_shards: tuple[Path, ...]
    accepted_ids: frozenset[str]
    source_identity_sha256: str
    source_schema_sha256: str
    source_content_sha256: str
    source_files: tuple[SourceFileManifest, ...]
    run_instance_id: str
    validation_policy_version: str
    publication_id: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_id(
    accepted_ids: set[str],
    accepted_shards: tuple[Path, ...],
    source_identities: dict[str, SourceIdentity],
    original_source: str,
    total_rows: int,
    source_identity_sha256: str,
    source_schema_sha256: str,
    source_content_sha256: str,
    run_manifest: RunManifest,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"source={original_source}\n".encode())
    digest.update(f"total={total_rows}\n".encode())
    digest.update(f"source_identity_sha256={source_identity_sha256}\n".encode())
    digest.update(f"source_schema_sha256={source_schema_sha256}\n".encode())
    digest.update(f"source_content_sha256={source_content_sha256}\n".encode())
    digest.update(f"run_instance_id={run_manifest.run_instance_id}\n".encode())
    digest.update(
        f"validation_policy={run_manifest.validation_policy_version}\n".encode()
    )
    for source_record_id, identity in sorted(source_identities.items()):
        digest.update(
            (
                f"{source_record_id}\0{identity.source_run_id}\0"
                f"{identity.source_task_id}\n"
            ).encode()
        )
    for synthetic_id in sorted(accepted_ids):
        digest.update(synthetic_id.encode())
        digest.update(b"\n")
    for shard in accepted_shards:
        digest.update(shard.name.encode())
        digest.update(file_sha256(shard).encode())
    return f"publication-{digest.hexdigest()[:20]}"


def validate_accepted_rows(
    accepted_dir: Path,
    output_schema: pa.Schema,
    source_identities: dict[str, SourceIdentity],
    *,
    expected_validation_policy: str,
) -> tuple[set[str], tuple[Path, ...]]:
    accepted_ids, row_count, _accepted_fingerprints = scan_accepted_shards(
        accepted_dir,
        output_schema,
        expected_validation_policy=expected_validation_policy,
    )
    shards = tuple(sorted(accepted_dir.glob("accepted-*.parquet")))
    checked_rows = 0
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        for batch in parquet_file.iter_batches(batch_size=1_024):
            for row in batch.to_pylist():
                checked_rows += 1
                synthetic_id = row.get("run_id")
                source_record_id = row.get("source_record_id")
                source_file = row.get("source_file")
                source_row_index = row.get("source_row_index")
                source_trial_name = row.get("source_trial_name")
                source_conversation_sha256 = row.get(
                    "source_conversation_fingerprint"
                )
                refined_conversation_sha256 = row.get(
                    "refined_conversation_fingerprint"
                )
                source_run_id = row.get("source_run_id")
                source_task_id = row.get("source_task_id")
                refinement_index = row.get("refinement_index")
                if (
                    row.get("refinement_validation_policy")
                    != expected_validation_policy
                ):
                    raise ValueError(
                        "accepted validation policy does not match the run manifest "
                        f"for {synthetic_id!r}"
                    )
                identity = source_identities.get(source_record_id)
                if (
                    identity is None
                    or identity.source_file != source_file
                    or identity.source_row_index != source_row_index
                    or identity.source_trial_name != source_trial_name
                    or (
                        identity.source_conversation_fingerprint
                        != source_conversation_sha256
                    )
                    or identity.source_run_id != source_run_id
                    or identity.source_task_id != source_task_id
                    or row.get("task") != source_task_id
                ):
                    raise ValueError(
                        f"missing or mismatched source lookup for {synthetic_id!r}"
                    )
                if refinement_index not in (0, 1):
                    raise ValueError(
                        f"invalid refinement_index for {synthetic_id!r}"
                    )
                expected_id = stable_synthetic_id(
                    source_record_id, refinement_index
                )
                if synthetic_id != expected_id:
                    raise ValueError(
                        f"synthetic ID does not match its assigned slot: {synthetic_id!r}"
                    )
                if row.get("trial_name") != synthetic_id:
                    raise ValueError(
                        f"trial_name does not match synthetic ID: {synthetic_id!r}"
                    )
                if row.get("is_synthetic_augmentation") is not True:
                    raise ValueError(
                        f"synthetic augmentation flag is not true: {synthetic_id!r}"
                    )
                conversations = normalize_conversations(row.get("conversations"))
                if (
                    conversation_fingerprint(conversations)
                    != refined_conversation_sha256
                ):
                    raise ValueError(
                        "refined conversation fingerprint mismatch for "
                        f"{synthetic_id!r}"
                    )
    if checked_rows != row_count:
        raise ValueError(
            f"accepted metadata count changed during validation: {checked_rows} != {row_count}"
        )
    return accepted_ids, shards


def preflight_publication(
    original_source: str,
    accepted_dir: Path,
    *,
    expected_new_rows: int,
    expected_total_rows: int | None = None,
    expected_source_identity_sha256: str | None = None,
    expected_source_schema_sha256: str | None = None,
    expected_source_content_sha256: str | None = None,
) -> PublicationPreflight:
    verify_refinement_state_inventory(accepted_dir.parent)
    source_manifest = load_source_manifest(accepted_dir.parent / "source_manifest.json")
    run_manifest = load_run_manifest(accepted_dir.parent / "run_manifest.json")
    if original_source not in {
        source_manifest.requested_source,
        source_manifest.resolved_source,
    }:
        raise ValueError("publication source conflicts with the refinement manifest")
    if (
        run_manifest.source_content_sha256
        != source_manifest.source_content_sha256
        or run_manifest.source_schema_sha256 != source_manifest.source_schema_sha256
    ):
        raise ValueError("run manifest does not match the immutable source manifest")
    if run_manifest.target_rows != expected_new_rows:
        raise ValueError(
            "publication synthetic target does not match the run manifest: "
            f"{expected_new_rows:,} != {run_manifest.target_rows:,}"
        )
    if (
        run_manifest.validation_policy_version
        != REFINEMENT_VALIDATION_POLICY_VERSION
    ):
        raise ValueError("run manifest uses an unsupported validation policy")
    source_files = verify_source_manifest_content(source_manifest, original_source)
    original_schema, source_identities = load_source_identities(original_source)
    output_schema = build_augmented_schema(original_schema)
    accepted_ids, accepted_shards = validate_accepted_rows(
        accepted_dir,
        output_schema,
        source_identities,
        expected_validation_policy=run_manifest.validation_policy_version,
    )
    original_rows = len(source_identities)
    source_identity_sha256 = source_identity_digest(source_identities.values())
    source_schema_sha256 = source_schema_digest(original_schema)
    source_content_sha256 = source_manifest.source_content_sha256
    if (
        expected_source_identity_sha256 is not None
        and source_identity_sha256 != expected_source_identity_sha256
    ):
        raise ValueError(
            "source identities do not match the refinement source manifest"
        )
    if (
        expected_source_schema_sha256 is not None
        and source_schema_sha256 != expected_source_schema_sha256
    ):
        raise ValueError("source schema does not match the refinement source manifest")
    if source_schema_sha256 != source_manifest.source_schema_sha256:
        raise ValueError("source schema does not match the refinement source manifest")
    if source_identity_sha256 != source_manifest.source_identity_sha256:
        raise ValueError("source identities do not match the refinement source manifest")
    if (
        expected_source_content_sha256 is not None
        and source_content_sha256 != expected_source_content_sha256
    ):
        raise ValueError("source content does not match the refinement source manifest")
    synthetic_rows = len(accepted_ids)
    total_rows = original_rows + synthetic_rows
    required_total_rows = (
        expected_total_rows
        if expected_total_rows is not None
        else original_rows + expected_new_rows
    )
    if synthetic_rows != expected_new_rows:
        raise ValueError(
            f"expected exactly {expected_new_rows:,} synthetic rows; "
            f"found {synthetic_rows:,}"
        )
    if total_rows != required_total_rows:
        raise ValueError(
            f"expected exactly {required_total_rows:,} total rows; "
            f"found {original_rows:,} original + {synthetic_rows:,} synthetic"
        )

    secondary_ids = choose_secondary_source_ids(
        list(source_identities), expected_new_rows
    )
    expected_ids = {
        slot.synthetic_id
        for identity in source_identities.values()
        for slot in refinement_slots_for_source(identity, secondary_ids)
    }
    missing = expected_ids.difference(accepted_ids)
    unexpected = accepted_ids.difference(expected_ids)
    if missing or unexpected:
        raise ValueError(
            "accepted refinement slots do not match the deterministic assignment: "
            f"missing={len(missing):,}, unexpected={len(unexpected):,}"
        )

    return PublicationPreflight(
        original_source=original_source,
        accepted_dir=accepted_dir,
        original_schema=original_schema,
        output_schema=output_schema,
        original_rows=original_rows,
        synthetic_rows=synthetic_rows,
        total_rows=total_rows,
        accepted_shards=accepted_shards,
        accepted_ids=frozenset(accepted_ids),
        source_identity_sha256=source_identity_sha256,
        source_schema_sha256=source_schema_sha256,
        source_content_sha256=source_content_sha256,
        source_files=source_files,
        run_instance_id=run_manifest.run_instance_id,
        validation_policy_version=run_manifest.validation_policy_version,
        publication_id=_publication_id(
            accepted_ids,
            accepted_shards,
            source_identities,
            original_source,
            total_rows,
            source_identity_sha256,
            source_schema_sha256,
            source_content_sha256,
            run_manifest,
        ),
    )


def augment_original_batch(
    batch: pa.RecordBatch,
    output_schema: pa.Schema,
) -> pa.Table:
    original_table = pa.Table.from_batches([batch])
    row_count = original_table.num_rows
    arrays = [original_table.column(name) for name in batch.schema.names]
    arrays.extend(
        [
            pa.array([False] * row_count, type=pa.bool_()),
            pa.nulls(row_count, type=output_schema.field("source_record_id").type),
            pa.nulls(row_count, type=output_schema.field("source_file").type),
            pa.nulls(row_count, type=output_schema.field("source_row_index").type),
            pa.nulls(row_count, type=output_schema.field("source_trial_name").type),
            pa.nulls(
                row_count,
                type=output_schema.field("source_conversation_fingerprint").type,
            ),
            pa.nulls(
                row_count,
                type=output_schema.field("refined_conversation_fingerprint").type,
            ),
            pa.nulls(row_count, type=output_schema.field("source_task_id").type),
            pa.nulls(row_count, type=output_schema.field("source_run_id").type),
            pa.nulls(row_count, type=pa.int8()),
            pa.nulls(
                row_count,
                type=output_schema.field("refinement_validation_policy").type,
            ),
        ]
    )
    return pa.Table.from_arrays(arrays, schema=output_schema)


class ShardedParquetWriter:
    """Write exact-size deterministic shards without retaining prior rows."""

    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema,
        *,
        total_rows: int,
        rows_per_shard: int,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.output_dir = output_dir
        self.schema = schema
        self.total_rows = total_rows
        self.rows_per_shard = rows_per_shard
        self.expected_shards = math.ceil(total_rows / rows_per_shard)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._writer: pq.ParquetWriter | None = None
        self._rows_in_shard = 0
        self._rows_written = 0
        self._shard_index = 0
        self._paths: list[Path] = []

    def _open_shard(self) -> None:
        path = self.output_dir / (
            f"train-{self._shard_index:05d}-of-{self.expected_shards:05d}.parquet"
        )
        self._writer = pq.ParquetWriter(path, self.schema)
        self._paths.append(path)
        self._rows_in_shard = 0

    def _close_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._shard_index += 1
            self._rows_in_shard = 0

    def write_table(self, table: pa.Table) -> None:
        if not table.schema.equals(self.schema, check_metadata=False):
            raise TypeError("publication table does not match the output schema")
        offset = 0
        while offset < table.num_rows:
            if self._writer is None:
                self._open_shard()
            available = self.rows_per_shard - self._rows_in_shard
            length = min(available, table.num_rows - offset)
            self._writer.write_table(table.slice(offset, length))
            self._rows_in_shard += length
            self._rows_written += length
            offset += length
            if self._rows_in_shard == self.rows_per_shard:
                self._close_shard()

    def finish(self) -> tuple[Path, ...]:
        self._close_shard()
        if self._rows_written != self.total_rows:
            raise ValueError(
                f"publication writer expected {self.total_rows:,} rows but wrote "
                f"{self._rows_written:,}"
            )
        if len(self._paths) != self.expected_shards:
            raise ValueError(
                f"expected {self.expected_shards} shards but wrote {len(self._paths)}"
            )
        return tuple(self._paths)

    def abort(self) -> None:
        """Close the active shard after a failed, unpromoted write."""
        self._close_shard()


def validate_publication_shards(
    shards: tuple[Path, ...], expected_schema: pa.Schema, expected_rows: int
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    row_count = 0
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        actual_schema = parquet_file.schema_arrow.remove_metadata()
        if not actual_schema.equals(expected_schema, check_metadata=False):
            raise TypeError(f"publication schema mismatch: {shard}")
        shard_rows = parquet_file.metadata.num_rows
        row_count += shard_rows
        files.append(
            {
                "name": shard.name,
                "rows": shard_rows,
                "bytes": shard.stat().st_size,
                "sha256": file_sha256(shard),
            }
        )
    if row_count != expected_rows:
        raise ValueError(
            f"physical publication row count is {row_count:,}, expected {expected_rows:,}"
        )
    return files


def _write_manifest(
    run_dir: Path,
    preflight: PublicationPreflight,
    files: list[dict[str, Any]],
) -> Path:
    manifest = {
        "publication_id": preflight.publication_id,
        "original_source": preflight.original_source,
        "accepted_directory": str(preflight.accepted_dir),
        "original_rows": preflight.original_rows,
        "synthetic_rows": preflight.synthetic_rows,
        "total_rows": preflight.total_rows,
        "source_identity_sha256": preflight.source_identity_sha256,
        "source_schema_sha256": preflight.source_schema_sha256,
        "source_content_sha256": preflight.source_content_sha256,
        "source_files": [
            {
                "path": source_file.path,
                "size": source_file.size,
                "sha256": source_file.sha256,
            }
            for source_file in preflight.source_files
        ],
        "run_instance_id": preflight.run_instance_id,
        "validation_policy_version": preflight.validation_policy_version,
        "schema": preflight.output_schema.to_string(),
        "files": files,
    }
    path = run_dir / "publication_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_current_pointer(output_root: Path, preflight: PublicationPreflight) -> None:
    current = {
        "publication_id": preflight.publication_id,
        "path": f"publications/{preflight.publication_id}",
        "total_rows": preflight.total_rows,
        "source_identity_sha256": preflight.source_identity_sha256,
        "source_schema_sha256": preflight.source_schema_sha256,
        "source_content_sha256": preflight.source_content_sha256,
        "source_files": [
            {
                "path": source_file.path,
                "size": source_file.size,
                "sha256": source_file.sha256,
            }
            for source_file in preflight.source_files
        ],
        "run_instance_id": preflight.run_instance_id,
        "validation_policy_version": preflight.validation_policy_version,
    }
    temporary = output_root / ".current.json.tmp"
    temporary.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root / "current.json")


def validate_existing_publication(
    final_dir: Path, preflight: PublicationPreflight
) -> None:
    manifest_path = final_dir / "publication_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid existing publication manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"existing publication manifest is not an object: {manifest_path}")
    expected_manifest_values = {
        "publication_id": preflight.publication_id,
        "original_rows": preflight.original_rows,
        "synthetic_rows": preflight.synthetic_rows,
        "total_rows": preflight.total_rows,
        "source_identity_sha256": preflight.source_identity_sha256,
        "source_schema_sha256": preflight.source_schema_sha256,
        "source_content_sha256": preflight.source_content_sha256,
        "source_files": [
            {
                "path": source_file.path,
                "size": source_file.size,
                "sha256": source_file.sha256,
            }
            for source_file in preflight.source_files
        ],
        "run_instance_id": preflight.run_instance_id,
        "validation_policy_version": preflight.validation_policy_version,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"existing publication manifest has unexpected {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or not entries
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError("existing publication manifest contains no files")
    shards = tuple(final_dir / "data" / str(entry.get("name")) for entry in entries)
    if any(not shard.is_file() for shard in shards):
        raise FileNotFoundError("existing publication is missing a declared shard")
    actual_entries = validate_publication_shards(
        shards,
        preflight.output_schema,
        preflight.total_rows,
    )
    if actual_entries != entries:
        raise ValueError("existing publication shard checksums do not match its manifest")


def write_local_publication(
    preflight: PublicationPreflight,
    output_root: Path,
    *,
    rows_per_shard: int,
) -> tuple[Path, bool]:
    """Write and atomically promote a fresh immutable local publication."""
    publications_root = output_root / "publications"
    final_dir = publications_root / preflight.publication_id
    manifest = final_dir / "publication_manifest.json"
    if final_dir.exists():
        if not manifest.is_file():
            raise FileExistsError(
                f"existing publication is incomplete: {final_dir}"
            )
        validate_existing_publication(final_dir, preflight)
        _write_current_pointer(output_root, preflight)
        return final_dir, False

    work_root = output_root / ".work"
    publications_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f"{preflight.publication_id}-", dir=work_root)
    )
    data_dir = work_dir / "data"
    writer = ShardedParquetWriter(
        data_dir,
        preflight.output_schema,
        total_rows=preflight.total_rows,
        rows_per_shard=rows_per_shard,
    )
    try:
        observed_source_identities: list[SourceIdentity] = []
        for source_file, row_offset, batch in iter_source_batches_with_coordinates(
            preflight.original_source
        ):
            observed_source_identities.extend(
                source_identity_for_row(source_file, index, row)
                for index, row in enumerate(
                    batch.to_pylist(), start=row_offset
                )
            )
            writer.write_table(
                augment_original_batch(batch, preflight.output_schema)
            )
        observed_source_digest = source_identity_digest(observed_source_identities)
        if (
            len(observed_source_identities) != preflight.original_rows
            or observed_source_digest != preflight.source_identity_sha256
        ):
            raise ValueError("source changed between publication preflight and write")
        observed_source_files = snapshot_source_files(preflight.original_source)
        if (
            observed_source_files != preflight.source_files
            or source_content_digest(observed_source_files)
            != preflight.source_content_sha256
        ):
            raise ValueError(
                "source content changed between publication preflight and write"
            )
        for shard in preflight.accepted_shards:
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=1_024):
                writer.write_table(pa.Table.from_batches([batch]))
        shards = writer.finish()
        files = validate_publication_shards(
            shards,
            preflight.output_schema,
            preflight.total_rows,
        )
        _write_manifest(work_dir, preflight, files)
        os.replace(work_dir, final_dir)
        _write_current_pointer(output_root, preflight)
    except Exception:
        # Keep the unpromoted work directory for diagnosis. It is never synced.
        writer.abort()
        raise
    return final_dir, True
