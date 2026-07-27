"""Local Stage 5-to-Stage 6 publication safety tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

import augment_150k_rows
from publication_pipeline import preflight_publication, write_local_publication
from refinement_pipeline import (
    build_augmented_schema,
    build_refined_row,
    choose_secondary_source_ids,
    ensure_source_manifest,
    load_source_identities,
    load_source_manifest,
    refinement_slots_for_source,
    write_accepted_shard,
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


def source_row(index: int) -> dict:
    source_id = f"source-{index}"
    return {
        "conversations": [
            {"content": f"Inspect project {index}", "role": "user"},
            {"content": f"Project {index} is ready", "role": "assistant"},
        ],
        "model": "source-model",
        "model_provider": "source-provider",
        "run_id": source_id,
        "task": f"task-{index}",
        "trial_name": source_id,
    }


def create_fixture(root: Path) -> tuple[Path, Path]:
    source_dir = root / "source"
    accepted_dir = root / "refined" / "accepted"
    source_dir.mkdir(parents=True)
    rows = [source_row(index) for index in range(4)]
    pq.write_table(
        pa.Table.from_pylist(rows[:2], schema=SOURCE_SCHEMA),
        source_dir / "train-00000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(rows[2:], schema=SOURCE_SCHEMA),
        source_dir / "train-00001.parquet",
    )

    source_schema, identities = load_source_identities(str(source_dir))
    ensure_source_manifest(
        accepted_dir.parent / "source_manifest.json",
        requested_source=str(source_dir),
        resolved_source=str(source_dir),
        schema=source_schema,
        identities=identities,
    )
    by_trial_name = {
        identity.source_trial_name: identity for identity in identities.values()
    }
    secondary = choose_secondary_source_ids(list(identities), target_rows=6)
    accepted_rows = []
    for row in rows:
        for slot in refinement_slots_for_source(
            by_trial_name[row["trial_name"]], secondary
        ):
            accepted_rows.append(
                build_refined_row(
                    row,
                    slot,
                    [
                        {
                            "role": "user",
                            "content": f"Refine {row['task']} carefully",
                        },
                        {
                            "role": "assistant",
                            "content": f"Refined {row['task']} successfully",
                        },
                    ],
                    SOURCE_SCHEMA,
                    model="refiner-model",
                    provider="gemini",
                )
            )
    output_schema = build_augmented_schema(SOURCE_SCHEMA)
    write_accepted_shard(accepted_dir, accepted_rows[:3], output_schema)
    write_accepted_shard(accepted_dir, accepted_rows[3:], output_schema)
    return source_dir, accepted_dir


class PublicationPipelineTests(unittest.TestCase):
    def test_data_directory_controls_stage_six_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with mock.patch.dict(
                os.environ, {"PIPELINE_DATA_DIR": str(data_dir)}, clear=False
            ):
                args = augment_150k_rows.parse_args(["--dry-run"])
        self.assertEqual(args.data_dir, data_dir)
        self.assertEqual(args.refined_dir, data_dir / "refined" / "accepted")
        self.assertEqual(args.upload_dir, data_dir / "upload")
        self.assertIsNone(args.expected_total_rows)

    def test_streams_exact_rows_into_fresh_versioned_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            upload_dir = root / "upload"
            upload_dir.mkdir()
            stale = upload_dir / "train-stale.parquet"
            stale.write_bytes(b"stale")
            preflight = preflight_publication(
                str(source_dir),
                accepted_dir,
                expected_new_rows=6,
                expected_total_rows=10,
            )

            publication_dir, created = write_local_publication(
                preflight,
                upload_dir,
                rows_per_shard=2,
            )
            data_shards = sorted((publication_dir / "data").glob("*.parquet"))
            physical_rows = sum(
                pq.ParquetFile(path).metadata.num_rows for path in data_shards
            )
            manifest = json.loads(
                (publication_dir / "publication_manifest.json").read_text()
            )
            current = json.loads((upload_dir / "current.json").read_text())

            self.assertTrue(created)
            self.assertEqual(len(data_shards), 5)
            self.assertEqual(physical_rows, 10)
            self.assertEqual(manifest["original_rows"], 4)
            self.assertEqual(manifest["synthetic_rows"], 6)
            self.assertEqual(
                manifest["source_identity_sha256"],
                preflight.source_identity_sha256,
            )
            self.assertEqual(current["publication_id"], preflight.publication_id)
            self.assertEqual(stale.read_bytes(), b"stale")
            self.assertNotIn(stale, data_shards)

    def test_preflight_derives_total_from_physical_source_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)

            preflight = preflight_publication(
                str(source_dir),
                accepted_dir,
                expected_new_rows=6,
            )

            self.assertEqual(preflight.original_rows, 4)
            self.assertEqual(preflight.synthetic_rows, 6)
            self.assertEqual(preflight.total_rows, 10)

    def test_preflight_rejects_a_source_manifest_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            manifest = load_source_manifest(
                accepted_dir.parent / "source_manifest.json"
            )

            with self.assertRaisesRegex(ValueError, "source identities"):
                preflight_publication(
                    str(source_dir),
                    accepted_dir,
                    expected_new_rows=6,
                    expected_source_identity_sha256="0" * 64,
                    expected_source_schema_sha256=manifest.source_schema_sha256,
                )

    def test_source_change_after_preflight_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            upload_dir = root / "upload"
            preflight = preflight_publication(
                str(source_dir),
                accepted_dir,
                expected_new_rows=6,
                expected_total_rows=10,
            )
            source_path = source_dir / "train-00000.parquet"
            rows = pq.read_table(source_path).to_pylist()
            rows[0]["conversations"][0]["content"] = "Changed after preflight"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA), source_path
            )

            with self.assertRaisesRegex(ValueError, "source changed"):
                write_local_publication(
                    preflight,
                    upload_dir,
                    rows_per_shard=2,
                )

            self.assertFalse((upload_dir / "current.json").exists())
            self.assertFalse(
                (upload_dir / "publications" / preflight.publication_id).exists()
            )

    def test_preflight_rejects_missing_source_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            shard = sorted(accepted_dir.glob("*.parquet"))[0]
            table = pq.read_table(shard)
            rows = table.to_pylist()
            rows[0]["source_run_id"] = "unknown-source"
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)

            with self.assertRaisesRegex(ValueError, "source lookup"):
                preflight_publication(
                    str(source_dir),
                    accepted_dir,
                    expected_new_rows=6,
                    expected_total_rows=10,
                )

    def test_preflight_rejects_changed_task_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            shard = sorted(accepted_dir.glob("*.parquet"))[0]
            table = pq.read_table(shard)
            rows = table.to_pylist()
            rows[0]["task"] = "different-task"
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)

            with self.assertRaisesRegex(ValueError, "source lookup"):
                preflight_publication(
                    str(source_dir),
                    accepted_dir,
                    expected_new_rows=6,
                )

    def test_dry_run_writes_nothing_and_no_sync_skips_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            upload_dir = root / "upload"
            dry_args = augment_150k_rows.parse_args(
                [
                    "--original-source",
                    str(source_dir),
                    "--refined-dir",
                    str(accepted_dir),
                    "--upload-dir",
                    str(upload_dir),
                    "--expected-new-rows",
                    "6",
                    "--expected-total-rows",
                    "10",
                    "--dry-run",
                ]
            )
            with mock.patch.object(augment_150k_rows, "sync_bucket") as sync:
                self.assertEqual(augment_150k_rows.run(dry_args), 0)
                sync.assert_not_called()
            self.assertFalse(upload_dir.exists())

            local_args = augment_150k_rows.parse_args(
                [
                    "--original-source",
                    str(source_dir),
                    "--refined-dir",
                    str(accepted_dir),
                    "--upload-dir",
                    str(upload_dir),
                    "--expected-new-rows",
                    "6",
                    "--expected-total-rows",
                    "10",
                    "--rows-per-shard",
                    "2",
                    "--no-sync",
                ]
            )
            with mock.patch.object(augment_150k_rows, "sync_bucket") as sync:
                self.assertEqual(augment_150k_rows.run(local_args), 0)
                sync.assert_not_called()
            self.assertTrue((upload_dir / "current.json").is_file())

    def test_final_sync_failure_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir, accepted_dir = create_fixture(root)
            args = [
                "--original-source",
                str(source_dir),
                "--refined-dir",
                str(accepted_dir),
                "--upload-dir",
                str(root / "upload"),
                "--expected-new-rows",
                "6",
                "--expected-total-rows",
                "10",
                "--rows-per-shard",
                "2",
            ]
            with mock.patch.object(
                augment_150k_rows,
                "sync_bucket",
                side_effect=RuntimeError("network unavailable"),
            ):
                self.assertEqual(augment_150k_rows.main(args), 1)


if __name__ == "__main__":
    unittest.main()
