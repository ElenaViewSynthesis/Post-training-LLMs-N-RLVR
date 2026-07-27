"""Tests for the bounded-memory source-row refinement primitives."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from refinement_pipeline import (
    SourceIdentity,
    build_augmented_schema,
    build_refined_row,
    choose_secondary_source_ids,
    ensure_source_manifest,
    load_source_identities,
    load_source_manifest,
    normalize_conversations,
    pin_source_revision,
    refinement_slots_for_source,
    scan_accepted_shards,
    source_conversation_fingerprint,
    source_identity_digest,
    stable_synthetic_id,
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


def source_row(run_id: str, task: str = "task-a") -> dict:
    return {
        "conversations": [
            {"content": "Inspect the project", "role": "user"},
            {"content": "The project is ready", "role": "assistant"},
        ],
        "model": "source-model",
        "model_provider": "source-provider",
        "run_id": run_id,
        "task": task,
        "trial_name": run_id,
    }


def source_identity(
    record_id: str, run_id: str, task: str = "task-a", row_index: int = 0
) -> SourceIdentity:
    row = source_row(run_id, task)
    return SourceIdentity(
        source_record_id=record_id,
        source_file="fixture/train.parquet",
        source_row_index=row_index,
        source_trial_name=run_id,
        source_conversation_fingerprint=source_conversation_fingerprint(
            row["conversations"]
        ),
        source_run_id=run_id,
        source_task_id=task,
    )


class RefinementPipelineTests(unittest.TestCase):
    def test_assigns_exact_deterministic_secondary_slots(self) -> None:
        source_ids = [f"source-{index}" for index in range(10)]
        selected = choose_secondary_source_ids(source_ids, target_rows=15)
        reordered = choose_secondary_source_ids(
            list(reversed(source_ids)), target_rows=15
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, reordered)
        slot_count = sum(
            len(
                refinement_slots_for_source(
                    source_identity(source_id, source_id, row_index=index), selected
                )
            )
            for index, source_id in enumerate(source_ids)
        )
        self.assertEqual(slot_count, 15)

    def test_rejects_invalid_target_or_duplicate_source_ids(self) -> None:
        for target in (1, 5):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "target rows"):
                    choose_secondary_source_ids(["a", "b"], target)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            choose_secondary_source_ids(["a", "a"], 2)

    def test_synthetic_ids_are_stable_and_slot_specific(self) -> None:
        first = stable_synthetic_id("source-1", 0)
        self.assertEqual(first, stable_synthetic_id("source-1", 0))
        self.assertNotEqual(first, stable_synthetic_id("source-1", 1))
        self.assertNotEqual(first, stable_synthetic_id("source-2", 0))

    def test_conversation_validation_is_strict(self) -> None:
        valid = [
            {"role": "user", "content": "Inspect"},
            {"role": "assistant", "content": "Done"},
        ]
        self.assertEqual(
            normalize_conversations(valid),
            [
                {"content": "Inspect", "role": "user"},
                {"content": "Done", "role": "assistant"},
            ],
        )
        with self.assertRaisesRegex(TypeError, "list-like"):
            normalize_conversations(str(valid))
        with self.assertRaisesRegex(ValueError, "empty content"):
            normalize_conversations(
                [
                    {"role": "user", "content": "Inspect"},
                    {"role": "assistant", "content": ""},
                ]
            )
        with self.assertRaisesRegex(ValueError, "end with an assistant"):
            normalize_conversations(
                [
                    {"role": "assistant", "content": "Inspect"},
                    {"role": "tool", "content": "Done"},
                ]
            )

    def test_loads_unique_source_identities_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory) / "source"
            source_dir.mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [source_row("source-1"), source_row("source-2", "task-b")],
                    schema=SOURCE_SCHEMA,
                ),
                source_dir / "train-00000.parquet",
            )

            schema, identities = load_source_identities(str(source_dir))

        self.assertTrue(schema.equals(SOURCE_SCHEMA, check_metadata=False))
        by_trial = {
            identity.source_trial_name: identity for identity in identities.values()
        }
        self.assertEqual(set(by_trial), {"source-1", "source-2"})
        self.assertEqual(by_trial["source-1"].source_run_id, "source-1")
        self.assertEqual(by_trial["source-2"].source_task_id, "task-b")

    def test_source_identity_changes_when_conversation_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory) / "source"
            source_dir.mkdir()
            source_path = source_dir / "train-00000.parquet"
            original = source_row("source-1")
            pq.write_table(
                pa.Table.from_pylist([original], schema=SOURCE_SCHEMA),
                source_path,
            )
            schema, before = load_source_identities(str(source_dir))
            manifest_path = Path(temporary_directory) / "refined/source_manifest.json"
            ensure_source_manifest(
                manifest_path,
                requested_source=str(source_dir),
                resolved_source=str(source_dir),
                schema=schema,
                identities=before,
            )

            changed = source_row("source-1")
            changed["conversations"][1]["content"] = "Changed upstream content"
            pq.write_table(
                pa.Table.from_pylist([changed], schema=SOURCE_SCHEMA),
                source_path,
            )
            changed_schema, after = load_source_identities(str(source_dir))

            self.assertNotEqual(set(before), set(after))
            self.assertNotEqual(
                source_identity_digest(before.values()),
                source_identity_digest(after.values()),
            )
            with self.assertRaisesRegex(ValueError, "immutable run manifest"):
                ensure_source_manifest(
                    manifest_path,
                    requested_source=str(source_dir),
                    resolved_source=str(source_dir),
                    schema=changed_schema,
                    identities=after,
                )

    def test_source_manifest_round_trips_and_hf_source_is_pinned(self) -> None:
        source = "hf://datasets/example/data/data/train-*.parquet"
        revision = "a" * 40
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.dataset_info.return_value.sha = revision
            resolved = pin_source_revision(source)

        self.assertEqual(
            resolved,
            f"hf://datasets/example/data@{revision}/data/train-*.parquet",
        )
        api.return_value.dataset_info.assert_called_once_with(
            "example/data", revision=None
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            source_dir.mkdir()
            pq.write_table(
                pa.Table.from_pylist([source_row("source-1")], schema=SOURCE_SCHEMA),
                source_dir / "train-00000.parquet",
            )
            schema, identities = load_source_identities(str(source_dir))
            path = root / "refined/source_manifest.json"
            written = ensure_source_manifest(
                path,
                requested_source=str(source_dir),
                resolved_source=str(source_dir),
                schema=schema,
                identities=identities,
            )

            self.assertEqual(load_source_manifest(path), written)

    def test_physical_identity_distinguishes_repeated_logical_ids(self) -> None:
        first = source_row("shared-run", "task-a")
        first["trial_name"] = "shared-trial"
        second = source_row("shared-run", "task-b")
        second["trial_name"] = "shared-trial"
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory) / "source"
            source_dir.mkdir()
            pq.write_table(
                pa.Table.from_pylist([first, second], schema=SOURCE_SCHEMA),
                source_dir / "train-00000.parquet",
            )

            _, identities = load_source_identities(str(source_dir))
            secondary = choose_secondary_source_ids(list(identities), target_rows=2)
            slots = [
                refinement_slots_for_source(identity, secondary)[0]
                for identity in identities.values()
            ]

        self.assertEqual(len(identities), 2)
        self.assertEqual(
            {identity.source_trial_name for identity in identities.values()},
            {"shared-trial"},
        )
        self.assertEqual({slot.source_run_id for slot in slots}, {"shared-run"})
        self.assertEqual(len({slot.synthetic_id for slot in slots}), 2)

    def test_builds_and_resumes_explicit_schema_shards(self) -> None:
        output_schema = build_augmented_schema(SOURCE_SCHEMA)
        identity = source_identity("record-1", "source-1")
        secondary = frozenset({identity.source_record_id})
        slot = refinement_slots_for_source(identity, secondary)[0]
        row = build_refined_row(
            source_row("source-1"),
            slot,
            [
                {"role": "user", "content": "Inspect carefully"},
                {"role": "assistant", "content": "Completed carefully"},
            ],
            SOURCE_SCHEMA,
            model="refiner-model",
            provider="gemini",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            accepted_dir = Path(temporary_directory) / "accepted"
            shard = write_accepted_shard(accepted_dir, [row], output_schema)
            same_shard = write_accepted_shard(accepted_dir, [row], output_schema)
            completed, row_count = scan_accepted_shards(accepted_dir, output_schema)

            self.assertEqual(shard, same_shard)
            self.assertEqual(completed, {slot.synthetic_id})
            self.assertEqual(row_count, 1)
            actual_schema = pq.read_schema(shard).remove_metadata()

        self.assertTrue(actual_schema.equals(output_schema, check_metadata=False))
        self.assertEqual(row["run_id"], slot.synthetic_id)
        self.assertEqual(row["source_record_id"], "record-1")
        self.assertEqual(row["source_run_id"], "source-1")
        self.assertEqual(row["refinement_index"], 0)
        self.assertEqual(row["model"], "refiner-model")

    def test_resume_scan_rejects_duplicate_ids_across_shards(self) -> None:
        output_schema = build_augmented_schema(SOURCE_SCHEMA)
        slot = refinement_slots_for_source(
            source_identity("record-1", "source-1"), frozenset()
        )[0]
        row = build_refined_row(
            source_row("source-1"),
            slot,
            [
                {"role": "user", "content": "Inspect carefully"},
                {"role": "assistant", "content": "Completed carefully"},
            ],
            SOURCE_SCHEMA,
            model="refiner-model",
            provider="gemini",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            accepted_dir = Path(temporary_directory) / "accepted"
            shard = write_accepted_shard(accepted_dir, [row], output_schema)
            shutil.copyfile(shard, accepted_dir / "accepted-duplicate.parquet")
            with self.assertRaisesRegex(ValueError, "duplicate accepted"):
                scan_accepted_shards(accepted_dir, output_schema)


if __name__ == "__main__":
    unittest.main()
