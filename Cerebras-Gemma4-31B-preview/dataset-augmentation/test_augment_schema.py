"""Tests that Stage 6 preserves the source dataset's nested Arrow schema."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from augment_150k_rows import (
    assert_output_arrow_schema,
    build_output_arrow_schema,
    dataframe_from_synthetic_rows,
    normalize_conversations,
    read_original_arrow_schema,
    require_exact_row_counts,
)


CONVERSATIONS_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("content", pa.string()),
            pa.field("role", pa.string()),
        ]
    )
)
ORIGINAL_SCHEMA = pa.schema(
    [
        pa.field("task", pa.string()),
        pa.field("conversations", CONVERSATIONS_TYPE),
    ]
)


class AugmentSchemaTests(unittest.TestCase):
    def test_rejects_stringified_conversations(self) -> None:
        with self.assertRaisesRegex(TypeError, "list-like sequence"):
            normalize_conversations('[{"role":"user","content":"Inspect"}]')

    def test_normalizes_parquet_list_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source.parquet"
            turns = [
                {"content": "Inspect the files", "role": "user"},
                {"content": "Done", "role": "assistant"},
            ]
            table = pa.Table.from_pylist(
                [{"task": "task-1", "conversations": turns}],
                schema=ORIGINAL_SCHEMA,
            )
            pq.write_table(table, path)
            parquet_value = pd.read_parquet(path).iloc[0]["conversations"]

        self.assertEqual(normalize_conversations(parquet_value), turns)

    def test_dask_handoff_keeps_conversations_as_nested_objects(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "task": "task-1",
                    "conversations": [
                        {"content": "Inspect", "role": "user"},
                        {"content": "Done", "role": "assistant"},
                    ],
                    "is_synthetic_augmentation": True,
                    "source_task_id": "task-1",
                }
            ]
        )

        frame = dataframe_from_synthetic_rows(rows)
        value = frame.head(1).iloc[0]["conversations"]

        self.assertEqual(frame.dtypes["conversations"], object)
        self.assertNotIsInstance(value, str)

    def test_combined_parquet_round_trip_matches_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()

            turns = [
                {"content": "Inspect the files", "role": "user"},
                {"content": "Done", "role": "assistant"},
            ]
            original_table = pa.Table.from_pylist(
                [{"task": "task-1", "conversations": turns}],
                schema=ORIGINAL_SCHEMA,
            )
            pq.write_table(original_table, source_dir / "train-00000.parquet")

            original_schema = read_original_arrow_schema(
                str(source_dir / "train-*.parquet")
            )
            output_schema = build_output_arrow_schema(original_schema)
            original = dd.read_parquet(source_dir).assign(
                is_synthetic_augmentation=False,
                source_task_id=None,
            )
            synthetic = dataframe_from_synthetic_rows(
                pd.DataFrame(
                    [
                        {
                            "task": "task-1",
                            "conversations": turns,
                            "is_synthetic_augmentation": True,
                            "source_task_id": "task-1",
                        }
                    ]
                )
            )
            combined = dd.concat([original, synthetic])[output_schema.names]
            combined.to_parquet(
                output_dir,
                write_index=False,
                engine="pyarrow",
                schema=output_schema,
                name_function=lambda i: f"train-{i:05d}-of-00002.parquet",
            )

            self.assertEqual(
                assert_output_arrow_schema(output_dir, output_schema), 2
            )
            reloaded = pd.read_parquet(output_dir)
            synthetic_conversations_type = (
                pq.read_schema(output_dir / "train-00001-of-00002.parquet")
                .field("conversations")
                .type
            )

        self.assertEqual(len(reloaded), 2)
        self.assertTrue(
            all(not isinstance(value, str) for value in reloaded["conversations"])
        )
        self.assertEqual(synthetic_conversations_type, CONVERSATIONS_TYPE)

    def test_requires_exact_publication_row_counts(self) -> None:
        self.assertEqual(
            require_exact_row_counts(
                100,
                150,
                expected_new_rows=150,
                expected_total_rows=250,
            ),
            250,
        )

        for synthetic_count in (149, 225):
            with self.subTest(synthetic_count=synthetic_count):
                with self.assertRaisesRegex(
                    ValueError, "validated synthetic row count must be exactly 150"
                ):
                    require_exact_row_counts(
                        100,
                        synthetic_count,
                        expected_new_rows=150,
                        expected_total_rows=250,
                    )

        with self.assertRaisesRegex(ValueError, "final row count must be exactly 250"):
            require_exact_row_counts(
                99,
                150,
                expected_new_rows=150,
                expected_total_rows=250,
            )


if __name__ == "__main__":
    unittest.main()
