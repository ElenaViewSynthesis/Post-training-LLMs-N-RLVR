"""Tests for the isolated streaming-pipeline pilot harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from refinement_pilot import (
    build_pilot_report,
    main,
    prepare_pilot_snapshot,
)
from refinement_pipeline import load_source_manifest
from test_refinement_pipeline import SOURCE_SCHEMA, source_row


class RefinementPilotTests(unittest.TestCase):
    def _write_source(self, root: Path, rows: int = 20) -> Path:
        source_dir = root / "source-upstream"
        source_dir.mkdir()
        pq.write_table(
            pa.Table.from_pylist(
                [source_row(f"run-{index}", f"task-{index}") for index in range(rows)],
                schema=SOURCE_SCHEMA,
            ),
            source_dir / "train.parquet",
        )
        return source_dir

    def test_preparation_is_deterministic_and_creates_manifest_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write_source(root)
            pilot_a = root / "pilot-a"
            pilot_b = root / "pilot-b"

            first = prepare_pilot_snapshot(
                str(source), pilot_a, sample_size=10, seed=17
            )
            second = prepare_pilot_snapshot(
                str(source), pilot_b, sample_size=10, seed=17
            )
            rows_a = pq.read_table(pilot_a / "source/pilot-00000.parquet").to_pylist()
            rows_b = pq.read_table(pilot_b / "source/pilot-00000.parquet").to_pylist()
            manifest = load_source_manifest(
                pilot_a / "refined/source_manifest.json"
            )

            self.assertEqual(rows_a, rows_b)
            self.assertEqual(len(rows_a), 10)
            self.assertEqual(first["resolved_source"], second["resolved_source"])
            self.assertEqual(manifest.version, 2)

    def test_report_counts_rejections_provider_errors_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write_source(root)
            pilot = root / "pilot"
            prepare_pilot_snapshot(str(source), pilot, sample_size=10, seed=17)
            attempts = [
                {
                    "synthetic_id": "slot-a",
                    "status": "rejected",
                    "rejection_codes": ["task_drift", "duplicate_conversation"],
                    "error_code": None,
                },
                {
                    "synthetic_id": "slot-a",
                    "status": "provider_error",
                    "rejection_codes": [],
                    "error_code": 429,
                },
            ]
            (pilot / "refined/attempts.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in attempts),
                encoding="utf-8",
            )

            report = build_pilot_report(pilot)

            self.assertEqual(report["provider_requests"], 2)
            self.assertEqual(report["accepted_slots"], 0)
            self.assertEqual(report["duplicate_collisions"], 1)
            self.assertEqual(report["rejection_codes"]["task_drift"], 1)
            self.assertEqual(report["provider_errors"], {"429": 1})
            self.assertEqual(report["attempts_per_slot"], {"slot-a": 2})

    def test_execution_is_opt_in_and_forces_no_sync_with_exact_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pilot = root / "pilot"
            observed: dict[str, object] = {}

            async def fake_worker(args: object) -> int:
                observed["args"] = args
                return 0

            with (
                mock.patch("refinement_pilot.prepare_pilot_snapshot"),
                mock.patch(
                    "refinement_pilot.build_pilot_report",
                    return_value={"accepted_slots": 0},
                ),
                mock.patch(
                    "refinement_pilot.run_streaming_worker",
                    side_effect=fake_worker,
                ) as worker,
            ):
                exit_code = main(
                    [
                        "--data-dir",
                        str(root / "data"),
                        "--pilot-dir",
                        str(pilot),
                        "--sample-size",
                        "10",
                        "--execute",
                    ]
                )

            self.assertEqual(exit_code, 0)
            worker.assert_called_once()
            args = observed["args"]
            self.assertTrue(args.no_sync)
            self.assertEqual(args.max_provider_requests, 10)
            self.assertEqual(args.output_dir, pilot / "refined")

    def test_production_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            self.assertEqual(
                main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "--pilot-dir",
                        str(data_dir / "refined"),
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
