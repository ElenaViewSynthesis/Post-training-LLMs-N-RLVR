"""Tests for authoritative pipeline paths and pre-status HF synchronization."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gemini_trajectory_worker as worker
import validate_n_dedup as validator


class PipelinePathConfigurationTests(unittest.TestCase):
    def test_stage_five_normalizes_provider_specific_turns(self) -> None:
        parsed = validator.parse_result_line(
            json.dumps(
                {
                    "variant_id": "task-v0-hash",
                    "status": "ok",
                    "conversations": [
                        {
                            "role": "assistant",
                            "name": "run_bash",
                            "arguments": {"command": "pwd"},
                        },
                        {"role": "assistant", "content": ""},
                        {
                            "role": "tool_result",
                            "observation": "/workspace",
                        },
                    ],
                }
            )
        )

        self.assertEqual(parsed["status"], "parsed")
        self.assertEqual(
            [turn["role"] for turn in parsed["conversations"]],
            ["assistant", "assistant", "tool"],
        )
        self.assertIn('"command":"pwd"', parsed["conversations"][0]["content"])
        self.assertEqual(parsed["conversations"][1]["content"], "")
        self.assertIn("/workspace", parsed["conversations"][2]["content"])
        self.assertTrue(
            all(set(turn) == {"role", "content"} for turn in parsed["conversations"])
        )

    def test_hf_sync_uses_bucket_python_api(self) -> None:
        sync_bucket = mock.Mock()
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.sync_bucket = sync_bucket  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            worker.sync_to_hf(
                Path("/pipeline/raw_results/trajectories.jsonl"),
                "hf://buckets/example/data/raw_results",
            )

        sync_bucket.assert_called_once_with(
            "/pipeline/raw_results", "hf://buckets/example/data/raw_results"
        )

    def test_worker_uses_configured_data_dir_for_default_paths(self) -> None:
        data_dir = Path("/mnt/c/example/pipeline/data")
        with mock.patch.dict(
            os.environ, {"PIPELINE_DATA_DIR": str(data_dir)}, clear=False
        ):
            args = worker.parse_args(["--status-only"])

        self.assertEqual(args.data_dir, data_dir)
        self.assertEqual(args.plan, data_dir / "variant_plan.parquet")
        self.assertEqual(args.tasks, data_dir / "tasks")
        self.assertEqual(args.output, data_dir / "raw_results" / "trajectories.jsonl")

    def test_stage_five_uses_configured_data_dir_for_default_paths(self) -> None:
        data_dir = Path("/mnt/c/example/pipeline/data")
        with mock.patch.dict(
            os.environ, {"PIPELINE_DATA_DIR": str(data_dir)}, clear=False
        ):
            args = validator.parse_args([])

        self.assertEqual(args.data_dir, data_dir)
        self.assertEqual(args.raw_results_dir, data_dir / "raw_results")
        self.assertEqual(
            args.output,
            data_dir / "validated" / "validated_trajectories.parquet",
        )

    def test_pre_status_sync_requires_status_only(self) -> None:
        with self.assertRaises(SystemExit):
            worker.parse_args(["--sync-to-hf-before-status"])

    def test_pre_status_sync_runs_before_resume_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            output = data_dir / "raw_results" / "trajectories.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text("", encoding="utf-8")
            args = worker.parse_args(
                [
                    "--data-dir",
                    str(data_dir),
                    "--status-only",
                    "--sync-to-hf-before-status",
                ]
            )
            calls: list[str] = []

            def fake_sync(output_path: Path, bucket: str) -> None:
                self.assertEqual(output_path, output)
                self.assertEqual(bucket, worker.DEFAULT_HF_BUCKET)
                calls.append("sync")

            def fake_load(*_args: object) -> tuple[object, ...]:
                calls.append("load")
                return ["v1", "v2"], [], {}, {"v1"}, 1, 0

            with (
                mock.patch.object(worker, "sync_to_hf", side_effect=fake_sync),
                mock.patch.object(worker, "load_work", side_effect=fake_load),
            ):
                exit_code = asyncio.run(worker.async_main(args))

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["sync", "load"])

    def test_pre_status_sync_releases_lock_when_upload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            output = data_dir / "raw_results" / "trajectories.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text("", encoding="utf-8")
            args = worker.parse_args(
                [
                    "--data-dir",
                    str(data_dir),
                    "--status-only",
                    "--sync-to-hf-before-status",
                ]
            )
            lock = mock.Mock()

            with (
                mock.patch.object(worker, "acquire_run_lock", return_value=lock),
                mock.patch.object(
                    worker,
                    "sync_to_hf",
                    side_effect=worker.HFSyncError("test sync failure"),
                ),
                self.assertRaises(worker.HFSyncError),
            ):
                asyncio.run(worker.async_main(args))

        lock.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
