"""Focused tests for incremental Hugging Face checkpoint synchronization."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from codex_hf_checkpoint_sidecar import (
    _default_verify_remote_files,
    checkpoint_once,
    cleanup_stale_stages,
)
from refinement_pipeline import write_refinement_state_inventory


class CaptureRemote:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[list[str]] = []

    def sync(self, source: Path, destination: str) -> None:
        del destination
        paths: list[str] = []
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                self.files[relative] = path.read_bytes()
                paths.append(relative)
        self.calls.append(paths)

    def verify(self, remote: str, expected) -> None:
        del remote
        for relative, entry in expected.items():
            data = self.files[relative]
            self.assert_entry(data, entry)

    @staticmethod
    def assert_entry(data: bytes, entry) -> None:
        if len(data) != entry["size"]:
            raise AssertionError("captured remote size mismatch")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise AssertionError("captured remote digest mismatch")


class CodexHfCheckpointSidecarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.state = self.root / "sidecar"
        self.output.mkdir()
        self.run_id = str(uuid.uuid4())
        self._write_json(
            self.output / "run_manifest.json",
            {
                "version": 1,
                "run_instance_id": self.run_id,
                "source_content_sha256": "a" * 64,
                "source_schema_sha256": "b" * 64,
                "target_rows": 30,
                "assignment_algorithm_version": "source-slot-v1",
                "model": "test-model",
                "generation_config": {"agent_batch_size": 4},
                "validation_policy_version": "quality-v1",
            },
        )
        self._write_json(self.output / "source_manifest.json", {"fixture": True})
        self.remote = CaptureRemote()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _add_shard(self, name: str, start: int, rows: int) -> None:
        accepted = self.output / "accepted"
        accepted.mkdir(exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [{"run_id": f"synthetic-{index}"} for index in range(start, start + rows)]
            ),
            accepted / f"accepted-{name}.parquet",
        )

    def _seal(self, completed: int) -> None:
        self._write_json(
            self.output / "progress.json",
            {
                "run_instance_id": self.run_id,
                "completed_rows": completed,
                "remaining_rows": 30 - completed,
                "target_rows": 30,
            },
        )
        write_refinement_state_inventory(self.output)

    def _checkpoint(self) -> dict:
        return checkpoint_once(
            self.output,
            self.state,
            "hf://buckets/example/refined",
            10,
            sync_directory=self.remote.sync,
            verify_remote_files=self.remote.verify,
        )

    def test_syncs_only_after_each_ten_row_threshold_and_resumes(self) -> None:
        self._add_shard("0001", 0, 6)
        self._add_shard("0002", 6, 6)
        self._seal(12)

        first = self._checkpoint()
        self.assertEqual(first["status"], "synchronized")
        self.assertEqual(first["checkpoint_threshold"], 10)
        self.assertEqual(len(self.remote.calls), 3)
        self.assertEqual(
            self.remote.calls[0],
            [
                "accepted/accepted-0001.parquet",
                "accepted/accepted-0002.parquet",
                "run_manifest.json",
                "source_manifest.json",
            ],
        )
        state_after_first = json.loads((self.state / "state.json").read_text())
        self.assertEqual(state_after_first["synced_completed_rows"], 12)
        self.assertEqual(state_after_first["last_checkpoint_threshold"], 10)

        self._add_shard("0003", 12, 4)
        self._seal(16)
        waiting = self._checkpoint()
        self.assertEqual(waiting["status"], "waiting")
        self.assertEqual(waiting["next_checkpoint"], 20)
        self.assertEqual(len(self.remote.calls), 3)

        self._add_shard("0004", 16, 4)
        self._seal(20)
        second = self._checkpoint()
        self.assertEqual(second["status"], "synchronized")
        self.assertEqual(second["checkpoint_threshold"], 20)
        self.assertEqual(
            self.remote.calls[3],
            [
                "accepted/accepted-0003.parquet",
                "accepted/accepted-0004.parquet",
            ],
        )
        marker = json.loads(self.remote.files[second["checkpoint"]])
        self.assertEqual(marker["completed_rows"], 20)
        self.assertEqual(marker["previous_checkpoint"], first["checkpoint"])
        self.assertEqual(len(marker["new_accepted_shards"]), 2)
        latest = json.loads(self.remote.files["latest.json"])
        self.assertEqual(latest["checkpoint"], second["checkpoint"])

    def test_local_cursor_does_not_advance_before_latest_readback(self) -> None:
        self._add_shard("0001", 0, 10)
        self._seal(10)

        def fail_latest(remote: str, expected) -> None:
            self.remote.verify(remote, expected)
            if "latest.json" in expected:
                raise OSError("simulated latest-marker readback failure")

        with self.assertRaisesRegex(OSError, "latest-marker"):
            checkpoint_once(
                self.output,
                self.state,
                "hf://buckets/example/refined",
                10,
                sync_directory=self.remote.sync,
                verify_remote_files=fail_latest,
            )
        self.assertFalse((self.state / "state.json").exists())

        recovered = self._checkpoint()
        self.assertEqual(recovered["status"], "synchronized")
        state = json.loads((self.state / "state.json").read_text())
        self.assertEqual(state["synced_completed_rows"], 10)
        self.assertEqual(state["last_checkpoint_threshold"], 10)

    def test_cleanup_removes_only_sidecar_staging_directories(self) -> None:
        self.state.mkdir()
        stale = self.state / ".stage-data-interrupted"
        stale.mkdir()
        (stale / "partial.parquet").write_bytes(b"partial")
        durable = self.state / "state.json"
        durable.write_text("{}", encoding="utf-8")

        cleanup_stale_stages(self.state)

        self.assertFalse(stale.exists())
        self.assertTrue(durable.exists())

    def test_remote_readback_retries_with_uncached_filesystem(self) -> None:
        payload = b"sealed checkpoint shard"
        expected = {
            "accepted/accepted-fixture.parquet": {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }

        missing = mock.Mock()
        missing.open.side_effect = FileNotFoundError("not visible yet")
        visible = mock.Mock()
        visible.open.side_effect = lambda path, mode: io.BytesIO(payload)
        with (
            mock.patch(
                "fsspec.core.url_to_fs",
                side_effect=[(missing, "bucket/root"), (visible, "bucket/root")],
            ) as url_to_fs,
            mock.patch("codex_hf_checkpoint_sidecar.time.sleep") as sleep,
        ):
            _default_verify_remote_files("hf://buckets/example/run", expected)

        self.assertEqual(url_to_fs.call_count, 2)
        for call in url_to_fs.call_args_list:
            self.assertTrue(call.kwargs["skip_instance_cache"])
        sleep.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
