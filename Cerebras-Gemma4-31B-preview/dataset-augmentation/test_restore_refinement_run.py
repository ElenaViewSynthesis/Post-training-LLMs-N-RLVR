"""Tests for the refinement remote-backup restore drill."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from refinement_pipeline import load_run_manifest
from restore_refinement_run import restore_refinement_run
from test_stream_refinement_worker import write_completed_fixture, write_source_fixture


class RestoreRefinementRunTests(unittest.TestCase):
    def test_restore_into_fresh_directory_reconstructs_status_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            staging = root / "staging"
            write_completed_fixture(source_dir, staging)
            run_manifest = load_run_manifest(staging / "run_manifest.json")
            remote = root / "bucket/runs" / run_manifest.run_instance_id
            remote.parent.mkdir(parents=True)
            shutil.move(str(staging), remote)
            restored = root / "restored"

            def copy_remote(source: str, destination: str) -> None:
                shutil.copytree(source, destination)

            with (
                mock.patch(
                    "restore_refinement_run.verify_remote_refinement_state"
                ) as verify_remote,
                mock.patch(
                    "restore_refinement_run.sync_bucket",
                    side_effect=copy_remote,
                ),
            ):
                report = restore_refinement_run(str(remote), restored)

            verify_remote.assert_called_once_with(str(remote))
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["accepted_rows"], 1)
            self.assertEqual(report["target_rows"], 1)
            self.assertEqual(report["run_instance_id"], run_manifest.run_instance_id)
            self.assertTrue((restored / "complete.json").is_file())

    def test_restore_requires_a_fresh_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "existing"
            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must not already exist"):
                restore_refinement_run("hf://buckets/example/runs/id", destination)


if __name__ == "__main__":
    unittest.main()
