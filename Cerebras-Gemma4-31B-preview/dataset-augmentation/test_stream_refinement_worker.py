"""Tests for bounded-concurrency, retry-per-slot refinement generation."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from refinement_pipeline import (
    SourceIdentity,
    build_augmented_schema,
    build_refined_row,
    conversation_fingerprint,
    ensure_run_manifest,
    ensure_source_manifest,
    load_source_identities,
    load_source_manifest,
    load_run_manifest,
    refinement_slots_for_source,
    scan_accepted_shards,
    source_conversation_fingerprint,
    write_accepted_shard,
    write_refinement_state_inventory,
)
from stream_refinement_worker import (
    ProviderPreflightError,
    ProviderRequestBudget,
    RefinementSyncError,
    Settings,
    SlotResult,
    _safe_error,
    acquire_run_lock,
    async_main,
    build_refinement_prompt,
    load_attempt_counts,
    parse_args,
    provider_preflight,
    refine_slot,
    resolve_batch_conversation_collisions,
    sync_output,
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


def make_source_row(run_id: str, task: str) -> dict:
    return {
        "conversations": [
            {"content": "Inspect the repository", "role": "user"},
            {"content": "Inspection complete", "role": "assistant"},
        ],
        "model": "source-model",
        "model_provider": "source-provider",
        "run_id": run_id,
        "task": task,
        "trial_name": run_id,
    }


def make_source_identity(run_id: str, task: str, row_index: int = 0) -> SourceIdentity:
    row = make_source_row(run_id, task)
    return SourceIdentity(
        source_record_id=f"record-{row_index}-{run_id}",
        source_file="fixture/train.parquet",
        source_row_index=row_index,
        source_trial_name=run_id,
        source_conversation_fingerprint=source_conversation_fingerprint(
            row["conversations"]
        ),
        source_run_id=run_id,
        source_task_id=task,
    )


def write_source_fixture(root: Path, row_count: int = 1) -> Path:
    source_dir = root / "source"
    source_dir.mkdir()
    rows = [make_source_row(f"source-{index}", f"task-{index}") for index in range(row_count)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA),
        source_dir / "train-00000.parquet",
    )
    return source_dir


def write_completed_fixture(source_dir: Path, output_dir: Path) -> None:
    original_schema, identities = load_source_identities(str(source_dir))
    source_manifest = ensure_source_manifest(
        output_dir / "source_manifest.json",
        requested_source=str(source_dir),
        resolved_source=str(source_dir),
        schema=original_schema,
        identities=identities,
    )
    ensure_run_manifest(
        output_dir / "run_manifest.json",
        source_manifest=source_manifest,
        target_rows=1,
        model="test-model",
        generation_config={
            "concurrency": 2,
            "request_batch_size": 4,
            "max_output_tokens": 4096,
            "max_attempts_per_run": 1,
            "timeout_seconds": 180,
        },
    )
    identity = next(iter(identities.values()))
    slot = refinement_slots_for_source(identity, frozenset())[0]
    row = make_source_row(identity.source_run_id, identity.source_task_id)
    accepted = build_refined_row(
        row,
        slot,
        [
            {"role": "user", "content": "Inspect the repository carefully"},
            {"role": "assistant", "content": "Careful inspection complete"},
        ],
        original_schema,
        model="test-model",
        provider="gemini",
    )
    write_accepted_shard(
        output_dir / "accepted",
        [accepted],
        build_augmented_schema(original_schema),
    )
    write_refinement_state_inventory(output_dir)


def worker_args(source_dir: Path, output_dir: Path, target_rows: int) -> object:
    return parse_args(
        [
            "--original-source",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--target-rows",
            str(target_rows),
            "--model",
            "test-model",
            "--concurrency",
            "2",
            "--request-batch-size",
            "4",
            "--max-attempts-per-run",
            "1",
        ]
    )


class ClosingAsyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.preflight_calls = 0
        self.models = SimpleNamespace(get=self.get_model)

    async def get_model(self, **_: object) -> SimpleNamespace:
        self.preflight_calls += 1
        return SimpleNamespace(name="test-model")

    async def aclose(self) -> None:
        self.closed = True


def settings(max_attempts: int = 3) -> Settings:
    return Settings(
        model="test-model",
        concurrency=2,
        request_batch_size=4,
        max_output_tokens=512,
        max_attempts_per_run=max_attempts,
        timeout_ms=1_000,
        sync_every_shards=2,
        hf_bucket="hf://buckets/example/refined",
        sync_enabled=False,
    )


class SequenceInteractions:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = 0

    async def create(self, **_: object) -> SimpleNamespace:
        output = self.outputs[self.calls]
        self.calls += 1
        return SimpleNamespace(output_text=output, id=f"interaction-{self.calls}")


class TrackingInteractions:
    def __init__(self, output: str):
        self.output = output
        self.active = 0
        self.max_active = 0

    async def create(self, **_: object) -> SimpleNamespace:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return SimpleNamespace(output_text=self.output, id="interaction")


class StreamRefinementWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_status_code_marks_provider_configuration_errors_fatal(self) -> None:
        class BadRequestError(Exception):
            status_code = 400

        class FailingInteractions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **_: object) -> object:
                self.calls += 1
                raise BadRequestError("invalid secret-key")

        interactions = FailingInteractions()
        client = SimpleNamespace(interactions=interactions)
        source = make_source_row("source-1", "task-a")
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]

        result = await refine_slot(
            client,
            source,
            slot,
            SOURCE_SCHEMA,
            settings(max_attempts=3),
            "secret-key",
            asyncio.Semaphore(1),
            starting_attempt=0,
        )

        self.assertIsNone(result.accepted_row)
        self.assertEqual(interactions.calls, 1)
        self.assertEqual(result.attempts[0]["status"], "provider_error")
        self.assertEqual(result.attempts[0]["error_code"], 400)
        self.assertNotIn("secret-key", result.attempts[0]["error"])
        self.assertEqual(_safe_error(BadRequestError("failed"), "")[1], 400)

    async def test_data_directory_controls_default_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with mock.patch.dict(
                os.environ, {"PIPELINE_DATA_DIR": str(data_dir)}, clear=False
            ):
                args = parse_args(["--status-only"])
        self.assertEqual(args.data_dir, data_dir)
        self.assertEqual(args.output_dir, data_dir / "refined")

    async def test_run_lock_prevents_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "refined"
            first = acquire_run_lock(output_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "another refinement worker"):
                    acquire_run_lock(output_dir)
            finally:
                first.close()

    async def test_sync_only_skips_source_and_gemini(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "refined"
            output_dir.mkdir()
            (output_dir / "progress.json").write_text("{}\n", encoding="utf-8")
            args = parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--hf-bucket",
                    "hf://buckets/example/refined",
                    "--sync-only",
                ]
            )
            with (
                mock.patch(
                    "stream_refinement_worker.load_source_identities"
                ) as load_source,
                mock.patch("stream_refinement_worker.sync_output") as sync,
                mock.patch(
                    "stream_refinement_worker.create_gemini_client"
                ) as create_client,
            ):
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 0)
        sync.assert_called_once_with(
            output_dir,
            "hf://buckets/example/refined",
            verify_hashes=True,
        )
        load_source.assert_not_called()
        create_client.assert_not_called()

    async def test_provider_preflight_only_never_reads_source_or_generates(self) -> None:
        args = parse_args(
            ["--provider-preflight-only", "--model", "test-model"]
        )
        client = ClosingAsyncClient()
        client.interactions = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            mock.patch(
                "stream_refinement_worker.create_gemini_client",
                return_value=SimpleNamespace(aio=client),
            ),
            mock.patch(
                "stream_refinement_worker.load_source_identities"
            ) as load_source,
        ):
            exit_code = await async_main(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(client.preflight_calls, 1)
        self.assertTrue(client.closed)
        load_source.assert_not_called()
        client.interactions.create.assert_not_called()

    async def test_fresh_runs_use_distinct_remote_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dirs = [root / "refined-a", root / "refined-b"]
            for output_dir in output_dirs:
                args = worker_args(source_dir, output_dir, target_rows=1)
                args.status_only = True
                self.assertEqual(await async_main(args), 0)

            manifests = [
                load_run_manifest(output_dir / "run_manifest.json")
                for output_dir in output_dirs
            ]
            self.assertNotEqual(
                manifests[0].run_instance_id,
                manifests[1].run_instance_id,
            )
            with (
                mock.patch("huggingface_hub.sync_bucket") as bucket_sync,
                mock.patch(
                    "stream_refinement_worker.verify_remote_refinement_state"
                ) as verify_remote,
            ):
                for output_dir in output_dirs:
                    sync_output(
                        output_dir,
                        "hf://buckets/example/refined",
                        verify_hashes=True,
                    )

            destinations = [call.args[1] for call in bucket_sync.call_args_list]
            self.assertEqual(len(destinations), 2)
            self.assertEqual(verify_remote.call_count, 2)
            self.assertNotEqual(destinations[0], destinations[1])
            self.assertTrue(
                destinations[0].endswith(
                    f"/runs/{manifests[0].run_instance_id}"
                )
            )
            self.assertTrue(
                destinations[1].endswith(
                    f"/runs/{manifests[1].run_instance_id}"
                )
            )

    async def test_worker_materializes_and_reuses_local_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root, row_count=2)
            output_dir = root / "refined"
            snapshot_dir = root / "source-snapshot"
            args = worker_args(source_dir, output_dir, target_rows=2)
            args.status_only = True
            args.source_snapshot_dir = snapshot_dir

            self.assertEqual(await async_main(args), 0)
            source_manifest = load_source_manifest(
                output_dir / "source_manifest.json"
            )
            _, identities = load_source_identities(str(snapshot_dir))

            self.assertEqual(source_manifest.resolved_source, str(snapshot_dir))
            self.assertEqual(len(source_manifest.source_files), 1)
            self.assertEqual(
                {identity.source_file for identity in identities.values()},
                {str(source_dir / "train-00000.parquet")},
            )
            self.assertEqual(await async_main(args), 0)

    async def test_sync_only_reuses_the_existing_run_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            status_args = worker_args(source_dir, output_dir, target_rows=1)
            status_args.status_only = True
            self.assertEqual(await async_main(status_args), 0)
            run_manifest = load_run_manifest(output_dir / "run_manifest.json")
            sync_args = parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--hf-bucket",
                    "hf://buckets/example/refined",
                    "--sync-only",
                ]
            )
            with (
                mock.patch("huggingface_hub.sync_bucket") as bucket_sync,
                mock.patch(
                    "stream_refinement_worker.load_source_identities"
                ) as load_source,
                mock.patch(
                    "stream_refinement_worker.create_gemini_client"
                ) as create_client,
                mock.patch(
                    "stream_refinement_worker.verify_remote_refinement_state"
                ) as verify_remote,
            ):
                self.assertEqual(await async_main(sync_args), 0)

            bucket_sync.assert_called_once_with(
                str(output_dir),
                "hf://buckets/example/refined/runs/"
                f"{run_manifest.run_instance_id}",
            )
            load_source.assert_not_called()
            create_client.assert_not_called()
            verify_remote.assert_called_once_with(
                "hf://buckets/example/refined/runs/"
                f"{run_manifest.run_instance_id}"
            )

    async def test_remote_readback_failure_fails_refinement_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            write_completed_fixture(source_dir, output_dir)
            with (
                mock.patch("huggingface_hub.sync_bucket") as bucket_sync,
                mock.patch(
                    "stream_refinement_worker.verify_remote_refinement_state",
                    side_effect=ValueError("remote checksum mismatch"),
                ) as verify_remote,
            ):
                with self.assertRaisesRegex(
                    RefinementSyncError,
                    "remote checksum mismatch",
                ):
                    sync_output(
                        output_dir,
                        "hf://buckets/example/refined",
                        verify_hashes=True,
                    )

            bucket_sync.assert_called_once()
            verify_remote.assert_called_once()

    async def test_provider_preflight_precedes_source_content_reverification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            status_args = worker_args(source_dir, output_dir, target_rows=1)
            status_args.status_only = True
            self.assertEqual(await async_main(status_args), 0)

            source_path = source_dir / "train-00000.parquet"
            rows = pq.read_table(source_path).to_pylist()
            rows[0]["model_provider"] = "changed-provider"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA),
                source_path,
            )
            run_args = worker_args(source_dir, output_dir, target_rows=1)
            client = ClosingAsyncClient()
            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ) as create_client,
            ):
                with self.assertRaisesRegex(ValueError, "immutable run manifest"):
                    await async_main(run_args)
            create_client.assert_called_once()
            self.assertEqual(client.preflight_calls, 1)
            self.assertTrue(client.closed)

    async def test_provider_preflight_failure_skips_source_hashing(self) -> None:
        class UnauthorizedError(Exception):
            status_code = 401

        class UnauthorizedModels:
            async def get(self, **_: object) -> object:
                raise UnauthorizedError("invalid test-key")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "refined"
            args = worker_args(root / "source", output_dir, target_rows=1)
            client = ClosingAsyncClient()
            client.models = UnauthorizedModels()
            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
                mock.patch(
                    "stream_refinement_worker.load_source_identities"
                ) as load_source,
            ):
                with self.assertRaisesRegex(
                    ProviderPreflightError,
                    "HTTP 401",
                ):
                    await async_main(args)

        load_source.assert_not_called()
        self.assertTrue(client.closed)

    async def test_provider_preflight_uses_model_get_without_spending_budget(self) -> None:
        client = ClosingAsyncClient()
        budget = ProviderRequestBudget(1)

        await provider_preflight(client, settings(), "test-key")

        self.assertEqual(client.preflight_calls, 1)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.remaining, 1)

    async def test_request_budget_is_exact_under_concurrency_and_retries(self) -> None:
        class RejectingInteractions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                await asyncio.sleep(0)
                return SimpleNamespace(
                    output_text="not-json",
                    id=f"interaction-{self.calls}",
                )

        interactions = RejectingInteractions()
        client = SimpleNamespace(interactions=interactions)
        request_budget = ProviderRequestBudget(10)
        semaphore = asyncio.Semaphore(3)
        source = make_source_row("source-1", "task-a")
        slots = [
            refinement_slots_for_source(
                make_source_identity(f"source-{index}", f"task-{index}", index),
                frozenset(),
            )[0]
            for index in range(8)
        ]

        results = await asyncio.gather(
            *(
                refine_slot(
                    client,
                    source,
                    slot,
                    SOURCE_SCHEMA,
                    settings(max_attempts=5),
                    "test-key",
                    semaphore,
                    starting_attempt=0,
                    request_budget=request_budget,
                )
                for slot in slots
            )
        )

        self.assertEqual(interactions.calls, 10)
        self.assertEqual(request_budget.used, 10)
        self.assertEqual(request_budget.remaining, 0)
        self.assertEqual(sum(len(result.attempts) for result in results), 10)
        self.assertTrue(any(result.request_budget_exhausted for result in results))
        self.assertEqual(
            sorted(
                attempt["provider_request_number"]
                for result in results
                for attempt in result.attempts
            ),
            list(range(1, 11)),
        )

    async def test_worker_scheduler_stops_at_exact_request_cap(self) -> None:
        class RejectingInteractions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                await asyncio.sleep(0)
                return SimpleNamespace(output_text="not-json", id=str(self.calls))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root, row_count=4)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=4)
            args.no_sync = True
            args.max_attempts_per_run = 3
            args.max_provider_requests = 10
            client = ClosingAsyncClient()
            interactions = RejectingInteractions()
            client.interactions = interactions
            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
            ):
                exit_code = await async_main(args)

            attempts = [
                json.loads(line)
                for line in (output_dir / "attempts.jsonl").read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(interactions.calls, 10)
        self.assertEqual(len(attempts), 10)
        self.assertEqual(
            sorted(attempt["provider_request_number"] for attempt in attempts),
            list(range(1, 11)),
        )

    async def test_completed_startup_syncs_without_gemini(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            write_completed_fixture(source_dir, output_dir)
            args = worker_args(source_dir, output_dir, target_rows=1)
            with (
                mock.patch("stream_refinement_worker.sync_output") as sync,
                mock.patch(
                    "stream_refinement_worker.create_gemini_client"
                ) as create_client,
            ):
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 0)
        sync.assert_called_once_with(output_dir, args.hf_bucket)
        create_client.assert_not_called()

    async def test_failed_completed_sync_can_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            write_completed_fixture(source_dir, output_dir)
            args = worker_args(source_dir, output_dir, target_rows=1)
            with (
                mock.patch(
                    "stream_refinement_worker.sync_output",
                    side_effect=[RefinementSyncError("offline"), None],
                ) as sync,
                mock.patch(
                    "stream_refinement_worker.create_gemini_client"
                ) as create_client,
            ):
                with self.assertRaisesRegex(RefinementSyncError, "offline"):
                    await async_main(args)
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(sync.call_count, 2)
        create_client.assert_not_called()

    async def test_failed_final_sync_recovers_without_another_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=1)
            client = ClosingAsyncClient()

            async def accepted_result(
                *call_args: object, **_: object
            ) -> SlotResult:
                source = call_args[1]
                slot = call_args[2]
                original_schema = call_args[3]
                assert isinstance(source, dict)
                return SlotResult(
                    slot,
                    build_refined_row(
                        source,
                        slot,
                        [
                            {"role": "user", "content": "Inspect carefully"},
                            {
                                "role": "assistant",
                                "content": "Inspection complete",
                            },
                        ],
                        original_schema,
                        model="test-model",
                        provider="gemini",
                    ),
                    [
                        {
                            "synthetic_id": slot.synthetic_id,
                            "status": "accepted",
                            "error_code": None,
                        }
                    ],
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ) as create_client,
                mock.patch(
                    "stream_refinement_worker.refine_slot",
                    side_effect=accepted_result,
                ) as refine,
                mock.patch(
                    "stream_refinement_worker.sync_output",
                    side_effect=[RefinementSyncError("offline"), None],
                ) as sync,
            ):
                with self.assertRaisesRegex(RefinementSyncError, "offline"):
                    await async_main(args)
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 0)
        self.assertTrue(client.closed)
        self.assertEqual(create_client.call_count, 1)
        self.assertEqual(refine.call_count, 1)
        self.assertEqual(sync.call_count, 2)

    async def test_fatal_batch_syncs_before_failure_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=1)
            client = ClosingAsyncClient()

            async def fatal_result(*call_args: object, **_: object) -> SlotResult:
                slot = call_args[2]
                assert hasattr(slot, "synthetic_id")
                return SlotResult(
                    slot,
                    None,
                    [
                        {
                            "synthetic_id": slot.synthetic_id,
                            "status": "provider_error",
                            "error_code": 401,
                        }
                    ],
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
                mock.patch(
                    "stream_refinement_worker.refine_slot",
                    side_effect=fatal_result,
                ),
                mock.patch("stream_refinement_worker.sync_output") as sync,
            ):
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 2)
        self.assertTrue(client.closed)
        sync.assert_called_once_with(output_dir, args.hf_bucket)

    async def test_fully_rejected_batch_syncs_before_failure_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=1)
            client = ClosingAsyncClient()

            async def rejected_result(*call_args: object, **_: object) -> SlotResult:
                slot = call_args[2]
                assert hasattr(slot, "synthetic_id")
                return SlotResult(
                    slot,
                    None,
                    [
                        {
                            "synthetic_id": slot.synthetic_id,
                            "status": "rejected",
                            "error_code": None,
                        }
                    ],
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
                mock.patch(
                    "stream_refinement_worker.refine_slot",
                    side_effect=rejected_result,
                ),
                mock.patch("stream_refinement_worker.sync_output") as sync,
            ):
                exit_code = await async_main(args)

        self.assertEqual(exit_code, 2)
        self.assertTrue(client.closed)
        sync.assert_called_once_with(output_dir, args.hf_bucket)

    async def test_partial_batch_is_persisted_and_synced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root, row_count=2)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=2)
            client = ClosingAsyncClient()

            async def partial_result(*call_args: object, **_: object) -> SlotResult:
                source_row = call_args[1]
                slot = call_args[2]
                original_schema = call_args[3]
                assert isinstance(source_row, dict)
                if slot.source_row_index == 0:
                    row = build_refined_row(
                        source_row,
                        slot,
                        [
                            {"role": "user", "content": "Inspect carefully"},
                            {"role": "assistant", "content": "Inspection complete"},
                        ],
                        original_schema,
                        model="test-model",
                        provider="gemini",
                    )
                    return SlotResult(
                        slot,
                        row,
                        [
                            {
                                "synthetic_id": slot.synthetic_id,
                                "status": "accepted",
                                "error_code": None,
                            }
                        ],
                    )
                return SlotResult(
                    slot,
                    None,
                    [
                        {
                            "synthetic_id": slot.synthetic_id,
                            "status": "rejected",
                            "error_code": None,
                        }
                    ],
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
                mock.patch(
                    "stream_refinement_worker.refine_slot",
                    side_effect=partial_result,
                ),
                mock.patch("stream_refinement_worker.sync_output") as sync,
            ):
                exit_code = await async_main(args)

            schema = build_augmented_schema(SOURCE_SCHEMA)
            completed, row_count, fingerprints = scan_accepted_shards(
                output_dir / "accepted", schema
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(len(completed), 1)
        self.assertEqual(row_count, 1)
        self.assertEqual(len(fingerprints), 1)
        self.assertTrue(client.closed)
        sync.assert_called_once_with(output_dir, args.hf_bucket)

    async def test_retries_rejected_result_for_the_same_slot(self) -> None:
        valid_output = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect carefully"},
                    {"role": "assistant", "content": "Completed carefully"},
                ]
            }
        )
        interactions = SequenceInteractions(["not-json", valid_output])
        client = SimpleNamespace(interactions=interactions)
        source = make_source_row("source-1", "task-a")
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]

        result = await refine_slot(
            client,
            source,
            slot,
            SOURCE_SCHEMA,
            settings(),
            "secret-key",
            asyncio.Semaphore(1),
            starting_attempt=4,
        )

        self.assertIsNotNone(result.accepted_row)
        self.assertEqual(interactions.calls, 2)
        self.assertEqual(
            [attempt["status"] for attempt in result.attempts],
            ["rejected", "accepted"],
        )
        self.assertEqual(
            [attempt["attempt_number"] for attempt in result.attempts], [5, 6]
        )
        self.assertEqual(result.accepted_row["run_id"], slot.synthetic_id)

    async def test_quality_rejection_codes_are_recorded(self) -> None:
        refusal = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect the repository carefully"},
                    {
                        "role": "assistant",
                        "content": "I am sorry, I cannot help with this request.",
                    },
                ]
            }
        )
        drift = json.dumps(
            {
                "conversations": [
                    {
                        "role": "user",
                        "content": "Write a detailed poem about flowers and summer rain.",
                    },
                    {
                        "role": "assistant",
                        "content": "Here is a complete poem about a bright garden.",
                    },
                ]
            }
        )
        valid = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect the repository carefully"},
                    {
                        "role": "assistant",
                        "content": "The repository inspection is complete and verified.",
                    },
                ]
            }
        )
        interactions = SequenceInteractions([refusal, drift, valid])
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]

        result = await refine_slot(
            SimpleNamespace(interactions=interactions),
            make_source_row("source-1", "task-a"),
            slot,
            SOURCE_SCHEMA,
            settings(),
            "secret-key",
            asyncio.Semaphore(1),
            starting_attempt=0,
        )

        self.assertIsNotNone(result.accepted_row)
        self.assertEqual(result.attempts[0]["status"], "rejected")
        self.assertEqual(result.attempts[0]["rejection_codes"], ["refusal_only"])
        self.assertEqual(result.attempts[1]["status"], "rejected")
        self.assertEqual(result.attempts[1]["rejection_codes"], ["task_drift"])
        self.assertEqual(result.attempts[2]["status"], "accepted")

    async def test_batch_duplicates_retry_the_losing_slot_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = write_source_fixture(root)
            output_dir = root / "refined"
            args = worker_args(source_dir, output_dir, target_rows=2)
            args.max_attempts_per_run = 2
            client = ClosingAsyncClient()
            calls: Counter[str] = Counter()

            async def duplicate_then_unique(
                *call_args: object, **_: object
            ) -> SlotResult:
                source = call_args[1]
                slot = call_args[2]
                original_schema = call_args[3]
                starting_attempt = call_args[7]
                assert isinstance(source, dict)
                calls[slot.synthetic_id] += 1
                if calls[slot.synthetic_id] == 1:
                    conversations = [
                        {
                            "role": "user",
                            "content": "Inspect the repository using the shared approach.",
                        },
                        {
                            "role": "assistant",
                            "content": "The shared repository inspection is complete.",
                        },
                    ]
                else:
                    conversations = [
                        {
                            "role": "user",
                            "content": (
                                "Inspect the repository using the alternate approach "
                                f"for {slot.synthetic_id}."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "The alternate repository inspection is complete for "
                                f"{slot.synthetic_id}."
                            ),
                        },
                    ]
                row = build_refined_row(
                    source,
                    slot,
                    conversations,
                    original_schema,
                    model="test-model",
                    provider="gemini",
                )
                return SlotResult(
                    slot,
                    row,
                    [
                        {
                            "synthetic_id": slot.synthetic_id,
                            "attempt_number": starting_attempt + 1,
                            "status": "accepted",
                            "error": None,
                            "error_code": None,
                            "conversation_fingerprint": conversation_fingerprint(
                                conversations
                            ),
                            "rejection_codes": [],
                        }
                    ],
                )

            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                mock.patch(
                    "stream_refinement_worker.create_gemini_client",
                    return_value=SimpleNamespace(aio=client),
                ),
                mock.patch(
                    "stream_refinement_worker.refine_slot",
                    side_effect=duplicate_then_unique,
                ),
                mock.patch("stream_refinement_worker.sync_output"),
            ):
                exit_code = await async_main(args)

            schema = build_augmented_schema(SOURCE_SCHEMA)
            completed, row_count, fingerprints = scan_accepted_shards(
                output_dir / "accepted", schema
            )
            attempts = [
                json.loads(line)
                for line in (output_dir / "attempts.jsonl").read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(completed), 2)
        self.assertEqual(row_count, 2)
        self.assertEqual(len(fingerprints), 2)
        self.assertEqual(sorted(calls.values()), [1, 2])
        duplicate_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("rejection_codes") == ["duplicate_conversation"]
        ]
        self.assertEqual(len(duplicate_attempts), 1)
        self.assertEqual(
            duplicate_attempts[0]["synthetic_id"],
            max(completed),
        )
        self.assertTrue(client.closed)

    async def test_existing_fingerprint_rejects_a_new_candidate(self) -> None:
        source = make_source_row("source-1", "task-a")
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]
        conversations = [
            {"role": "user", "content": "Inspect the repository carefully."},
            {
                "role": "assistant",
                "content": "The careful repository inspection is complete.",
            },
        ]
        row = build_refined_row(
            source,
            slot,
            conversations,
            SOURCE_SCHEMA,
            model="test-model",
            provider="gemini",
        )
        fingerprint = conversation_fingerprint(conversations)
        result = SlotResult(
            slot,
            row,
            [
                {
                    "synthetic_id": slot.synthetic_id,
                    "status": "accepted",
                    "error": None,
                    "error_code": None,
                    "conversation_fingerprint": fingerprint,
                    "rejection_codes": [],
                }
            ],
        )

        accepted, duplicates = resolve_batch_conversation_collisions(
            [result], {fingerprint}
        )

        self.assertEqual(accepted, set())
        self.assertEqual(duplicates, {slot.synthetic_id})
        self.assertIsNone(result.accepted_row)
        self.assertEqual(
            result.attempts[-1]["rejection_codes"], ["duplicate_conversation"]
        )

    async def test_stops_slot_after_first_success(self) -> None:
        valid_output = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect carefully"},
                    {"role": "assistant", "content": "Completed carefully"},
                ]
            }
        )
        interactions = SequenceInteractions([valid_output, valid_output, valid_output])
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]

        result = await refine_slot(
            SimpleNamespace(interactions=interactions),
            make_source_row("source-1", "task-a"),
            slot,
            SOURCE_SCHEMA,
            settings(),
            "secret-key",
            asyncio.Semaphore(1),
            starting_attempt=0,
        )

        self.assertIsNotNone(result.accepted_row)
        self.assertEqual(interactions.calls, 1)
        self.assertEqual(len(result.attempts), 1)

    async def test_retries_an_unchanged_source_conversation(self) -> None:
        source = make_source_row("source-1", "task-a")
        unchanged = json.dumps({"conversations": source["conversations"]})
        refined = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect more carefully"},
                    {"role": "assistant", "content": "Inspection improved"},
                ]
            }
        )
        interactions = SequenceInteractions([unchanged, refined])
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]

        result = await refine_slot(
            SimpleNamespace(interactions=interactions),
            source,
            slot,
            SOURCE_SCHEMA,
            settings(),
            "secret-key",
            asyncio.Semaphore(1),
            starting_attempt=0,
        )

        self.assertIsNotNone(result.accepted_row)
        self.assertEqual(interactions.calls, 2)
        self.assertIn("identical to the source", result.attempts[0]["error"])

    async def test_requests_are_concurrent_but_semaphore_bounded(self) -> None:
        valid_output = json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Inspect carefully"},
                    {"role": "assistant", "content": "Completed carefully"},
                ]
            }
        )
        interactions = TrackingInteractions(valid_output)
        client = SimpleNamespace(interactions=interactions)
        semaphore = asyncio.Semaphore(2)
        requests = []
        for index in range(4):
            source_id = f"source-{index}"
            task_id = f"task-{index}"
            slot = refinement_slots_for_source(
                make_source_identity(source_id, task_id, index), frozenset()
            )[0]
            requests.append(
                refine_slot(
                    client,
                    make_source_row(source_id, task_id),
                    slot,
                    SOURCE_SCHEMA,
                    settings(),
                    "secret-key",
                    semaphore,
                    starting_attempt=0,
                )
            )

        results = await asyncio.gather(*requests)

        self.assertTrue(all(result.accepted_row for result in results))
        self.assertEqual(interactions.max_active, 2)

    async def test_prompt_contains_source_conversation_and_slot(self) -> None:
        slot = refinement_slots_for_source(
            make_source_identity("source-1", "task-a"), frozenset()
        )[0]
        prompt = build_refinement_prompt(
            make_source_row("source-1", "task-a"), slot
        )
        self.assertIn("Task identifier: task-a", prompt)
        self.assertIn("Refinement index: 0", prompt)
        self.assertIn("Inspect the repository", prompt)

    async def test_attempt_scan_treats_non_objects_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attempts.jsonl"
            path.write_text(
                "[]\n"
                '{"synthetic_id":"refined-1","status":"rejected"}\n'
                "not-json\n",
                encoding="utf-8",
            )
            counts, malformed = load_attempt_counts(path)

        self.assertEqual(counts["refined-1"], 1)
        self.assertEqual(malformed, 2)


if __name__ == "__main__":
    unittest.main()
