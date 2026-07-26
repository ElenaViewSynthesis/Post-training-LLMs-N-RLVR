"""Tests for bounded-concurrency, retry-per-slot refinement generation."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa

from refinement_pipeline import SourceIdentity, refinement_slots_for_source
from stream_refinement_worker import (
    Settings,
    acquire_run_lock,
    build_refinement_prompt,
    load_attempt_counts,
    parse_args,
    refine_slot,
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
    return SourceIdentity(
        source_record_id=f"record-{row_index}-{run_id}",
        source_file="fixture/train.parquet",
        source_row_index=row_index,
        source_trial_name=run_id,
        source_run_id=run_id,
        source_task_id=task,
    )


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
