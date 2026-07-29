"""Tests for Codex-only trajectory quality policy and micro-batching."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

import pyarrow as pa

from codex_refinement_worker import (
    DEFAULT_BATCH_SIZE,
    CodexResponse,
    CodexRuntime,
    PendingItem,
    SYSTEM_INSTRUCTIONS,
    build_batch_prompt,
    parse_args,
    process_batch,
    validate_codex_trajectory_quality,
)
from refinement_pipeline import (
    SourceIdentity,
    refinement_slots_for_source,
    source_conversation_fingerprint,
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


def source_row(conversations: list[dict[str, str]]) -> dict:
    return {
        "conversations": conversations,
        "model": "source-model",
        "model_provider": "source-provider",
        "run_id": "source-run",
        "task": "source-task",
        "trial_name": "source-trial",
    }


def pending_item(conversations: list[dict[str, str]]) -> PendingItem:
    return pending_items(conversations)[0]


def pending_items(conversations: list[dict[str, str]]) -> list[PendingItem]:
    row = source_row(conversations)
    identity = SourceIdentity(
        source_record_id="source-record",
        source_file="fixture/train.parquet",
        source_row_index=0,
        source_trial_name="source-trial",
        source_conversation_fingerprint=source_conversation_fingerprint(
            conversations
        ),
        source_run_id="source-run",
        source_task_id="source-task",
    )
    slots = refinement_slots_for_source(
        identity, frozenset({identity.source_record_id})
    )
    return [PendingItem(slot=slot, source_row=row) for slot in slots]


class CodexRefinementWorkerTests(unittest.TestCase):
    def test_default_micro_batch_size_is_four(self) -> None:
        self.assertEqual(DEFAULT_BATCH_SIZE, 4)
        self.assertEqual(parse_args([]).batch_size, 4)

    def test_prompt_preserves_legacy_terminal_feedback_envelope(self) -> None:
        self.assertIn("user -> assistant pairs", SYSTEM_INSTRUCTIONS)
        self.assertIn("do not relabel those source user turns", SYSTEM_INSTRUCTIONS)

    def test_prompt_requires_assistant_to_carry_forward_terminal_evidence(
        self,
    ) -> None:
        normalized_instructions = " ".join(SYSTEM_INSTRUCTIONS.split())
        self.assertIn(
            "following assistant response must explicitly interpret and carry "
            "forward",
            normalized_instructions,
        )
        self.assertIn(
            "paths, commands, symbols, failures, test counts, warnings, and "
            "findings instead of a generic acknowledgment",
            normalized_instructions,
        )

    def test_batch_prompt_deduplicates_shared_source_conversation(self) -> None:
        conversations = [
            {"role": "user", "content": "Fix src/app.py"},
            {"role": "assistant", "content": "I will inspect src/app.py."},
            {
                "role": "user",
                "content": "New Terminal Output: UNIQUE_EVIDENCE_MARKER",
            },
            {
                "role": "assistant",
                "content": "The output confirms the src/app.py failure.",
            },
        ]
        items = pending_items(conversations)

        prompt = build_batch_prompt(items)
        payload = json.loads(prompt.split("Input batch JSON:\n", 1)[1])

        self.assertEqual(len(items), 2)
        self.assertEqual(len(payload["sources"]), 1)
        self.assertEqual(len(payload["requests"]), 2)
        self.assertEqual(
            {request["synthetic_id"] for request in payload["requests"]},
            {item.slot.synthetic_id for item in items},
        )
        self.assertTrue(
            all(
                request["source_record_id"] == "source-record"
                for request in payload["requests"]
            )
        )
        self.assertEqual(prompt.count("UNIQUE_EVIDENCE_MARKER"), 1)

    def test_codex_policy_rejects_observed_synthetic_pilot_failures(self) -> None:
        source = [
            {"role": "user", "content": "Inspect the repository"},
            {"role": "assistant", "content": "Inspection complete"},
        ]
        candidate = [
            {
                "role": "user",
                "content": "Inspect the repository and summarize the findings",
            },
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "assistant", "content": "The inspection is complete."},
        ]

        codes = {
            issue.code
            for issue in validate_codex_trajectory_quality(source, candidate)
        }

        self.assertEqual(
            codes,
            {"added_deliverable", "adjacent_same_role", "vague_completion"},
        )

    def test_codex_policy_preserves_tool_grounding(self) -> None:
        source = [
            {"role": "user", "content": "Inspect src/app.py"},
            {"role": "assistant", "content": "I will read src/app.py."},
            {"role": "tool", "content": "PORT = 8080"},
            {
                "role": "assistant",
                "content": "`src/app.py` configures port 8080.",
            },
        ]
        candidate = [
            {"role": "user", "content": "Inspect src/app.py"},
            {
                "role": "assistant",
                "content": "`src/app.py` configures port 8080.",
            },
        ]

        codes = {
            issue.code
            for issue in validate_codex_trajectory_quality(source, candidate)
        }

        self.assertIn("missing_tool_evidence", codes)

    def test_codex_policy_accepts_concrete_source_supported_trajectory(self) -> None:
        source = [
            {"role": "user", "content": "Inspect src/app.py"},
            {"role": "assistant", "content": "I will read src/app.py."},
            {"role": "tool", "content": "PORT = 8080"},
            {
                "role": "assistant",
                "content": "src/app.py configures port 8080.",
            },
        ]
        candidate = [
            {"role": "user", "content": "Inspect src/app.py"},
            {"role": "assistant", "content": "I will inspect `src/app.py`."},
            {"role": "tool", "content": "PORT = 8080"},
            {
                "role": "assistant",
                "content": "`src/app.py` sets `PORT` to 8080.",
            },
        ]

        self.assertEqual(validate_codex_trajectory_quality(source, candidate), ())

    def test_process_batch_applies_codex_policy(self) -> None:
        source = [
            {"role": "user", "content": "Inspect the repository"},
            {"role": "assistant", "content": "Inspection complete"},
        ]
        item = pending_item(source)
        candidate = [
            {
                "role": "user",
                "content": "Inspect the repository and summarize the findings",
            },
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "assistant", "content": "The inspection is complete."},
        ]
        response = CodexResponse(
            payload={
                "results": [
                    {
                        "synthetic_id": item.slot.synthetic_id,
                        "conversations": candidate,
                    }
                ]
            },
            thread_id="thread-1",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        runtime = CodexRuntime(
            executable="codex", version="codex 1.0", environment={}
        )

        with mock.patch(
            "codex_refinement_worker.invoke_codex",
            new=mock.AsyncMock(return_value=response),
        ):
            result = asyncio.run(
                process_batch(
                    runtime,
                    model="gpt-5.6-sol",
                    items=[item],
                    attempt_numbers={item.slot.synthetic_id: 1},
                    agent_call_number=1,
                    timeout_seconds=60,
                    original_schema=SOURCE_SCHEMA,
                )
            )

        self.assertEqual(result.candidates, [])
        self.assertEqual(result.incomplete, [item])
        codes = set(result.attempts[0]["rejection_codes"])
        self.assertTrue(
            {"added_deliverable", "adjacent_same_role", "vague_completion"}
            <= codes
        )
        self.assertEqual(
            result.attempts[0]["codex_validation_policy"], "codex-quality-v1"
        )


if __name__ == "__main__":
    unittest.main()
