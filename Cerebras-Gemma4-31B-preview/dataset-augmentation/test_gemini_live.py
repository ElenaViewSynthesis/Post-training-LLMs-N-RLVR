"""Opt-in Gemini checks; skipped until credentials are deliberately enabled."""

from __future__ import annotations

import asyncio
import os
import unittest

import pyarrow as pa

from refinement_pipeline import refinement_slots_for_source, source_identity_for_row
from stream_refinement_worker import (
    ProviderRequestBudget,
    Settings,
    create_gemini_client,
    provider_preflight,
    refine_slot,
)


LIVE_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-3.6-flash")
LIVE_ENABLED = os.getenv("RUN_GEMINI_INTEGRATION") == "1"
PAID_ENABLED = os.getenv("RUN_GEMINI_PAID_TESTS") == "1"

CONVERSATIONS_TYPE = pa.list_(
    pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())])
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


def live_settings() -> Settings:
    return Settings(
        model=LIVE_MODEL,
        concurrency=1,
        request_batch_size=1,
        max_output_tokens=4096,
        max_attempts_per_run=1,
        timeout_ms=180_000,
        sync_every_shards=1,
        hf_bucket="unused",
        sync_enabled=False,
    )


class GeminiLiveTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(LIVE_ENABLED, "set RUN_GEMINI_INTEGRATION=1")
    async def test_provider_preflight(self) -> None:
        api_key = os.environ["GEMINI_API_KEY"]
        client = create_gemini_client(api_key, live_settings()).aio
        try:
            await provider_preflight(client, live_settings(), api_key)
        finally:
            await client.aclose()

    @unittest.skipUnless(
        LIVE_ENABLED and PAID_ENABLED,
        "set RUN_GEMINI_INTEGRATION=1 and RUN_GEMINI_PAID_TESTS=1",
    )
    async def test_exactly_one_real_refinement_request(self) -> None:
        api_key = os.environ["GEMINI_API_KEY"]
        source = {
            "conversations": [
                {
                    "role": "user",
                    "content": "Inspect src/parser.py and run pytest tests/test_parser.py.",
                },
                {
                    "role": "assistant",
                    "content": "I inspected src/parser.py and the parser tests passed.",
                },
            ],
            "model": "fixture",
            "model_provider": "fixture",
            "run_id": "live-source",
            "task": "inspect-parser",
            "trial_name": "live-source",
        }
        identity = source_identity_for_row("live-fixture.parquet", 0, source)
        slot = refinement_slots_for_source(identity, frozenset())[0]
        budget = ProviderRequestBudget(1)
        client = create_gemini_client(api_key, live_settings()).aio
        try:
            await provider_preflight(client, live_settings(), api_key)
            result = await refine_slot(
                client,
                source,
                slot,
                SOURCE_SCHEMA,
                live_settings(),
                api_key,
                asyncio.Semaphore(1),
                starting_attempt=0,
                request_budget=budget,
            )
        finally:
            await client.aclose()

        self.assertEqual(budget.used, 1)
        self.assertEqual(len(result.attempts), 1)
        self.assertNotEqual(result.attempts[0]["status"], "provider_error")


if __name__ == "__main__":
    unittest.main()
