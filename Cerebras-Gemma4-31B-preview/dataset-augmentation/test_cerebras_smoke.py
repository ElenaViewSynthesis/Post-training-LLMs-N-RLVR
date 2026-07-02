"""
Smoke tests for the Cerebras Inference backend used in Stage 3 (gemma4_31b_agent.py).

Run all implemented tests:
    uv run --with cerebras-cloud-sdk --with python-dotenv test_cerebras_smoke.py

Run a single test:
    uv run --with cerebras-cloud-sdk --with python-dotenv test_cerebras_smoke.py --test 1

Tests:
    1. bare_completion — one chat completion, max_completion_tokens=20. Confirms
       auth, model availability, and response shape. (implemented)
    2. sdk_client      — AsyncCerebras client, concurrent calls under a semaphore,
                          matching the call shape used in gemma4_31b_agent.py's
                          synthesize_trajectory(). (implemented)
    3. single_task     — run one real task from extract_tasks.py through the full
                          synthesis prompt + parsing logic. (TODO)
    4. mini_pipeline   — plan_variants.py -> gemma4_31b_agent.py -> validate_n_dedup.py
                          end-to-end on n=1 task. (TODO)
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = os.getenv("CEREBRAS_MODEL_ID", "gemma-4-31b")
SDK_CLIENT_CONCURRENCY = 2
SDK_CLIENT_CALLS = 3


def test_1_bare_completion() -> bool:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
        max_completion_tokens=20,
    )
    content = response.choices[0].message.content or ""
    print(f"[1/bare_completion] model={MODEL} response={content!r}")
    return bool(content.strip())


async def _sdk_client_probe() -> list[str]:
    from cerebras.cloud.sdk import AsyncCerebras

    client = AsyncCerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    sem = asyncio.Semaphore(SDK_CLIENT_CONCURRENCY)

    async def one_call(i: int) -> str:
        async with sem:
            # Same call shape as synthesize_trajectory()'s Cerebras branch in
            # gemma4_31b_agent.py: system+user messages, temperature, max_completion_tokens.
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a terse test responder."},
                    {"role": "user", "content": f"Reply with exactly one word: PONG{i}"},
                ],
                temperature=0.7,
                max_completion_tokens=20,
            )
            return response.choices[0].message.content or ""

    return await asyncio.gather(*(one_call(i) for i in range(SDK_CLIENT_CALLS)))


def test_2_sdk_client() -> bool:
    results = asyncio.run(_sdk_client_probe())
    for i, content in enumerate(results):
        print(f"[2/sdk_client] call={i} response={content!r}")
    return len(results) == SDK_CLIENT_CALLS and all(r.strip() for r in results)


def test_3_single_task() -> bool:
    raise NotImplementedError("run one real task from extract_tasks.py through synthesis + parsing")


def test_4_mini_pipeline() -> bool:
    raise NotImplementedError("plan_variants.py -> gemma4_31b_agent.py -> validate_n_dedup.py on n=1")


TESTS = {
    1: ("bare_completion", test_1_bare_completion),
    2: ("sdk_client", test_2_sdk_client),
    3: ("single_task", test_3_single_task),
    4: ("mini_pipeline", test_4_mini_pipeline),
}


def main():
    parser = argparse.ArgumentParser(description="Cerebras backend smoke tests")
    parser.add_argument(
        "--test", type=int, choices=sorted(TESTS), default=None,
        help="Run only this test number (default: run all)",
    )
    args = parser.parse_args()
    to_run = [args.test] if args.test else sorted(TESTS)

    results = {}
    for n in to_run:
        name, fn = TESTS[n]
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except NotImplementedError as e:
            results[name] = f"SKIP ({e})"
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"

    print("\n--- Results ---")
    for name, status in results.items():
        print(f"{name}: {status}")

    if any(status.startswith(("FAIL", "ERROR")) for status in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
