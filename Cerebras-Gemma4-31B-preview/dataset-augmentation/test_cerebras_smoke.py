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
    3. single_task     — pull one real row from the live dataset (extract_tasks.py's
                          SRC/TRAJECTORY_COL), build the real synthesis prompt
                          (gemma4_31b_agent.SYSTEM_PROMPT/build_synthesis_prompt),
                          call Cerebras, and parse the response with the same
                          logic as synthesize_trajectory(). (implemented)
    4. mini_pipeline   — plan_variants.plan_variants() -> gemma4_31b_agent.
                          synthesize_trajectory() -> validate_n_dedup.
                          structural_check()/conversations_fingerprint(),
                          end-to-end on one real task, n=1 variant, using the
                          actual production functions (not reimplemented
                          logic). (implemented)
"""
import argparse
import asyncio
import json
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


SINGLE_TASK_MAX_TOKENS = 4096  # matches gemma4_31b_agent.py's MAX_COMPLETION_TOKENS


def test_3_single_task() -> bool:
    import dask.dataframe as dd
    from cerebras.cloud.sdk import Cerebras

    from extract_tasks import SRC, TRAJECTORY_COL, extract_instruction
    from gemma4_31b_agent import SYSTEM_PROMPT, build_synthesis_prompt

    print("[3/single_task] loading one real row from the live dataset...")
    df = dd.read_parquet(SRC)
    row = df.head(1).iloc[0]
    task = row.drop(labels=[TRAJECTORY_COL]).to_dict()
    task["instruction"] = extract_instruction(row[TRAJECTORY_COL])
    print(f"[3/single_task] task fields: {list(task.keys())}")

    # build_synthesis_prompt reads task["instruction"|"prompt"|"task_description"]; the real
    # instruction is derived by extract_tasks.extract_instruction() from conversations[0], not
    # a flat column. Surface it loudly if that came back empty (e.g. an unrecognized row format).
    if not task.get("instruction"):
        print("[3/single_task] WARNING: extract_instruction() returned empty "
              "-> build_synthesis_prompt will fall back to an empty task description")
    else:
        print(f"[3/single_task] instruction preview: {task['instruction'][:150]!r}")

    variant = {
        "task_id": task.get("task", "smoke-test-task"),
        "variant_id": "smoke-test-v0",
        "temperature": 0.7,
        "instruction_framing": None,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_synthesis_prompt(task, variant["instruction_framing"])},
    ]
    print(f"[3/single_task] prompt preview: {messages[1]['content'][:200]!r}")

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=variant["temperature"],
        max_completion_tokens=SINGLE_TASK_MAX_TOKENS,
    )
    raw = response.choices[0].message.content or ""

    # Same parse logic as synthesize_trajectory()'s Cerebras branch in gemma4_31b_agent.py.
    try:
        parsed = json.loads(raw)
        conversations = parsed.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("missing or empty 'conversations' list")
        status = "ok"
    except (json.JSONDecodeError, ValueError) as e:
        conversations = None
        status = f"parse_error:{e}"

    n_turns = len(conversations) if conversations else 0
    print(f"[3/single_task] status={status} turns={n_turns} raw_len={len(raw)}")
    if status != "ok":
        print(f"[3/single_task] raw response (truncated): {raw[:300]!r}")

    return status == "ok"


async def _mini_pipeline_probe() -> dict:
    import dask.dataframe as dd
    import pandas as pd

    from extract_tasks import SRC, TRAJECTORY_COL, extract_instruction
    from plan_variants import plan_variants
    from gemma4_31b_agent import synthesize_trajectory
    from validate_n_dedup import structural_check, conversations_fingerprint

    print("[4/mini_pipeline] loading one real row from the live dataset...")
    df = dd.read_parquet(SRC)
    row = df.head(1).iloc[0]
    task = row.drop(labels=[TRAJECTORY_COL]).to_dict()
    task["instruction"] = extract_instruction(row[TRAJECTORY_COL])

    # Stage 2: plan_variants.py — plan exactly 1 variant for this 1 task.
    tasks_pdf = pd.DataFrame([task])
    plan = plan_variants(tasks_pdf, n_requests=1, seed=42)
    variant = plan.iloc[0].to_dict()
    print(f"[4/mini_pipeline] planned variant: {variant}")

    # Stage 3: gemma4_31b_agent.py — the real synthesize_trajectory(), not a stand-in.
    sem = asyncio.Semaphore(1)
    result = await synthesize_trajectory(variant, task, sem)
    n_turns = len(result["conversations"]) if result.get("conversations") else 0
    print(f"[4/mini_pipeline] synthesis status={result['status']} turns={n_turns}")

    # Stage 5: validate_n_dedup.py — same structural check + fingerprint as the real dedup job.
    passed_structural = structural_check(result.get("conversations"))
    fingerprint = conversations_fingerprint(result["conversations"]) if passed_structural else None
    print(f"[4/mini_pipeline] structural_check={passed_structural} fingerprint={fingerprint}")

    return {
        "synthesis_status": result["status"],
        "structural_ok": passed_structural,
    }


def test_4_mini_pipeline() -> bool:
    outcome = asyncio.run(_mini_pipeline_probe())
    return outcome["synthesis_status"] == "ok" and outcome["structural_ok"]


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
