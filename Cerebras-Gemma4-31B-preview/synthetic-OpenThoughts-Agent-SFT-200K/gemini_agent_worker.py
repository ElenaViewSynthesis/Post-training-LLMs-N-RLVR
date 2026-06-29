"""
Stage 3: Async multi-turn agent-loop worker — Cerebras Inference + Gemma-4-31B.

Uses the Cerebras Inference API (OpenAI-compatible) to drive Gemma-4-31B as
the teacher model. Each variant gets a real multi-turn conversation: the model
issues tool calls (run_bash, edit_file), receives observations, and continues
until it declares completion or MAX_TURNS is reached.

Because Cerebras has no server-side conversation state, the full message
history is rebuilt and sent on every turn. At Cerebras throughput speeds this
is fine — the bottleneck is sandbox I/O, not token generation.

Diversity comes from temperature + instruction framing (Stage 2), NOT
thinking_level (that was Gemini-specific and does not apply here).

CONCURRENCY is the primary throughput knob. Cerebras can sustain very high
token/sec rates — your real ceiling is sandbox container capacity (BACKEND B)
or RPM limits. Start at 32, watch for 429s, ramp up.

Requires: pip install cerebras-cloud-sdk
"""
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from cerebras.cloud.sdk import AsyncCerebras
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CEREBRAS_MODEL_ID", "cerebras/Gemma4-31B-preview")
BACKEND = "custom_sandbox"      # or "stub" for local testing without containers
CONCURRENCY = 32                # ramp up gradually; watch for 429s
MAX_TURNS = 20
OUT_DIR = Path("/home/claude/pipeline/data/raw_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = AsyncCerebras(api_key=os.environ["CEREBRAS_API_KEY"])

SYSTEM_PROMPT = (
    "You are a terminal/software agent solving a task in a sandboxed Linux "
    "environment. Use the provided tools to inspect and modify the system. "
    "Issue one concrete tool call at a time and reason briefly before each. "
    "Stop when the task is complete."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command in the task sandbox and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Overwrite a file at the given path with new contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
    },
]


async def execute_tool(sandbox, name: str, args: dict) -> str:
    """Wire to your real per-task sandbox. Returns the tool observation string."""
    if name == "run_bash":
        return await sandbox.run_bash(args["command"])          # implement
    if name == "edit_file":
        return await sandbox.edit_file(args["path"], args["contents"])  # implement
    return f"ERROR: unknown tool {name}"


def build_user_message(task: dict, framing: str | None) -> str:
    instruction = task.get("instruction") or task.get("prompt") or task.get("task_description", "")
    env = task.get("environment") or task.get("dockerfile") or task.get("setup_script", "")
    text = instruction if not framing else f"{framing}\n\n{instruction}"
    return f"## Task\n{text}\n\n## Environment\n{env or 'Generic Ubuntu container.'}"


async def run_agent_loop(variant: dict, task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        sandbox = None  # = await spin_up_sandbox(task)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(task, variant["instruction_framing"])},
        ]
        trajectory = []
        try:
            for _ in range(MAX_TURNS):
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    temperature=variant["temperature"],
                    max_completion_tokens=1024,
                )
                msg = response.choices[0].message
                messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

                tool_calls = msg.tool_calls or []
                for tc in tool_calls:
                    trajectory.append({
                        "type": "function_call",
                        "content": json.dumps({
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                            "id": tc.id,
                        }),
                    })

                if not tool_calls:
                    # Model gave a final answer with no further tool call
                    trajectory.append({"type": "text", "content": msg.content or ""})
                    break

                # Execute tools and feed observations back
                for tc in tool_calls:
                    args = json.loads(tc.function.arguments)
                    obs = await execute_tool(sandbox, tc.function.name, args)
                    trajectory.append({"type": "function_result", "content": obs})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs,
                    })

            return {
                "variant_id": variant["variant_id"],
                "task_id": variant["task_id"],
                "status": "ok",
                "trajectory": trajectory,
                "final_text": next(
                    (t["content"] for t in reversed(trajectory) if t["type"] == "text"), None
                ),
            }
        except Exception as e:
            return {
                "variant_id": variant["variant_id"],
                "task_id": variant["task_id"],
                "status": f"error:{type(e).__name__}",
                "trajectory": trajectory,
                "final_text": None,
            }
        finally:
            pass  # await sandbox.teardown()


async def main():
    plan = pd.read_parquet("/home/claude/pipeline/data/variant_plan.parquet")
    tasks = pd.read_parquet("/home/claude/pipeline/data/tasks/")
    id_col = "task_id" if "task_id" in tasks.columns else "id"
    task_lookup = {row[id_col]: row.to_dict() for _, row in tasks.iterrows()}

    sem = asyncio.Semaphore(CONCURRENCY)

    # Resume-safe: skip variants already written
    out_path = OUT_DIR / "trajectories.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["variant_id"])
    print(f"Resuming: {len(done)} already done, {len(plan) - len(done)} remaining")

    todo = [v for _, v in plan.iterrows() if v["variant_id"] not in done]
    coros = [run_agent_loop(v, task_lookup[v["task_id"]], sem) for v in todo]

    with open(out_path, "a") as f:
        for fut in asyncio.as_completed(coros):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    asyncio.run(main())
