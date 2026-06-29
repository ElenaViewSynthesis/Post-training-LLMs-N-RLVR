"""
Stage 3 (Gemini rewrite): Async multi-turn agent-loop worker.

This REPLACES the old 03_submit_batches.py + 04_poll_and_fetch.py. With
unlimited Gemini requests there's no cost reason to use a batch API or to fake
trajectories single-shot -- you run the real agent loop and capture actually-
executed tool calls/observations, matching how the original dataset was built
(Terminus-2 harness, real environments, real verifiers).

Two execution backends are sketched here; pick one:

  BACKEND A -- "antigravity" (lowest effort, Google-hosted sandbox):
    Use the managed Antigravity agent (antigravity-preview-05-2026). It runs
    Bash/Python/Node in a Google-hosted Linux sandbox, so you build zero
    container orchestration. Requires background=True + polling. Caveat: you
    don't control the exact environment, so tasks that depend on a specific
    Dockerfile/setup may not reproduce faithfully. Great for the nl2bash-style
    and generic-Ubuntu tasks; weaker for tasks with bespoke environments.

  BACKEND B -- "custom_sandbox" (most faithful, your infra):
    You spin the task's real Docker environment, expose run_bash/edit_file/
    run_tests as custom function-calling tools, and let Gemini drive it via
    the Interactions API (previous_interaction_id carries state server-side so
    you don't resend history each turn). Run the pytest verifier at the end.
    This reproduces the original generation method exactly. You own the
    sandbox lifecycle. Since you already run vLLM/SGLang infra, container
    orchestration here is familiar territory.

Diversity comes from thinking_level + framing (Stage 2), NOT sampling params.

"Unlimited requests" still has a throughput ceiling (RPM/concurrency, and for
BACKEND B your sandbox capacity). CONCURRENCY below is the real tuning knob --
start conservative, watch for 429s, ramp gradually (Gemini, like most APIs,
penalizes sharp traffic spikes).

Requires: pip install -U "google-genai>=2.3.0"
"""
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from google import genai

MODEL = "gemini-3.5-flash"
ANTIGRAVITY_AGENT = "antigravity-preview-05-2026"
BACKEND = "custom_sandbox"          # or "antigravity"
CONCURRENCY = 64                     # tune to your RPM/sandbox ceiling; ramp up gradually
MAX_TURNS = 20                       # matches the original Terminus-2 turn cap
OUT_DIR = Path("/home/claude/pipeline/data/raw_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_INSTRUCTION = (
    "You are a terminal/software agent solving a task in a sandboxed Linux "
    "environment. Use the provided tools to inspect and modify the system. "
    "Issue one concrete tool call at a time and reason briefly before each. "
    "Stop when the task is complete."
)

# --- Custom tools exposed to the model (BACKEND B). Wire these to your real
# sandbox; the bodies below are stubs showing the contract. ---
RUN_BASH = {
    "type": "function",
    "name": "run_bash",
    "description": "Run a bash command in the task's sandbox and return stdout/stderr/exit code.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to run."}},
        "required": ["command"],
    },
}
EDIT_FILE = {
    "type": "function",
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
}
TOOLS = [RUN_BASH, EDIT_FILE]


async def execute_tool(sandbox, name: str, args: dict) -> str:
    """Wire this to your real per-task sandbox (Docker exec, etc.).
    Returns a string the model receives as the tool observation."""
    if name == "run_bash":
        return await sandbox.run_bash(args["command"])          # implement
    if name == "edit_file":
        return await sandbox.edit_file(args["path"], args["contents"])  # implement
    return f"ERROR: unknown tool {name}"


def build_input(task: dict, framing: str | None) -> str:
    instruction = task.get("instruction") or task.get("prompt") or task.get("task_description", "")
    env = task.get("environment") or task.get("dockerfile") or task.get("setup_script", "")
    text = instruction if not framing else f"{framing}\n\n{instruction}"
    return f"## Task\n{text}\n\n## Environment\n{env or 'Generic Ubuntu container.'}"


async def run_custom_sandbox(variant: dict, task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        sandbox = None  # = await spin_up_sandbox(task)   <-- your orchestration
        trajectory = []
        try:
            interaction = await client.aio.interactions.create(
                model=MODEL,
                input=build_input(task, variant["instruction_framing"]),
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={"thinking_level": variant["thinking_level"]},
            )
            for _ in range(MAX_TURNS):
                # capture every step (thoughts, function calls, text) as trajectory turns
                calls = [s for s in interaction.steps if s.type == "function_call"]
                for s in interaction.steps:
                    trajectory.append({"type": s.type, "content": _serialize_step(s)})
                if not calls:
                    break  # model produced a final answer with no further tool call

                results = []
                for call in calls:
                    obs = await execute_tool(sandbox, call.name, dict(call.arguments))
                    results.append({
                        "type": "function_result",
                        "call_id": call.id,
                        "name": call.name,
                        "result": obs,
                    })
                interaction = await client.aio.interactions.create(
                    model=MODEL,
                    previous_interaction_id=interaction.id,  # server-side history
                    input=results,
                    tools=TOOLS,
                    generation_config={"thinking_level": variant["thinking_level"]},
                )
            return {"variant_id": variant["variant_id"], "task_id": variant["task_id"],
                    "status": "ok", "trajectory": trajectory,
                    "final_text": getattr(interaction, "output_text", None)}
        except Exception as e:
            return {"variant_id": variant["variant_id"], "task_id": variant["task_id"],
                    "status": f"error:{type(e).__name__}", "trajectory": trajectory, "final_text": None}
        finally:
            pass  # await sandbox.teardown()


async def run_antigravity(variant: dict, task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            interaction = await client.aio.interactions.create(
                agent=ANTIGRAVITY_AGENT,
                input=build_input(task, variant["instruction_framing"]),
                environment="remote",
                background=True,                 # agents require background execution
            )
            # poll until done
            while interaction.status not in ("completed", "failed", "cancelled"):
                await asyncio.sleep(5)
                interaction = await client.aio.interactions.retrieve(interaction.id)
            trajectory = [{"type": s.type, "content": _serialize_step(s)} for s in interaction.steps]
            return {"variant_id": variant["variant_id"], "task_id": variant["task_id"],
                    "status": "ok" if interaction.status == "completed" else interaction.status,
                    "trajectory": trajectory,
                    "final_text": getattr(interaction, "output_text", None)}
        except Exception as e:
            return {"variant_id": variant["variant_id"], "task_id": variant["task_id"],
                    "status": f"error:{type(e).__name__}", "trajectory": [], "final_text": None}


def _serialize_step(step) -> str:
    if step.type == "function_call":
        return json.dumps({"name": step.name, "arguments": dict(step.arguments), "id": step.id})
    return getattr(step, "text", "") or json.dumps(getattr(step, "__dict__", {}), default=str)


async def main():
    plan = pd.read_parquet("/home/claude/pipeline/data/variant_plan.parquet")
    tasks = pd.read_parquet("/home/claude/pipeline/data/tasks/")
    id_col = "task_id" if "task_id" in tasks.columns else "id"
    task_lookup = {row[id_col]: row.to_dict() for _, row in tasks.iterrows()}

    runner = run_antigravity if BACKEND == "antigravity" else run_custom_sandbox
    sem = asyncio.Semaphore(CONCURRENCY)

    # Resume-safe: skip variants already written to the shard file
    out_path = OUT_DIR / "trajectories.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["variant_id"])
    print(f"Resuming: {len(done)} already done")

    todo = [v for _, v in plan.iterrows() if v["variant_id"] not in done]
    coros = [runner(v, task_lookup[v["task_id"]], sem) for v in todo]

    with open(out_path, "a") as f:
        for fut in asyncio.as_completed(coros):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    asyncio.run(main())
