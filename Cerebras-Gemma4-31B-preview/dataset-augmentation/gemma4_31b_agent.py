"""
Stage 3: Structured trajectory synthesis — Gemma-4-31B via SageMaker, Together.ai, or Cerebras.

ARCHITECTURE NOTE — trajectory synthesis, not agent execution:

The original OpenThoughts-Agent-SFT-100K dataset was built with a live agent
runtime (Terminus-2 harness): a model actually executed bash commands in a
real sandboxed environment and the tool call/observation turns were captured.

This pipeline does NOT replicate that. Instead it uses structured trajectory
synthesis: Gemma-4-31B generates a complete synthetic conversation in a single
inference call, conditioned on the original task and augmentation strategy.
The model produces all turns — reasoning, tool calls, observations, and the
final answer — without any live environment.

Why this is the right approach for SFT augmentation at this scale:
- Agent execution requires container orchestration and is 10–100× slower.
- Trajectory synthesis runs at GPU throughput speeds, making 150K+ new
  samples practical within a single pipeline run.
- Quality is enforced post-generation by the Stage 5 validation pipeline.
- For SFT, the student learns from the conversation pattern — well-structured
  synthetic trajectories are sufficient signal.

Objective: expand OpenThoughts-Agent-SFT-100K → 250K total (100K original +
150K synthetic) by augmenting the `conversations` column.

Inference backend (set in .env, checked in priority order):
  SageMaker   — set SAGEMAKER_ENDPOINT_NAME (deploy via deploy_sagemaker.py)
  Together.ai — set TOGETHER_API_KEY (~$15–20 for 150K samples, recommended)
  Cerebras    — set CEREBRAS_API_KEY (fallback)

Requires: pip install cerebras-cloud-sdk 'sagemaker<3.0.0' boto3 openai
"""
import asyncio
import json
import os
from pathlib import Path
import sys

import boto3
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import sync_bucket
from tqdm.asyncio import tqdm as atqdm

if __name__ == "__main__":
    print(
        "This 225K oversampling entry point is retired. "
        "Use stream_refinement_worker.py for deterministic source-row refinement.",
        file=sys.stderr,
    )
    raise SystemExit(2)

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Inference backend ─────────────────────────────────────────────────────────
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")
TOGETHER_API_KEY        = os.getenv("TOGETHER_API_KEY")
AWS_REGION              = os.getenv("AWS_REGION", "us-east-1")

USE_SAGEMAKER = bool(SAGEMAKER_ENDPOINT_NAME)
USE_TOGETHER  = not USE_SAGEMAKER and bool(TOGETHER_API_KEY)

if USE_SAGEMAKER:
    sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    print(f"Backend: SageMaker endpoint '{SAGEMAKER_ENDPOINT_NAME}'")
elif USE_TOGETHER:
    from openai import AsyncOpenAI
    TOGETHER_MODEL = os.getenv("TOGETHER_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    client = AsyncOpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )
    print(f"Backend: Together.ai model '{TOGETHER_MODEL}'")
else:
    from cerebras.cloud.sdk import AsyncCerebras
    MODEL  = os.getenv("CEREBRAS_MODEL_ID", "gemma-4-31b")   # Gemma-4-31B (paid tier)
    client = AsyncCerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    print(f"Backend: Cerebras model '{MODEL}'")

# ── Config ────────────────────────────────────────────────────────────────────
CONCURRENCY          = 16 if USE_TOGETHER else 4   # Together.ai handles higher concurrency
BATCH_SIZE           = 60     # process plan in chunks of 60 variants
MAX_RETRIES          = 5      # retries on rate-limit / transient errors
RETRY_BACKOFF_BASE   = 2      # seconds; doubles each retry (2, 4, 8, 16, 32)

# gemma-4-31b's free tier caps at 1,000,000 tokens/day (see
# ../cerebras-gemma4-31b-quota.md), so completion size should stay small --
# but 1,024 was tested empirically (70-request real batch, 2026-07-04) and
# produced a 2.9% accept rate (68/70 failed, mostly truncated mid-JSON:
# "unterminated string" / empty response). Real successful trajectories in
# that same test and earlier pilots consumed 1,602-2,423 completion tokens
# (18-31 turns), all well above 1,024. 2,500 leaves headroom for that
# observed range without swinging back to the 4,096+ default that made the
# daily token budget the bottleneck again.
MAX_COMPLETION_TOKENS = 2500

OUT_DIR = Path("~/pipeline/data/raw_results").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_BUCKET          = "hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k/raw_results"
SYNC_EVERY_N_BATCHES = 10    # sync to HF bucket every N batches (~600 records)


def sync_to_hf():
    try:
        sync_bucket(str(OUT_DIR), HF_BUCKET)
    except Exception as e:
        print(f"\n[hf sync warning] {e} — continuing without sync")


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert software/terminal agent and dataset curator.
Your task is to synthesize a realistic, high-quality agent trajectory for the given task.

Output a JSON object with a single key "conversations" whose value is a list of turns.
Each turn is an object with:
  - "role": one of "system", "user", "assistant", "tool"
  - "content": the text of that turn

The trajectory must:
1. Begin with a brief reasoning step before each tool call ("I'll start by...")
2. Include realistic bash commands (run_bash) or file edits (edit_file) as tool calls
3. Follow each tool call with a plausible observation result
4. End with a clear summary confirming the task is complete
5. Match the style of the OpenThoughts-Agent-SFT dataset: concise, technical, step-by-step

Respond with ONLY the JSON object. No markdown fences, no commentary."""


def build_synthesis_prompt(task: dict, framing: str | None) -> str:
    instruction = task.get("instruction") or task.get("prompt") or task.get("task_description", "")
    env         = task.get("environment") or task.get("dockerfile") or task.get("setup_script", "")
    text        = instruction if not framing else f"{framing}\n\n{instruction}"
    return (
        f"## Task\n{text}\n\n"
        f"## Environment\n{env or 'Generic Ubuntu 22.04 container.'}\n\n"
        "Synthesize a complete, realistic agent trajectory that solves this task."
    )


# ── SageMaker call (sync, run in executor) ────────────────────────────────────
def _sm_invoke(messages: list, temperature: float) -> str:
    body = {
        "messages":   messages,
        "temperature": temperature,
        "max_tokens":  MAX_COMPLETION_TOKENS,
    }
    resp = sm_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(body),
    )
    result = json.loads(resp["Body"].read())
    return result["choices"][0]["message"]["content"]


async def _sm_generate(messages: list, temperature: float) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sm_invoke, messages, temperature)


# ── Core synthesis ────────────────────────────────────────────────────────────
async def synthesize_trajectory(variant: dict, task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_synthesis_prompt(task, variant["instruction_framing"])},
        ]

        for attempt in range(MAX_RETRIES):
            try:
                if USE_SAGEMAKER:
                    raw = await _sm_generate(messages, variant["temperature"])
                elif USE_TOGETHER:
                    response = await client.chat.completions.create(
                        model=TOGETHER_MODEL,
                        messages=messages,
                        temperature=variant["temperature"],
                        max_tokens=MAX_COMPLETION_TOKENS,
                    )
                    raw = response.choices[0].message.content or ""
                else:
                    response = await client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=variant["temperature"],
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                    )
                    raw = response.choices[0].message.content or ""

                try:
                    parsed        = json.loads(raw)
                    conversations = parsed.get("conversations")
                    if not isinstance(conversations, list):
                        raise ValueError("missing or non-list 'conversations' key")
                    status = "ok"
                except (json.JSONDecodeError, ValueError) as e:
                    conversations = None
                    status        = f"parse_error:{e}"

                return {
                    "variant_id":   variant["variant_id"],
                    "task_id":      variant["task_id"],
                    "temperature":  variant["temperature"],
                    "status":       status,
                    "conversations": conversations,
                    "raw":          raw if status != "ok" else None,
                }

            except Exception as e:
                err_str      = str(e)
                is_transient = (
                    "429" in err_str
                    or "500" in err_str
                    or "503" in err_str
                    or "rate_limit" in err_str.lower()
                    or "too many"   in err_str.lower()
                    or "throttling" in err_str.lower()
                )
                if is_transient and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                    continue

                return {
                    "variant_id":   variant["variant_id"],
                    "task_id":      variant["task_id"],
                    "temperature":  variant["temperature"],
                    "status":       f"error:{type(e).__name__}:{e}",
                    "conversations": None,
                    "raw":          None,
                }

        return {
            "variant_id":   variant["variant_id"],
            "task_id":      variant["task_id"],
            "temperature":  variant["temperature"],
            "status":       "error:MaxRetriesExceeded",
            "conversations": None,
            "raw":          None,
        }


# ── Batch runner ──────────────────────────────────────────────────────────────
async def run_batch(batch: list, task_lookup: dict, sem: asyncio.Semaphore, f) -> tuple[int, int]:
    coros = [synthesize_trajectory(v, task_lookup.get(v["task_id"], {}), sem) for v in batch]
    ok = err = 0
    async for fut in atqdm(asyncio.as_completed(coros), total=len(coros), unit="req", leave=False):
        rec = await fut
        if rec["status"] == "ok":
            ok += 1
        else:
            err += 1
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return ok, err


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    plan  = pd.read_parquet(Path("~/pipeline/data/variant_plan.parquet").expanduser())
    tasks = pd.read_parquet(Path("~/pipeline/data/tasks/").expanduser())
    id_col      = next((c for c in ("task_id", "id", "task", "run_id") if c in tasks.columns), tasks.columns[0])
    task_lookup = {row[id_col]: row.to_dict() for _, row in tasks.iterrows()}

    sem = asyncio.Semaphore(CONCURRENCY)

    out_path = OUT_DIR / "trajectories.jsonl"
    done     = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok":
                        done.add(rec["variant_id"])
                except json.JSONDecodeError:
                    pass

    todo          = [v for _, v in plan.iterrows() if v["variant_id"] not in done]
    total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Resuming: {len(done)} already done, {len(todo)} remaining "
          f"({total_batches} batches of {BATCH_SIZE})")

    total_ok = total_err = 0
    with open(out_path, "a") as f:
        for batch_num, start in enumerate(range(0, len(todo), BATCH_SIZE), 1):
            batch = todo[start:start + BATCH_SIZE]
            print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} variants)...")
            ok, err       = await run_batch(batch, task_lookup, sem, f)
            total_ok     += ok
            total_err    += err
            processed     = total_ok + total_err
            print(f"  ok={ok} err={err} | total: ok={total_ok} err={total_err} "
                  f"err_rate={total_err / max(processed, 1):.1%}")

            if batch_num % SYNC_EVERY_N_BATCHES == 0:
                print(f"[batch {batch_num}] syncing to HF bucket...")
                sync_to_hf()

    print(f"\nDone. ok={total_ok} err={total_err}  Results -> {out_path}")
    print("Final sync to HF bucket...")
    sync_to_hf()
