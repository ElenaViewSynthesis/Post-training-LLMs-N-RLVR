"""
Pilot run — Stages 2, 3, 5 on a small task sample, per README's "Before
running at full scale: pilot first" guidance. Calibrates OVERSAMPLE_FACTOR,
measures degenerate-output rate, and prints a sample conversation for manual
inspection, using the real production functions (plan_variants,
synthesize_trajectory, structural_check, conversations_fingerprint) rather
than reimplemented logic.

Deliberately small default (24 tasks -> ~36 requests): the Cerebras free
tier for gemma-4-31b caps at 5 requests/minute, 150/hour, 2,400/day. This
size stays comfortably under all three even accounting for a few 429
retries, so the accept-rate signal reflects real validation outcomes rather
than quota throttling. NOTE: at 225,000 requests, the planned full-scale run
would take ~94 days on this quota (2,400/day) -- resolve that separately
before scaling past the pilot.

Writes to ~/pipeline/data/pilot/ — a separate path from the production
~/pipeline/data/variant_plan.parquet and raw_results/, and never calls
gemma4_31b_agent.sync_to_hf(), so a pilot run cannot touch production data
or the HF bucket.

Usage:
    uv run python run_pilot.py --n-tasks 24
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from extract_tasks import TASKS_OUT
from plan_variants import plan_variants, OVERSAMPLE_FACTOR
from gemma4_31b_agent import synthesize_trajectory, CONCURRENCY
from validate_n_dedup import structural_check, conversations_fingerprint, DEGENERATE_RE

PILOT_DIR = Path("~/pipeline/data/pilot").expanduser()


async def run_pilot(n_tasks: int, seed: int) -> list[dict]:
    tasks_pdf = pd.read_parquet(TASKS_OUT / "tasks.parquet")
    sample = tasks_pdf.sample(n=min(n_tasks, len(tasks_pdf)), random_state=seed).reset_index(drop=True)
    print(f"[pilot] sampled {len(sample)} tasks from {len(tasks_pdf)} total")

    n_requests = max(len(sample), int(len(sample) * OVERSAMPLE_FACTOR))
    plan = plan_variants(sample, n_requests, seed=seed)
    print(f"[pilot] planned {len(plan)} variant requests ({OVERSAMPLE_FACTOR}x oversample)")

    id_col = next((c for c in ("task_id", "id", "task", "run_id") if c in sample.columns), sample.columns[0])
    task_lookup = {row[id_col]: row.to_dict() for _, row in sample.iterrows()}
    sem = asyncio.Semaphore(CONCURRENCY)

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PILOT_DIR / "trajectories.jsonl"

    coros = [synthesize_trajectory(v, task_lookup[v["task_id"]], sem) for _, v in plan.iterrows()]
    results = []
    with open(out_path, "w") as f:
        for i, fut in enumerate(asyncio.as_completed(coros), 1):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            results.append(rec)
            print(f"[pilot] {i}/{len(coros)} status={rec['status']}")
    print(f"[pilot] raw results -> {out_path}")
    return results


def analyze(results: list[dict]) -> None:
    n = len(results)
    status_counts = Counter(r["status"] for r in results)
    ok = [r for r in results if r["status"] == "ok"]

    structural_pass = [r for r in ok if structural_check(r["conversations"])]

    def is_degenerate(r: dict) -> bool:
        text = " ".join(t.get("content", "") for t in r["conversations"] if isinstance(t, dict))
        return bool(DEGENERATE_RE.search(text))

    degenerate = [r for r in ok if r["conversations"] and is_degenerate(r)]

    # Per-task-id fingerprint dedup, matching validate_n_dedup.main()'s
    # drop_duplicates(subset=["task_id", "fingerprint"]).
    pairs = [(r["task_id"], conversations_fingerprint(r["conversations"])) for r in structural_pass]
    dup_count = len(pairs) - len(set(pairs))

    turn_counts = [len(r["conversations"]) for r in ok if r["conversations"]]

    print("\n=== Pilot report ===")
    print(f"Total requests:          {n}")
    print(f"Status breakdown:        {dict(status_counts)}")
    print(f"Parsed ok:               {len(ok)} ({len(ok) / max(n, 1):.1%})")
    print(f"Passed structural_check: {len(structural_pass)} ({len(structural_pass) / max(n, 1):.1%})")
    print(f"Degenerate outputs:      {len(degenerate)} ({len(degenerate) / max(len(ok), 1):.1%} of parsed)")
    print(f"Near-dup rate:           {dup_count}/{len(pairs)}")
    if turn_counts:
        print(f"Turn count:              min={min(turn_counts)} max={max(turn_counts)} "
              f"avg={sum(turn_counts) / len(turn_counts):.1f}")

    accept_rate = len(structural_pass) / n if n else 0.0
    print(f"\nOverall accept rate: {accept_rate:.1%}")
    if accept_rate > 0:
        print(f"Suggested OVERSAMPLE_FACTOR for full run: ~{round(1 / accept_rate, 2)} "
              f"(current default in plan_variants.py: {OVERSAMPLE_FACTOR})")

    if structural_pass:
        print("\n--- Sample accepted conversation (first 2 turns) ---")
        for turn in structural_pass[0]["conversations"][:2]:
            print(f"[{turn.get('role')}] {turn.get('content', '')[:200]}")


async def _main(n_tasks: int, seed: int) -> None:
    results = await run_pilot(n_tasks, seed)
    analyze(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot batch (stages 2-5) to calibrate before full-scale generation")
    parser.add_argument("--n-tasks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(_main(args.n_tasks, args.seed))
