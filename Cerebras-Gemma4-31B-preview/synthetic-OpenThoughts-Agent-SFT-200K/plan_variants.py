"""
Stage 2: Perturbation planner.

Rollout diversity comes from three axes:
  1. temperature — Cerebras/Gemma-4-31B has no restriction on sampling params
     (unlike Gemini 3.x), so varying temperature is a legitimate and effective
     diversity knob. Low values produce more deterministic, focused rollouts;
     high values produce more exploratory ones.
  2. instruction_framing — light paraphrase of the *framing*, not task semantics.
     Keeps the task distribution stable while varying the surface presentation.
  3. inherent run-to-run nondeterminism — even at fixed temperature the model
     is not fully deterministic, so repeated (task, temp, framing) tuples still
     differ. Weak on its own; 1+2 carry the load.

Target: ~150K accepted new rows (100K original + 150K new = 250K total).
OVERSAMPLE_FACTOR=1.5 → generates 225K requests; calibrate from your Stage 5
pilot rejection rate and adjust before the full run.
"""
import hashlib
import random

import dask.dataframe as dd
import pandas as pd

TARGET_NEW_ROWS = 150_000   # 100K original + 150K new = 250K total
OVERSAMPLE_FACTOR = 1.5     # generate 225K, accept ~150K after Stage 5 filtering
N_REQUESTS = int(TARGET_NEW_ROWS * OVERSAMPLE_FACTOR)

TEMPERATURES = [0.7, 0.85, 1.0]
TEMPERATURE_WEIGHTS = [0.3, 0.5, 0.2]   # bias toward 0.85 (quality/diversity sweet spot)

INSTRUCTION_FRAMINGS = [
    None,   # verbatim original instruction
    "Solve this task step by step, verifying each command's effect before continuing.",
    "Approach this as you would in a real production terminal.",
]


def stable_variant_id(task_id: str, variant_idx: int) -> str:
    h = hashlib.sha256(f"{task_id}::{variant_idx}".encode()).hexdigest()[:16]
    return f"{task_id}-v{variant_idx}-{h}"


def plan_variants(tasks_pdf: pd.DataFrame, n_requests: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    n_tasks = len(tasks_pdf)
    base_per_task = n_requests // n_tasks
    remainder = n_requests % n_tasks

    id_col = next((c for c in ("task_id", "id", "task", "run_id") if c in tasks_pdf.columns), tasks_pdf.columns[0])
    rows = []
    for i, (_, task_row) in enumerate(tasks_pdf.iterrows()):
        k = base_per_task + (1 if i < remainder else 0)
        for v in range(k):
            rows.append({
                "task_id": task_row[id_col],
                "variant_id": stable_variant_id(str(task_row[id_col]), v),
                "temperature": rng.choices(TEMPERATURES, weights=TEMPERATURE_WEIGHTS)[0],
                "instruction_framing": rng.choice(INSTRUCTION_FRAMINGS),
            })
    plan = pd.DataFrame(rows)
    # Shuffle so any single worker shard isn't dominated by one task's variants
    return plan.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path
    TASKS_PATH = str(Path("~/pipeline/data/tasks/").expanduser())
    PLAN_PATH  = str(Path("~/pipeline/data/variant_plan.parquet").expanduser())

    tasks = dd.read_parquet(TASKS_PATH).compute()
    plan = plan_variants(tasks, N_REQUESTS)
    plan.to_parquet(PLAN_PATH, index=False)
    print(f"Planned {len(plan)} generation requests across {len(tasks)} tasks")
    print(plan["temperature"].value_counts())
