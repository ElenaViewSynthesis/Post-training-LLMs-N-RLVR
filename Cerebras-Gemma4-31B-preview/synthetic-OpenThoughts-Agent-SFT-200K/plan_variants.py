"""
Stage 2 (Gemini 3.x rewrite): Perturbation planner.

CHANGED vs the original plan: we do NOT perturb temperature/top_p/top_k.
For Gemini 3.x, Google explicitly recommends leaving sampling params at their
defaults -- the reasoning path is tuned for them, and nudging them degrades
trajectory quality rather than usefully diversifying it. So the rollout
diversity (same task, new rollout) has to come from axes that are actually
legitimate for a 3.x reasoning model:

  1. thinking_level in {"low", "medium", "high"} -- this genuinely changes
     trajectory character: low gives terse, fewer-turn rollouts; high gives
     longer deliberation with more intermediate tool calls. This is your
     single best controlled diversity knob now.
  2. instruction_framing -- light paraphrase of *framing*, not task semantics.
     Keep it minimal; Gemini 3.x responds best to direct instructions and can
     over-analyze verbose prompt-engineering, so these stay short.
  3. inherent run-to-run nondeterminism -- even at fixed config the model is
     not fully deterministic, so repeated runs of the same (task, level,
     framing) tuple still differ. This is weak on its own, which is why 1+2
     carry the load.

Target is still ~150K accepted new rows (100K -> 250K). With unlimited Gemini
requests, cost is no longer the constraint, so you can afford a higher
oversample and lean on Stage 5 to be strict. The constraint is now throughput
(RPM/concurrency + sandbox capacity), handled in Stage 3.
"""
import hashlib
import random

import dask.dataframe as dd
import pandas as pd

TARGET_NEW_ROWS = 150_000
OVERSAMPLE_FACTOR = 1.5  # higher than before -- requests are free now, be strict in Stage 5
N_REQUESTS = int(TARGET_NEW_ROWS * OVERSAMPLE_FACTOR)

THINKING_LEVELS = ["low", "medium", "high"]
# Bias toward medium: docs note medium is the best quality/latency default for
# most tasks; low/high give you the spread at the tails.
THINKING_WEIGHTS = [0.25, 0.5, 0.25]

INSTRUCTION_FRAMINGS = [
    None,  # verbatim original instruction
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

    id_col = "task_id" if "task_id" in tasks_pdf.columns else "id"
    rows = []
    for i, (_, task_row) in enumerate(tasks_pdf.iterrows()):
        k = base_per_task + (1 if i < remainder else 0)
        for v in range(k):
            rows.append({
                "task_id": task_row[id_col],
                "variant_id": stable_variant_id(str(task_row[id_col]), v),
                "thinking_level": rng.choices(THINKING_LEVELS, weights=THINKING_WEIGHTS)[0],
                "instruction_framing": rng.choice(INSTRUCTION_FRAMINGS),
            })
    plan = pd.DataFrame(rows)
    # Shuffle so any single worker shard isn't dominated by one task's variants
    return plan.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    tasks = dd.read_parquet("/home/claude/pipeline/data/tasks/").compute()
    plan = plan_variants(tasks, N_REQUESTS)
    plan.to_parquet("/home/claude/pipeline/data/variant_plan.parquet", index=False)
    print(f"Planned {len(plan)} generation requests across {len(tasks)} tasks")
    print(plan["thinking_level"].value_counts())
