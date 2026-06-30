"""
Stage 5: Validation, dedup, and filtering — the stage that actually determines
data quality, so spend real attention here rather than treating it as a
formality.

Checks, in order of cheapest-to-most-expensive (fail fast):
  1. Parse check        — is the model output valid JSON matching the schema?
  2. Structural check    — reasonable turn count, no empty/truncated trajectory,
                            ends in a final/summary turn.
  3. Near-dup filter      — catch cases where high-temperature sampling still
                            converged to ~the same trajectory as another
                            variant of the same task (wasted diversity).
  4. Verifier replay      — IF the task has a pytest verifier, this is the
                            strongest signal: actually try to extract the
                            final proposed solution/commands and replay them
                            against the verifier. Tasks without a verifier
                            skip this (per the dataset card, verifiers are
                            optional in the SFT setting).
  5. Length/degenerate    — drop repetition loops, refusals, or "I cannot
                            access a real environment" disclaimers (the
                            single-shot teacher will sometimes do this -
                            watch for it in your pilot batch and tighten the
                            regex/classifier below as needed).

This runs with dask since the result set is large (~200K rows) and dedup +
embedding similarity benefits from partitioned parallelism.
"""
import hashlib
import json
import re
from pathlib import Path

import dask.dataframe as dd
import dask.bag as db
import pandas as pd

RAW_RESULTS_DIR = Path("~/pipeline/data/raw_results").expanduser()
OUT_PATH = Path("~/pipeline/data/validated").expanduser()

DEGENERATE_PATTERNS = [
    r"i (don't|do not) have access to (a|an) (real|actual) (environment|terminal|shell)",
    r"as an ai( language model)?,? i (cannot|can't|am unable)",
    r"i'm unable to actually execute",
]
DEGENERATE_RE = re.compile("|".join(DEGENERATE_PATTERNS), re.IGNORECASE)


def parse_result_line(line: str) -> dict:
    """Parse one line of trajectories.jsonl written by gemma4_31b_agent.py.
    Each record: {variant_id, task_id, temperature, status,
    conversations: [{role, content}, ...], raw}."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return {"variant_id": None, "status": "json_error", "conversations": None}
    if rec.get("status") != "ok":
        return {"variant_id": rec.get("variant_id"), "status": rec.get("status", "unknown"),
                "conversations": None}
    return {"variant_id": rec["variant_id"], "status": "parsed",
            "conversations": rec.get("conversations")}


def structural_check(conversations: list | None) -> bool:
    if not conversations or not isinstance(conversations, list):
        return False
    if len(conversations) < 2 or len(conversations) > 40:
        return False
    full_text = " ".join(turn.get("content", "") for turn in conversations if isinstance(turn, dict))
    if DEGENERATE_RE.search(full_text):
        return False
    if len(full_text.strip()) < 50:
        return False
    return True


def conversations_fingerprint(conversations: list) -> str:
    """Cheap near-dup signal: hash of normalized content, ignoring whitespace
    and casing. For stronger dedup, swap this for an embedding-similarity
    pass (e.g. sentence-transformers + cosine threshold) — fine as a first
    pass given volume, but log your collision rate and upgrade if it's high."""
    norm = " ".join(turn.get("content", "").strip().lower() for turn in conversations if isinstance(turn, dict))
    norm = re.sub(r"\s+", " ", norm)
    return hashlib.sha256(norm.encode()).hexdigest()


def process_partition(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        if rec["status"] != "parsed":
            continue
        if not structural_check(rec["conversations"]):
            continue
        rec["fingerprint"] = conversations_fingerprint(rec["conversations"])
        out.append(rec)
    return out


def main():
    files = sorted(RAW_RESULTS_DIR.glob("trajectories*.jsonl"))
    print(f"Found {len(files)} result chunks")

    bag = db.read_text([str(f) for f in files]).map(parse_result_line)
    parsed = bag.map_partitions(process_partition)
    records = parsed.compute()
    print(f"Passed parse + structural checks: {len(records)}")

    df = pd.DataFrame(records)

    # Global near-dup filter (per task_id, since variant_id encodes task_id prefix)
    df["task_id"] = df["variant_id"].str.extract(r"^(.*)-v\d+-")
    before = len(df)
    df = df.drop_duplicates(subset=["task_id", "fingerprint"])
    print(f"Dropped {before - len(df)} near-duplicate trajectories")

    # TODO: plug in verifier replay here for tasks that have a pytest verifier.
    # This needs the task's environment + verifier columns from Stage 1's task
    # table, plus a sandboxed exec step — keep it out of this dask job and
    # run it as a separate containerized step if your verifiers need Docker,
    # then merge accepted variant_ids back in before the final write.

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH / "validated_trajectories.parquet", index=False)
    print(f"Final validated count: {len(df)} (target 150,000 new rows → 250K total)")
    if len(df) < 150_000:
        print("WARNING: under target. Either raise OVERSAMPLE_FACTOR in plan_variants.py "
              "and re-run the worker, or loosen structural_check if it's overly strict.")


if __name__ == "__main__":
    main()
