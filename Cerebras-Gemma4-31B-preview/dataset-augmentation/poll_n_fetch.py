"""
OBSOLETE for the Gemini path.

The old 04_poll_and_fetch.py existed only to poll the Anthropic Message
Batches API. With the Gemini async agent-loop worker (03_gemini_agent_worker.py),
there's no separate batch to poll -- trajectories stream to
data/raw_results/trajectories.jsonl as each completes, and the worker is
itself resume-safe (re-run it and it skips already-finished variant_ids).

Keep this file only as a marker. Delete it once you've committed to the Gemini
path. If you instead use the Gemini *Batch* API (single-shot, not multi-turn),
reintroduce a polling stage modeled on this -- but note batch can't run the
multi-turn tool loop, so you'd be back to simulated trajectories.
"""
