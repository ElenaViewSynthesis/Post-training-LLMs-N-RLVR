# Cerebras Quota — gemma-4-31b (Preview)

| | |
|---|---|
| **Max Context Length** | 65,536 tokens |
| **Type** | Preview |

## Requests

| Window | Limit |
|---|---|
| Per minute | 5 |
| Per hour | 150 |
| Per day | 2,400 |

## Tokens

| Window | Limit |
|---|---|
| Per minute | 30,000 |
| Per hour | 1,000,000 |
| Per day | 1,000,000 |

## What actually halts a run: the daily *cumulative* token cap

Observed empirically on 2026-07-04 (see
`assets/cerebras-tokens/july4-1M-tokens.png`, a screenshot of the Cerebras
Analytics dashboard for that day):

- **Total tokens for the day: 1.25M** — which *overshoots* the 1,000,000/day
  cap by ~250K. The overshoot happens because in-flight concurrent requests
  (CONCURRENCY=4) are already dispatched when the cap is crossed.
- The per-minute token graph peaked around **20-24K/min, under the 30K/min
  cap** — so the per-minute token limit was *never* the binding constraint.
  It was the running **daily cumulative total crossing 1M** that started
  returning errors.
- The per-minute request graph sat pinned at the **5/min** dashed quota line
  during active batches, confirming request throttling (but this only slows
  the run; it doesn't stop it).

**Failure signatures, in the order they appeared** (all `429 RateLimitError`,
NOT `402` — a `402`/`payment_required` is a separate hard billing block seen
only in older stale data):

| Order | Code | Message |
|---|---|---|
| First (~380 requests in) | `request_quota_exceeded` | Requests per hour limit exceeded |
| Then (~445 requests in)  | `token_quota_exceeded`  | Tokens per day limit exceeded |

Once `token_quota_exceeded` starts, every subsequent request is rejected and
the accept rate collapses — the day is effectively over. `watch_generation.sh`
greps for these signatures so a live run surfaces them immediately.

## Implication for the full-scale augmentation run

`plan_variants.py` targets **225,000 total requests** (150,000 accepted rows ×
`OVERSAMPLE_FACTOR=1.5`). But the **daily token cap (1M/day), not the request
cap, is the real ceiling on daily throughput**: at ~2,500 completion tokens +
~1,000 prompt tokens per request (~3.5K total), 1M tokens/day covers only
**~285 requests/day**, far below the 2,400/day request cap. On 2026-07-04 the
run reached ~432 requests before quota exhaustion (higher than 285 because
many early requests were short truncation/parse failures that burned fewer
tokens). At a realistic sustained rate, reaching 225,000 requests would take
**many months** on the free tier.

**Open decision before scaling past a pilot:** switch the backend to
Together.ai (already supported in `gemma4_31b_agent.py`'s `USE_TOGETHER`
path; README estimates ~$15-20 for 150K samples), request a Cerebras quota
increase, or accept the multi-month timeline on the current tier.

> Verified 2026-07-04
