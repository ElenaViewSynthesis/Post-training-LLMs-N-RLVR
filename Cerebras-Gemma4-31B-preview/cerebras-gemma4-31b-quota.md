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

## Implication for the full-scale augmentation run

`plan_variants.py` targets **225,000 total requests** (150,000 accepted rows ×
`OVERSAMPLE_FACTOR=1.5`). At the 2,400/day request cap, generating all
225,000 requests would take **~94 days** of sustained generation — the
150/hour and 5/min limits are consistent with this being the real, binding
constraint (not a burst-only throttle).

**Open decision before scaling past a pilot:** switch the backend to
Together.ai (already supported in `gemma4_31b_agent.py`'s `USE_TOGETHER`
path; README estimates ~$15-20 for 150K samples), request a Cerebras quota
increase, or accept the multi-month timeline on the current tier.

> Verified 2026-07-04
