#!/usr/bin/env bash
# Watch a gemma4_31b_agent.py background run for progress and quota-exhaustion
# signals.
#
# Usage:
#   ./watch_generation.sh <path-to-output-log>
#
# Failure signatures learned from real runs (2026-07-04):
#   429 / RateLimitError / quota_exceeded -- this is the signal that actually
#     matters. Cerebras reports both the hourly request cap and the daily
#     token cap as a 429 RateLimitError, e.g.:
#       {'code': 'request_quota_exceeded', 'message': 'Requests per hour limit exceeded'}
#       {'code': 'token_quota_exceeded', 'message': 'Tokens per day limit exceeded'}
#     A first pass at this script only grepped for "402"/"payment_required"
#     (the error shape seen in older stale data) and completely missed a live
#     run that hit both quota caps for ~130 requests before being caught --
#     don't narrow this pattern back down to just that one shape.
#   402 / payment_required -- hard billing block (different failure mode,
#     kept as a fallback in case it recurs).
#   Traceback -- anything that crashes the script outright.
#   Batch / ok= / Resuming / Done. -- normal progress markers.
set -euo pipefail

LOGFILE="${1:?Usage: $0 <output-log-path>}"

tail -f -n +1 "$LOGFILE" | grep -E --line-buffered \
  "Batch|ok=|Resuming|Traceback|429|402|quota_exceeded|RateLimitError|payment_required|Done\."
