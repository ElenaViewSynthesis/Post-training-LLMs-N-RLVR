#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing pipeline dependencies ==="
python3 -m pip install -r requirements.txt

echo ""
echo "=== Enabling HuggingFace fast transfer ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -qxF "HF_HUB_ENABLE_HF_TRANSFER=1" .env 2>/dev/null || echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> .env

echo ""
echo "=== Creating local pipeline directories ==="
mkdir -p "$HOME/pipeline/data/tasks"
mkdir -p "$HOME/pipeline/data/raw_results"
mkdir -p "$HOME/pipeline/data/validated"
echo "Pipeline data root: $HOME/pipeline/data"

echo ""
echo "=== Done. Next steps ==="
echo "1. Confirm CEREBRAS_API_KEY and HF_TOKEN are set in .env"
echo "2. Run: python3 extract_tasks.py    (inspect schema output before continuing)"
echo "3. Run: python3 plan_variants.py    (generates 225K variant plan)"
echo "4. Run: python3 gemini_agent_worker.py  (pilot on 1K rows first)"
echo "5. Run: python3 validate_n_dedup.py"
echo "6. Run: python3 augment_150k_rows.py"
