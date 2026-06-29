#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing pipeline dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Enabling HuggingFace fast transfer ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> .env

echo ""
echo "=== Creating local pipeline directories ==="
mkdir -p /home/claude/pipeline/data/tasks
mkdir -p /home/claude/pipeline/data/raw_results
mkdir -p /home/claude/pipeline/data/validated

echo ""
echo "=== Done. Next steps ==="
echo "1. Add CEREBRAS_API_KEY and HF_TOKEN to .env"
echo "2. Run: python extract_tasks.py    (inspect schema output before continuing)"
echo "3. Run: python plan_variants.py    (generates 225K variant plan)"
echo "4. Run: python gemini_agent_worker.py  (pilot on 1K rows first)"
echo "5. Run: python validate_n_dedup.py"
echo "6. Run: python augment_150k_rows.py"
