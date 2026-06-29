#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "=== Checking for python3-full ==="
if ! python3 -m venv --help &>/dev/null; then
    echo "Installing python3-full (required for venv)..."
    sudo apt update && sudo apt install -y python3-full
fi

echo ""
echo "=== Creating virtual environment: $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "=== Installing pipeline dependencies into venv ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Enabling HuggingFace fast transfer (GCP EU CDN) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -qxF "HF_HUB_ENABLE_HF_TRANSFER=1" .env 2>/dev/null || echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> .env

echo ""
echo "=== Logging into HuggingFace CLI ==="
echo "    (uses HF_TOKEN from .env if set, otherwise prompts interactively)"
if grep -q "^HF_TOKEN=" .env 2>/dev/null; then
    HF_TOKEN_VAL=$(grep "^HF_TOKEN=" .env | cut -d= -f2)
    huggingface-cli login --token "$HF_TOKEN_VAL" --add-to-git-credential
else
    huggingface-cli login --add-to-git-credential
fi

echo ""
echo "=== Creating local pipeline directories ==="
mkdir -p "$HOME/pipeline/data/tasks"
mkdir -p "$HOME/pipeline/data/raw_results"
mkdir -p "$HOME/pipeline/data/validated"
echo "Pipeline data root: $HOME/pipeline/data"

echo ""
echo "=== Done. Activate the venv before every session ==="
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "=== Then run stages in order ==="
echo "    python3 extract_tasks.py"
echo "    python3 plan_variants.py"
echo "    python3 gemini_agent_worker.py"
echo "    python3 validate_n_dedup.py"
echo "    python3 augment_150k_rows.py"
