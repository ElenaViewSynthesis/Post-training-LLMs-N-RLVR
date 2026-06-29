#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "=== Checking for python3-full ==="
if ! python3 -m venv --help &>/dev/null; then
    echo "Installing python3-full (required for venv)..."
    sudo apt update && sudo apt install -y python3-full
fi

echo ""
echo "=== Installing uv (fast Python package manager) ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "=== Installing HuggingFace CLI via uv ==="
uv tool install hf
export PATH="$HOME/.local/bin:$PATH"

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
echo "=== HuggingFace auth ==="
echo "    If not already logged in, run: hf auth login"
echo ""
echo "=== Creating local pipeline directories ==="
mkdir -p "$HOME/pipeline/data/tasks"
mkdir -p "$HOME/pipeline/data/raw_results"
mkdir -p "$HOME/pipeline/data/validated"
mkdir -p "$HOME/pipeline/data/upload"
echo "Pipeline data root: $HOME/pipeline/data"

echo ""
echo "=== Done. Activate the venv before every session ==="
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "=== Bucket sync commands ==="
echo "    Upload: hf sync ./data hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k"
echo "    Download: hf sync hf://buckets/borntobeignored/OpenThoughts-Agents-SFT-250k ./local"
echo ""
echo "=== Then run stages in order ==="
echo "    python3 extract_tasks.py"
echo "    python3 plan_variants.py"
echo "    python3 gemini_agent_worker.py"
echo "    python3 validate_n_dedup.py"
echo "    python3 augment_150k_rows.py"
