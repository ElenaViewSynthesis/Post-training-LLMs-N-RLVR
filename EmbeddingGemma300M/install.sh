#!/usr/bin/env bash
set -euo pipefail

# ── 1. Install uv if not present ──────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.cargo/env"
fi

# ── 2. Create / activate venv ─────────────────────────────────────────────────
uv venv .venv --python 3.11
source .venv/bin/activate

# ── 3. Install torch first (required before xformers detection) ───────────────
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# ── 4. Detect torch version → resolve xformers pin ───────────────────────────
TORCH_VER=$(python - <<'EOF'
import torch, re
v = re.match(r'\d+\.\d+', torch.__version__).group(0)
xmap = {'2.10': '0.0.34', '2.9': '0.0.33.post1', '2.8': '0.0.32.post2'}
print(xmap.get(v, '0.0.34'))
EOF
)
echo "→ xformers pin: ${TORCH_VER}"

# ── 5. Core deps (with full resolver) ─────────────────────────────────────────
uv pip install \
  sentencepiece protobuf \
  "datasets==4.3.0" \
  "huggingface_hub>=0.34.0" \
  hf_transfer \
  "transformers==4.56.2"

# ── 6. No-deps installs (avoid resolver conflicts) ────────────────────────────
uv pip install --no-deps \
  unsloth_zoo \
  bitsandbytes \
  accelerate \
  "xformers==${TORCH_VER}" \
  peft \
  "trl==0.22.2" \
  triton \
  unsloth

# ── 7. torchao — upgrade-only, no-deps ────────────────────────────────────────
uv pip install --no-deps --upgrade "torchao>=0.16.0"

echo "✓ Environment ready"
