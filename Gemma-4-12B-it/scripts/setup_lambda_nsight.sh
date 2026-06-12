#!/usr/bin/env bash
set -euo pipefail

# One-time Nsight Systems setup for Lambda GPU Ubuntu images.
# Installs nsight-systems through apt, then extracts a newer libssh locally so
# Nsight's QDSTRM importer can satisfy LIBSSH_4_9_0 without replacing system libs.

NSYS_LIB_ROOT="${NSYS_LIB_ROOT:-$HOME/nsys-libs}"
NSYS_HOST_DIR="${NSYS_HOST_DIR:-/usr/lib/nsight-systems/host-linux-x64}"
LIBSSH_DEB_URL="${LIBSSH_DEB_URL:-https://archive.ubuntu.com/ubuntu/pool/main/libs/libssh/libssh-4_0.10.6-2build2_amd64.deb}"
LIBSSH_DEB_NAME="${LIBSSH_DEB_URL##*/}"
LIBSSH_DIR="$NSYS_LIB_ROOT/usr/lib/x86_64-linux-gnu"
LIBSSH_SO="$LIBSSH_DIR/libssh.so.4"
ENV_FILE="$NSYS_LIB_ROOT/env.sh"

log() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
  printf '[setup-lambda-nsight] %s\n' "$*"
}

download() {
  local url="$1"                                          
  local output="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
  else
    log "Installing wget so the libssh package can be downloaded."
    sudo apt-get update
    sudo apt-get install -y wget ca-certificates
    wget -q --show-progress -O "$output" "$url"
  fi
}

log "Installing Nsight Systems if needed."
sudo apt-get update
sudo apt-get install -y nsight-systems

if [[ ! -x "$NSYS_HOST_DIR/QdstrmImporter" ]]; then
  log "ERROR: QdstrmImporter was not found at $NSYS_HOST_DIR/QdstrmImporter"
  log "Check the Nsight install with: dpkg -L nsight-systems | grep QdstrmImporter"
  exit 1
fi

if [[ ! -f "$LIBSSH_SO" ]] || ! strings "$LIBSSH_SO" | grep -q 'LIBSSH_4_9_0'; then
  log "Installing newer libssh locally under $NSYS_LIB_ROOT."
  mkdir -p "$NSYS_LIB_ROOT"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  download "$LIBSSH_DEB_URL" "$tmpdir/$LIBSSH_DEB_NAME"
  dpkg-deb -x "$tmpdir/$LIBSSH_DEB_NAME" "$NSYS_LIB_ROOT"
else
  log "Local libssh already has LIBSSH_4_9_0."
fi

if ! strings "$LIBSSH_SO" | grep -q 'LIBSSH_4_9_0'; then
  log "ERROR: $LIBSSH_SO does not provide LIBSSH_4_9_0."
  log "Set LIBSSH_DEB_URL to a newer libssh-4 .deb and rerun this script."
  exit 1
fi

cat > "$ENV_FILE" <<EOF
# Source this before running nsys profile on Lambda:
#   source "$ENV_FILE"
export NSYS_LIB_ROOT="$NSYS_LIB_ROOT"
export NSYS_HOST_DIR="$NSYS_HOST_DIR"
export LD_LIBRARY_PATH="$LIBSSH_DIR:$NSYS_HOST_DIR:\${LD_LIBRARY_PATH:-}"
EOF

log "Wrote $ENV_FILE"
log "Use this before any Nsight profile run:"
printf '\n  source "%s"\n\n' "$ENV_FILE"
log "Then run, for example:"
cat <<'EOF'

  CUDA_VISIBLE_DEVICES=0 nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    -o profiles/nsys/mitre_sft_smoke \
    accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 20

EOF
