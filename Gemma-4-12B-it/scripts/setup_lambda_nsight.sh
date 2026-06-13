#!/usr/bin/env bash
set -euo pipefail

# One-time Nsight Systems setup for Lambda GPU Ubuntu images.
# Installs nsight-systems through apt, then extracts a newer libssh locally so
# Nsight's QDSTRM importer can satisfy LIBSSH_4_9_0 without replacing system libs.

LAMBDA_SSH_LOGIN="${LAMBDA_SSH_LOGIN:-ubuntu@192.222.59.12}"
LAMBDA_INSTANCE_TYPE="${LAMBDA_INSTANCE_TYPE:-gpu_1x_gh200}"
NSYS_LIB_ROOT="${NSYS_LIB_ROOT:-$HOME/nsys-libs}"
NSYS_HOST_DIR="${NSYS_HOST_DIR:-}"
NSYS_TARGET_DIR="${NSYS_TARGET_DIR:-}"
REQUIRED_LIBSSH_SYMBOL="${REQUIRED_LIBSSH_SYMBOL:-LIBSSH_4_9_0}"
DEB_ARCH="$(dpkg --print-architecture)"

case "$DEB_ARCH" in
  amd64)
    DEFAULT_LIBSSH_DEB_URL="https://archive.ubuntu.com/ubuntu/pool/main/libs/libssh/libssh-4_0.10.6-2build2_amd64.deb"
    DEFAULT_LIBBPF_DEB_URL=""
    LIBSSH_MULTIARCH="x86_64-linux-gnu"
    ;;
  arm64)
    DEFAULT_LIBSSH_DEB_URL="https://ports.ubuntu.com/ubuntu-ports/pool/main/libs/libssh/libssh-4_0.10.6-2build2_arm64.deb"
    DEFAULT_LIBBPF_DEB_URL=""
    LIBSSH_MULTIARCH="aarch64-linux-gnu"
    ;;
  *)
    echo "[setup-lambda-nsight] ERROR: Unsupported Ubuntu architecture: $DEB_ARCH" >&2
    exit 1
    ;;
esac

LIBSSH_DEB_URL="${LIBSSH_DEB_URL:-$DEFAULT_LIBSSH_DEB_URL}"
LIBBPF_DEB_URL="${LIBBPF_DEB_URL:-$DEFAULT_LIBBPF_DEB_URL}"
LIBSSH_DEB_NAME="${LIBSSH_DEB_URL##*/}"
LOCAL_LIBSSH_DIR="$NSYS_LIB_ROOT/usr/lib/$LIBSSH_MULTIARCH"
LOCAL_LIBSSH_SO="$LOCAL_LIBSSH_DIR/libssh.so.4"
ACTIVE_LIBSSH_DIR=""
LOCAL_LIBBPF_DIR="$NSYS_LIB_ROOT/usr/lib/$LIBSSH_MULTIARCH"
LOCAL_LIBBPF_SO="$LOCAL_LIBBPF_DIR/libbpf.so.1"
ACTIVE_LIBBPF_DIR=""
ENV_FILE="$NSYS_LIB_ROOT/env.sh"

log() {
  printf '[setup-lambda-nsight] %s\n' "$*"
}

detect_nsys_host_dir() {
  local candidate
  for candidate in \
    /usr/lib/aarch64-linux-gnu/nsight-systems/host-linux-sbsa-armv8 \
    /usr/lib/aarch64-linux-gnu/nsight-systems/host-linux-armv8 \
    /usr/lib/aarch64-linux-gnu/nsight-systems/host-linux-aarch64 \
    /usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64 \
    /usr/lib/nsight-systems/host-linux-x64 \
    /usr/lib/nsight-systems/host-linux-armv8 \
    /usr/lib/nsight-systems/host-linux-aarch64
  do
    if [[ -x "$candidate/QdstrmImporter" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local importer
  importer="$(find /usr/lib/aarch64-linux-gnu/nsight-systems /usr/lib/x86_64-linux-gnu/nsight-systems /usr/lib/nsight-systems /opt/nvidia/nsight-systems -path '*/QdstrmImporter' -type f -executable -print -quit 2>/dev/null || true)"
  if [[ -n "$importer" ]]; then
    dirname "$importer"
    return 0
  fi

  return 1
}

detect_nsys_target_dir() {
  local candidate
  for candidate in \
    /usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-sbsa-armv8 \
    /usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-armv8 \
    /usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-aarch64 \
    /usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64 \
    /usr/lib/nsight-systems/target-linux-x64 \
    /usr/lib/nsight-systems/target-linux-armv8 \
    /usr/lib/nsight-systems/target-linux-aarch64 \
    /usr/lib/nsight-systems/target-linux-sbsa-armv8
  do
    if [[ -f "$candidate/libToolsInjectionMemoryAllocator.so" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local injection_lib
  injection_lib="$(find /usr/lib/aarch64-linux-gnu/nsight-systems /usr/lib/x86_64-linux-gnu/nsight-systems /usr/lib/nsight-systems /opt/nvidia/nsight-systems -name libToolsInjectionMemoryAllocator.so -type f -print -quit 2>/dev/null || true)"
  if [[ -n "$injection_lib" ]]; then
    dirname "$injection_lib"
    return 0
  fi

  return 1
}

find_system_libssh() {
  local libssh_so
  libssh_so="$(ldconfig -p 2>/dev/null | awk '/libssh\.so\.4 / { print $NF; exit }')"
  if [[ -n "$libssh_so" && -f "$libssh_so" ]]; then
    printf '%s\n' "$libssh_so"
    return 0
  fi

  for libssh_so in \
    "/usr/lib/$LIBSSH_MULTIARCH/libssh.so.4" \
    /usr/lib/libssh.so.4
  do
    if [[ -f "$libssh_so" ]]; then
      printf '%s\n' "$libssh_so"
      return 0
    fi
  done

  return 1
}

lib_has_symbol() {
  local libssh_so="$1"
  [[ -f "$libssh_so" ]] && strings "$libssh_so" | grep -q "$REQUIRED_LIBSSH_SYMBOL"
}

find_libbpf_so_1() {
  local libbpf_so
  libbpf_so="$(ldconfig -p 2>/dev/null | awk '/libbpf\.so\.1 / { print $NF; exit }')"
  if [[ -n "$libbpf_so" && -f "$libbpf_so" ]]; then
    printf '%s\n' "$libbpf_so"
    return 0
  fi

  for libbpf_so in \
    "/usr/lib/$LIBSSH_MULTIARCH/libbpf.so.1" \
    /usr/lib/libbpf.so.1 \
    "$LOCAL_LIBBPF_SO"
  do
    if [[ -f "$libbpf_so" ]]; then
      printf '%s\n' "$libbpf_so"
      return 0
    fi
  done

  return 1
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

log "Lambda login: $LAMBDA_SSH_LOGIN"
log "Lambda instance type: $LAMBDA_INSTANCE_TYPE"
log "Detected Ubuntu architecture: $DEB_ARCH"
log "Installing Nsight Systems if needed."
sudo apt-get update
sudo apt-get install -y nsight-systems binutils libssh-4

if ! find_libbpf_so_1 >/dev/null; then
  if apt-cache show libbpf1 >/dev/null 2>&1; then
    sudo apt-get install -y libbpf1
  else
    log "libbpf1 is not available from the configured apt sources."
  fi
fi

if ! find_libbpf_so_1 >/dev/null && [[ -n "$LIBBPF_DEB_URL" ]]; then
  log "Installing libbpf locally under $NSYS_LIB_ROOT."
  mkdir -p "$NSYS_LIB_ROOT"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  libbpf_deb_name="${LIBBPF_DEB_URL##*/}"
  download "$LIBBPF_DEB_URL" "$tmpdir/$libbpf_deb_name"
  dpkg-deb -x "$tmpdir/$libbpf_deb_name" "$NSYS_LIB_ROOT"
fi

LIBBPF_SO="$(find_libbpf_so_1 || true)"
if [[ -n "$LIBBPF_SO" ]]; then
  ACTIVE_LIBBPF_DIR="$(dirname "$LIBBPF_SO")"
  log "Found libbpf.so.1: $LIBBPF_SO"
else
  log "WARNING: libbpf.so.1 was not found. nsys may fail to start."
  log "If needed, rerun with LIBBPF_DEB_URL set to a libbpf1 .deb for $DEB_ARCH."
fi

if [[ -z "$NSYS_HOST_DIR" ]]; then
  if ! NSYS_HOST_DIR="$(detect_nsys_host_dir)"; then
    log "ERROR: Could not find Nsight Systems QdstrmImporter."
    log "Check the Nsight install with: dpkg -L nsight-systems | grep QdstrmImporter"
    exit 1
  fi
fi

if [[ ! -x "$NSYS_HOST_DIR/QdstrmImporter" ]]; then
  log "ERROR: QdstrmImporter was not found at $NSYS_HOST_DIR/QdstrmImporter"
  log "Check the Nsight install with: dpkg -L nsight-systems | grep QdstrmImporter"
  exit 1
fi

if [[ -z "$NSYS_TARGET_DIR" ]]; then
  if ! NSYS_TARGET_DIR="$(detect_nsys_target_dir)"; then
    log "ERROR: Could not find Nsight Systems libToolsInjectionMemoryAllocator.so."
    log "Check the Nsight install with: find /usr/lib/aarch64-linux-gnu/nsight-systems /usr/lib/x86_64-linux-gnu/nsight-systems /usr/lib/nsight-systems /opt/nvidia/nsight-systems -name libToolsInjectionMemoryAllocator.so"
    exit 1
  fi
fi

if [[ ! -f "$NSYS_TARGET_DIR/libToolsInjectionMemoryAllocator.so" ]]; then
  log "ERROR: libToolsInjectionMemoryAllocator.so was not found at $NSYS_TARGET_DIR/libToolsInjectionMemoryAllocator.so"
  exit 1
fi

SYSTEM_LIBSSH_SO="$(find_system_libssh || true)"
if [[ -n "$SYSTEM_LIBSSH_SO" ]] && lib_has_symbol "$SYSTEM_LIBSSH_SO"; then
  ACTIVE_LIBSSH_DIR="$(dirname "$SYSTEM_LIBSSH_SO")"
  log "System libssh provides $REQUIRED_LIBSSH_SYMBOL: $SYSTEM_LIBSSH_SO"
elif ! lib_has_symbol "$LOCAL_LIBSSH_SO"; then
  log "Installing newer libssh locally under $NSYS_LIB_ROOT."
  mkdir -p "$NSYS_LIB_ROOT"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  download "$LIBSSH_DEB_URL" "$tmpdir/$LIBSSH_DEB_NAME"
  dpkg-deb -x "$tmpdir/$LIBSSH_DEB_NAME" "$NSYS_LIB_ROOT"
else
  log "Local libssh already has $REQUIRED_LIBSSH_SYMBOL."
fi

if [[ -z "$ACTIVE_LIBSSH_DIR" ]] && lib_has_symbol "$LOCAL_LIBSSH_SO"; then
  ACTIVE_LIBSSH_DIR="$LOCAL_LIBSSH_DIR"
  log "Local libssh provides $REQUIRED_LIBSSH_SYMBOL: $LOCAL_LIBSSH_SO"
fi

if [[ -z "$ACTIVE_LIBSSH_DIR" ]]; then
  log "WARNING: No libssh.so.4 with $REQUIRED_LIBSSH_SYMBOL was found."
  log "Continuing without a local libssh override. If nsys reports a libssh symbol error,"
  log "set LIBSSH_DEB_URL to a newer libssh-4 .deb for $DEB_ARCH and rerun this script."
fi

LD_LIBRARY_PATH_PARTS=()
[[ -n "$ACTIVE_LIBSSH_DIR" ]] && LD_LIBRARY_PATH_PARTS+=("$ACTIVE_LIBSSH_DIR")
[[ -n "$ACTIVE_LIBBPF_DIR" ]] && LD_LIBRARY_PATH_PARTS+=("$ACTIVE_LIBBPF_DIR")
LD_LIBRARY_PATH_PARTS+=("$NSYS_HOST_DIR")
LD_LIBRARY_PATH_PARTS+=("$NSYS_TARGET_DIR")
LD_LIBRARY_PATH_VALUE="$(IFS=:; printf '%s' "${LD_LIBRARY_PATH_PARTS[*]}"):\${LD_LIBRARY_PATH:-}"

cat > "$ENV_FILE" <<EOF
# Source this before running nsys profile on Lambda:
#   source "$ENV_FILE"
export NSYS_LIB_ROOT="$NSYS_LIB_ROOT"
export NSYS_HOST_DIR="$NSYS_HOST_DIR"
export NSYS_TARGET_DIR="$NSYS_TARGET_DIR"
export LAMBDA_SSH_LOGIN="$LAMBDA_SSH_LOGIN"
export LAMBDA_INSTANCE_TYPE="$LAMBDA_INSTANCE_TYPE"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE"
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
