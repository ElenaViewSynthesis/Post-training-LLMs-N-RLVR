# Lambda GH200 Connection and Nsight Workflow

## Instance details

- **SSH login:** `ssh ubuntu@192.222.59.12`
- **Instance type:** `gpu_1x_gh200`
- **Project dir on instance:** `~/Gemma-4-12B-it`

---

## 1. Sync the project from local WSL

Run this in the local WSL terminal:

```bash
rsync -av --exclude .venv --exclude .uv-cache --exclude .git \
  /mnt/c/Users/proxi/Documents/ccsyntheticdata/Gemma-4-12B-it/ \
  ubuntu@192.222.59.12:~/Gemma-4-12B-it/
```

---

## 2. SSH into Lambda

Run this in the local WSL terminal:

```bash
ssh ubuntu@192.222.59.12
```

Everything below runs in the Lambda SSH terminal.

---

## 3. Set up Nsight

```bash
cd ~/Gemma-4-12B-it
bash scripts/setup_lambda_nsight.sh
source ~/nsys-libs/env.sh
nsys --version
```

---

## 4. Build compatible libbpf if Nsight fails

Only do this if `nsys --version` fails with a `libbpf.so.1`, `LIBBPF_0.8.0`, or `GLIBC_2.38` error.

```bash
rm -f ~/nsys-libs/usr/lib/aarch64-linux-gnu/libbpf.so*

sudo apt-get update
sudo apt-get install -y build-essential pkg-config libelf-dev zlib1g-dev wget ca-certificates

mkdir -p /tmp/libbpf-build
cd /tmp/libbpf-build

wget -O libbpf-v0.8.0.tar.gz \
  https://github.com/libbpf/libbpf/archive/refs/tags/v0.8.0.tar.gz

tar -xf libbpf-v0.8.0.tar.gz
cd libbpf-0.8.0/src

make -j"$(nproc)"

mkdir -p ~/nsys-libs/usr/lib/aarch64-linux-gnu
cp libbpf.so.0.8.0 ~/nsys-libs/usr/lib/aarch64-linux-gnu/
ln -sf libbpf.so.0.8.0 ~/nsys-libs/usr/lib/aarch64-linux-gnu/libbpf.so.1
```

Then rerun the Nsight setup:

```bash
cd ~/Gemma-4-12B-it
bash scripts/setup_lambda_nsight.sh
source ~/nsys-libs/env.sh
nsys --version
```

---

## 5. Create the Python environment

```bash
cd ~/Gemma-4-12B-it
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

---

## 6. Authenticate

Run this after `pip install -r requirements.txt` succeeds and the virtual environment is active:

```bash
wandb login
hf auth login
```

---

## 7. Smoke test training

```bash
cd ~/Gemma-4-12B-it
source ~/nsys-libs/env.sh
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py --max-steps 20
```

---

## 8. Run real training with Nsight profiling

Start a tmux session:

```bash
tmux new -s mitre-nsys
```

Inside tmux:

```bash
cd ~/Gemma-4-12B-it
source ~/nsys-libs/env.sh
source .venv/bin/activate
mkdir -p profiles/nsys

CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiles/nsys/mitre_sft_timeline \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx
```

Detach from tmux while training continues:

```text
Ctrl-b then d
```

Reattach later:

```bash
tmux attach -t mitre-nsys
```

---

## 9. Copy profiling results back to local WSL

Run this in the local WSL terminal:

```bash
scp -r ubuntu@192.222.59.12:~/Gemma-4-12B-it/profiles .
```

---

## Later sessions

Run this after SSH-ing into Lambda:

```bash
cd ~/Gemma-4-12B-it
source ~/nsys-libs/env.sh
source .venv/bin/activate
```
