# Troubleshooting

## Lambda GPU Python / NumPy / SciPy Issues

If fine-tuning fails with imports coming from:

```text
/home/ubuntu/.local/lib/python3.10/
```

or with a NumPy/SciPy binary compatibility error, the active environment is wrong or polluted by system/user-site packages.

The project requires Python 3.12 or newer.

## 1. Verify the Active Environment

Run from the Lambda instance:

```bash
cd ~/Gemma-4-12B-it
source .venv/bin/activate
which python
python --version
```

Expected:

```text
/home/ubuntu/Gemma-4-12B-it/.venv/bin/python
Python 3.12.x
```

Also check that Python is not loading user-site packages:

```bash
python -c "import sys, site; print(sys.executable); print(site.ENABLE_USER_SITE)"
```

Expected:

```text
/home/ubuntu/Gemma-4-12B-it/.venv/bin/python
False
```

## 2. Recreate the Virtualenv with Python 3.12

If `python --version` shows Python 3.10, recreate the environment:

```bash
cd ~/Gemma-4-12B-it
deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
rm -rf .venv

sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 3. If Python Is Already 3.12

If the venv is correct but NumPy/SciPy still fail, reinstall the scientific stack inside the venv:

```bash
cd ~/Gemma-4-12B-it
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME
pip install -U numpy scipy scikit-learn
```

## 4. Verify CUDA and Imports

```bash
python -c "import numpy, scipy, sklearn, torch; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('sklearn ok'); print('cuda', torch.cuda.is_available())"
```

Expected:

```text
sklearn ok
cuda True
```

## 5. Sync Project and Restart Fine-Tuning

If the script is missing or outdated on the instance, re-sync from local WSL first.

**Copy a single file (faster for quick script updates):**
```bash
scp /mnt/c/Users/proxi/Documents/ccsyntheticdata/Gemma-4-12B-it/06_mitre_sft.py \
  ubuntu@129.146.110.174:~/Gemma-4-12B-it/06_mitre_sft.py
```

> Replace `129.146.110.174` with the current instance IP if on a new machine.

**Or re-sync the full project:**

```bash
rsync -av --exclude .venv --exclude .uv-cache --exclude .git \
  /mnt/c/Users/proxi/Documents/ccsyntheticdata/Gemma-4-12B-it/ \
  ubuntu@129.146.110.174:~/Gemma-4-12B-it/
```

> If you are on a different GPU instance, replace `129.146.110.174` with that instance's IP.

Then on the Lambda SSH terminal:

```bash
cd ~/Gemma-4-12B-it
ls -la 06_mitre_sft.py
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

## 6. Restart QLoRA Fine-Tuning

```bash
cd ~/Gemma-4-12B-it
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

For long runs, use `tmux`:

```bash
tmux new -s mitre-sft
cd ~/Gemma-4-12B-it
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py
```

Detach with `Ctrl-b`, then `d`.

Reattach later:

```bash
tmux attach -t mitre-sft
```

