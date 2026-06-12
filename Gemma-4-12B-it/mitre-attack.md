# Gemma 4 12B — MITRE ATT&CK Fine-tuning

Fine-tunes Gemma 4 12B on the MITRE ATT&CK Enterprise dataset using QLoRA.
The script downloads `enterprise-attack.json` from GitHub, converts STIX objects
into instruction Q&A pairs, and trains with HuggingFace TRL.



## 1. Data

The script fetches the MITRE ATT&CK Enterprise STIX bundle automatically:

```
Source: mitre-attack/attack-stix-data
File:   enterprise-attack/enterprise-attack.json
Cache:  ./data/mitre/enterprise-attack.json  (downloaded once, reused on subsequent runs)
```

Or download it manually before running:

```bash
mkdir -p data/mitre
wget -O data/mitre/enterprise-attack.json \
  https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
```

### Example model output

After fine-tuning, the model produces structured ATT&CK analysis like:

```json
{
  "tactic": "Execution",
  "technique_id": "T1059.001",
  "technique_name": "PowerShell",
  "is_subtechnique": true,
  "platforms": ["Windows"],
  "summary": "Adversaries may abuse PowerShell commands and scripts for execution.",
  "recommended_detection": "Monitor PowerShell logs and suspicious command-line activity.",
  "data_sources": ["Command", "Process"],
  "recommended_mitigation": ["Execution Prevention", "Privileged Account Management"]
}
```

### What gets converted to training examples

| STIX type | Examples generated |
|---|---|
| `attack-pattern` (technique) | Explain technique · Detect it · Structured JSON summary · Mitigations |
| `intrusion-set` (group) | Describe group · List techniques used |
| `malware` / `tool` | Describe the software |
| `x-mitre-tactic` | Describe the tactic |

Revoked and deprecated objects are skipped automatically.

---

## 3. Run fine-tuning

### QLoRA (recommended — fits on A100 40 GB)

```bash
accelerate launch 06_mitre_sft.py
```

### Full-precision LoRA

```bash
accelerate launch 06_mitre_sft.py --no-qlora
```


---

## 4. Training config

| Parameter | Value |
|---|---|
| Base model | `google/gemma-4-12B-it` |
| Method | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o proj + gate/up/down proj |
| Epochs | 3 |
| Batch size | 1 (+ 8 gradient accumulation steps) |
| Learning rate | 2e-4 (cosine schedule) |
| Max seq length | 2048 |
| Packing | Disabled by default |
| Optimizer | paged_adamw_8bit |
| Output | `./gemma4-mitre-sft/` |

---

## 5. Profiling

Prefer Nsight Systems first for whole-training timeline analysis, then Nsight Compute for a small number of kernels.

### Nsight Systems timeline

```bash
mkdir -p profiles/nsys
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiles/nsys/mitre_sft_timeline \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

### Nsight Compute kernel metrics

```bash
mkdir -p profiles/ncu
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set basic \
  --launch-skip 20 \
  --launch-count 10 \
  --force-overwrite \
  -o profiles/ncu/mitre_sft_basic \
  accelerate launch --num_processes 1 06_mitre_sft.py --nvtx --max-steps 30
```

Copy reports back to local WSL:

```bash
scp -r ubuntu@132.226.76.207:~/Gemma-4-12B-it/profiles .
```

---

## 6. After training

Merge the LoRA adapter into the base model and export:

```bash
python 05_merge_and_export.py
```

Push to HuggingFace Hub (optional):

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./gemma4-mitre-sft')
model.push_to_hub('your-hf-username/gemma4-mitre-attack')
"
```

---

## 7. Terminate the Lambda instance when done

```bash
python lambda_gpu.py list
python lambda_gpu.py terminate <instance_id>
```

Or from the Lambda Cloud dashboard: https://cloud.lambda.ai/instances
