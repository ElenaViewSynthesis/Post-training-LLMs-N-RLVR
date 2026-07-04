# Post-training LLMs & RLVR

A research monorepo of independent **LLM post-training** experiments. Each top-level
folder is a self-contained project — its own `uv`/`pyproject.toml` environment, its own
scripts, and its own operational docs — covering the full spectrum from **synthetic data
generation** through **SFT / DPO / GRPO / RLVR** to **merging, serving, and GPU profiling**.

Two families live here:

- **Generative post-training** (`Cerebras-Gemma4-31B-preview`, `Gemma-4-12B-it`) — a
  numbered-stage pipeline: inference → SFT (LoRA/QLoRA) → preference/RL (DPO / GRPO / RLVR)
  → merge & export → serve.
- **Embedding fine-tuning** (`EmbeddingGemma300M`, `Qwen3Embedding4B`) — a lighter
  `finetune → train → training_logs` loop over financial-QA datasets.

## Projects at a glance

| Project | Model | Focus | Pipeline / entry points |
|---|---|---|---|
| **[Cerebras-Gemma4-31B-preview](Cerebras-Gemma4-31B-preview/)** | Gemma-4-31B (Early Preview) | Full post-training + a large **synthetic-data augmentation** sub-pipeline, teacher served on **Cerebras Inference** | `01_inference` → `02_sft_lora` → `03_grpo_rlvr` → `04_merge_and_export`; plus `dataset-augmentation/` and `gemma-reasoning-endpoint/` (SageMaker/TGI) |
| **[Gemma-4-12B-it](Gemma-4-12B-it/)** | Gemma-4-12B-IT | Instruction-tuned model fine-tuned on **MITRE ATT&CK** for defensive security analysis, run on Lambda GPUs with Nsight profiling | `01_inference` → `02_sft_lora` → `03_dpo` → `04_grpo` → `05_merge_and_export` → `06_mitre_sft`; `serve_mitre_endpoint.py` |
| **[EmbeddingGemma300M](EmbeddingGemma300M/)** | EmbeddingGemma-300M | Embedding + reranker fine-tune on **FinQA** via Unsloth + SentenceTransformers | `finetune.py` / `train.py` / `demo.py` |
| **[Qwen3Embedding4B](Qwen3Embedding4B/)** | Qwen3-Embedding-4B | Embedding fine-tune on **ConvFinQA** (`grasson/t2-ragbench`) via Unsloth + LoRA | `finetune.py` / `train.py` |

## Conventions

- **Per-project environments.** Each folder has its own `pyproject.toml` (some with
  `uv.lock`, `.python-version`, `install.sh`). Scope your `uv run` to the project you're in;
  the augmentation sub-pipeline is even scoped separately from its parent so the heavy
  training stack (torch/transformers/trl) isn't pulled in for pure API generation.
- **Numbered stages.** In the generative projects, `NN_*.py` files run in order and map
  onto the post-training stages.
- **Docs live beside code.** Operational notes — GPU cost, quota, deploy, troubleshooting,
  profiling — are `.md` files next to the scripts they describe rather than centralized.
- **Secrets via `.env`.** Projects ship a `.env.example`; copy to `.env` and fill in keys
  (`CEREBRAS_API_KEY`, `HF_TOKEN`, `LAMBDA_API_KEY`, W&B, etc.). `.env` is git-ignored.

---

## Cerebras-Gemma4-31B-preview — deep dive

Post-training **Gemma-4-31B Early Preview** with the teacher/inference path served through
the **Cerebras Inference API** (OpenAI-compatible `cerebras-cloud-sdk`), which delivers
100k+ tokens/sec on wafer-scale hardware — fast enough to make large-scale synthetic
generation practical.

### Post-training stages (project root)

| Stage | File | What it does |
|---|---|---|
| 1 | `01_inference.py` | Fast hosted inference via the Cerebras Cloud API — the reference path for sampling the model. |
| 2 | `02_sft_lora.py` | Supervised fine-tuning with **QLoRA (4-bit NF4)**, sized to fit the 31B model on a single A100-80GB or 2× A100-40GB. |
| 3 | `03_grpo_rlvr.py` | **GRPO / RLVR** post-training. GRPO (the RL method behind DeepSeek-R1) scores *groups* of sampled completions relative to each other — no critic/value network. **RLVR** pairs it with **verifiable reward functions** (math, code, structured-output correctness) rather than a learned reward model. |
| 4 | `04_merge_and_export.py` | Merges the LoRA/GRPO adapter into the base weights, producing a stand-alone checkpoint to push to the Hub or serve directly. |

Supporting docs: `cerebras-pricing.md`, `cerebras-gemma4-31b-quota.md`, `GPU-cost-estimates.md`.

### `dataset-augmentation/` — OpenThoughts-Agent-SFT 100K → 250K

A standalone, resume-safe data-engineering pipeline that expands
**OpenThoughts-Agent-SFT-100K** to **~250K traces** by regenerating the `conversations`
column with Gemma-4-31B on Cerebras. The **task distribution is preserved** — the same 100K
tasks receive new synthetic trajectories, not new tasks.

**Core idea — structured trajectory synthesis (not agent execution).** Instead of running a
live sandboxed agent loop (the original dataset used the Terminus-2 harness), Gemma-4-31B
generates a *complete* multi-turn conversation — reasoning, tool calls, observations, final
answer — in a **single inference call** conditioned on the task and an augmentation strategy.
This runs at Cerebras throughput (10–100× cheaper than container-orchestrated agent
execution) and is sufficient supervision signal for SFT; quality is enforced *after*
generation by the Stage-5 validator.

| Stage | File | Runtime | Role |
|---|---|---|---|
| 0–1 | `extract_tasks.py` | Dask | Stream dataset from HF, introspect the real schema, split *task* vs. *trajectory* columns. |
| 2 | `plan_variants.py` | pandas | Build the diversity plan — temperature `[0.7, 0.85, 1.0]` (weighted toward 0.85) × instruction framing; `OVERSAMPLE_FACTOR ≈ 1.5×`. |
| 3 | `gemma4_31b_agent.py` | async API | Structured trajectory synthesis via Cerebras; async, checkpointed, resume-safe workers streaming results as they finish. |
| 4 | `poll_n_fetch.py` | — | Obsolete on the Cerebras path (no batch to poll); the Stage-3 worker is itself resume-safe. |
| 5 | `validate_n_dedup.py` | Dask | JSON-schema + structural checks, near-dup filter, degenerate-output/refusal regex, diversity scoring, metadata enrichment. |
| 6 | `augment_150k_rows.py` | Dask | Reshape to original schema, concatenate original 100K + ~150K accepted synthetic rows, write sharded parquet. |

**Quality gate (Stage 5):** schema validation → structural/truncation checks → fingerprint
near-dup filter → global dedup → safety/PII + refusal regex → reasoning-quality (length,
coherent tool-call/observation structure) → metadata (`is_synthetic_augmentation`,
`source_task_id`, `variant_id`).

**Output:** synced via `hf sync` (with `HF_HUB_ENABLE_HF_TRANSFER=1`, GCP EU CDN) to the
public HF bucket
[`borntobeignored/OpenThoughts-Agents-SFT-250k`](https://huggingface.co/buckets/borntobeignored/OpenThoughts-Agents-SFT-250k).

**Pilot before full scale.** Run stages 2–5 on 500–1,000 tasks first to calibrate the
rejection rate (→ `OVERSAMPLE_FACTOR`), degenerate-output rate (→ regex filter), the best
temperature weights, and synthesis-prompt quality before committing the full ~150K request
budget. See `README.md`, `CLAUDE.md`, and `design_notes.md` in that folder.

### `gemma-reasoning-endpoint/`

Deploys a reasoning-configured Gemma endpoint on **SageMaker** via a **TGI** container
pulled from ECR — `deploy_sagemaker.py`, `check_quotas.py`, `auto_terminate.py`,
`test_reasoning.py`, plus deploy/monitoring/parameter docs.

---

## Gemma-4-12B-it — MITRE ATT&CK set

Fine-tunes **`google/gemma-4-12b-it`** into a **defensive-security analyst** for
**MITRE ATT&CK** techniques, then serves it behind a FastAPI endpoint. Everything runs on
**Lambda Cloud** GPUs (GH200 / A100), with **W&B** monitoring and **Nsight** profiling.

### The dataset (`mitre-attack.md` / `06_mitre_sft.py`)

`06_mitre_sft.py` downloads the **MITRE ATT&CK Enterprise STIX bundle**
(`mitre-attack/attack-stix-data → enterprise-attack.json`, cached under `./data/mitre/`) and
converts STIX objects into instruction Q&A pairs with HuggingFace **TRL**. Revoked and
deprecated objects are skipped.

| STIX type | Training examples generated |
|---|---|
| `attack-pattern` (technique) | Explain the technique · Detect it · Structured JSON summary · Mitigations |
| `intrusion-set` (group) | Describe the group · List techniques it uses |
| `malware` / `tool` | Describe the software |
| `x-mitre-tactic` | Describe the tactic |

### Training configuration

| Parameter | Value |
|---|---|
| Base model | `google/gemma-4-12b-it` |
| Method | **QLoRA** (4-bit NF4) — `--no-qlora` for full-precision LoRA |
| LoRA rank / alpha | 16 / 32 |
| Target modules | q/k/v/o proj + gate/up/down proj |
| Epochs | 3 |
| Batch size | 1 (× 8 gradient-accumulation steps) |
| Learning rate | 2e-4, cosine schedule |
| Max seq length | 2048 |
| Optimizer | `paged_adamw_8bit` |
| Output adapter | `./gemma4-mitre-sft/` |

A full run is **1,185 steps**. On a Lambda **GH200** (ARM64, 96 GB unified) it completes in
**~57 min (~$2.29)** vs. **~1h37m (~$3.22)** on an A100 40 GB SXM4 — the unified memory
removes the gradient-checkpointing pressure that squeezes the A100. See `README.md` for the
full runtime benchmark and W&B curves (loss ~4.5 → ~0.4, token accuracy ~0.4 → ~0.85).

### Running it

```bash
# QLoRA fine-tune (recommended; fits A100 40 GB)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 06_mitre_sft.py

# With Flash Attention 2 (preferred once installed)
accelerate launch 06_mitre_sft.py --attn-implementation flash_attention_2
```

### Serving — `serve_mitre_endpoint.py`

Loads the base model in 4-bit QLoRA and attaches the `./gemma4-mitre-sft` PEFT adapter,
exposing three routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Browser UI — question textarea with a prompt guide |
| `/analyze` | POST | JSON inference endpoint (`question`, `max_new_tokens`) |
| `/health` | GET | Model-load status + CUDA availability |

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn serve_mitre_endpoint:app --host 0.0.0.0 --port 8000
```

### Sample outputs — `sample_responses/`

Real endpoint outputs demonstrating what the fine-tuned model produces:

| File | Technique | Shape |
|---|---|---|
| `T1059_command_and_scripting_interpreter.md` | T1059 Command & Scripting Interpreter | Prose analysis (`max_new_tokens: 700`) — attacker behavior, detection ideas, mitigations |
| `T1059_structured_json.md` | T1059 (same technique) | **Structured JSON** — tactic, platforms, abuse patterns, detection logic, telemetry (Sysmon/EID/AMSI), data sources, mitigations, Atomic Red Team test cases |
| `T1078_valid_accounts.md` | T1078 Valid Accounts | Prose analysis |
| `T1566_phishing.md` | T1566 Phishing | Prose analysis |

The structured-JSON path is the headline capability: given a technique ID the model returns a
complete, schema-consistent ATT&CK record (tactic, platforms, detection logic, telemetry,
mitigations, and defensive test cases) suitable for feeding SIEM/EDR detection workflows.

### Profiling & ops docs

`GPUprofiler.md`, `GPU_util_improve.md`, `mitre-attack.md`, `lambdaGB200_connect.md`,
`troubleshooting.md`, and `scripts/setup_lambda_nsight.sh` (handles the ARM64 `libbpf`/`libssh`
symbol issues for Nsight Systems / Nsight Compute on GH200).
