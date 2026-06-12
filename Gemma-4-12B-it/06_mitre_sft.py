"""
Gemma 4 — SFT on MITRE ATT&CK Enterprise data
Loads enterprise-attack.json from the mitre-attack/attack-stix-data GitHub repo,
converts STIX objects into instruction-following Q&A pairs, then fine-tunes
Gemma 4 12B with QLoRA.

Run:
    accelerate launch 06_mitre_sft.py
    accelerate launch 06_mitre_sft.py --no-qlora   # full-precision LoRA

Dependencies:
    pip install transformers datasets peft trl bitsandbytes accelerate requests python-dotenv
"""

import argparse
import json
import os
import random
from pathlib import Path

import requests
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_ID", "google/gemma-4-12B-it")
HF_TOKEN   = os.getenv("HF_TOKEN")
OUTPUT_DIR = "./gemma4-mitre-sft"
CACHE_PATH = Path("./data/mitre/enterprise-attack.json")
STIX_URL   = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
    "/master/enterprise-attack/enterprise-attack.json"
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

SYSTEM_PROMPT = (
    "You are a cybersecurity expert with deep knowledge of the MITRE ATT&CK framework. "
    "Answer questions about adversary tactics, techniques, threat groups, malware, "
    "and mitigations accurately and concisely."
)


# ── Download / cache STIX bundle ─────────────────────────────────────────────

def load_stix() -> dict:
    if not CACHE_PATH.exists():
        print(f"Downloading enterprise-attack.json from GitHub ...")
        r = requests.get(STIX_URL, timeout=120)
        r.raise_for_status()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_bytes(r.content)
        print(f"Saved to {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1_000_000:.1f} MB)")
    else:
        print(f"Using cached {CACHE_PATH}")
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


# ── STIX parsing helpers ──────────────────────────────────────────────────────

def index_objects(bundle: dict) -> dict:
    """Return a dict keyed by STIX id for fast relationship lookups."""
    return {obj["id"]: obj for obj in bundle["objects"] if "id" in obj}


def get_attack_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def tactic_names(obj: dict) -> list[str]:
    return [p["phase_name"].replace("-", " ").title()
            for p in obj.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"]


def parse_techniques(bundle: dict) -> list[dict]:
    """Extract structured records from all non-revoked attack-pattern objects."""
    records = []
    for obj in bundle["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        attack_id = get_attack_id(obj)
        if not attack_id:
            continue
        records.append({
            "attack_id":      attack_id,
            "name":           obj.get("name"),
            "description":    obj.get("description", ""),
            "tactics":        [p["phase_name"] for p in obj.get("kill_chain_phases", [])
                               if p.get("kill_chain_name") == "mitre-attack"],
            "platforms":      obj.get("x_mitre_platforms", []),
            "data_sources":   obj.get("x_mitre_data_sources", []),
            "detection":      obj.get("x_mitre_detection", ""),
            "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
            "_stix_id":       obj["id"],
        })
    print(f"Parsed {len(records)} techniques from bundle "
          f"({sum(1 for r in records if r['is_subtechnique'])} sub-techniques).")
    return records


def relationships_for(stix_id: str, rels: list[dict], direction: str = "source") -> list[dict]:
    key = "source_ref" if direction == "source" else "target_ref"
    return [r for r in rels if r.get(key) == stix_id]


def clean(text: str) -> str:
    return (text or "").strip()


# ── Example generators ────────────────────────────────────────────────────────

def examples_from_technique(rec: dict, idx: dict, rels: list[dict]) -> list[dict]:
    tid    = rec["attack_id"]
    name   = rec["name"] or ""
    desc   = clean(rec["description"])
    plats  = ", ".join(rec["platforms"]) or "various platforms"
    tacs   = ", ".join(t.replace("-", " ").title() for t in rec["tactics"]) or "unknown tactic"
    detect = clean(rec["detection"])
    sources = rec["data_sources"]

    if not desc:
        return []

    samples = []
    stix_id  = rec["_stix_id"]
    mit_rels = [r for r in rels
                if r.get("relationship_type") == "mitigates"
                and r.get("target_ref") == stix_id]
    mits = [idx[r["source_ref"]].get("name", "")
            for r in mit_rels if r.get("source_ref") in idx]

    # basic explanation
    samples.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Explain the ATT&CK technique {name} ({tid})."},
            {"role": "assistant", "content": f"{desc}\n\n**Tactic(s):** {tacs}\n**Platforms:** {plats}"},
        ]
    })

    # detection
    if detect:
        src_note = "\n**Data sources:** " + ", ".join(sources) if sources else ""
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"How can defenders detect {name} ({tid})?"},
                {"role": "assistant", "content": detect + src_note},
            ]
        })

    # structured JSON summary
    structured = {
        "tactic": tacs,
        "technique_id": tid,
        "technique_name": name,
        "is_subtechnique": rec["is_subtechnique"],
        "platforms": rec["platforms"],
        "summary": desc,
        "recommended_detection": detect,
        "data_sources": sources,
        "recommended_mitigation": mits,
    }
    samples.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Return a structured JSON summary for ATT&CK technique {name} ({tid})."},
            {"role": "assistant", "content": json.dumps(structured, ensure_ascii=False, indent=2)},
        ]
    })

    # mitigations via relationships
    if mits:
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"What mitigations apply to {name} ({tid})?"},
                {"role": "assistant", "content": "\n".join(f"- {m}" for m in mits)},
            ]
        })

    return samples


def examples_from_group(obj: dict, idx: dict, rels: list[dict]) -> list[dict]:
    name  = obj.get("name", "")
    desc  = clean(obj.get("description", ""))
    aliases = ", ".join(obj.get("aliases", [])) or "none known"

    if not desc:
        return []

    samples = []

    samples.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Who is the threat group {name}?"},
            {"role": "assistant", "content": f"{desc}\n\n**Also known as:** {aliases}"},
        ]
    })

    # techniques used by this group
    use_rels = [r for r in rels
                if r.get("relationship_type") == "uses"
                and r.get("source_ref") == obj["id"]]
    techs = [idx[r["target_ref"]].get("name", "")
             for r in use_rels
             if r.get("target_ref") in idx
             and idx[r["target_ref"]].get("type") == "attack-pattern"]
    if techs:
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"What ATT&CK techniques does {name} use?"},
                {"role": "assistant", "content": "\n".join(f"- {t}" for t in techs[:30])},
            ]
        })

    return samples


def examples_from_software(obj: dict) -> list[dict]:
    name  = obj.get("name", "")
    desc  = clean(obj.get("description", ""))
    kind  = "malware" if obj["type"] == "malware" else "tool"
    plats = ", ".join(obj.get("x_mitre_platforms", [])) or "various platforms"

    if not desc:
        return []

    return [{
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"What is the {kind} known as {name}?"},
            {"role": "assistant", "content": f"{desc}\n\n**Platforms:** {plats}"},
        ]
    }]


def examples_from_tactic(obj: dict) -> list[dict]:
    name     = obj.get("name", "")
    desc     = clean(obj.get("description", ""))
    shortname = obj.get("x_mitre_shortname", "")

    if not desc:
        return []

    return [{
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Describe the ATT&CK tactic: {name}."},
            {"role": "assistant", "content": desc},
        ]
    }]


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset(bundle: dict) -> Dataset:
    idx        = index_objects(bundle)
    rels       = [o for o in bundle["objects"] if o.get("type") == "relationship"]
    techniques = parse_techniques(bundle)

    # technique examples from structured records
    examples = []
    for rec in techniques:
        examples.extend(examples_from_technique(rec, idx, rels))

    # group / software / tactic examples from raw bundle
    for obj in bundle["objects"]:
        t = obj.get("type")
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if t == "intrusion-set":
            examples.extend(examples_from_group(obj, idx, rels))
        elif t in ("malware", "tool"):
            examples.extend(examples_from_software(obj))
        elif t == "x-mitre-tactic":
            examples.extend(examples_from_tactic(obj))

    random.seed(42)
    random.shuffle(examples)
    print(f"Built {len(examples)} training examples from STIX bundle.")
    return Dataset.from_list(examples)


# ── Format for SFT ───────────────────────────────────────────────────────────

def format_sample(sample, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qlora", action="store_true", help="Use full-precision LoRA")
    parser.add_argument(
        "--attn-implementation",
        default=os.getenv("ATTN_IMPLEMENTATION"),
        help="Optional Transformers attention implementation, e.g. flash_attention_2 if flash-attn is installed.",
    )
    args = parser.parse_args()
    use_qlora = not args.no_qlora

    # dataset
    bundle  = load_stix()
    dataset = build_dataset(bundle)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    dataset   = dataset.map(lambda s: format_sample(s, tokenizer))

    # quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if use_qlora else None

    # model
    model_kwargs = {
        "token": HF_TOKEN,
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.enable_input_require_grads()
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # training
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name="gemma4-mitre-sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
