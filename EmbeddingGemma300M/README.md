# synEmbeddingGemma

Embedding and reranker model fine-tuning and inference using [Unsloth](https://github.com/unslothai/unsloth) and [SentenceTransformers](https://github.com/huggingface/sentence-transformers).

## Quick Start (no GPU needed)

Just want to see it run? You only need Python and one package:

```bash
pip install sentence-transformers
python demo.py
```

The demo downloads the models automatically on first run and works on CPU. No CUDA, no Unsloth, no extra setup.

---

## Models

| Model | Type | Parameters | VRAM (QLoRA 4-bit) | VRAM (LoRA 16-bit) | Notes |
|---|---|---:|---:|---:|---|
| `google/embeddinggemma-300m` | Embedding (decoder-based) | 300M | **3 GB** | **6 GB** | Official Unsloth benchmark [[1]](#references) |
| `Qwen/Qwen3-Embedding-0.6B` | Embedding | 600M | **6–8 GB** | **10–14 GB** | Supports 32k context, multilingual retrieval [[2]](#references) |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding (encoder-only) | 22M | **<2 GB** | **2–3 GB** | Extremely lightweight; ideal for rapid experiments [[1]](#references) |
| `BAAI/bge-reranker-v2-m3` | Reranker (cross-encoder) | ~568M | **8–10 GB** | **14–18 GB** | Cross-encoders are more expensive than embedding models [[1]](#references) |

## Setup

### WSL / Linux

```bash
chmod +x install.sh
bash install.sh
```

This creates a `.venv` with Python 3.11 and installs all dependencies (Unsloth, SentenceTransformers, TRL, etc.).

Then activate the environment and run the demo:

```bash
source .venv/bin/activate
python demo.py
```

### Manual

```bash
pip install sentence-transformers unsloth
```

## Usage

### Run the demo

```bash
python demo.py
```

The demo encodes a query against a set of documents, ranks them by similarity for each embedding model, then re-ranks them using the cross-encoder reranker.

### Embedding models

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

query_emb = model.encode("your query")
doc_embs  = model.encode(["doc 1", "doc 2", "doc 3"])
scores    = model.similarity(query_emb, doc_embs)
```

### Reranker model

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores   = reranker.predict([("your query", doc) for doc in documents])
```

### Fine-tuning with Unsloth (`finetune.py`)

| Step | What happens |
|---:|---|
| 1 | Load model via `FastSentenceTransformer` |
| 2 | Apply LoRA (`r=16`) to attention projections |
| 3 | Load `natural-questions-hard-negatives` dataset |
| 4 | Train with `MultipleNegativesRankingLoss` |
| 5 | Save LoRA adapters → `output/lora-adapters/` |
| 6 | Merge adapters into base model → `output/merged-model/` |
| 7 | (Optional) push both to Hugging Face Hub |
| 8 | Load merged model into vLLM with `task="embed"` |
| 9 | Run cosine similarity on query / document pairs |

```bash
python finetune.py
```

To switch models or tune hyperparameters, edit the `Config` dataclass at the top of `finetune.py`. To push to the Hub, set `hub_repo = "your-username/model-name"` and run `hf auth login` first.

```python
from unsloth import FastSentenceTransformer

model = FastSentenceTransformer.from_pretrained("google/embeddinggemma-300m")

# ... SentenceTransformers training loop ...

model.save_pretrained("output/lora-adapters")       # LoRA adapters
model.save_pretrained_merged("output/merged-model") # merged model
model.push_to_hub("your-hf-username/model-name")    # upload to Hub
```

> **Note:** When loading for inference, always pass `for_inference=True`:
> ```python
> model = FastSentenceTransformer.from_pretrained("...", for_inference=True)
> ```

## References

- \[1\] [Unsloth Embedding Fine-tuning Docs](https://unsloth.ai/docs/basics/embedding-finetuning)
- \[2\] [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (arXiv:2506.05176)](https://arxiv.org/abs/2506.05176)
- [SentenceTransformers Docs](https://www.sbert.net)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
