# Fine-tuning Embedding Models with Unsloth Guide

Source: https://unsloth.ai/docs/basics/embedding-finetuning

Fine-tuning embedding models can largely improve retrieval and RAG performance on specific tasks. It aligns the model's vectors with your domain and the kind of 'similarity' that matters for your use case, which improves search, RAG, clustering, and recommendations on your data.

**Example:** The headlines "Google launches Pixel 10" and "Qwen releases Qwen3" might be embedded as similar if you're just labeling both as 'Tech,' but not similar if you're doing semantic search because they're about different things. Fine-tuning helps the model make the 'right' kind of similarity for your use case, reducing errors and improving results.

[Unsloth](https://github.com/unslothai/unsloth) now supports training embedding, **classifier**, **BERT**, **reranker** models **~1.8-3.3x faster** with 20% less memory and 2x longer context than other Flash Attention 2 implementations — no accuracy degradation. EmbeddingGemma-300M works on just **3GB VRAM**. You can use your trained **model anywhere**: transformers, LangChain, Ollama, vLLM, llama.cpp etc.

Unsloth uses [SentenceTransformers](https://github.com/huggingface/sentence-transformers) to support compatible models like Qwen3-Embedding, BERT and more. **Even if there's no notebook or upload, it's still supported.**

---

## Models Used in This Project

This project focuses on four models:

| Model | Type | Notebook |
|---|---|---|
| `google/embeddinggemma-300m` | Embedding (decoder-based) | [EmbeddingGemma (300M) Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb) |
| `BAAI/bge-reranker-v2-m3` | Reranker / cross-encoder | — |
| `Qwen/Qwen3-Embedding-0.6B` | Embedding (compact) | [Qwen3-Embedding 0.6B Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(0_6B).ipynb) |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding (compact) | [All-MiniLM-L6-v2 Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/All_MiniLM_L6_v2.ipynb) |

### google/embeddinggemma-300m

- Decoder-based embedding model (not encoder-only)
- Runs on **3GB VRAM** with QLoRA, 6GB VRAM with LoRA
- Trained via `FastSentenceTransformer` with LoRA/QLoRA

### BAAI/bge-reranker-v2-m3

- Cross-encoder reranker model
- Confirmed to train correctly under Unsloth's fallback path
- Used to re-rank retrieved candidates for improved RAG precision

### Qwen/Qwen3-Embedding-0.6B

- Compact decoder-based embedding model
- Strong multilingual and instruction-following embedding capability
- Good balance between size and retrieval quality

### sentence-transformers/all-MiniLM-L6-v2

- Lightweight encoder-only embedding model
- Fast and memory-efficient; ideal for domain-specific fine-tuning on custom data
- Produces compact sentence embeddings for semantic search, retrieval, and clustering

---

## Unsloth Features

- LoRA/QLoRA or full fine-tuning for embeddings, without needing to rewrite your pipeline
- Best support for encoder-only `SentenceTransformer` models (with a `modules.json`)
- Cross-encoder models are confirmed to train properly even under the fallback path
- Supports `transformers v5`

**Notes:**
- Limited support for models without `modules.json` (Unsloth will auto-assign default SentenceTransformers pooling modules).
- If you're doing something custom (custom heads, nonstandard pooling), double-check the pooled embedding behavior.
- Some models like MPNet or DistilBERT needed custom patches for gradient checkpointing.

---

## Fine-tuning Workflow

The fine-tuning flow is centered around `FastSentenceTransformer`.

### Save / Push Methods

```python
model.save_pretrained()           # save LoRA adapters to a local folder
model.save_pretrained_merged()    # save merged model to a local folder
model.push_to_hub()               # push LoRA adapters to Hugging Face
model.push_to_hub_merged()        # push merged model to Hugging Face
```

### Loading for Inference

> **Critical:** To load a model for inference using `FastSentenceTransformer`, you **must** pass `for_inference=True`.

`from_pretrained()` is similar to Unsloth's other fast classes, with one exception: inference requires the explicit flag.

```python
from unsloth import FastSentenceTransformer

# EmbeddingGemma-300M
model = FastSentenceTransformer.from_pretrained(
    "google/embeddinggemma-300m",
    for_inference=True,
)

# BGE Reranker v2 M3
model = FastSentenceTransformer.from_pretrained(
    "BAAI/bge-reranker-v2-m3",
    for_inference=True,
)

# Qwen3-Embedding 0.6B
model = FastSentenceTransformer.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B",
    for_inference=True,
)

# All-MiniLM-L6-v2
model = FastSentenceTransformer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    for_inference=True,
)
```

### Hugging Face Authentication

Run this once in the same virtualenv before calling hub methods — then `push_to_hub()` and `push_to_hub_merged()` do **not** require a token argument:

```bash
hf auth login
```

---

## Inference and Deploy Anywhere

Your fine-tuned Unsloth model works with all major tools with no vendor lock-in:
transformers, LangChain, Weaviate, sentence-transformers, Text Embeddings Inference (TEI), vLLM, llama.cpp, pgvector, FAISS / vector databases, and any RAG framework.

```python
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("<your-unsloth-finetuned-model>")

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

# 2. Encode via encode_query and encode_document to automatically use the right prompts
query_embedding    = model.encode_query(query)
document_embedding = model.encode_document(documents)
print(query_embedding.shape, document_embedding.shape)

# 3. Compute similarity via the built-in similarity helper
similarity = model.similarity(query_embedding, document_embedding)
print(similarity)
```

---

## Benchmarks

Unsloth is consistently **1.8–3.3x faster** across a wide variety of embedding models and sequence lengths (128 to 2048+).

| Fine-tuning mode | Speedup vs SentenceTransformers + FA2 |
|---|---|
| 4-bit QLoRA | 1.8x – 2.6x faster |
| 16-bit LoRA | 1.2x – 3.3x faster |

**VRAM requirements for EmbeddingGemma-300M:**
- QLoRA: 3GB VRAM
- LoRA: 6GB VRAM

---

## Supported Models (Reference)

This project uses four models. The full list of Unsloth-supported models for reference:

```
BAAI/bge-reranker-v2-m3                    ← used in this project
Qwen/Qwen3-Embedding-0.6B                  ← used in this project
google/embeddinggemma-300m                  ← used in this project
sentence-transformers/all-MiniLM-L6-v2     ← used in this project
```

Most [common SentenceTransformer models](https://huggingface.co/models?library=sentence-transformers) are already supported. For unsupported encoder-only models, open a [GitHub issue](https://github.com/unslothai/unsloth/issues).
