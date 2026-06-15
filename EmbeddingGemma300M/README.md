# synEmbeddingGemma

Embedding and reranker model fine-tuning and inference using [Unsloth](https://github.com/unslothai/unsloth) and [SentenceTransformers](https://github.com/huggingface/sentence-transformers).

## Models

| Model | Type | VRAM (QLoRA / LoRA) |
|---|---|---|
| `google/embeddinggemma-300m` | Embedding (decoder-based) | 3GB / 6GB |
| `Qwen/Qwen3-Embedding-0.6B` | Embedding (compact) | — |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding (encoder-only) | — |
| `BAAI/bge-reranker-v2-m3` | Reranker (cross-encoder) | — |

## Setup

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

### Fine-tuning with Unsloth

```python
from unsloth import FastSentenceTransformer

model = FastSentenceTransformer.from_pretrained("google/embeddinggemma-300m")

# ... SentenceTransformers training loop ...

model.save_pretrained("output/embeddinggemma-finetuned")        # LoRA adapters
model.save_pretrained_merged("output/embeddinggemma-merged")    # merged model
model.push_to_hub("your-hf-username/embeddinggemma-finetuned") # upload to Hub
```

> **Note:** When loading for inference, always pass `for_inference=True`:
> ```python
> model = FastSentenceTransformer.from_pretrained("...", for_inference=True)
> ```

## Fine-tuning Notebooks

| Model | Colab |
|---|---|
| EmbeddingGemma 300M | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb) |
| Qwen3-Embedding 0.6B | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(0_6B).ipynb) |
| All-MiniLM-L6-v2 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/All_MiniLM_L6_v2.ipynb) |

## References

- [Unsloth Embedding Fine-tuning Docs](https://unsloth.ai/docs/basics/embedding-finetuning)
- [SentenceTransformers Docs](https://www.sbert.net)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
