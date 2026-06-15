from sentence_transformers import SentenceTransformer, CrossEncoder

query = "Which planet is known as the Red Planet?"
documents = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Jupiter, the largest planet, has a prominent red spot.",
    "Saturn is famous for its rings.",
]

# ── Embedding models ──────────────────────────────────────────────────────────

EMBEDDING_MODELS = [
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
]

for model_id in EMBEDDING_MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    model = SentenceTransformer(model_id)

    query_emb = model.encode(query, prompt_name="query") if hasattr(model, "prompts") and "query" in (model.prompts or {}) else model.encode(query)
    doc_embs  = model.encode(documents)

    scores = model.similarity(query_emb, doc_embs)[0]
    ranked = sorted(zip(scores.tolist(), documents), reverse=True)

    print(f"Query: {query}")
    for score, doc in ranked:
        print(f"  {score:.4f}  {doc}")

# ── Reranker model ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Model: BAAI/bge-reranker-v2-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

pairs  = [(query, doc) for doc in documents]
scores = reranker.predict(pairs)
ranked = sorted(zip(scores.tolist(), documents), reverse=True)

print(f"Query: {query}")
for score, doc in ranked:
    print(f"  {score:.4f}  {doc}")
