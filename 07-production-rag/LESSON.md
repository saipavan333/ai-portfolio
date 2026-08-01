# Lesson 07 - RAG: Retrieval-Augmented Generation

> Read with `src/rag_demo.py` open; run `notebooks/07_rag_from_scratch.ipynb` for the similarity
> and 2D-map diagrams. RAG is the most-deployed LLM pattern in production - and your data edge.

## 0. What you'll be able to do after this
- Explain why we bolt **retrieval** onto an LLM, and the chunk -> embed -> search -> generate flow.
- Run a working mini-RAG (`python -m src.rag_demo`) and read the production pipeline (`rag_pipeline.py`).

## 1. The big picture (why this project exists)
An LLM alone answers from fuzzy memory and **hallucinates**. RAG retrieves the *actual* relevant
text from your documents and asks the LLM to answer grounded in it - accurate, citable, updatable.
Crucially, RAG quality is mostly a **data/retrieval** problem (chunking, embeddings, ranking,
evaluation), which plays directly to data-engineering strengths.

## 2. Foundations from scratch (the basics)
- **Chunking:** split long documents into passage-sized pieces (with overlap so meaning isn't cut).
  `src/rag_demo.py::chunk` does this.
- **Embeddings:** turn each chunk into a vector so *similar meaning = nearby*. The demo uses TF-IDF
  vectors as a simple stand-in; production uses a neural embedding model.
- **Semantic search:** embed the question, find the chunks with the highest **cosine similarity**.
  The demo retrieves the refund passage for a refund question (asserted).
- **Augmented prompt:** paste the retrieved chunks + the question into the LLM prompt, instruct it
  to answer **only** from that context and cite sources. `build_prompt` shows the exact text.
- **Make it good:** hybrid search (keyword BM25 + vectors) + a cross-encoder **re-ranker** for
  precision, and **RAGAS** to measure faithfulness/relevance (not vibes).

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                       (production - rag_pipeline.py)
  docs -> chunk -> TF-IDF vectors -> cosine search     docs -> chunk -> NEURAL embeddings -> vector DB
              (rag_demo.py)            |                              |
                                       v                              v
                          augmented prompt (context+question)   retrieve top-k (+rerank) -> LLM answer
                                       |                              |
                                   (an LLM answers)         FastAPI (api.py) + Docker; RAGAS eval
```

## 4. Code reading order
1. `src/rag_demo.py` - `TinyRAG` (chunk, retrieve, build_prompt) - run it first.
2. `notebooks/07_rag_from_scratch.ipynb` - similarity bar + 2D knowledge map.
3. `src/rag_pipeline.py` - production ingest/index/query with neural embeddings + a vector store.
4. `src/api.py` + `Dockerfile` - serve it; then add hybrid search + RAGAS evaluation.

## 5. Common mistakes (and how to avoid them)
- **Chunks too big** -> noisy context; **too small** -> lost meaning. Tune size + overlap.
- **Vector-only retrieval** -> misses exact keywords; add BM25 (hybrid) + a re-ranker.
- **No evaluation** -> you can't tell if a change helped; measure faithfulness with RAGAS.
- **No "I don't know" path** -> a good RAG system declines when the answer isn't retrieved.

## 6. Check your understanding
1. Why does retrieval reduce hallucination compared to asking the LLM directly?
2. What does chunk overlap protect against?
3. Why combine keyword (BM25) and vector search instead of vectors alone?
4. What does RAGAS' "faithfulness" measure?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
RAG, chunk, embedding, semantic search, cosine similarity, vector database, augmented prompt,
hallucination, hybrid search, re-ranking, RAGAS.

## 8. Going deeper
- LlamaIndex docs: https://docs.llamaindex.ai/
- RAGAS evaluation: https://docs.ragas.io/
