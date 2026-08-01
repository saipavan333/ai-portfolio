# Project 07 - Production RAG System (Retrieval-Augmented Generation)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Build a question-answering system over your own documents: chunk, embed, store in a vector DB, retrieve, and generate grounded answers with citations.
**Phase 4: LLMs, RAG & Fine-tuning**  |  **Difficulty:** Intermediate-Advanced  |  **Est. time:** Week 11-13 (~45 hrs)

---

## Why (the point of this project)
RAG is THE most-deployed LLM pattern in production in 2026 - 'roles mentioning LLM or RAG grew 340%.' It is also where data-engineering skills transfer directly here: RAG quality is mostly a data/retrieval problem (chunking, embeddings, indexing, evaluation), not a model problem. It's arguably one of the most employable projects here.

## What you will build
A RAG service that ingests a document corpus (PDFs/markdown/web), builds a vector index, retrieves relevant chunks for a query, and uses an LLM to answer WITH citations - exposed via a FastAPI endpoint and a Gradio chat UI, with a retrieval-quality evaluation harness.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Embeddings & semantic search** | Map text to vectors so similar meaning -> nearby vectors; retrieve by nearest-neighbor instead of keywords. |
| **Chunking** | Split documents into passages sized for retrieval; chunk strategy hugely affects quality (careful parsing matters). |
| **Vector databases** | Stores + indexes embeddings for fast similarity search (FAISS, Chroma, Qdrant, pgvector). |
| **Retrieval-augmented generation** | Retrieve relevant context, stuff it into the prompt, and have the LLM answer grounded in it - reduces hallucination and adds citations. |
| **Hybrid search & re-ranking** | Combine keyword (BM25) + vector search, then re-rank with a cross-encoder for precision. |
| **RAG evaluation** | Measure faithfulness, answer relevance, and context precision/recall (e.g. RAGAS) - not vibes. |

## How - step by step
1. Choose a corpus (a set of course notes, a docs set, or public PDFs). Build a robust ingestion/parsing pipeline.
2. Chunk thoughtfully (size + overlap + metadata); embed with a strong open embedding model.
3. Index in a vector DB (start with Chroma/FAISS; optionally pgvector to show DB skill).
4. Implement retrieval -> prompt assembly -> LLM answer with inline citations to source chunks.
5. Add hybrid search (BM25 + vectors) and a cross-encoder re-ranker; measure the lift.
6. Build an evaluation set of Q/A pairs; score faithfulness + relevance with RAGAS; log to W&B.
7. Expose a FastAPI `/query` endpoint and a Gradio chat UI; containerize with Docker.
8. Document failure modes (missing context, contradictory sources) and your mitigations.

## Tech stack
LangChain/LlamaIndex (or hand-rolled), FAISS/Chroma/pgvector, sentence-transformers, an LLM (open or API), RAGAS, FastAPI, Gradio, Docker, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | Embedding + generation models; optional Space for the chat UI. |
| **GitHub** | Service code, eval harness, Dockerfile, architecture diagram. |
| **Docker** | Containerize the API + vector DB (compose) - an ops strength on display. |
| **Weights & Biases** | Log retrieval/eval metrics across chunking & retriever configs. |

## Portfolio artifact
GitHub repo + live demo (Space or deployed container) + W&B eval report. Resume line: 'Built and evaluated a production RAG system (hybrid retrieval + re-ranking) with a RAGAS-measured faithfulness of X.XX.'

## Definition of Done
- [ ] Answers are grounded with citations to retrieved source chunks.
- [ ] Hybrid search + re-ranking measurably beats naive vector-only retrieval.
- [ ] RAGAS (or equivalent) scores logged for at least 2 configurations.
- [ ] FastAPI endpoint + chat UI both work; whole thing runs via docker compose.
- [ ] README has an architecture diagram and an honest failure-mode section.

## Stretch goals
- Add query rewriting / multi-query retrieval.
- Add streaming responses and conversation memory.
- Swap in a 2026 small open model (e.g. Qwen3.6-class) for fully local RAG and compare cost/quality.

## Resources
- [LlamaIndex docs](https://docs.llamaindex.ai/)
- [RAGAS evaluation](https://docs.ragas.io/)
- [Pinecone RAG learning hub](https://www.pinecone.io/learn/)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
