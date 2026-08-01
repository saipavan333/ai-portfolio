"""
rag_pipeline.py - PRODUCTION RAG: neural embeddings + a vector DB + an LLM.

    python -m src.rag_pipeline --docs data/policies

WHAT : ingests documents, embeds them with a real model, stores them in a vector index, and
       answers questions grounded in retrieved chunks (with citations).
WHY  : the same shape as rag_demo.py, but with neural embeddings and a scalable vector store -
       this is what you deploy.
HOW  : load -> chunk+embed -> index -> retrieve (optionally hybrid + re-rank) -> LLM answer.
WHERE: production layer; serve it via api.py (FastAPI) + the Dockerfile. Evaluate with RAGAS.

Requires: llama-index, llama-index-embeddings-huggingface, an LLM (open or API).
"""
from __future__ import annotations
import argparse


def build_engine(docs_dir: str, embed_model: str = "BAAI/bge-small-en-v1.5", top_k: int = 3):
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(embed_model)     # real neural embeddings
    documents = SimpleDirectoryReader(docs_dir).load_data()      # your PDFs/markdown
    index = VectorStoreIndex.from_documents(documents)           # chunk + embed + store
    return index.as_query_engine(similarity_top_k=top_k)         # retrieval + LLM answer


def evaluate_with_ragas(engine, qa_pairs):
    """Measure faithfulness + answer relevance (not vibes). qa_pairs: list of (question, ground_truth)."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    # build a dataset of {question, answer, contexts, ground_truth} from the engine, then:
    # return evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    raise NotImplementedError("wire up your eval dataset, then call ragas.evaluate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="folder of documents to index")
    args = ap.parse_args()
    engine = build_engine(args.docs)
    print(engine.query("How long do refunds take?"))   # grounded, cited answer
    # TODO: add hybrid search (BM25 + vectors) + a cross-encoder re-ranker; RAGAS eval.


if __name__ == "__main__":
    main()
