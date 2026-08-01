"""FastAPI service exposing the RAG pipeline (containerize with the Dockerfile)."""
from fastapi import FastAPI

app = FastAPI(title="RAG service")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from src.rag_pipeline import build_engine
        _engine = build_engine("data/policies")
    return _engine


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
def query(q: str):
    return {"answer": str(get_engine().query(q))}
