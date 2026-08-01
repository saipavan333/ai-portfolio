"""
rag_demo.py - a working mini-RAG in pure scikit-learn (runnable, no GPU, no API key).

    python -m src.rag_demo

WHAT : builds a tiny knowledge base, retrieves the most relevant passage for a question with
       TF-IDF cosine similarity, and assembles the 'augmented prompt' an LLM would answer from.
WHY  : RAG quality is mostly a DATA/retrieval problem (chunking, embeddings, ranking) - a data-engineering
       strength. This demo makes the retrieval step concrete and verifiable.
HOW  : chunk -> embed (TF-IDF) -> cosine-similarity search -> build a cited prompt.
WHERE: the concept layer. rag_pipeline.py is production (neural embeddings + a vector DB + an LLM).
       See notebooks/07_rag_from_scratch.ipynb for the similarity + 2D-map diagrams.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = [
    "Refunds are processed within 5 to 7 business days after we receive the returned item.",
    "Standard shipping takes 3 to 5 business days; express shipping arrives next business day.",
    "Our products come with a 2 year warranty covering manufacturing defects.",
    "You can cancel your subscription anytime from the account settings page.",
    "Customer support is available 24/7 via live chat and email.",
]


def chunk(text: str, size: int = 12, overlap: int = 4):
    """Split long text into overlapping word windows (overlap keeps cross-boundary meaning)."""
    words, out, i = text.split(), [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


class TinyRAG:
    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer().fit(docs)      # 'embeddings' = TF-IDF vectors
        self.matrix = self.vec.transform(docs)      # the search index

    def retrieve(self, query: str, k: int = 2):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i]), self.docs[i]) for i in top]

    def build_prompt(self, query: str, k: int = 2) -> str:
        hits = self.retrieve(query, k)
        context = "\n".join(f"[{i}] {text}" for i, _, text in hits)
        return (f"Answer using ONLY the context, cite sources like [0]. If unknown, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")


def main() -> None:
    rag = TinyRAG(DOCS)
    q = "How long does it take to get my money back after a return?"
    hits = rag.retrieve(q)
    print("Question:", q)
    for i, score, text in hits:
        print(f"  retrieved [{i}] (score {score:.2f}): {text}")
    # the refund passage (doc 0) must be the top hit
    assert hits[0][0] == 0, "RAG should retrieve the refund passage for a refund question"
    print("\n--- augmented prompt the LLM would answer ---\n")
    print(rag.build_prompt(q))
    print("\n[rag] OK - retrieval returned the correct passage")


if __name__ == "__main__":
    main()
