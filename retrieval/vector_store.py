"""FAISS-backed vector store for government document retrieval."""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    import faiss
except Exception:  # pragma: no cover - fallback if faiss is unavailable
    faiss = None

from retrieval.chunking import DocumentChunk
from utils.io import ensure_parent_dir, iter_jsonl, write_jsonl


@dataclass
class SearchResult:
    """A scored retrieval result."""

    score: float
    chunk: DocumentChunk


class FaissTextStore:
    """Store TF-IDF embeddings in a FAISS index with metadata persistence."""

    def __init__(self, vectorizer: TfidfVectorizer | None = None) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.chunks: List[DocumentChunk] = []

    @property
    def is_built(self) -> bool:
        """Return whether the index has been initialized."""
        return self.index is not None or self.embeddings is not None

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        """Fit the vectorizer and construct the retrieval index."""
        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]
        if not texts:
            raise ValueError("No chunks provided to build the vector store")
        matrix = self.vectorizer.fit_transform(texts)
        dense = matrix.astype(np.float32).toarray()
        dense = normalize(dense)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dense.shape[1])
            self.index.add(dense)
            self.embeddings = None
        else:
            self.index = None
            self.embeddings = dense

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the index for the most relevant document chunks."""
        if not self.chunks:
            raise ValueError("The vector store is empty")
        query_vector = self.vectorizer.transform([query]).astype(np.float32).toarray()
        query_vector = normalize(query_vector)
        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_vector, top_k)
            results: List[SearchResult] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                results.append(SearchResult(score=float(score), chunk=self.chunks[int(index)]))
            return results
        if self.embeddings is None:
            raise ValueError("Fallback embeddings are missing")
        scores = np.matmul(query_vector, self.embeddings.T)[0]
        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(score=float(scores[index]), chunk=self.chunks[int(index)]) for index in top_indices]

    def save(self, index_path: str | Path, meta_path: str | Path, vectorizer_path: str | Path | None = None) -> None:
        """Persist the vector store to disk."""
        index_file = Path(index_path)
        meta_file = Path(meta_path)
        vectorizer_file = Path(vectorizer_path) if vectorizer_path is not None else index_file.with_suffix(".pkl")
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(meta_file, [asdict(chunk) for chunk in self.chunks])
        with vectorizer_file.open("wb") as handle:
            pickle.dump(self.vectorizer, handle)
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(index_file))
        else:
            np.save(index_file.with_suffix(".npy"), self.embeddings)

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path, vectorizer_path: str | Path | None = None) -> "FaissTextStore":
        """Load a persisted vector store from disk."""
        index_file = Path(index_path)
        meta_file = Path(meta_path)
        vectorizer_file = Path(vectorizer_path) if vectorizer_path is not None else index_file.with_suffix(".pkl")
        store = cls()
        with vectorizer_file.open("rb") as handle:
            store.vectorizer = pickle.load(handle)
        store.chunks = [DocumentChunk(**record) for record in iter_jsonl(meta_file)]
        if faiss is not None and index_file.exists():
            store.index = faiss.read_index(str(index_file))
        else:
            npy_path = index_file.with_suffix(".npy")
            if not npy_path.exists():
                raise FileNotFoundError(f"Fallback embedding file not found: {npy_path}")
            store.embeddings = np.load(npy_path)
        return store
