"""Government document retrieval helpers and CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from retrieval.chunking import build_chunks
from retrieval.vector_store import FaissTextStore, SearchResult
from utils.config import load_config
from utils.io import iter_jsonl
from utils.lang import normalize_language_code
from utils.text import clean_text


class GovernmentRetriever:
    """High-level retrieval wrapper for government and legal documents."""

    def __init__(self, store: FaissTextStore) -> None:
        self.store = store

    @classmethod
    def from_documents(cls, documents: List[dict], chunk_size: int = 80, overlap: int = 20) -> "GovernmentRetriever":
        """Create a retriever from raw government document records."""
        chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        store = FaissTextStore()
        store.build(chunks)
        return cls(store)

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path, vectorizer_path: str | Path | None = None) -> "GovernmentRetriever":
        """Load a retriever from saved artifacts."""
        return cls(FaissTextStore.load(index_path, meta_path, vectorizer_path=vectorizer_path))

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Return the most relevant document chunks for a query."""
        return self.store.search(clean_text(query), top_k=top_k)

    def save(self, index_path: str | Path, meta_path: str | Path, vectorizer_path: str | Path | None = None) -> None:
        """Persist the retrieval artifacts to disk."""
        self.store.save(index_path, meta_path, vectorizer_path=vectorizer_path)


def load_government_documents(path: str | Path) -> List[dict]:
    """Load official document records from JSONL."""
    return [record for record in iter_jsonl(path)]


def filter_documents_by_languages(documents: List[dict], supported_languages: List[str]) -> List[dict]:
    """Filter retrieval documents to configured languages only."""
    normalized_supported = {normalize_language_code(language) for language in supported_languages}
    output: List[dict] = []
    for item in documents:
        language = normalize_language_code(str(item.get("language", "en")))
        if language not in normalized_supported:
            continue
        normalized_item = dict(item)
        normalized_item["language"] = language
        output.append(normalized_item)
    return output


def resolve_retrieval_documents(config: dict) -> List[dict]:
    """Resolve retrieval source with a preference for processed government corpus."""
    paths = config.get("paths", {})
    supported_languages = config.get("languages", {}).get("supported", ["en", "hi", "gu"])
    gov_documents_path = Path(paths.get("gov_documents", "data/dummy/gov_docs.jsonl"))
    raw_corpus_path = Path(paths.get("raw_corpus", "data/processed/raw_corpus.jsonl"))

    if raw_corpus_path.exists():
        raw_items = [item for item in iter_jsonl(raw_corpus_path)]
        # Prefer government-derived records when available.
        government_only = [
            item
            for item in raw_items
            if str(item.get("source", "")).startswith("gov:") or str(item.get("source", "")).startswith("gov:web")
        ]
        if government_only:
            return filter_documents_by_languages(government_only, supported_languages)
        if raw_items:
            return filter_documents_by_languages(raw_items, supported_languages)

    return filter_documents_by_languages(load_government_documents(gov_documents_path), supported_languages)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build or query the government retriever")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--build-index", action="store_true", help="Build the retrieval index")
    parser.add_argument("--query", type=str, default="", help="Optional query to test retrieval")
    return parser.parse_args()


def main() -> None:
    """Build the retrieval index or run a retrieval query from the command line."""
    args = parse_args()
    config = load_config(args.config)
    index_path = Path(config["paths"]["retrieval_index"])
    meta_path = Path(config["paths"]["retrieval_meta"])
    vectorizer_path = index_path.with_suffix(".pkl")
    documents = resolve_retrieval_documents(config)
    if not documents:
        raise RuntimeError("No retrieval documents found for configured languages")
    retriever = GovernmentRetriever.from_documents(
        documents,
        chunk_size=int(config["retrieval"]["chunk_size"]),
        overlap=int(config["retrieval"]["chunk_overlap"]),
    )
    if args.build_index:
        retriever.save(index_path, meta_path, vectorizer_path=vectorizer_path)
        print(f"Saved retrieval index to {index_path}")
    if args.query:
        results = retriever.retrieve(args.query, top_k=int(config["retrieval"]["top_k"]))
        for result in results:
            print(f"[{result.score:.4f}] {result.chunk.title}: {result.chunk.text}")


if __name__ == "__main__":
    main()
