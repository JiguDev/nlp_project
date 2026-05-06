"""Interactive chatbot inference pipeline (Multilingual Government Chatbot)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from retrieval.retriever import GovernmentRetriever, resolve_retrieval_documents
from utils.io import iter_jsonl
from utils.config import load_config
from utils.lang import detect_language, normalize_language_code
from inference.generator import ChatbotGenerator


# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multilingual Government Chatbot")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()





# =========================
def _select_language_matched_contexts(results, language: str, top_k: int) -> List[str]:
    """Prefer same-language chunks."""

    same_lang = [
        r for r in results
        if normalize_language_code(r.chunk.language) == language
    ]

    selected = same_lang[:top_k]

    # fallback if not enough same-language
    if len(selected) < top_k:
        for r in results:
            if r not in selected:
                selected.append(r)
            if len(selected) >= top_k:
                break

    return [f"{r.chunk.title}: {r.chunk.text}" for r in selected]


# =========================
def load_retriever(config: dict) -> GovernmentRetriever:
    index_path = Path(config["paths"]["retrieval_index"])
    meta_path = Path(config["paths"]["retrieval_meta"])
    vectorizer_path = index_path.with_suffix(".pkl")

    if index_path.exists() and meta_path.exists() and vectorizer_path.exists():
        print("[OK] Loading existing FAISS index...")
        return GovernmentRetriever.load(index_path, meta_path, vectorizer_path=vectorizer_path)

    print("[WARN] Building retriever from documents...")

    documents = resolve_retrieval_documents(config)

    if not documents:
        documents_path = Path(config["paths"]["gov_documents"])
        documents = list(iter_jsonl(documents_path))

    retriever = GovernmentRetriever.from_documents(
        documents,
        chunk_size=int(config["retrieval"]["chunk_size"]),
        overlap=int(config["retrieval"]["chunk_overlap"]),
    )

    retriever.save(index_path, meta_path, vectorizer_path=vectorizer_path)

    return retriever


# =========================
def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("\n==============================")
    print("Starting Chatbot")
    print("==============================\n")

    retriever = load_retriever(config)
    generator = ChatbotGenerator(config)

    top_k = int(config["retrieval"]["top_k"])

    supported_languages = {
        normalize_language_code(lang)
        for lang in config["languages"]["supported"]
    }

    default_lang = normalize_language_code(config["languages"]["default"])

    print("Chatbot ready (en / hi / gu). Type 'exit' to quit.\n")

    # =========================
    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # Detect language
        language = normalize_language_code(
            detect_language(user_text, default=default_lang),
            default=default_lang,
        )

        if language not in supported_languages:
            language = default_lang

        # 🔥 RETRIEVAL (more candidates = better filtering)
        results = retriever.retrieve(user_text, top_k=top_k * 5)

        contexts = _select_language_matched_contexts(results, language, top_k)

        # 🔥 SAFE RESPONSE
        response = generator.generate(language, user_text, contexts)

        print(f"Bot: {response}\n")


# =========================
if __name__ == "__main__":
    main()