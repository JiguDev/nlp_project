"""Interactive chatbot inference pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from retrieval.retriever import GovernmentRetriever, resolve_retrieval_documents
from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.engine import build_model, get_device, load_checkpoint
from utils.io import iter_jsonl
from utils.config import load_config
from utils.lang import add_language_token, detect_language, normalize_language_code, strip_language_token
from utils.text import clean_text, join_non_empty


def parse_args() -> argparse.Namespace:
    """Parse chatbot command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the multilingual government chatbot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    return parser.parse_args()


LANGUAGE_PROMPT_TEMPLATES = {
    "en": {
        "question": "Question",
        "context": "Context",
        "answer": "Answer",
    },
    "hi": {
        "question": "प्रश्न",
        "context": "संदर्भ",
        "answer": "उत्तर",
    },
    "gu": {
        "question": "પ્રશ્ન",
        "context": "સંદર્ભ",
        "answer": "જવાબ",
    },
}


def build_context_prompt(question: str, language: str, retrieved_chunks: list[str]) -> str:
    """Create a prompt that includes the user question and retrieved evidence."""
    labels = LANGUAGE_PROMPT_TEMPLATES.get(language, LANGUAGE_PROMPT_TEMPLATES["en"])
    context_text = join_non_empty(retrieved_chunks, separator="\n")
    prompt = f"{labels['question']}: {question}\n{labels['context']}: {context_text}\n{labels['answer']}:"
    return add_language_token(prompt, language)


def _retrieval_fallback_response(language: str, question: str, chunks: List[str]) -> str:
    """Return a context-grounded response when no trained checkpoint is available."""
    context = chunks[0] if chunks else ""
    if language == "hi":
        if context:
            return f"उपलब्ध संदर्भ के आधार पर: {context}"
        return f"मुझे इस प्रश्न का सटीक उत्तर देने के लिए अधिक सरकारी संदर्भ चाहिए: {question}"
    if language == "gu":
        if context:
            return f"મળેલા સંદર્ભના આધાર પર: {context}"
        return f"આ પ્રશ્નનો ચોક્કસ જવાબ આપવા વધુ સરકારી સંદર્ભ જોઈએ: {question}"
    if context:
        return f"Based on the retrieved context: {context}"
    return f"I need more government context to answer this precisely: {question}"


def _select_language_matched_contexts(results, language: str, top_k: int) -> list[str]:
    """Prefer same-language retrieved chunks and then backfill if needed."""
    same_language = [item for item in results if normalize_language_code(item.chunk.language) == language]
    selected = same_language[:top_k]
    if len(selected) < top_k:
        selected_ids = {item.chunk.chunk_id for item in selected}
        for item in results:
            if item.chunk.chunk_id in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break
    return [f"{item.chunk.title}: {item.chunk.text}" for item in selected]


def load_retriever(config: dict) -> GovernmentRetriever:
    """Load an existing retriever or build one from the dummy documents if needed."""
    index_path = Path(config["paths"]["retrieval_index"])
    meta_path = Path(config["paths"]["retrieval_meta"])
    vectorizer_path = index_path.with_suffix(".pkl")
    if index_path.exists() and meta_path.exists() and vectorizer_path.exists():
        loaded = GovernmentRetriever.load(index_path, meta_path, vectorizer_path=vectorizer_path)
        supported_languages = {
            normalize_language_code(language)
            for language in config.get("languages", {}).get("supported", ["en", "hi", "gu"])
        }
        if loaded.store.chunks and all(normalize_language_code(chunk.language) in supported_languages for chunk in loaded.store.chunks):
            return loaded

    documents = resolve_retrieval_documents(config)
    if not documents:
        documents_path = Path(config["paths"]["gov_documents"])
        documents = [record for record in iter_jsonl(documents_path)]
    retriever = GovernmentRetriever.from_documents(documents, chunk_size=int(config["retrieval"]["chunk_size"]), overlap=int(config["retrieval"]["chunk_overlap"]))
    retriever.save(index_path, meta_path, vectorizer_path=vectorizer_path)
    return retriever


def main() -> None:
    """Start an interactive multilingual chatbot session."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device(str(config["training"]["device"]))

    tokenizer_model_path = Path(config["paths"]["tokenizer_model"])
    tokenizer = SentencePieceTokenizer(tokenizer_model_path) if tokenizer_model_path.exists() else None
    model = build_model(tokenizer, config["model"]).to(device) if tokenizer is not None else None
    checkpoint_loaded = False
    checkpoint_path = Path(config["paths"]["finetune_checkpoint"])
    if model is not None and checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, map_location=device)
        checkpoint_loaded = True
    else:
        pretrain_path = Path(config["paths"]["pretrain_checkpoint"])
        if model is not None and pretrain_path.exists():
            load_checkpoint(pretrain_path, model, map_location=device)
            checkpoint_loaded = True

    retriever = load_retriever(config)
    max_new_tokens = int(config["inference"]["max_new_tokens"])
    temperature = float(config["inference"]["temperature"])
    top_k = int(config["inference"]["top_k"])
    beam_width = int(config["inference"]["beam_width"])
    decoding_strategy = str(config["inference"].get("decoding_strategy", "beam")).lower()
    supported_languages = {
        normalize_language_code(language)
        for language in config.get("languages", {}).get("supported", ["en", "hi", "gu"])
    }
    default_language = normalize_language_code(config["languages"]["default"])

    if tokenizer is None:
        print("Tokenizer model not found. Running retrieval-grounded fallback mode.")
    elif not checkpoint_loaded:
        print("No trained checkpoint found. Running retrieval-grounded fallback mode.")

    print("Tri-lingual government chatbot ready (en/hi/gu). Type 'exit' to stop.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        language = normalize_language_code(detect_language(user_text, default=default_language), default=default_language)
        if language not in supported_languages:
            language = default_language
        raw_results = retriever.retrieve(user_text, top_k=max(int(config["retrieval"]["top_k"]) * 3, int(config["retrieval"]["top_k"])))
        retrieved_texts = _select_language_matched_contexts(raw_results, language, int(config["retrieval"]["top_k"]))

        if tokenizer is not None and model is not None and checkpoint_loaded:
            prompt = build_context_prompt(clean_text(user_text), language, retrieved_texts)
            src_tokens = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=True)], dtype=torch.long, device=device)
            if decoding_strategy == "sample":
                generated = model.sample_decode(src_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            elif decoding_strategy == "greedy":
                generated = model.greedy_decode(src_tokens, max_new_tokens=max_new_tokens)
            else:
                generated = model.beam_search_decode(src_tokens, max_new_tokens=max_new_tokens, beam_width=beam_width)
            response = strip_language_token(tokenizer.decode(generated[0].tolist())).strip()
            if not response:
                response = _retrieval_fallback_response(language, clean_text(user_text), retrieved_texts)
        else:
            response = _retrieval_fallback_response(language, clean_text(user_text), retrieved_texts)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
