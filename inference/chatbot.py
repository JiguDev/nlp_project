"""Interactive chatbot inference pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from retrieval.retriever import GovernmentRetriever
from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.engine import build_model, get_device, load_checkpoint
from utils.io import iter_jsonl
from utils.config import load_config
from utils.lang import add_language_token, detect_language, strip_language_token
from utils.text import clean_text, join_non_empty


def parse_args() -> argparse.Namespace:
    """Parse chatbot command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the multilingual government chatbot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    return parser.parse_args()


def build_context_prompt(question: str, language: str, retrieved_chunks: list[str]) -> str:
    """Create a prompt that includes the user question and retrieved evidence."""
    context_text = join_non_empty(retrieved_chunks, separator="\n")
    prompt = f"Question: {question}\nContext: {context_text}\nAnswer:"
    return add_language_token(prompt, language)


def load_retriever(config: dict) -> GovernmentRetriever:
    """Load an existing retriever or build one from the dummy documents if needed."""
    index_path = Path(config["paths"]["retrieval_index"])
    meta_path = Path(config["paths"]["retrieval_meta"])
    vectorizer_path = index_path.with_suffix(".pkl")
    if index_path.exists() and meta_path.exists() and vectorizer_path.exists():
        return GovernmentRetriever.load(index_path, meta_path, vectorizer_path=vectorizer_path)
    documents_path = Path(config["paths"]["gov_documents"])
    retriever = GovernmentRetriever.from_documents(
        [record for record in iter_jsonl(documents_path)],
        chunk_size=int(config["retrieval"]["chunk_size"]),
        overlap=int(config["retrieval"]["chunk_overlap"]),
    )
    retriever.save(index_path, meta_path, vectorizer_path=vectorizer_path)
    return retriever


def main() -> None:
    """Start an interactive multilingual chatbot session."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device(str(config["training"]["device"]))

    tokenizer = SentencePieceTokenizer(config["paths"]["tokenizer_model"])
    model = build_model(tokenizer, config["model"]).to(device)
    checkpoint_path = Path(config["paths"]["finetune_checkpoint"])
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, map_location=device)
    else:
        pretrain_path = Path(config["paths"]["pretrain_checkpoint"])
        if pretrain_path.exists():
            load_checkpoint(pretrain_path, model, map_location=device)

    retriever = load_retriever(config)
    max_new_tokens = int(config["inference"]["max_new_tokens"])
    temperature = float(config["inference"]["temperature"])
    top_k = int(config["inference"]["top_k"])
    beam_width = int(config["inference"]["beam_width"])
    decoding_strategy = str(config["inference"].get("decoding_strategy", "beam")).lower()

    print("Multilingual government chatbot ready. Type 'exit' to stop.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        language = detect_language(user_text, default=config["languages"]["default"])
        results = retriever.retrieve(user_text, top_k=int(config["retrieval"]["top_k"]))
        retrieved_texts = [f"{result.chunk.title}: {result.chunk.text}" for result in results]
        prompt = build_context_prompt(clean_text(user_text), language, retrieved_texts)
        src_tokens = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=True)], dtype=torch.long, device=device)
        if decoding_strategy == "sample":
            generated = model.sample_decode(src_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        elif decoding_strategy == "greedy":
            generated = model.greedy_decode(src_tokens, max_new_tokens=max_new_tokens)
        else:
            generated = model.beam_search_decode(src_tokens, max_new_tokens=max_new_tokens, beam_width=beam_width)
        response = tokenizer.decode(generated[0].tolist())
        response = strip_language_token(response)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
