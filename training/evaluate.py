"""Evaluation script for BLEU, perplexity, and retrieval recall@k."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from retrieval.retriever import GovernmentRetriever
from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.dataset import SequencePairDataset
from training.engine import build_dataloader, build_model, get_device, load_checkpoint, train_one_epoch
from utils.config import load_config
from utils.io import iter_jsonl
from utils.lang import add_language_token, detect_language, normalize_language_code, strip_language_token
from utils.metrics import compute_bleu, perplexity_from_loss
from utils.text import clean_text, join_non_empty


LANGUAGE_PROMPT_TEMPLATES = {
    "en": {"question": "Question", "context": "Context", "answer": "Answer"},
    "hi": {"question": "प्रश्न", "context": "संदर्भ", "answer": "उत्तर"},
    "gu": {"question": "પ્રશ્ન", "context": "સંદર્ભ", "answer": "જવાબ"},
}


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate chatbot model and retriever")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--split", type=str, default="finetune", choices=["finetune", "pretrain"], help="Dataset split to evaluate")
    return parser.parse_args()


def load_retriever(config: dict) -> GovernmentRetriever:
    """Load persisted retrieval artifacts."""
    index_path = Path(config["paths"]["retrieval_index"])
    meta_path = Path(config["paths"]["retrieval_meta"])
    vectorizer_path = index_path.with_suffix(".pkl")
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Retrieval index not found. Run retrieval/retriever.py --build-index first.")
    return GovernmentRetriever.load(index_path, meta_path, vectorizer_path=vectorizer_path)


def build_prompt(question: str, language: str, contexts: List[str]) -> str:
    """Construct generation prompt with retrieved evidence."""
    labels = LANGUAGE_PROMPT_TEMPLATES.get(language, LANGUAGE_PROMPT_TEMPLATES["en"])
    context_block = join_non_empty(contexts, separator="\n")
    return add_language_token(
        f"{labels['question']}: {question}\n{labels['context']}: {context_block}\n{labels['answer']}:",
        language,
    )


def select_language_contexts(results, language: str, top_k: int) -> List[str]:
    """Prefer same-language contexts during evaluation generation."""
    same_language = [item for item in results if normalize_language_code(item.chunk.language) == language]
    selected = same_language[:top_k]
    if len(selected) < top_k:
        seen = {item.chunk.chunk_id for item in selected}
        for item in results:
            if item.chunk.chunk_id in seen:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break
    return [f"{result.chunk.title}: {result.chunk.text}" for result in selected]


def evaluate_retrieval_recall(config: dict, retriever: GovernmentRetriever, top_k: int) -> float:
    """Compute retrieval recall@k using QA records with optional gold_doc_id."""
    qa_path = Path(config["paths"].get("raw_qa", "data/processed/raw_qa.jsonl"))
    if not qa_path.exists():
        return 0.0

    records = [item for item in iter_jsonl(qa_path)]
    if not records:
        return 0.0

    hits = 0
    valid = 0
    for item in records:
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        gold_doc_id = str(item.get("gold_doc_id", "")).strip()
        if not question:
            continue
        results = retriever.retrieve(question, top_k=top_k)
        if not results:
            continue
        valid += 1
        if gold_doc_id:
            if any(result.chunk.document_id == gold_doc_id for result in results):
                hits += 1
        else:
            # Fallback proxy: at least one retrieved chunk overlaps with answer text.
            if answer and any(answer[:40] in result.chunk.text for result in results):
                hits += 1
    return hits / valid if valid else 0.0


def generate_predictions(
    config: dict,
    model,
    tokenizer: SentencePieceTokenizer,
    retriever: GovernmentRetriever,
    device: torch.device,
    max_examples: int,
) -> tuple[List[str], List[str]]:
    """Generate model predictions for QA-style evaluation."""
    qa_path = Path(config["paths"].get("raw_qa", "data/processed/raw_qa.jsonl"))
    if not qa_path.exists():
        return [], []

    references: List[str] = []
    hypotheses: List[str] = []
    decode_mode = str(config.get("inference", {}).get("decoding_strategy", "beam"))
    temperature = float(config.get("inference", {}).get("temperature", 0.8))
    sample_top_k = int(config.get("inference", {}).get("top_k", 50))
    beam_width = int(config.get("inference", {}).get("beam_width", 4))
    max_new_tokens = int(config.get("inference", {}).get("max_new_tokens", 80))
    default_language = normalize_language_code(config.get("languages", {}).get("default", "en"))
    supported_languages = {
        normalize_language_code(language)
        for language in config.get("languages", {}).get("supported", ["en", "hi", "gu"])
    }

    for index, item in enumerate(iter_jsonl(qa_path)):
        if index >= max_examples:
            break
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        if not question or not answer:
            continue
        language = normalize_language_code(str(item.get("language", "")) or detect_language(question, default=default_language), default=default_language)
        if language not in supported_languages:
            language = default_language
        requested_top_k = int(config.get("retrieval", {}).get("top_k", 4))
        retrieval_results = retriever.retrieve(question, top_k=max(requested_top_k * 3, requested_top_k))
        contexts = select_language_contexts(retrieval_results, language, requested_top_k)
        prompt = build_prompt(question, language, contexts)
        src_tokens = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=True)], dtype=torch.long, device=device)

        if decode_mode == "sample":
            generated = model.sample_decode(src_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=sample_top_k)
        elif decode_mode == "greedy":
            generated = model.greedy_decode(src_tokens, max_new_tokens=max_new_tokens)
        else:
            generated = model.beam_search_decode(src_tokens, max_new_tokens=max_new_tokens, beam_width=beam_width)

        predicted = strip_language_token(tokenizer.decode(generated[0].tolist()))
        references.append(answer)
        hypotheses.append(predicted)

    return references, hypotheses


def main() -> None:
    """Run evaluation and print aggregate metrics."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device(str(config["training"]["device"]))

    tokenizer = SentencePieceTokenizer(config["paths"]["tokenizer_model"])
    model = build_model(tokenizer, config["model"]).to(device)

    checkpoint_path = Path(config["paths"]["finetune_checkpoint"])
    if not checkpoint_path.exists():
        checkpoint_path = Path(config["paths"]["pretrain_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found for evaluation")
    load_checkpoint(checkpoint_path, model, map_location=device)

    dataset_path = Path(config["paths"]["finetune_pairs" if args.split == "finetune" else "pretrain_pairs"])
    dataset = SequencePairDataset.from_jsonl(
        dataset_path,
        tokenizer=tokenizer,
        max_source_length=int(config["training"]["max_source_length"]),
        max_target_length=int(config["training"]["max_target_length"]),
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        device=device,
        pad_id=tokenizer.pad_id,
        label_smoothing=float(config["training"]["label_smoothing"]),
        grad_clip=float(config["training"]["grad_clip"]),
    )
    perplexity = perplexity_from_loss(loss)

    retriever = load_retriever(config)
    references, hypotheses = generate_predictions(
        config=config,
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        device=device,
        max_examples=int(config.get("evaluation", {}).get("max_generation_examples", 100)),
    )
    bleu = compute_bleu(references, hypotheses) if references else 0.0
    retrieval_recall = evaluate_retrieval_recall(config, retriever, top_k=int(config.get("retrieval", {}).get("top_k", 4)))

    print(f"eval_loss: {loss:.4f}")
    print(f"perplexity: {perplexity:.4f}")
    print(f"bleu: {bleu:.4f}")
    print(f"retrieval_recall@k: {retrieval_recall:.4f}")
    print(f"evaluated_generations: {len(references)}")


if __name__ == "__main__":
    main()
