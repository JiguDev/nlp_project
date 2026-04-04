# Domain-Adaptive Multilingual Chatbot for Indian Government Services

This project implements a multilingual government-service chatbot with a transformer model built from scratch in PyTorch, a shared SentencePiece tokenizer, and a lightweight FAISS-based retrieval layer for up-to-date official information.

## Features

- Multilingual input and output across Indian languages
- Transformer encoder-decoder implemented manually in PyTorch
- Shared SentencePiece tokenizer with language tokens such as `<hi>`, `<gu>`, and `<ta>`
- Retrieval-Augmented Generation using document chunking, TF-IDF embeddings, and FAISS
- Pretraining and fine-tuning pipelines
- Beam search and temperature sampling for inference
- Basic metrics such as perplexity and BLEU
- Retrieval recall@k evaluation for RAG quality
- Scalable ingestion pipeline for Indic, Samanantar, and government corpora

## Thesis Documentation

- Architecture and methodology: `docs/architecture_methodology.md`

## Project Structure

- `data/`: raw, dummy, and processed corpora
- `tokenizer/`: SentencePiece tokenizer wrapper and training script
- `model/`: attention, encoder, decoder, and transformer modules
- `training/`: data preparation and training scripts
- `retrieval/`: chunking, embeddings, FAISS index, and retrieval logic
- `inference/`: chatbot entry point
- `utils/`: shared helpers
- `configs/`: YAML configuration files

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Review `configs/default.yaml` and adjust paths or model sizes if needed.

## Data Preparation

The repository ships with a minimal dummy dataset under `data/dummy/` so the pipeline can run end-to-end.

To build processed training data:

```bash
python training/data_pipeline.py --data-dir data --processed-dir data/processed
```

### Scalable Real-Data Ingestion

Use this when you have real IndicCorp/Samanantar and government datasets:

```bash
python training/ingest_real_data.py --config configs/default.yaml
```

To additionally scrape government URLs listed in `data/government/gov_urls.txt`:

```bash
python training/ingest_real_data.py --config configs/default.yaml --scrape
```

The ingestion step creates:

- `data/processed/raw_corpus.jsonl`
- `data/processed/raw_qa.jsonl`
- `data/processed/pretrain_pairs.jsonl`
- `data/processed/finetune_pairs.jsonl`
- `data/processed/tokenizer_corpus.txt`

To train the tokenizer:

```bash
python tokenizer/train_tokenizer.py --input-dir data --output-corpus data/processed/tokenizer_corpus.txt --model-prefix data/processed/indic_gov_tokenizer
```

## Training

### Pretraining

```bash
python training/train_pretrain.py --config configs/default.yaml
```

### Fine-tuning

```bash
python training/train_finetune.py --config configs/default.yaml
```

Both scripts load the same transformer architecture and save checkpoints under `checkpoints/`.

## Retrieval Index

Build the government-document index from the dummy corpus:

```bash
python retrieval/retriever.py --config configs/default.yaml --build-index
```

## Chatbot Inference

Run the interactive chatbot:

```bash
python inference/chatbot.py --config configs/default.yaml
```

The chatbot will:

1. Detect the input language
2. Retrieve top-k relevant government passages
3. Append the retrieved context to the prompt
4. Generate a response in the same language

## Evaluation

The training scripts report token-level loss and perplexity. Generation quality can be evaluated with BLEU on held-out question-answer pairs.

Run full evaluation (perplexity + BLEU + retrieval recall@k):

```bash
python training/evaluate.py --config configs/default.yaml --split finetune
```

## Stability and Scalability Recommendations

- Keep retrieval index updates decoupled from model retraining for faster refresh cycles.
- Use periodic evaluation to track regression on multilingual quality and domain grounding.
- Store large corpora as sharded JSONL files and use glob-based ingestion.
- Increase model size and batch size gradually based on GPU memory limits.

## Notes

- The model is intentionally implemented without HuggingFace transformers.
- The retrieval stack uses TF-IDF embeddings and FAISS for a lightweight, production-friendly baseline.
- Replace the dummy datasets with IndicCorp, Samanantar, India Code, PRS, state government portals, and ministry FAQs for a full-scale project.
