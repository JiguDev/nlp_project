"""Transformer generation wrapper for chatbot inference."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.engine import build_model, get_device, load_checkpoint
from utils.lang import add_language_token


class ChatbotGenerator:
    """Wrapper to handle tokenization, RAG formatting, and transformer decoding."""

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device(str(config.get("inference", {}).get("device", "cpu")))
        
        # Load Tokenizer
        tokenizer_model_path = Path(config["paths"]["tokenizer_model"])
        if not tokenizer_model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_model_path}")
        self.tokenizer = SentencePieceTokenizer(tokenizer_model_path)
        
        # Load Model
        self.model = build_model(self.tokenizer, config["model"]).to(self.device)
        
        # Load Fine-tuned Checkpoint
        checkpoint_path = Path(config["paths"]["finetune_checkpoint"])
        if not checkpoint_path.exists():
            print(f"⚠️ Fine-tuned checkpoint not found at {checkpoint_path}. Attempting to use pretrain checkpoint.")
            checkpoint_path = Path(config["paths"]["pretrain_checkpoint"])
            if not checkpoint_path.exists():
                raise FileNotFoundError("Neither finetune nor pretrain checkpoints found. Please train the model first.")
        
        print(f"📦 Loading generation model: {checkpoint_path}")
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

    def generate(self, language: str, question: str, contexts: List[str]) -> str:
        """Generate an answer using RAG context and the TransformerSeq2Seq."""
        
        # Formulate prompt with context
        context_str = " ".join(contexts)
        if context_str:
            prompt_text = f"Context: {context_str} Question: {question}"
        else:
            prompt_text = question
            
        # Add language token
        tagged_prompt = add_language_token(prompt_text, language)
        
        # Tokenize source
        max_source_length = int(self.config["training"]["max_source_length"])
        source_ids = self.tokenizer.encode(tagged_prompt, add_bos=True, add_eos=True)[:max_source_length]
        
        # Convert to tensor
        src_tokens = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        
        # Generation settings
        inf_cfg = self.config.get("inference", {})
        max_new_tokens = int(inf_cfg.get("max_new_tokens", 80))
        decoding_strategy = str(inf_cfg.get("decoding_strategy", "beam")).lower()
        beam_width = int(inf_cfg.get("beam_width", 4))
        temperature = float(inf_cfg.get("temperature", 0.8))
        top_k = int(inf_cfg.get("top_k", 50))
        
        # Decode
        try:
            with torch.no_grad():
                if decoding_strategy == "beam":
                    generated_tokens = self.model.beam_search_decode(
                        src_tokens, 
                        max_new_tokens=max_new_tokens, 
                        beam_width=beam_width
                    )
                elif decoding_strategy == "sample":
                    generated_tokens = self.model.sample_decode(
                        src_tokens, 
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature,
                        top_k=top_k
                    )
                else: # greedy
                    generated_tokens = self.model.greedy_decode(
                        src_tokens, 
                        max_new_tokens=max_new_tokens
                    )
        except Exception as e:
            print(f"⚠️ Error during generation: {e}")
            if language == "hi":
                return "माफ़ कीजिए, मैं अभी उत्तर उत्पन्न करने में असमर्थ हूँ।"
            elif language == "gu":
                return "માફ કરશો, હું અત્યારે જવાબ ઉત્પન્ન કરવામાં અસમર્થ છું."
            return "Sorry, I am currently unable to generate a response."
            
        # Decode tokens to text
        generated_ids = generated_tokens[0].cpu().tolist()
        
        # Remove bos, eos, pad, and lang tokens if present
        cleaned_ids = [
            tid for tid in generated_ids 
            if tid not in (self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)
        ]
        
        output_text = self.tokenizer.decode(cleaned_ids)
        
        # Clean potential bleeding of language tokens from generation
        output_text = output_text.replace(f"<{language}>", "").strip()
        
        # If output is completely empty, provide fallback
        if not output_text:
            if not context_str:
                if language == "hi": return "माफ़ कीजिए, इस प्रश्न के लिए जानकारी उपलब्ध नहीं है।"
                elif language == "gu": return "માફ કરશો, આ પ્રશ્ન માટે માહિતી ઉપલબ્ધ નથી."
                return "Sorry, no information is available for this question."
            else:
                # If we have context but generation failed, return context directly
                if language == "hi": return f"उपलब्ध जानकारी: {context_str}"
                elif language == "gu": return f"ઉપલબ્ધ માહિતી: {context_str}"
                return f"Available information: {context_str}"
                
        return output_text
