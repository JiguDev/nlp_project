import sys
import os
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import streamlit as st
from utils.lang import detect_language, normalize_language_code
from utils.config import load_config
from retrieval.retriever import GovernmentRetriever
from inference.chatbot import _select_language_matched_contexts, load_retriever
from inference.generator import ChatbotGenerator


# =========================
# Load config + retriever (only once)
@st.cache_resource
def init_system():
    config = load_config("configs/default.yaml")
    retriever = load_retriever(config)
    generator = ChatbotGenerator(config)
    return config, retriever, generator


config, retriever, generator = init_system()

top_k = int(config["retrieval"]["top_k"])

supported_languages = {
    normalize_language_code(lang)
    for lang in config["languages"]["supported"]
}

default_lang = normalize_language_code(config["languages"]["default"])


# =========================
# UI Layout
st.set_page_config(page_title="Gov Chatbot", page_icon="🇮🇳")

st.title("🇮🇳 Multilingual Government Chatbot")
st.write("Ask questions in English, Hindi, or Gujarati")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# User input
user_input = st.chat_input("Ask your question...")

if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Detect language
    language = normalize_language_code(
        detect_language(user_input, default=default_lang),
        default=default_lang,
    )

    if language not in supported_languages:
        language = default_lang

    # Retrieval
    results = retriever.retrieve(user_input, top_k=top_k * 5)
    contexts = _select_language_matched_contexts(results, language, top_k)

    # Response
    response = generator.generate(language, user_input, contexts)

    # Show bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        full_response = ""
        words = response.split()

        time.sleep(0.08)
        for word in words:
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.08)  # speed control (reduce for faster)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": response})