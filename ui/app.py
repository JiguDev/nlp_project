import sys
import os
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import streamlit as st
from utils.lang import detect_language, normalize_language_code, detect_target_language_override
from utils.config import load_config
from retrieval.retriever import GovernmentRetriever
from inference.chatbot import _select_language_matched_contexts, load_retriever
from inference.generator import ChatbotGenerator
from retrieval.web_search import WebRetriever


# =========================
# Load config + retriever
def init_system():
    config = load_config("configs/default.yaml")
    retriever = load_retriever(config)
    generator = ChatbotGenerator(config)
    web_retriever = WebRetriever()
    return config, retriever, generator, web_retriever


config, retriever, generator, web_retriever = init_system()

top_k = int(config["retrieval"]["top_k"])

supported_languages = {
    normalize_language_code(lang)
    for lang in config["languages"]["supported"]
}

default_lang = normalize_language_code(config["languages"]["default"])


# =========================
# UI Layout
st.set_page_config(page_title="Gov Chatbot", page_icon="🇮🇳", layout="centered")

# Sidebar for controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Custom CSS for Theme-Adaptive Premium Look
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background: transparent;
    }
    
    /* Center the chat and add overall margins */
    .main .block-container {
        max-width: 900px;
        padding-right: 3rem;
        padding-left: 3rem;
    }

    /* Message styling - Adaptive */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.05);
        margin-right: 8%; /* Balanced right margin */
    }
    
    /* Custom Status Pill - Adaptive */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 18px;
        background: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 50px;
        font-size: 0.85rem;
        margin-bottom: 15px;
        backdrop-filter: blur(8px);
        animation: fadeInOut 2s infinite ease-in-out;
    }

    /* Target st.info (Summary box) to add more right margin and spacing */
    [data-testid="stAlert"], [data-testid="stNotification"] {
        margin-right: 8% !important;
        margin-bottom: 20px !important;
        border-radius: 12px !important;
    }

    /* Paragraph styling for better readability and alignment */
    [data-testid="stChatMessage"] p {
        line-height: 1.6;
        padding-right: 8%; /* Align with the Summary box margin */
    }

    /* Input box styling */
    .stChatInputContainer {
        border-radius: 20px;
    }

    @keyframes fadeInOut {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🇮🇳 Multilingual Government Chatbot")
st.write("Ask questions in English, Hindi, or Gujarati")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Chat Display Container
chat_container = st.container()

# Display chat history from state
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                st.info(f"**📌 Summary:**\n\n{msg['content']['summary']}")
                if msg["content"]["details"]:
                    st.markdown(f"**🔍 Details:**\n\n{msg['content']['details']}")
            else:
                st.markdown(msg["content"])


# =========================
# User input
user_input = st.chat_input("Ask your question...")

if user_input:
    # 1. Immediately show user message in the UI and state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # 2. Process response
    # Status Placeholder for Pill UI
    status_placeholder = st.empty()
    
    def update_status(text):
        status_placeholder.markdown(f"""
            <div class="status-pill">
                {text}
            </div>
            """, unsafe_allow_html=True)

    update_status("Thinking...")
    
    # Logic remains same...
    language = normalize_language_code(
        detect_language(user_input, default=default_lang),
        default=default_lang,
    )
    if language not in supported_languages:
        language = default_lang
    target_language = detect_target_language_override(user_input, language)

    update_status("🌐 Searching the web for live information...")
    web_contexts = web_retriever.search_all(user_input, top_k=3)

    if web_contexts:
        combined_contexts = web_contexts
    else:
        update_status("📚 Web search returned no results. Checking local database...")
        time.sleep(1)
        results = retriever.retrieve(user_input, top_k=top_k * 5)
        local_contexts = _select_language_matched_contexts(results, language, top_k)
        combined_contexts = local_contexts

    update_status("🧠 Synthesizing response...")
    response = generator.generate(target_language, user_input, combined_contexts)
    status_placeholder.empty()

    # 3. Show assistant response with streaming effect
    with chat_container:
        with st.chat_message("assistant"):
            summary_placeholder = st.empty()
            details_placeholder = st.empty()

            summary_placeholder.info(f"**📌 Summary:**\n\n{response['summary']}")
            
            if response["details"]:
                full_details = ""
                words = response["details"].split()
                for word in words:
                    full_details += word + " "
                    details_placeholder.markdown(f"**🔍 Details:**\n\n{full_details}▌")
                    time.sleep(0.04)
                details_placeholder.markdown(f"**🔍 Details:**\n\n{full_details}")

    # 4. Save to history
    st.session_state.messages.append({"role": "assistant", "content": response})