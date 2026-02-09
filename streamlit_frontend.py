import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# -----------------------------
# Environment setup
# -----------------------------
load_dotenv()

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
        .main { padding: 0rem; }

        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }

        @media (max-width: 640px) {
            .chat-message {
                font-size: 0.9rem;
                padding: 0.75rem;
            }
        }

        h1 {
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load LangGraph backend
# -----------------------------
try:
    from langgraph_backend import chatbot
except ImportError as e:
    st.error(f"❌ Backend load error: {e}")
    st.info("Ensure `langgraph_backend.py` is present.")
    st.stop()
except ValueError:
    st.error("❌ GEMINI_API_KEY not configured")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.info("LangGraph-powered chatbot using Google Gemini")

    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ Gemini API Key loaded")
    else:
        st.warning("⚠️ Gemini API Key missing")

# -----------------------------
# Main UI
# -----------------------------
st.title("🤖 LangGraph Chatbot")

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

GRAPH_CONFIG = {
    "configurable": {"thread_id": "thread-1"}
}

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    try:
        with st.spinner("🤔 Thinking..."):
            response = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=GRAPH_CONFIG,
            )

            ai_reply = response["messages"][-1].content

        st.session_state["messages"].append(
            {"role": "assistant", "content": ai_reply}
        )

        with st.chat_message("assistant"):
            st.markdown(ai_reply)

    except Exception as err:
        st.error(f"❌ Error: {err}")
        st.session_state["messages"].pop()
