import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import requests
from io import BytesIO

# ======================================== Environment setup ========================================
load_dotenv()
# Import from backend
from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ======================================== Page configuration ========================================
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================== Custom CSS for Responsiveness ========================================
st.markdown(
    """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .main { 
            padding: 1rem; 
            max-width: 1200px;
            margin: 0 auto;
        }

        .chat-message {
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stChatMessage {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 0.5rem;
        }

        h1 {
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
        }
        
        h2 {
            color: #1f77b4;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .thread-button {
            margin: 0.5rem 0;
            width: 100%;
            text-align: left;
        }

        @media (max-width: 768px) {
            .main { 
                padding: 0.5rem;
            }
            
            h1 {
                font-size: 1.8rem;
                margin-bottom: 1rem;
            }

            .chat-message {
                font-size: 0.95rem;
                padding: 1rem;
            }
            
            .stChatMessage {
                padding: 0.75rem;
            }
        }

        @media (max-width: 480px) {
            .main { 
                padding: 0.25rem;
            }
            
            h1 {
                font-size: 1.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================== Utility Functions for Threading ========================================

def generate_thread_id():
    """Generate a unique thread ID for each conversation"""
    thread_id = str(uuid.uuid4())
    return thread_id


def reset_chat():
    """Reset the chat and create a new thread"""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    """Add a new thread to the chat threads list"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    """Load conversation history from a specific thread"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except Exception as e:
        st.warning(f"⚠️ Could not load conversation: {e}")
        return []


def delete_thread(thread_id):
    """Delete a thread from chat threads list"""
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
        if st.session_state['thread_id'] == thread_id:
            reset_chat()



# ======================================== Session State Setup ========================================

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_documents' not in st.session_state:
    st.session_state['thread_documents'] = {}

# Add initial thread to the threads list
add_thread(st.session_state['thread_id'])


# ============================ Sidebar - PDF Upload & Chat History ============================
st.sidebar.title("📚 LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{str(st.session_state['thread_id'])[:12]}...`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Show document info for current thread
thread_docs = st.session_state['thread_documents'].get(st.session_state['thread_id'], {})
if thread_docs:
    latest_doc = list(thread_docs.values())[-1] if thread_docs else None
    if latest_doc:
        st.sidebar.success(
            f"📄 Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
        )
else:
    st.sidebar.info("📭 No PDF indexed for this chat yet.")

# PDF uploader with URL option
st.sidebar.subheader("📄 Add PDF to Chat")

pdf_source = st.sidebar.radio("Choose PDF source:", ["Upload File", "Load from URL"], horizontal=True)

if pdf_source == "Upload File":
    uploaded_pdf = st.sidebar.file_uploader("📤 Upload a PDF for this chat", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name not in thread_docs:
            with st.sidebar.status("🔄 Indexing PDF…", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=st.session_state['thread_id'],
                        filename=uploaded_pdf.name,
                    )
                    
                    # Store in session state
                    if st.session_state['thread_id'] not in st.session_state['thread_documents']:
                        st.session_state['thread_documents'][st.session_state['thread_id']] = {}
                    st.session_state['thread_documents'][st.session_state['thread_id']][uploaded_pdf.name] = summary
                    
                    status_box.update(label="✅ PDF indexed successfully", state="complete", expanded=False)
                    st.sidebar.success(f"Added: {uploaded_pdf.name}")
                except Exception as e:
                    status_box.update(label="❌ Error indexing PDF", state="error", expanded=True)
                    st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.info(f"✓ `{uploaded_pdf.name}` already processed for this chat.")

elif pdf_source == "Load from URL":
    pdf_url = st.sidebar.text_input("📎 Enter PDF URL", placeholder="https://example.com/mathematics.pdf")
    
    if st.sidebar.button("📥 Load PDF from URL", use_container_width=True):
        if pdf_url.strip():
            with st.sidebar.status("🔄 Downloading and indexing PDF…", expanded=True) as status_box:
                try:
                    # Download PDF from URL
                    response = requests.get(pdf_url, timeout=30)
                    response.raise_for_status()
                    
                    pdf_bytes = BytesIO(response.content)
                    
                    # Extract filename from URL
                    filename = pdf_url.split('/')[-1]
                    if not filename.endswith('.pdf'):
                        filename = f"document_{len(thread_docs)+1}.pdf"
                    
                    # Ingest the PDF
                    summary = ingest_pdf(
                        pdf_bytes.getvalue(),
                        thread_id=st.session_state['thread_id'],
                        filename=filename,
                    )
                    
                    # Store in session state
                    if st.session_state['thread_id'] not in st.session_state['thread_documents']:
                        st.session_state['thread_documents'][st.session_state['thread_id']] = {}
                    st.session_state['thread_documents'][st.session_state['thread_id']][filename] = summary
                    
                    status_box.update(label="✅ PDF loaded and indexed successfully", state="complete", expanded=False)
                    st.sidebar.success(f"Added: {filename}")
                    st.rerun()
                
                except requests.exceptions.MissingSchema:
                    status_box.update(label="❌ Invalid URL", state="error", expanded=True)
                    st.sidebar.error("❌ Invalid URL format. Please enter a valid URL starting with http:// or https://")
                except requests.exceptions.RequestException as e:
                    status_box.update(label="❌ Download failed", state="error", expanded=True)
                    st.sidebar.error(f"❌ Could not download PDF: {str(e)}")
                except Exception as e:
                    status_box.update(label="❌ Error processing PDF", state="error", expanded=True)
                    st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter a valid PDF URL")

st.sidebar.divider()

# Past conversations
st.sidebar.subheader(f"💬 Conversations ({len(st.session_state['chat_threads'])})")
if not st.session_state['chat_threads']:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in reversed(st.session_state['chat_threads']):
        if st.sidebar.button(
            f"📌 {str(thread_id)[:12]}...",
            key=f"side-thread-{thread_id}",
            use_container_width=True
        ):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            
            st.session_state['message_history'] = temp_messages
            st.rerun()


# ============================ Main Layout ============================
st.title("🤖 LangGraph PDF Chatbot")
st.subheader("💬 Chat with Your Documents")

# Display chat history
if len(st.session_state['message_history']) == 0:
    st.info("👋 Start a new conversation! Upload a PDF in the sidebar, then ask questions about it.", icon="ℹ️")
else:
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Chat input
st.divider()
user_input = st.chat_input("Ask a question about your documents...", key="chat_input")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Prepare config with current thread
    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    },
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn",
}
    # Generate AI response with streaming
    try:
        with st.chat_message("assistant"):
            def ai_stream_generator():
                """Stream AI response tokens"""
                status_holder = {"box": None}  # Initialize status holder
                
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    # Lazily create & update the SAME status container when any tool runs
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )
                    
                    # Yield AI message content
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content
            
            ai_message = st.write_stream(ai_stream_generator())
        
        # Store the complete response in history
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        
    except Exception as err:
        st.error(f"❌ Error generating response: {err}")
        # Remove the user message if there was an error
        if st.session_state['message_history']:
            st.session_state['message_history'].pop()
        st.info("🔄 Please try again or start a new conversation.")

