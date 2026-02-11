from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

if not gemini_api_key:
    raise ValueError(
        "\n❌ GEMINI_API_KEY not found!\n"
        "Please set it in one of these ways:\n"
        "1. Create a .env file with: GEMINI_API_KEY=your_key\n"
        "2. Set environment variable: export GEMINI_API_KEY=your_key\n"
        "3. Get your key from: https://aistudio.google.com/app/apikey"
    )

logger.info("✅ Gemini API Key loaded successfully")

# ============================================
# INITIALIZE GEMINI LLM & EMBEDDINGS
# ============================================
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=gemini_api_key,
        temperature=0.7,
        max_tokens=1024
    )
    logger.info("✅ Gemini LLM initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    raise

# Using Google Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
logger.info("✅ Embeddings initialized successfully")

# ============================================
# PDF RETRIEVER STORE (per thread)
# ============================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# ============================================
# TOOLS DEFINITION
# ============================================
search_tool = DuckDuckGoSearchRun(region="us-en")

# Suppress warnings for langchain_google_genai schema validation
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)



def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        
        # Save metadata to persist across restarts
        save_thread_metadata()

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }



@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ============================================
# STATE DEFINITION
# ============================================
class ChatState(TypedDict):
    """State for the chat application"""
    messages: Annotated[list[BaseMessage], add_messages]

# ============================================
# CHAT NODE
# ============================================
def chat_node(state: ChatState, config=None) -> dict:
    """
    Process user message and generate AI response
    
    Args:
        state: Current chat state with messages
        config: LangGraph configuration with thread_id
    
    Returns:
        Updated state with AI response
    """
    try:
        # Extract thread_id for RAG tool context
        thread_id = None
        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")
        
        system_message = SystemMessage(
            content=(
                "You are a helpful PDF assistant. When answering questions:\n"
                "- If the user asks about a PDF document, ALWAYS use the rag_tool to retrieve relevant content\n"
                f"- Always pass thread_id='{thread_id}' when using the rag_tool\n"
                "- You can also use: web search (search_tool), calculator, and stock price lookup\n"
                "- If no PDF is available, let the user know they should upload one first"
            )
        )
        
        messages = [system_message] + state['messages']
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in chat_node: {e}")
        raise

# ============================================
# CHECKPOINTER & GRAPH
# ============================================
# Use MemorySaver - Streamlit manages session persistence
checkpointer = MemorySaver()
graph = StateGraph(ChatState)

tool_node = ToolNode(tools)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
#graph.add_edge("chat_node", END)

# Compile the chatbot
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def thread_document_metadata(thread_id: str) -> dict:
    """Get document metadata for a specific thread"""
    return _THREAD_METADATA.get(str(thread_id), {})

# ============================================
# PERSISTENCE FUNCTIONS (Save/Load Metadata)
# ============================================
METADATA_FILE = "thread_metadata.json"

def save_thread_metadata():
    """Save thread metadata to JSON file for persistence"""
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(_THREAD_METADATA, f, indent=2)
        logger.info(f"✅ Thread metadata saved to {METADATA_FILE}")
    except Exception as e:
        logger.error(f"❌ Error saving metadata: {e}")

def load_thread_metadata():
    """Load thread metadata from JSON file on startup"""
    global _THREAD_METADATA
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                _THREAD_METADATA = json.load(f)
            logger.info(f"✅ Thread metadata loaded from {METADATA_FILE}")
        else:
            logger.info("📭 No previous metadata found")
    except Exception as e:
        logger.error(f"❌ Error loading metadata: {e}")

def save_retrieved_documents_for_thread(thread_id: str, filename: str, doc_info: dict):
    """Hook to save document metadata when PDF is ingested"""
    _THREAD_METADATA[str(thread_id)] = doc_info
    save_thread_metadata()

# Load metadata on startup
load_thread_metadata()

logger.info("✅ Chatbot graph compiled successfully")