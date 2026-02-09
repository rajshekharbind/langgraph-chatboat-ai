from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging

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
# INITIALIZE LLAMA MODEL
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

# ============================================
# STATE DEFINITION
# ============================================
class ChatState(TypedDict):
    """State for the chat application"""
    messages: Annotated[list[BaseMessage], add_messages]

# ============================================
# CHAT NODE
# ============================================
def chat_node(state: ChatState) -> dict:
    """
    Process user message and generate AI response
    
    Args:
        state: Current chat state with messages
    
    Returns:
        Updated state with AI response
    """
    try:
        messages = state['messages']
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in chat_node: {e}")
        raise

# ============================================
# LANGGRAPH WORKFLOW
# ============================================
# Create checkpointer for conversation memory
checkpointer = InMemorySaver()

# Create graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile the chatbot
chatbot = graph.compile(checkpointer=checkpointer)

logger.info("✅ Chatbot graph compiled successfully")