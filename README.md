# 🤖 LangGraph Agentic Chatbot

A **Streamlit-based chatbot** built using **LangGraph**, **LangChain**, and **Google Gemini**.  
This project demonstrates how to build **agentic AI systems** with **graph-based workflows**, **typed state**, and **conversation memory**.

---

## 🚀 Features

- Agentic chatbot using **LangGraph**
- Interactive **Streamlit chat UI**
- **Google Gemini** LLM integration
- Typed graph state with `TypedDict`
- Automatic message handling
- In-memory conversation checkpointing
- Secure environment variable management
- Cloud-ready (Render deployment)

---

## 🏗️ Tech Stack

- **UI**: Streamlit  
- **Agent Framework**: LangGraph  
- **LLM**: Google Gemini  
- **Messages**: LangChain Core  
- **Memory**: InMemorySaver  
- **Config**: python-dotenv  

---
```
Langgraph-Agentic-Ai/
├── streamlit_frontend.py
├── langgraph_backend.py
├── requirements.txt
└── README.md
```
▶️ Run Locally
pip install -r requirements.txt
streamlit run streamlit_frontend.py

GEMINI_API_KEY=your_api_key


## 📁 Project Structure

