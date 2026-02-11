# 🚀 LangGraph Agentic AI — RAG Chatbot

A production-ready **Agentic AI + RAG (Retrieval-Augmented Generation)** chatbot built with **LangGraph**, **LangChain**, **Google Gemini**, and **FAISS**. This repository integrates document RAG (PDFs), web search tools, agentic workflows, and full observability via **LangSmith**.

---

## 🔎 Project Summary

This project demonstrates a full-stack LLM application architecture that:

* Uses **LangGraph** for graph-based agent workflows and tool orchestration.
* Uses **LangChain** utilities for document loading, text splitting, and vector retrieval.
* Uses **Google Gemini** models for chat & embeddings via `langchain-google-genai`.
* Stores embeddings locally using **FAISS** for fast retrieval.
* Integrates web-search tools (DuckDuckGo) for up-to-date information.
* Adds observability, tracing, and evaluation with **LangSmith**.

This README covers concepts, setup, architecture, usage, troubleshooting, and a ready `requirements.txt`.

---

## 🎯 Key Features

* Agentic, tool-calling workflows (LangGraph `StateGraph` + `ToolNode`).
* RAG pipeline: `PyPDFLoader` → `RecursiveCharacterTextSplitter` → embeddings → FAISS.
* Web search tool integration (`DuckDuckGoSearchRun`) for external knowledge.
* Stateful memory & checkpointing using `MemorySaver`.
* Observability & tracing with LangSmith (traces, monitoring, evals).
* Streamlit demo UI (optional) for quick prototyping.

---

## 🧭 Architecture Overview

```
User → Streamlit / CLI / API
    ↓
LangGraph Agent (StateGraph)
    ↓
Decide: RAG retrieval / web search / direct LLM call
    ├─ RAG: PyPDFLoader → TextSplitter → Embeddings → FAISS → Retriever
    ├─ DuckDuckGoSearchRun → Web results
    └─ ChatGoogleGenerativeAI → LLM response
    ↓
MemorySaver checkpoint → LangSmith trace → Response
```

---

## 📚 Core Concepts Explained

### LangGraph

* Graph-based agent framework to model agent workflows as nodes and edges (`StateGraph`, `ToolNode`).
* Ideal for building stateful multi-step agent logic, sub-agents, and memory checkpoints (`MemorySaver`).

### LangChain

* Utilities for document loading, prompt templates, text splitting, chains, and vectorstore adapters.
* Community tools (e.g., `langchain_community`) add connectors such as `PyPDFLoader`, `DuckDuckGoSearchRun`, and FAISS wrappers.

### RAG (Retrieval-Augmented Generation)

* Load documents (PDF) → chunk text → embed chunks → store in vector DB (FAISS).
* During query time: embed query → similarity search in FAISS → supply top-k contexts to the LLM.

### Google Gemini Integration

* `ChatGoogleGenerativeAI` for conversational LLM responses.
* `GoogleGenerativeAIEmbeddings` to create embeddings for document chunks.

### FAISS

* Local, high-performance vector index for storing and searching embeddings.
* Good for prototype and single-node deployments; consider Milvus/Weaviate/Elasticsearch for scale.

### Web Search (DuckDuckGo)

* `DuckDuckGoSearchRun` (requires the `ddgs` package) to fetch live web results as a tool for the agent.
* Useful for queries requiring up-to-date information.

### LangSmith (Observability)

* Observability platform for LLMs and agents: **tracing**, **monitoring**, **evaluation**, and **dashboards**.
* Each agent run produces a **trace** capturing the end-to-end execution (LLM calls, tool calls, intermediate steps).
* Use LangSmith to debug prompt failures, tool failures, latency, cost, and non-deterministic behaviors.
* Enable tracing by setting the appropriate environment variable (e.g., `LANGSMITH_TRACING=true`) if you integrate LangSmith tracing wrappers.

> Tip: Instrument your agent to emit traces for every request during dev & staging — it significantly reduces debugging time.

---

## ⚙️ Prerequisites

* Python 3.10+
* Optional: a virtual environment (venv / conda)
* Google API key for Gemini (set as env var)
* (Optional) LangSmith account & API key for observability

---

## 🔧 Quickstart — Local Development

1. Create & activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Save the `requirements.txt` in the repo (see the bottom of this README). Then install:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the repo root and add required keys:

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key
# Optional: LangSmith (observability)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

4. Run the prototype Streamlit app (if included):

```bash
streamlit run app.py
```

Or run your agent entrypoint (e.g., `python main.py` or your custom CLI script).

---

## 📁 Suggested Repository Structure

```
├─ data/                     # Raw PDFs, sample documents
├─ src/
│  ├─ agents/                # LangGraph state graphs & nodes
│  ├─ loaders/               # Document loaders & preprocessors
│  ├─ retriever/             # Embeddings & FAISS wrapper
│  ├─ tools/                 # Tool adapters (DuckDuckGo, custom tools)
│  ├─ webapp/                # Streamlit / FastAPI / UI code
│  └─ main.py                # App entrypoint
├─ tests/                    # Unit & integration tests
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## 🧪 Testing & CI

* Unit test chains, loader results, and FAISS indexing using `pytest`.
* Use a small, deterministic model or mock the LLM for unit tests.
* Add CI pipeline to:

  * Run linters (`black`, `pylint`).
  * Run tests and static analysis.

---

## 🛡 Security & Privacy

* Keep API keys in `.env` and out of version control. Add `.env` to `.gitignore`.
* If using LangSmith, review data retention and privacy settings. Mask or scrub sensitive data before tracing if required by policy.
* For production, consider self-hosting vector DB & observability if data residency is required.

---

## 🚀 Deployment Notes

* Containerize with Docker (example steps below).
* For production: use a managed vector DB (Milvus/Weaviate), model provider endpoints, and a secrets manager.
* Use a process manager (gunicorn / uvicorn) for API backends.

**Example Dockerfile (skeleton)**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "main.py"]
```

---

## 🔍 Troubleshooting

### `ImportError: Could not import ddgs` (DuckDuckGoSearchRun)

If you see an error about `ddgs` when using `DuckDuckGoSearchRun`, run:

```bash
python -m pip install -U ddgs
```

Or inside your venv:

```bash
pip install -U ddgs
```

### FAISS issues on Windows

`faiss-cpu` sometimes has wheel compatibility issues on Windows. If you run into errors, consider:

* Using WSL (Linux) for development, or
* Using an alternative vector store (sqlite + embeddings) for local testing.

### LangSmith tracing not appearing

* Ensure `LANGSMITH_TRACING=true` in your environment and that the LangSmith wrapper is enabled in your runtime.

---

## ✅ Contribution Guidelines

* Follow `git` feature-branch workflow.
* Write unit tests for new functionality.
* Run `black` and `pylint` before submitting a PR.

---

## 📄 License

This project uses the MIT License (or choose your preferred license). Include `LICENSE` in your repo.

---

## 📦 requirements.txt

Use the following pinned dependencies for reproducible environments:

```text
# ============================================
# LangGraph Chatbot Dependencies
# ============================================

# Core Framework
langgraph==0.2.28
langchain==0.2.1
langchain-core==0.2.38
langchain-community==0.2.0

# LLM Integration
langchain-google-genai==1.0.10
google-generativeai==0.7.2

# Vector Database & Document Processing
faiss-cpu==1.8.0
pypdf==4.0.1

# Tools & Utilities
python-dotenv==1.0.0
aiohttp==3.13.3
pydantic>=2.7.4
requests>=2.31.0

# Web Framework (Streamlit)
streamlit>=1.38.0
streamlit-chat>=0.1.1

# Optional: For better async support
greenlet>=3.3.1

# Development/Testing (optional)
pytest>=7.4.3
black>=24.1.0
pylint>=3.0.3
```

---

## 📚 Resources & References

* LangChain docs & LangSmith guides (tracing, observability, quickstarts).
* LangGraph documentation for agent graphs.

---

## 🙋 Need help?

I can:

* Generate `Dockerfile`, `docker-compose.yml`, and deployment docs.
* Create a Streamlit demo app (`app.py`) wired to this stack.
* Add CI workflow (GitHub Actions) including tests and linting.




