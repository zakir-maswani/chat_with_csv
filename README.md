# Chat with CSV

> A production-ready web application that enables natural language conversations with your CSV data through a Retrieval-Augmented Generation (RAG) pipeline.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?style=flat-square)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

**Chat with CSV** bridges the gap between raw tabular data and human understanding. Upload any CSV file and immediately begin asking plain-English questions — the system ingests your data, builds a semantic vector index, and uses a large language model to return accurate, context-aware answers.

The backend is powered by **FastAPI** for high-throughput request handling, **LangChain** for orchestrating the RAG pipeline, **Groq** for blazing-fast LLM inference, **HuggingFace** sentence-transformer embeddings for semantic search, and **FAISS** for efficient nearest-neighbor retrieval.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  FastAPI    │────▶│  RAG Pipeline    │────▶│   Groq LLM   │
│  Backend    │     │  (LangChain)     │     │  Inference   │
└─────────────┘     └──────────────────┘     └──────────────┘
       │                    │
       │            ┌───────┴────────┐
       │            │                │
       ▼            ▼                ▼
 CSV Upload    HuggingFace       FAISS
               Embeddings     Vector Store
```

---

## Features

| Feature | Description |
|---|---|
| **CSV Upload** | Drag-and-drop or browse to upload any `.csv` file |
| **Natural Language Queries** | Ask questions in plain English — no SQL required |
| **RAG Pipeline** | Combines semantic retrieval with LLM generation for accurate answers |
| **Fast Inference** | Powered by Groq for sub-second LLM response times |
| **Semantic Search** | HuggingFace embeddings ensure contextually relevant chunk retrieval |
| **Scalable API** | Async FastAPI backend handles concurrent requests efficiently |
| **Clean UI** | Minimal, responsive frontend — no framework dependencies |

---

## Project Structure

```
chat_with_csv/
│
├── core/                       # Core RAG pipeline components
│   ├── __init__.py
│   ├── document_analyzer.py    # Document parsing and structural analysis
│   ├── embedding_engine.py     # Text embedding generation and management
│   └── pdf_processor.py        # Multi-format document processing utilities
│
├── external/                   # External service integrations
│   ├── __init__.py
│   └── paperless_client.py     # Paperless-ngx integration client
│
├── frontend/                   # Web application interface
│   ├── index.html              # Application shell and layout
│   ├── script.js               # Upload handling, chat logic, API communication
│   └── style.css               # Responsive styles and component design
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── file_handler.py         # File I/O, validation, and temp management
│   └── logger_setup.py         # Structured logging configuration
│
├── main.py                     # FastAPI application entry point and routes
├── rag_pipeline.py             # RAG chain definition, retriever, and prompt logic
├── requirements.txt            # Python dependencies (pinned)
├── .env.example                # Environment variable template
├── .gitignore                  # Git exclusion rules
└── notebook.ipynb              # Development and experimentation notebook
```

---

## Tech Stack

### Backend
| Library | Version | Purpose |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | ≥0.100 | Async web framework and API routing |
| [LangChain](https://langchain.com/) | latest | RAG orchestration and chain management |
| [Groq](https://console.groq.com/) | latest | High-speed LLM inference |
| [HuggingFace `sentence-transformers`](https://www.sbert.net/) | latest | Text embedding generation |
| [FAISS](https://faiss.ai/) | latest | Efficient vector similarity search |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | latest | Environment variable management |
| [Uvicorn](https://www.uvicorn.org/) | latest | ASGI server for production serving |

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 | Application structure and semantic markup |
| CSS3 | Responsive layout, animations, and theming |
| Vanilla JavaScript | File upload, WebSocket/REST communication, chat UI |

---

## Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)
- A **Groq API key** — [get one free at console.groq.com](https://console.groq.com/)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zakir-maswani/chat_with_csv.git
cd chat_with_csv
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example file and add your credentials:

```bash
cp .env.example .env
```

Open `.env` and set the following:

```env
# Required
GROQ_API_KEY="your_groq_api_key_here"

# Optional — override defaults
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL="llama3-8b-8192"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> **Note:** Never commit your `.env` file. It is already excluded via `.gitignore`.

### 5. Run the Application

```bash
uvicorn main:app --reload
```

The server starts at **http://127.0.0.1:8000**.

For production deployments, omit `--reload` and configure workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Usage

1. Open **http://127.0.0.1:8000** in your browser.
2. Click **Upload CSV** and select your `.csv` file (or drag and drop).
3. Wait for the file to be ingested — the system will chunk, embed, and index it.
4. Type a question in natural language into the chat box and press **Send**.
5. Review the AI-generated answer, which is grounded in your CSV data.

### Example Queries

```
"What is the average sales revenue for Q3?"
"Which customer placed the most orders?"
"List all products with a price above $100."
"Summarize the trend in monthly signups."
```

---

## API Reference

### `POST /upload`
Upload a CSV file for processing.

| Parameter | Type | Description |
|---|---|---|
| `file` | `multipart/form-data` | The CSV file to upload |

**Response:**
```json
{
  "status": "success",
  "message": "File ingested and indexed successfully.",
  "chunks": 48
}
```

---

### `POST /query`
Submit a natural language question against the uploaded data.

**Request body:**
```json
{
  "question": "What is the total revenue by region?"
}
```

**Response:**
```json
{
  "answer": "The total revenue by region is: North $1.2M, South $980K, East $750K, West $1.1M.",
  "sources": ["row 12-45", "row 102-130"]
}
```

---

## RAG Pipeline Details

The `rag_pipeline.py` module implements the following steps:

1. **Document Loading** — CSV rows are loaded and converted to plain-text documents.
2. **Text Splitting** — Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve context across chunk boundaries.
3. **Embedding** — Each chunk is encoded using a HuggingFace sentence-transformer model, producing dense vector representations.
4. **Vector Store** — Embeddings are stored in a FAISS index for O(log n) approximate nearest-neighbor retrieval.
5. **Retrieval** — At query time, the question is embedded and the top-k most semantically similar chunks are retrieved.
6. **Generation** — Retrieved chunks and the original question are passed to the Groq-hosted LLM via a structured prompt, which returns a grounded answer.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | *(Required)* Your Groq API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model name |
| `LLM_MODEL` | `llama3-8b-8192` | Groq-hosted model identifier |
| `CHUNK_SIZE` | `1000` | Token count per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | `4` | Number of retrieved chunks per query |

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black . && isort .
```

### Linting

```bash
flake8 . --max-line-length 100
```

### Jupyter Notebook

The included `notebook.ipynb` is useful for rapid prototyping of new embedding strategies or prompt templates before integrating into the main pipeline.

```bash
jupyter notebook notebook.ipynb
```

---

## Troubleshooting

**`GROQ_API_KEY not found`**
Ensure your `.env` file exists in the project root and the variable name matches exactly.

**Slow embedding on first run**
The HuggingFace model is downloaded on first use (~90 MB). Subsequent runs use the local cache.

**`ModuleNotFoundError` after install**
Confirm your virtual environment is activated before running `pip install` and `uvicorn`.

**Empty or incorrect answers**
- Verify your CSV is well-formed (headers in first row, consistent delimiters).
- Try increasing `CHUNK_SIZE` or `TOP_K_RESULTS` for larger files.

---

## Contributing

Contributions are welcome and appreciated. To contribute:

1. **Fork** this repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a **Pull Request** with a clear description of what you changed and why.

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and ensure all tests pass before submitting.

---

## Acknowledgements

- [LangChain](https://langchain.com/) for the RAG orchestration framework
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [HuggingFace](https://huggingface.co/) for open-source embedding models
- [FAISS](https://faiss.ai/) (Meta AI Research) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the elegant async web framework

---

*Built with ❤️ by [Zakir Maswani](https://github.com/zakir-maswani)*
