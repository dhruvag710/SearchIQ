<p align="center">
  <h1 align="center">SearchIQ</h1>
  <p align="center">
    Intelligent Document Search and Question Answering using Retrieval-Augmented Generation (RAG)
  </p>
</p>

---

## Overview

SearchIQ is an intelligent document search and question-answering system that enables users to interact with documents using natural language.

The system automatically parses uploaded documents, indexes their contents, retrieves the most relevant information through semantic search, reranks the retrieved context, and generates grounded responses using a Large Language Model (LLM).

The project follows a modular architecture where document ingestion, indexing, retrieval, reranking, and answer generation are implemented as independent components, making the system easier to extend and maintain.

---

## Features

### Document Ingestion

- PDF upload with validation
- Automatic indexing after upload
- PDF parsing using PyMuPDF
- Modular parser architecture

### Document Processing

- Text cleaning pipeline
- Recursive chunking
- Configurable chunk overlap
- Page-aware metadata preservation

### Retrieval Pipeline

- Dense semantic retrieval
- Vector storage using PostgreSQL + pgvector
- Cross-encoder reranking
- Context selection for generation

### Question Answering

- Retrieval-Augmented Generation (RAG)
- OpenRouter LLM integration
- Context-aware prompt construction
- Citation-backed responses

---

## Architecture

```text
                PDF Upload
                     │
                     ▼
              Document Parser
                     │
                     ▼
               Text Cleaning
                     │
                     ▼
                 Chunking
                     │
                     ▼
          Sentence Embeddings
                     │
                     ▼
        PostgreSQL + pgvector
                     │
                     ▼
           Dense Retrieval
                     │
                     ▼
      Cross-Encoder Reranker
                     │
                     ▼
             Prompt Builder
                     │
                     ▼
              OpenRouter LLM
                     │
                     ▼
       Citation-backed Answer
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| FastAPI | Backend API |
| PostgreSQL | Metadata storage |
| pgvector | Vector similarity search |
| SQLAlchemy | ORM |
| Alembic | Database migrations |
| Sentence Transformers | Embedding generation |
| PyMuPDF | PDF parsing |
| OpenRouter | LLM inference |
| Uvicorn | ASGI server |

---

## Project Structure

```text
SearchIQ
│
├── backend
│   ├── app
│   │   ├── api
│   │   ├── chunking
│   │   ├── database
│   │   ├── embeddings
│   │   ├── generation
│   │   ├── ingestion
│   │   ├── models
│   │   ├── reranker
│   │   ├── retrieval
│   │   ├── schemas
│   │   ├── services
│   │   └── verifier
│   │
│   ├── alembic
│   ├── requirements.txt
│   └── .env.example
│
├── data
└── README.md
```

---

## Getting Started

### Clone the repository

```bash
git clone <repository-url>
cd SearchIQ/backend
```

### Create a virtual environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file inside the `backend` directory.

```env
DATABASE_URL=your_database_url

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_model_name
```

### Start the application

```bash
python -m uvicorn app.main:app --reload
```


---

## Author

**Dhruv Agarwal**

Built as a personal project to explore modern Retrieval-Augmented Generation (RAG) systems, semantic search, and LLM-powered document understanding.