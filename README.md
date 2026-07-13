# SearchIQ

SearchIQ is a document intelligence system that indexes documents and provides accurate, citation-backed answers using modern Retrieval-Augmented Generation (RAG) techniques.

> **Project Status:** 🚧 In Development

---

## Current Features

### Document Ingestion
- PDF upload with validation
- Modular parser architecture
- PDF parsing with PyMuPDF
- Extensible support for DOCX, TXT, and Markdown

### Document Processing
- Text cleaning pipeline
- Custom recursive chunking engine
- Page-aware chunk metadata
- Word-preserving chunk overlap

### Backend
- FastAPI
- PostgreSQL + pgvector infrastructure
- SQLAlchemy ORM
- Alembic migrations

---

## Planned Features

- BGE embedding generation
- Hybrid retrieval (Vector + BM25)
- Cross-encoder reranking
- Citation-backed answers
- React frontend
- Dockerized deployment

---

## Tech Stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- Alembic
- PyMuPDF

---

## Project Structure

```text
SearchIQ/
├── backend/
├── frontend/
├── docs/
├── data/
└── tests/
```

---

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```