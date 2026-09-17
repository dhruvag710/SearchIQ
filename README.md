<p align="center">
  <h1 align="center">SearchIQ</h1>
  <p align="center">
    <strong>Production-Grade Multimodal Hybrid Search & Retrieval-Augmented Generation (RAG)</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat&logo=postgresql&logoColor=white" alt="pgvector" />
  <img src="https://img.shields.io/badge/Tests-94%20Passing-brightgreen?style=flat" alt="Tests" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" />
</p>

---

## Overview

**SearchIQ** is an enterprise-grade **Multimodal Hybrid RAG** system designed for high-accuracy document intelligence and question answering across organizational knowledge bases.

Beyond standard text-only retrieval, SearchIQ understands both text and visual content embedded within documents (e.g., charts, diagrams, and figures in PDFs). It indexes document text alongside VLM-generated visual captions in a unified vector space, combines lexical search (BM25) and dense semantic search (BGE) via **Reciprocal Rank Fusion (RRF)**, rescores unified candidate pools using a **Cross-Encoder Reranker**, and synthesizes citation-backed answers via Large Language Models.

---

## Key Features

- **Multi-Format Ingestion**: Supports PDF, DOCX, TXT, and Markdown documents with format validation and immediate background indexing.
- **Multimodal PDF Understanding**: Automatically extracts embedded raster images and figures from PDFs during ingestion.
- **VLM Visual Captioning**: Employs an OpenRouter Vision-Language Model (`google/gemini-2.5-flash`) to generate detailed, factual descriptions of diagrams, tables, and charts.
- **Dense BGE Embeddings**: Generates 384-dimensional dense vector embeddings (`BAAI/bge-small-en-v1.5`) for both text chunks and visual captions.
- **PostgreSQL + pgvector**: Unified relational metadata and vector database storage with pgvector cosine distance indexing.
- **Hybrid Text Retrieval**: Combines in-memory BM25 keyword matching with dense vector retrieval using **Reciprocal Rank Fusion (RRF)** for optimal keyword and semantic recall.
- **Visual Similarity Search**: Retrieves relevant visual assets independently via cosine similarity search over image caption embeddings.
- **Unified Cross-Encoder Reranking**: Merges top text chunks and top visual candidates into a single candidate pool and rescores them jointly using `BAAI/bge-reranker-base`.
- **Grounded Answer Generation**: Assembles structured prompts with source badges and generates faithful, citation-backed answers via OpenRouter LLM.
- **Offline Evaluation Suite**: Built-in benchmarking framework calculating Recall@K, Precision@K, and MRR@K against annotated golden datasets.

---

## Architecture

```text
  Documents (PDF, DOCX, TXT, MD)
                ↓
      Multi-format Ingestion
                ↓
  ┌──────────────────────┬──────────────────────┐
  │ Text Pipeline        │ Visual Pipeline      │
  │ • Cleaning           │ • Image extraction   │
  │ • Recursive chunking │ • VLM captioning     │
  │ • BGE embeddings     │ • BGE embeddings     │
  │ • BM25 + pgvector    │ • pgvector           │
  └──────────┬───────────┴──────────┬───────────┘
             ↓                      ↓
       Hybrid Text Retrieval   Visual Retrieval
       (Dense + BM25 → RRF)   (pgvector)
             └──────────┬───────────┘
                        ↓
              Unified Candidate Pool
             (25 Text + 10 Visual)
                        ↓
              Cross-Encoder Reranking
             (BAAI/bge-reranker-base)
                        ↓
                  Prompt Builder
                        ↓
                 OpenRouter LLM
                        ↓
            Grounded Answer + Citations
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI / Uvicorn | Async REST API, OpenAPI docs, dependency injection |
| **Database & Vectors** | PostgreSQL 17 + pgvector | Relational metadata + 384-dim vector similarity storage |
| **ORM & Migrations** | SQLAlchemy 2.0 / Alembic | Type-safe schema definitions and migration management |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dimensional dense semantic representations |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder joint query-context scoring |
| **Lexical Search** | `rank-bm25` | In-memory BM25 retrieval for exact terminology |
| **Document Parsers** | PyMuPDF, python-docx, markdown | Multi-format parsing and PDF raster image extraction |
| **VLM & LLM** | OpenRouter (`gemini-2.5-flash`) | Visual asset captioning and grounded answer synthesis |
| **Testing** | pytest, pytest-asyncio | Unit, integration, and API route test suites |

---

## Project Structure

```text
SearchIQ
├── backend
│   ├── alembic/                # Database schema migrations
│   ├── app
│   │   ├── api/routes/         # REST API routes (upload, chat, health)
│   │   ├── chunking/           # Recursive page-aware text chunker
│   │   ├── core/               # App configuration & settings
│   │   ├── database/           # Session factory & base model
│   │   ├── embeddings/         # BGE embedding generator
│   │   ├── evaluation/         # Retrieval benchmark framework & metrics
│   │   ├── generation/         # LLM & VLM OpenRouter services
│   │   ├── ingestion/          # Multi-format & multimodal PDF parsers
│   │   ├── models/             # SQLAlchemy ORM (Document, Chunk, DocumentImage)
│   │   ├── prompting/          # Context prompt synthesis
│   │   ├── reranker/           # Cross-Encoder candidate reranker
│   │   ├── retrieval/          # Hybrid, BM25, vector & visual search services
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   └── services/           # Chat & indexing orchestration services
│   ├── scripts/                # Evaluation CLI scripts
│   ├── requirements.txt        # Backend dependencies
│   └── .env.example            # Environment template
├── data/
│   ├── raw/                    # Uploaded source documents
│   ├── images/                 # Extracted PDF images
│   └── eval/                   # Golden evaluation datasets
├── docker-compose.yml          # PostgreSQL + pgvector container
└── tests/                      # Automated test suite (94 tests)
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dhruvag710/SearchIQ.git
cd SearchIQ
```

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

### 3. Setup Python Virtual Environment

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env` inside the `backend/` directory:

```env
APP_ENV=development
APP_DEBUG=false
HOST=0.0.0.0
PORT=8000

POSTGRES_DB=searchiq
POSTGRES_USER=searchiq
POSTGRES_PASSWORD=searchiq
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.5-flash
```

### 5. Run Database Migrations

```bash
alembic upgrade head
```

### 6. Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API Reference

### `GET /health`
Health check endpoint.
```json
{
  "status": "healthy"
}
```

### `POST /documents/upload`
Upload and index a PDF, DOCX, TXT, or MD document (up to 50 MB). Extracts text, extracts raster images from PDFs, generates VLM captions, computes BGE embeddings, and indexes chunks.

- **Request**: `multipart/form-data` with `file` binary.
- **Response (`201 Created`)**:
```json
{
  "document_id": "66319144-585b-430b-8004-b97e12ce937e",
  "original_filename": "architecture_overview.pdf",
  "stored_filename": "66319144-585b-430b-8004-b97e12ce937e.pdf",
  "file_size_bytes": 1048576,
  "uploaded_at": "2026-09-17T10:00:00.000000+00:00",
  "status": "indexed"
}
```

### `POST /chat`
Ask natural language questions across indexed documents. Executes hybrid text retrieval and visual candidate search, performs unified cross-encoder reranking, and generates an answer with citations.

- **Request**:
```json
{
  "question": "According to the graph on page 6, what happens to NDCG@1 as the keep ratio decreases?"
}
```
- **Response (`200 OK`)**:
```json
{
  "answer": "According to the graph on page 6, as the keep ratio decreases from 1.0 to 0.1, NDCG@1 steadily declines...",
  "sources": [
    {
      "document_id": "66319144-585b-430b-8004-b97e12ce937e",
      "page_number": 6,
      "chunk_index": 0,
      "is_visual": true,
      "image_path": "data/images/66319144-585b-430b-8004-b97e12ce937e/page_6_img_0.png"
    },
    {
      "document_id": "66319144-585b-430b-8004-b97e12ce937e",
      "page_number": 6,
      "chunk_index": 1,
      "is_visual": false,
      "image_path": null
    }
  ]
}
```

---

## Retrieval Evaluation

SearchIQ includes an offline evaluation suite (`backend/scripts/run_eval.py`) to empirically benchmark retrieval strategies against annotated ground-truth datasets.

### Measured Results

Below are benchmark results measured across 4 retrieval strategies on the local golden evaluation dataset (`searchiq_resume_golden`, 10 annotated test cases across keyword, semantic, and complex queries):

| Strategy | Recall@5 | Recall@25 | MRR@5 | Precision@5 | Avg Latency |
|---|---|---|---|---|---|
| **A. Dense Retrieval** (BGE pgvector) | 0.800 | 0.950 | 0.293 | 0.180 | ~44 ms |
| **B. BM25 Retrieval** (Lexical) | 0.450 | 1.000 | 0.123 | 0.100 | ~0.2 ms |
| **C. Hybrid Retrieval** (Dense + BM25 RRF) | 0.450 | 0.950 | 0.158 | 0.100 | ~38 ms |
| **D. Hybrid + Reranker** (Cross-Encoder) | **0.950** | N/A | **0.468** | **0.220** | ~2490 ms |

> [!NOTE]
> *These measurements reflect the project's local golden evaluation dataset and demonstrate that the Cross-Encoder reranker substantially improves top-5 recall (+15% over dense) and ranking precision (MRR@5 +60% over dense). They should be interpreted as domain-specific verification rather than a universal IR benchmark.*

---

## Testing

The project maintains an automated test suite verifying all ingestion parsers, chunking strategies, embeddings, hybrid retrieval, visual search, cross-encoder reranking, chat orchestration, and API routes.

```bash
pytest
```

```text
94 passed, 1 warning in 21.76s
```

---

## Author

**Dhruv Agarwal** — Built to demonstrate end-to-end multimodal retrieval-augmented generation, hybrid lexical/dense search fusion, cross-encoder reranking, and production backend architecture.