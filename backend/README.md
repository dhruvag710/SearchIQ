# SearchIQ Backend

FastAPI backend for the **SearchIQ Multimodal Hybrid Search RAG** platform.

The backend provides production-grade document ingestion (PDF, DOCX, TXT, Markdown), automated visual asset extraction and VLM captioning, hybrid retrieval (Dense BGE + Lexical BM25 with Reciprocal Rank Fusion), visual similarity search, unified cross-encoder reranking, and citation-backed answer generation.

---

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for PostgreSQL + pgvector)

---

## Quickstart

### 1. Start Vector Database

From the project root:

```bash
docker compose up -d
```

This launches a PostgreSQL 17 container with the `pgvector` extension exposed on port `5432`.

### 2. Configure Environment

In the `backend/` directory, copy `.env.example` to `.env` and set your credentials:

```bash
cd backend
cp .env.example .env
```

Ensure your `.env` contains:

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

### 3. Install Dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Run Database Migrations

Apply Alembic migrations to create tables (`documents`, `chunks`, `document_images` with pgvector indexes):

```bash
alembic upgrade head
```

### 5. Start the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API Endpoints

### 1. `GET /`
Returns service welcome message.

**Response:**
```json
{
  "message": "Welcome to SearchIQ"
}
```

---

### 2. `GET /health`
Liveness and health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 3. `POST /documents/upload`
Upload and automatically index a document. Supports PDF, DOCX, TXT, and Markdown files (max 50 MB).

- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Binary file)

**Response (`201 Created`):**
```json
{
  "document_id": "66319144-585b-430b-8004-b97e12ce937e",
  "original_filename": "research_paper.pdf",
  "stored_filename": "66319144-585b-430b-8004-b97e12ce937e.pdf",
  "file_size_bytes": 1048576,
  "uploaded_at": "2026-09-17T10:00:00.000000+00:00",
  "status": "indexed"
}
```

---

### 4. `POST /chat`
Query indexed documents using multimodal hybrid RAG. Retrieves relevant text chunks and visual image captions, reranks them jointly with a Cross-Encoder, and returns a grounded answer with citations.

**Request Body (`application/json`):**
```json
{
  "question": "According to the graph on page 6, what happens to NDCG@1 as keep ratio decreases?"
}
```

**Response (`200 OK`):**
```json
{
  "answer": "As the keep ratio decreases from 1.0 to 0.1, NDCG@1 exhibits a steady decline...",
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

## Multimodal Pipeline Architecture

1. **Ingestion & Extraction**:
   - `MultimodalPdfParser` extracts text pages and raster images/diagrams from PDFs.
   - Text is normalized via `Cleaner` and segmented using recursive page-aware chunking.
   - Extracted images are saved under `data/images/<doc_id>/`.
2. **Visual Captioning & Indexing**:
   - `VisualCaptioner` invokes OpenRouter VLM (`google/gemini-2.5-flash`) to generate detailed factual descriptions of extracted diagrams and figures.
   - Embeddings for text chunks and image captions are computed via `BAAI/bge-small-en-v1.5` (384 dimensions) and stored in PostgreSQL using `pgvector`.
3. **Retrieval & Fusion**:
   - `HybridRetriever`: Combines dense semantic vector retrieval (pgvector) and lexical search (BM25) via **Reciprocal Rank Fusion (RRF)**.
   - `VisualSearchService`: Runs dense vector search over visual image captions.
4. **Unified Cross-Encoder Reranking**:
   - Candidate text chunks (top 25) and visual candidates (top 10) form a unified candidate pool (35 items).
   - `CrossEncoderReranker` (`BAAI/bge-reranker-base`) jointly scores all candidates against the user query to select the top 5 most relevant items.
5. **Generation**:
   - `PromptBuilder` formats selected text and visual contexts with clear source indicators.
   - `LLMService` generates a grounded response with citations.

---

## Testing & Evaluation

### Run Automated Unit & Integration Tests

```bash
pytest
```

*(94 unit and API route tests passing)*

### Run Offline Retrieval Benchmark

```bash
python scripts/run_eval.py
```
Evaluates retrieval strategies (Dense, BM25, Hybrid RRF, Hybrid + Cross-Encoder) on annotated golden datasets and outputs Recall@K, MRR@K, Precision@K, and latency metrics.
