# SearchIQ — Project Context

## What Is SearchIQ?

SearchIQ is an enterprise-grade **Hybrid Search RAG** platform designed for production workloads. It orchestrates document ingestion, hybrid retrieval (keyword + semantic), reranking, and answer verification to deliver accurate, grounded responses from organizational knowledge bases.

The platform prioritizes reliability, observability, and clear separation of concerns so teams can evolve each stage independently without compromising system integrity.

---

## Architecture Principles

1. **Modular pipeline** — Ingestion, retrieval, reranking, and verification are isolated modules with well-defined interfaces.
2. **Hybrid by default** — Combine lexical and vector search to maximize recall and precision across diverse query types.
3. **Production-first** — Favor explicit configuration, structured logging, health checks, and testable boundaries over premature abstraction.
4. **Schema-driven contracts** — Use Pydantic schemas at API and service boundaries to enforce type safety and validation.
5. **Stateless API layer** — The HTTP layer remains thin; business logic lives in services and domain modules.
6. **Incremental delivery** — Ship minimal, working slices; expand capabilities without rewriting foundations.

---

## Coding Standards

### Python

- Target **Python 3.11+**.
- Follow [PEP 8](https://peps.python.org/pep-0008/) naming and formatting conventions.
- Use **type hints** on all public functions, methods, and return values.
- Prefer **explicit imports**; avoid wildcard imports.
- Keep functions focused; extract helpers only when reuse or clarity demands it.
- Document non-obvious behavior with concise docstrings; let clear code speak for itself.

### API Design

- RESTful routes under `/api/v1/` as the API surface grows.
- Return consistent JSON response shapes; use appropriate HTTP status codes.
- Validate all inputs with Pydantic models in `schemas/`.
- Version breaking changes; never silently alter contract semantics.

### Configuration

- Environment-specific values live in `.env` (never committed).
- Use `pydantic-settings` for typed, validated configuration in `core/`.
- Provide defaults suitable for local development; require explicit values in production.

### Testing

- Place tests in `tests/` mirroring the backend package layout.
- Unit-test services and domain logic; integration-test API endpoints.
- No test should depend on external services without explicit fixtures or mocks.

### Git & Reviews

- Small, focused commits with descriptive messages.
- No secrets, credentials, or large binary artifacts in version control.
- Every PR should state what changed, why, and how it was verified.

---

## Folder Responsibilities

### `backend/`

The FastAPI application and all server-side logic.

| Path | Responsibility |
|------|----------------|
| `app/main.py` | Application entry point; mounts routers, health checks, and middleware. |
| `app/api/` | HTTP route handlers (`/documents/upload`, `/chat`, `/health`, `/`). |
| `app/core/` | Application configuration (`pydantic-settings`), logging, and global settings. |
| `app/database/` | PostgreSQL connection session factory and declarative base. |
| `app/models/` | SQLAlchemy ORM models (`Document`, `Chunk`, `DocumentImage`). |
| `app/schemas/` | Pydantic request/response schemas (`Upload`, `Chat`, DTO models). |
| `app/ingestion/` | Multi-format parsers (PDF, DOCX, TXT, Markdown), multimodal image extraction, text cleaning. |
| `app/chunking/` | Recursive page-aware text chunking with configurable overlap. |
| `app/embeddings/` | Dense vector embedding generation via `BAAI/bge-small-en-v1.5`. |
| `app/retrieval/` | Hybrid search (Dense BGE + BM25 + Reciprocal Rank Fusion) and Visual search (`VisualSearchService`). |
| `app/reranker/` | Cross-Encoder joint candidate reranking (`BAAI/bge-reranker-base`). |
| `app/prompting/` | Structured prompt builder synthesizing text chunks and visual descriptions. |
| `app/generation/` | LLM answer generation (`LLMService`) and VLM captioning (`VisualCaptioner`/`VLMService`) via OpenRouter. |
| `app/evaluation/` | Offline retrieval benchmarking suite (Golden dataset, evaluator, Recall@K, Precision@K, MRR@K). |
| `app/services/` | High-level orchestration services (`IndexingService`, `ChatService`). |

### `frontend/`

Web client interface for document management, search, and multimodal chat interaction. To be implemented next.

### `docs/`

Architecture Decision Records (`decisions.md`), technical specifications, and system documentation.

### `infrastructure/`

Docker Compose configurations (`pgvector/pgvector:pg17`), deployment manifests, and environment templates.

### `data/`

| Path | Responsibility |
|------|----------------|
| `data/raw/` | Uploaded raw source documents (UUID-keyed). |
| `data/processed/` | Intermediate chunked or enriched document artifacts. |
| `data/images/` | Extracted raster images and diagrams from uploaded PDFs. |
| `data/eval/` | Annotated golden datasets for offline retrieval evaluation benchmarks. |

Contents under `data/raw/`, `data/processed/`, and `data/images/` are gitignored; directory structure is tracked.

### `tests/`

Automated test suite (94 passing tests) covering unit and integration testing for ingestion, chunking, embeddings, hybrid retrieval, visual search, cross-encoder reranking, chat orchestration, evaluation, and API routes.

---

## Current Scope

The **SearchIQ Backend** is fully implemented, verified, and operational:

- **Multi-Format Ingestion**: Ingests PDF, DOCX, TXT, and Markdown files with validation and automatic indexing upon upload.
- **Multimodal Visual Pipeline**: Extracts embedded raster figures and diagrams from PDFs, generates rich factual captions using OpenRouter VLM (`google/gemini-2.5-flash`), and indexes 384-dimensional BGE embeddings in pgvector.
- **Hybrid Text + Visual Retrieval**: Combines Dense semantic search (BGE) and Lexical search (BM25) via Reciprocal Rank Fusion (RRF), alongside pgvector visual candidate search.
- **Unified Cross-Encoder Reranking**: Merges text chunks and visual candidates into a unified candidate pool and jointly rescores them with `BAAI/bge-reranker-base`.
- **Grounded Answer Synthesis**: Builds structured prompts citing text chunks and visual figures, generating verifiable, citation-backed answers via OpenRouter LLM.
- **Offline Retrieval Evaluation**: Benchmark framework measuring Recall@K, Precision@K, and MRR@K against golden evaluation datasets.
- **RESTful API**: Production endpoints for document upload (`POST /documents/upload`), natural language chat (`POST /chat`), health checks (`GET /health`), and root (`GET /`).

**Next Phase**: Frontend client development (UI for document uploading, multimodal conversation, visual source inspection, and citation display).

