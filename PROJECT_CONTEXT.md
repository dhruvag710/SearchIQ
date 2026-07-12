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
| `app/main.py` | Application entry point; mounts routes and middleware. |
| `app/api/` | HTTP route handlers and API versioning. |
| `app/core/` | Configuration, logging, middleware, and shared utilities. |
| `app/database/` | Database connections, migrations, and persistence adapters. |
| `app/ingestion/` | Document parsing, chunking, and indexing pipelines. |
| `app/retrieval/` | Hybrid search: keyword, vector, and fusion strategies. |
| `app/reranker/` | Cross-encoder and LLM-based result reranking. |
| `app/verifier/` | Groundedness checks and citation validation. |
| `app/services/` | Orchestration layer coordinating domain modules. |
| `app/models/` | ORM / database entity definitions. |
| `app/schemas/` | Pydantic request/response and internal DTO models. |

### `frontend/`

Web client for search, administration, and analytics. Not yet implemented.

### `docs/`

Architecture decision records (ADRs), API guides, runbooks, and onboarding material.

### `infrastructure/`

Deployment manifests, CI/CD configuration, and environment provisioning. Docker and cloud resources will live here when added.

### `data/`

| Path | Responsibility |
|------|----------------|
| `data/raw/` | Unprocessed source documents. |
| `data/processed/` | Chunked, enriched, or indexed artifacts. |
| `data/sample_docs/` | Small fixtures for local development and tests. |

Contents under `raw/` and `processed/` are gitignored; only directory structure is tracked.

### `tests/`

Automated test suite for backend services, API endpoints, and pipeline components.

---

## Current Scope

This initial scaffold provides:

- A minimal FastAPI application with root and health endpoints.
- A production-oriented directory layout ready for incremental feature work.
- Dependency and environment templates.

**Not yet implemented:** authentication, database layer, ingestion, retrieval, embeddings, reranking, verification, frontend, and infrastructure automation.

Refer to this document when adding new modules to ensure consistency with project conventions.
