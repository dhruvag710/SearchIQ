# SearchIQ

Enterprise-grade hybrid search RAG platform built for production.

## Overview

SearchIQ combines keyword search, vector retrieval, reranking, and answer verification into a single platform for building reliable retrieval-augmented generation systems.

## Project Structure

```
SearchIQ/
├── backend/          # FastAPI application
├── frontend/         # Web client (future)
├── docs/             # Documentation
├── infrastructure/   # Deployment and ops (future)
├── data/             # Raw, processed, and sample documents
└── tests/            # Test suite
```

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```



## License

Proprietary. All rights reserved.
