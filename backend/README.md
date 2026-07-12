# SearchIQ Backend

FastAPI backend for the SearchIQ hybrid search RAG platform.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

From the `backend` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path     | Description        |
|--------|----------|--------------------|
| GET    | `/`      | Welcome message    |
| GET    | `/health`| Health check       |

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
