from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.upload import router as upload_router

app = FastAPI(
    title="SearchIQ",
    description="Enterprise-grade Hybrid Search RAG platform",
    version="0.1.0",
)

app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to SearchIQ"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
