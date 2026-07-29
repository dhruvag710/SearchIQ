# SearchIQ Decisions

This document records the key architectural and technical decisions made during the development of SearchIQ.

---

## D001 - Backend Framework

**Choice:** FastAPI

**Reason:** Provides a lightweight, high-performance backend with asynchronous request handling, automatic OpenAPI documentation, and a clean dependency injection system.

---

## D002 - Document Storage

**Choice:** Store uploaded documents in `data/raw` using UUID-based filenames.

**Reason:** Prevents filename collisions while preserving the original filename as document metadata.

---

## D003 - PDF Parsing

**Choice:** PyMuPDF

**Reason:** Provides fast and reliable text extraction with page-level access, making it suitable for downstream chunking and citation generation.

---

## D004 - Parser Architecture

**Choice:** Keep document parsing independent of the API layer.

**Reason:** Separates business logic from HTTP handling, simplifies testing, and allows the parsing pipeline to be reused in different workflows.

---

## D005 - Multi-format Support

**Choice:** Use a common `DocumentParser` interface with one parser implementation per document type.

**Reason:** Makes the ingestion pipeline extensible without modifying existing parser implementations.

---

## D006 - Chunking Strategy

**Choice:** Implement a custom recursive chunking algorithm.

**Reason:** Preserves page boundaries, supports configurable overlap, and provides full control over chunk size and splitting behavior.

---

## D007 - Embedding Model

**Choice:** `BAAI/bge-small-en-v1.5`

**Reason:** Produces high-quality semantic embeddings while remaining lightweight enough for local inference. The 384-dimensional embeddings align with the pgvector schema.

---

## D008 - Embedding Architecture

**Choice:** Keep embedding generation independent from persistence.

**Reason:** Separates embedding computation from storage, improving modularity, testing, and future reuse.

---

## D009 - Vector Database

**Choice:** PostgreSQL with pgvector.

**Reason:** Stores both document metadata and vector embeddings in a single database while enabling efficient vector similarity search.

---

## D010 - Retrieval Strategy

**Choice:** Dense vector retrieval using pgvector followed by cross-encoder reranking.

**Reason:** Dense retrieval efficiently identifies semantically relevant chunks, while reranking improves result quality before passing context to the language model.

---

## D011 - Reranking Model

**Choice:** Cross-Encoder reranker from Sentence Transformers.

**Reason:** Scores retrieved chunks jointly with the user query, producing more accurate rankings than embedding similarity alone.

---

## D012 - LLM Provider

**Choice:** OpenRouter

**Reason:** Provides a provider-independent interface for accessing multiple language models without coupling the application to a single vendor.

---

## D013 - Generation Architecture

**Choice:** Retrieval-Augmented Generation (RAG)

**Reason:** Grounds generated responses in retrieved document context, reducing hallucinations and enabling citation-backed answers.

---

## D014 - Indexing Pipeline

**Choice:** Perform indexing immediately after document upload.

**Reason:** Ensures every uploaded document is immediately searchable and removes the need for a separate indexing workflow.

---

## D015 - Layered Project Structure

**Choice:** Separate ingestion, chunking, embeddings, retrieval, reranking, generation, and API into independent modules.

**Reason:** Promotes separation of concerns, improves maintainability, and allows individual components to evolve independently.