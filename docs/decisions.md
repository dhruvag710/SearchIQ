# SearchIQ Decisions

## D001 - Backend

**Choice:** FastAPI

**Reason:** Fast, async, automatic API docs, widely used for AI backends.

---

## D002 - PDF Storage

**Choice:** Store uploaded PDFs in `data/raw` using UUID filenames.

**Reason:** Prevent filename conflicts while keeping the original filename as metadata.

---

## D003 - PDF Parsing

**Choice:** PyMuPDF

**Reason:** Fast, reliable text extraction with page-level access.

---

## D004 - Architecture

**Choice:** Keep parsing logic independent from FastAPI.

**Reason:** Easier testing and future reuse.

## D005 - Multi-format Parsing

**Choice:** Use a common `DocumentParser` interface with one parser implementation per document type.

**Reason:** Keeps ingestion extensible and allows adding new document formats without changing existing parsers.

---

## D006 - Chunking Strategy

**Choice:** Implement a custom recursive chunking algorithm instead of relying on framework-provided splitters.

**Reason:** Preserves page boundaries, supports controlled overlap, and gives complete control over chunking behavior.

---

## D007 - Embedding Model

**Choice:** BAAI/bge-small-en-v1.5

**Reason:** Provides strong retrieval quality with 384-dimensional embeddings, matching the database schema while remaining lightweight enough for local inference.

---

## D008 - Embedding Architecture

**Choice:** Keep embedding generation independent from database writes.

**Reason:** Separates responsibilities, simplifies testing, and allows the embedding module to be reused without storage concerns.

---

## D009 - Database

**Choice:** PostgreSQL with pgvector.

**Reason:** Stores metadata and vector embeddings in one database while enabling efficient similarity search.