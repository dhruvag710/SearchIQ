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

**Choice:** Hybrid retrieval combining Dense vector search (`BAAI/bge-small-en-v1.5`) and BM25 lexical search via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking.

**Reason:** Combining lexical BM25 and semantic dense search maximizes candidate recall across keyword-heavy and concept-based queries, while Reciprocal Rank Fusion provides robust, scale-invariant score merging.

---

## D011 - Reranking Model

**Choice:** Cross-Encoder reranker (`BAAI/bge-reranker-base`) from Sentence Transformers.

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

---

## D016 - Multimodal PDF Understanding and Visual Indexing

**Choice:** Extract embedded raster images from PDFs using PyMuPDF, generate dense factual descriptions via OpenRouter VLM (`google/gemini-2.5-flash`), compute `bge-small-en-v1.5` embeddings for the captions, and store them in the `document_images` table with pgvector embeddings.

**Reason:** Captures critical document knowledge trapped in embedded figures, charts, and diagrams into searchable semantic representations without requiring high-latency multi-modal vector models at query time.

---

## D017 - Unified Text and Visual Candidate Reranking

**Choice:** Merge top text candidates from `HybridRetriever` with top visual candidates from `VisualSearchService` into a unified candidate pool, and jointly score them using `BAAI/bge-reranker-base`.

**Reason:** Allows the cross-encoder to evaluate text chunks and visual image captions against the query in the same scoring pass, ensuring the most relevant context wins regardless of its source modality.

---

## D018 - Multi-format Document Parsing

**Choice:** Implement dedicated, modular parsers for PDF (`MultimodalPdfParser` / `PdfParser`), DOCX (`DocxParser`), Markdown (`MarkdownParser`), and Plain Text (`TxtParser`) conforming to a common `DocumentParser` interface.

**Reason:** Broadens document support across common enterprise file types while isolating format-specific dependencies and producing standardized text pages for downstream chunking.

---

## D019 - Retrieval Evaluation Framework

**Choice:** Build an offline benchmark suite (`app/evaluation/`) calculating Recall@K, Precision@K, and MRR@K across Dense, BM25, Hybrid (RRF), and Hybrid + Reranker strategies using golden annotated test cases.

**Reason:** Provides empirical verification of retrieval accuracy and latency trade-offs, validating that reranking yields measurable precision improvements on domain data.