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