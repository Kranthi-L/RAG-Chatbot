# ingest.py
import os
import glob
import json
import shutil
import pickle
from enum import Enum
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain (new-style imports)
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

def sanitize_text(s: str) -> str:
    """Return a UTF-8 safe version of s, replacing invalid surrogates/etc."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")

load_dotenv()

DATA_DIR = "data"        # expects subfolders per course, e.g., data/networking/, data/architecture/
DB_DIR = "chroma_db"     # Chroma index folder (created automatically)
RANGES_FN = "ranges.json"  # optional, can be absent
COURSE_FILTER = os.getenv("COURSE_FILTER")

# Always rebuild a fresh DB to avoid duplicate entries
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

# ---- Embeddings -------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ConceptType(str, Enum):
    DEFINITION = "definition"
    EXAMPLE = "example"
    INTUITION = "intuition"
    PROCEDURE = "procedure"
    OTHER = "other"


USE_LLM_TAGGER = False  # placeholder for future tagging


def tag_chunk_level_and_type(text: str) -> Tuple[DifficultyLevel, ConceptType]:
    """
    Lightweight tagging hook. Currently returns defaults; can be replaced with LLM-based tagging.
    """
    _ = text
    return DifficultyLevel.INTERMEDIATE, ConceptType.OTHER


def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


# ---- Helpers ----------------------------------------------------------------
def parse_ranges(spec: str, max_pages: int) -> List[int]:
    """
    Parse a string like "1-40, 120-150, 200" (1-based page numbers) into
    a sorted list of 0-based page indices. If spec is empty/None, return [].
    """
    pages = set()
    if not spec:
        return []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = max(1, int(a)), min(max_pages, int(b))
            pages.update(range(a, b + 1))
        else:
            p = max(1, min(max_pages, int(part)))
            pages.add(p)
    # convert to zero-based indices
    return sorted([p - 1 for p in pages])

def load_pdf_subset(path: str, page_idxs: Optional[List[int]]) -> List[Document]:
    """
    If page_idxs is provided, extract only those pages as Documents.
    Otherwise, use PyPDFLoader to load the whole file.
    """
    if not page_idxs:
        return PyPDFLoader(path).load()

    reader = PdfReader(path)
    docs: List[Document] = []
    for i in page_idxs:
        if 0 <= i < len(reader.pages):
            text = reader.pages[i].extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": path, "page": i + 1}))
    return docs

# ---- Collect PDFs and PPTX ---------------------------------------------------
pdf_paths = sorted(glob.glob(f"{DATA_DIR}/**/*.pdf", recursive=True))
pptx_paths = sorted(glob.glob(f"{DATA_DIR}/**/*.pptx", recursive=True))
files = [(p, "pdf") for p in pdf_paths] + [(p, "pptx") for p in pptx_paths]
files = sorted(files, key=lambda x: x[0])

if COURSE_FILTER:
    cf_lower = COURSE_FILTER.lower()
    files = [f for f in files if os.path.basename(os.path.dirname(f[0])).lower() == cf_lower]
    print(f"Course filter: {COURSE_FILTER}")
else:
    print("Course filter: <none>")

if not files:
    raise SystemExit("No PDFs/PPTX found under ./data (tip: put files under data/<course>/YourFile.pdf).")

# Optional page ranges per file (key = basename, e.g., 'book1.pdf')
ranges = {}
if os.path.exists(RANGES_FN):
    with open(RANGES_FN, "r") as f:
        ranges = json.load(f)

# ---- Load raw page-documents with metadata ----------------------------------
raw_docs: List[Document] = []
print(f"Discovered {len(files)} file(s) (PDF + PPTX).")
for path, kind in files:
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path)).lower()  # course label from folder name
    course = parent if parent else "unknown"

    if kind == "pdf":
        try:
            n_pages = len(PdfReader(path).pages)
        except Exception:
            n_pages = None

        page_idxs = parse_ranges(ranges.get(base, ""), n_pages) if n_pages else None

        print(f"- Ingesting PDF: {path}")
        if page_idxs:
            preview = [i + 1 for i in page_idxs[:8]]
            print(f"  Using page ranges (1-based): {preview}{' ...' if len(page_idxs) > 8 else ''} (total {len(page_idxs)})")
        else:
            print("  Using: ALL pages")

        docs = load_pdf_subset(path, page_idxs)

        for d in docs:
            d.metadata["course"] = course
            d.metadata["filename"] = base
        raw_docs.extend(docs)

    elif kind == "pptx":
        print(f"- Ingesting PPTX: {path}")
        try:
            docs = UnstructuredPowerPointLoader(path).load()
        except Exception as e:
            print(f"  Skipped (failed to load): {e}")
            continue

        for i, d in enumerate(docs, start=1):
            if not hasattr(d, "metadata") or d.metadata is None:
                d.metadata = {}
            d.metadata.setdefault("course", course)
            d.metadata.setdefault("filename", base)
            d.metadata.setdefault("page", i)
        raw_docs.extend(docs)

print(f"Loaded {len(raw_docs)} page-doc(s) from {len(files)} file(s).")

# ---- Chunking ----------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunk(s).")

# ---- Debug: inspect raw page_content types per course -----------------------
course_type_counts = {}
non_str_examples = []
for i, doc in enumerate(chunks):
    pc = doc.page_content
    course = doc.metadata.get("course", "unknown")
    t = type(pc).__name__

    c_counts = course_type_counts.setdefault(course, {})
    c_counts[t] = c_counts.get(t, 0) + 1

    if not isinstance(pc, str) and len(non_str_examples) < 5:
        non_str_examples.append({
            "idx": i,
            "type": t,
            "course": course,
            "filename": doc.metadata.get("filename"),
            "sample": repr(pc)[:200],
        })

print("DEBUG: page_content type counts per course:", course_type_counts)
if non_str_examples:
    print("DEBUG: sample non-string page_content chunks:")
    for ex in non_str_examples:
        print("  -", ex)

# ---- Clean chunk contents before embedding ---------------------------------
clean_chunks: List[Document] = []
dropped = 0

for doc in chunks:
    pc = doc.page_content

    if pc is None:
        dropped += 1
        continue

    if not isinstance(pc, str):
        try:
            pc = str(pc)
        except Exception:
            dropped += 1
            continue

    pc = pc.strip()
    if not pc:
        dropped += 1
        continue

    doc.page_content = pc
    clean_chunks.append(doc)

print(f"Cleaned chunks: {len(chunks)} -> {len(clean_chunks)} (dropped {dropped})")

# Final safety check: ensure only strings remain
for i, d in enumerate(clean_chunks):
    if not isinstance(d.page_content, str):
        raise TypeError(
            f"Non-string after cleaning at index {i}: {type(d.page_content)} -> {d.page_content!r}"
        )

# ---- Sanitize text to ensure UTF-8 safety -----------------------------------
sanitized_docs = 0
for doc in clean_chunks:
    original_pc = doc.page_content
    safe_pc = sanitize_text(original_pc)
    if safe_pc != original_pc:
        sanitized_docs += 1
    doc.page_content = safe_pc

    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        for k, v in list(doc.metadata.items()):
            if isinstance(v, str):
                doc.metadata[k] = sanitize_text(v)

print(f"Sanitized UTF-8 for {sanitized_docs} document(s) before indexing.")

# ---- Tag difficulty level and concept type; assign chunk indices -----------
for idx, doc in enumerate(clean_chunks):
    level, ctype = tag_chunk_level_and_type(doc.page_content)
    doc.metadata["level"] = level.value
    doc.metadata["concept_type"] = ctype.value
    doc.metadata["chunk_index"] = idx

# ---- Build BM25 indices per course -----------------------------------------
bm25_by_course: dict[str, Tuple[BM25Okapi, List[dict]]] = {}
docs_by_course: dict[str, List[Document]] = {}
for d in clean_chunks:
    course = d.metadata.get("course", "unknown")
    docs_by_course.setdefault(course, []).append(d)

for course, docs in docs_by_course.items():
    tokenized = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_records = []
    for d in docs:
        rec = {
            "doc_id": f"{d.metadata.get('course')}|{d.metadata.get('filename')}|{d.metadata.get('page')}|{d.metadata.get('chunk_index')}",
            "text": d.page_content,
            "metadata": d.metadata,
        }
        bm25_records.append(rec)

    bm25_by_course[course] = (bm25, bm25_records)

# Persist BM25 artifacts
for course, (bm25, bm25_records) in bm25_by_course.items():
    idx_path = os.path.join(DB_DIR, f"bm25_{course}.pkl")
    docs_path = os.path.join(DB_DIR, f"bm25_docs_{course}.pkl")
    os.makedirs(DB_DIR, exist_ok=True)
    with open(idx_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(docs_path, "wb") as f:
        pickle.dump(bm25_records, f)
    print(f"Saved BM25 index for course '{course}' with {len(bm25_records)} docs -> {idx_path}")

# ---- Build / persist Chroma index -------------------------------------------
# Pass an Embeddings object (not a raw function) for compatibility
vs = Chroma.from_documents(
    documents=clean_chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
)
vs.persist()
print(f"Indexed {len(clean_chunks)} chunks → {DB_DIR}")
