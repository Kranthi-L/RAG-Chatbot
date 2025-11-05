# ingest.py
import os
import glob
import json
from typing import List, Optional

from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain (new-style imports)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = "data"        # expects subfolders per course, e.g., data/networking/, data/architecture/
DB_DIR = "chroma_db"     # Chroma index folder (created automatically)
RANGES_FN = "ranges.json"  # optional, can be absent

# ---- Embeddings (local, free) -----------------------------------------------
# all-MiniLM-L6-v2 is small & good; normalization helps cosine similarity
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

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

# ---- Collect PDFs ------------------------------------------------------------
# Find all PDFs under data/* (any depth) so you can have many PDFs per course
paths = sorted(glob.glob(f"{DATA_DIR}/**/*.pdf", recursive=True))
if not paths:
    raise SystemExit("No PDFs found under ./data (tip: put files under data/<course>/YourFile.pdf).")

# Optional page ranges per file (key = basename, e.g., 'book1.pdf')
ranges = {}
if os.path.exists(RANGES_FN):
    with open(RANGES_FN, "r") as f:
        ranges = json.load(f)

# ---- Load raw page-documents with metadata ----------------------------------
raw_docs: List[Document] = []
print(f"Discovered {len(paths)} PDF file(s).")
for p in paths:
    base = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p)).lower()  # course label from folder name
    course = parent if parent else "unknown"

    try:
        n_pages = len(PdfReader(p).pages)
    except Exception:
        n_pages = None

    # If ranges.json exists and has an entry for this basename, use it; else full file
    page_idxs = parse_ranges(ranges.get(base, ""), n_pages) if n_pages else None

    print(f"- Ingesting: {p}")
    if page_idxs:
        preview = [i + 1 for i in page_idxs[:8]]
        print(f"  Using page ranges (1-based): {preview}{' ...' if len(page_idxs) > 8 else ''} (total {len(page_idxs)})")
    else:
        print("  Using: ALL pages")

    docs = load_pdf_subset(p, page_idxs)

    # tag each page with metadata for filtering/citation later
    for d in docs:
        d.metadata["course"] = course
        d.metadata["filename"] = base
    raw_docs.extend(docs)

print(f"Loaded {len(raw_docs)} page-doc(s) from {len(paths)} file(s).")

# ---- Chunking ----------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunk(s).")

# ---- Build / persist Chroma index -------------------------------------------
# Pass an Embeddings object (not a raw function) for compatibility
vs = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
)
vs.persist()
print(f"Indexed {len(chunks)} chunks → {DB_DIR}")
