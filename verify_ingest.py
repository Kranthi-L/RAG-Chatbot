# verify_ingest.py
import os
from langchain_community.vectorstores import Chroma


vs = Chroma(persist_directory="chroma_db")

# Detect the metadata key your ingest used: "course" (folder-based) or "book" (filename-based)
meta_key = "course"
try:
    probe = vs.similarity_search("test", k=1)
    if probe and meta_key not in probe[0].metadata:
        meta_key = "book" if "book" in probe[0].metadata else meta_key
except Exception:
    pass

probes = [
    "application layer",      # networking
    "congestion control",     # networking
    "pipeline hazards",       # architecture
    "cache coherence"         # architecture
]

for q in probes:
    docs = vs.similarity_search(q, k=3)
    print(f"\nQuery: {q!r} → {len(docs)} chunks")
    for i, d in enumerate(docs, 1):
        label = d.metadata.get(meta_key, "unknown")
        src = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        print(f"  [{i}] {label} | {src} p.{page}")
        # optional peek at text:
        print("     ", (d.page_content[:300] + "...").replace("\n", " "))
