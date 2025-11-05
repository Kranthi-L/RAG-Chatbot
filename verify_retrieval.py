# verify_retrieval.py
import csv
from langchain_community.vectorstores import Chroma

vs = Chroma(persist_directory="chroma_db")

# auto-detect meta key
meta_key = "course"
probe = vs.similarity_search("test", k=1)
if probe and meta_key not in probe[0].metadata:
    if "book" in probe[0].metadata:
        meta_key = "book"

rows = list(csv.DictReader(open("eval/retrieval.csv")))

hits = 0
for r in rows:
    q = r["question"]
    kw = r["should_contain"].lower()
    target = r.get("course", "all").lower()

    if target in ("all", "", None):
        docs = vs.similarity_search(q, k=4)
    else:
        filt = {meta_key: {"$eq": target}}
        try:
            docs = vs.similarity_search(q, k=4, filter=filt)
        except TypeError:
            docs = vs.similarity_search(q, k=4)

    joined = " ".join([d.page_content.lower() for d in docs])
    ok = kw in joined
    hits += 1 if ok else 0
    print(f"[{target}] {q} → {'YES' if ok else 'NO'}")

print(f"\nOverall Hit@4 = {hits}/{len(rows)} = {hits/len(rows):.2f}")
