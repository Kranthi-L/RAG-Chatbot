"""
Generate weak relevance labels via LLM for retrieval evaluation.
Outputs JSONL at eval/data/relevance_labels.jsonl with fields:
course, question_id, question, doc_id, raw_score, label, metadata.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_core import retrieve_docs, RetrieverType

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "relevance_labels.jsonl"

COURSE_TO_CSV = {
    "networking": REPO_ROOT / "eval" / "networking_eval.csv",
    "architecture": REPO_ROOT / "eval" / "architecture_eval.csv",
    "machine_learning": REPO_ROOT / "eval" / "machine_learning_eval.csv",
}

MAX_CANDIDATES = 30
LLM_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")


def doc_id_from_meta(meta: Dict, fallback_idx: int) -> str:
    course = meta.get("course", "unknown")
    filename = meta.get("filename", meta.get("source", "unknown"))
    page = meta.get("page", "0")
    chunk_idx = meta.get("chunk_index", fallback_idx)
    return f"{course}|{filename}|{page}|{chunk_idx}"


def load_questions(course: str) -> List[Dict]:
    path = COURSE_TO_CSV[course]
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        import csv

        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            if not q or q.startswith("#"):
                continue
            rows.append({"question": q, "ideal_answer": (r.get("ideal_answer") or "").strip()})
    return rows


def judge_relevance(question: str, ideal_answer: str, chunk_text: str) -> float:
    """
    Ask LLM for relevance score in [0,1]. Returns 0.0 on failure.
    """
    prompt = (
        "You are scoring retrieval relevance.\n"
        "Given a question, an ideal answer, and a candidate chunk, output a JSON object with a single key 'score' in [0,1].\n"
        "Guidance:\n"
        "- 0.0: not relevant\n"
        "- ~0.3: weak/partial relevance\n"
        "- ~0.7+: strongly supports the ideal answer\n"
        "Respond ONLY with JSON like {\"score\": 0.75}.\n\n"
        f"Question: {question}\n"
        f"Ideal Answer: {ideal_answer}\n"
        f"Chunk: {chunk_text[:500]}\n"
    )
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
        resp = llm.invoke([("user", prompt)]).content
        data = json.loads(resp)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score
    except Exception:
        return 0.0


def label_from_score(score: float) -> int:
    if score >= 0.66:
        return 2
    if score >= 0.33:
        return 1
    return 0


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for course in COURSE_TO_CSV.keys():
            questions = load_questions(course)
            for qid, row in enumerate(questions):
                q = row["question"]
                ideal = row["ideal_answer"]
                docs = retrieve_docs(
                    query=q,
                    course=course,
                    top_k=MAX_CANDIDATES,
                    retriever_type=RetrieverType.HYBRID,
                    learner_level=None,
                )
                for idx, d in enumerate(docs):
                    doc_id = doc_id_from_meta(d.metadata, idx)
                    score = judge_relevance(q, ideal, d.page_content)
                    label = label_from_score(score)
                    rec = {
                        "course": course,
                        "question_id": qid,
                        "question": q,
                        "doc_id": doc_id,
                        "raw_score": score,
                        "label": label,
                        "metadata": d.metadata,
                        "retriever": "hybrid",
                    }
                    out_f.write(json.dumps(rec) + "\n")
                print(f"[{course}] labeled question {qid+1}/{len(questions)}")
    print(f"Wrote relevance labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
