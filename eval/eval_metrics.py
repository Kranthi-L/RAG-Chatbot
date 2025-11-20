# eval/eval_metrics.py
import os
import re
import csv
import time
from typing import List, Dict, Tuple

from dotenv import load_dotenv

# Metrics
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore
from sacrebleu.metrics import BLEU


# Optional: only needed if you want to generate missing answers on the fly
GENERATE_IF_MISSING = os.getenv("GENERATE_IF_MISSING", "0") == "1"
TOP_K = int(os.getenv("TOP_K", "6"))
COURSE = os.getenv("COURSE", "networking")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
PAUSE = float(os.getenv("PAUSE", "0.25"))  # brief pause between API calls

# File paths
INPUT_CSV = os.getenv("INPUT_CSV", "eval/networking_eval_filled.csv")
OUTPUT_PER_Q = os.getenv("OUTPUT_PER_Q", "eval/networking_metrics.csv")

# If generating missing answers, we’ll reuse your RAG bits
if GENERATE_IF_MISSING:
    from langchain_community.vectorstores import Chroma
    from app_cli import build_context, ask_gpt, ask_claude
    DB_DIR = os.getenv("DB_DIR", "chroma_db")
    SYSTEM_PATH = os.getenv("SYSTEM_PROMPT", "prompts/qa_system.md")
    vs = Chroma(persist_directory=DB_DIR)
    with open(SYSTEM_PATH, "r") as f:
        SYSTEM = f.read()

load_dotenv()


# ---------- Text normalization (for fair, reproducible metrics) ----------
def clean_text(s: str) -> str:
    if not s:
        return ""
    # Remove markdown
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)

    # Remove LaTeX math environments: \[ ... \], \( ... \), $$...$$
    s = re.sub(r"\\\[.*?\\\]", " ", s)
    s = re.sub(r"\\\(.*?\\\)", " ", s)
    s = re.sub(r"\$\$(.*?)\$\$", " ", s)
    s = re.sub(r"\$(.*?)\$", " ", s)

    # Remove LaTeX commands like \frac{a}{b}, \text{...}, \alpha, etc.
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)

    # Strip list markers and normalize whitespace
    s = re.sub(r"(?m)^\s*[-*]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()



# ---------- CSV loading ----------
def load_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"question", "gpt_response", "claude_response", "ideal_answer"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns {need}. Found: {reader.fieldnames}")
        for r in reader:
            rows.append(r)
    return rows


# ---------- Optional generation for missing answers ----------
def retrieve_docs(question: str):
    # Filter to course if possible
    filt = None if COURSE in (None, "", "all") else {"course": {"$eq": COURSE}}
    try:
        pairs = (
            vs.similarity_search_with_relevance_scores(question, k=TOP_K, filter=filt)
            if filt
            else vs.similarity_search_with_relevance_scores(question, k=TOP_K)
        )
        docs = [p[0] for p in pairs]
    except Exception:
        docs = vs.similarity_search(question, k=TOP_K) if filt is None else vs.similarity_search(question, k=TOP_K, filter=filt)
    return docs


def generate_if_missing(row: Dict[str, str]) -> Tuple[str, str]:
    """Return (gpt_resp, claude_resp) – use existing if present; optionally generate if empty."""
    gpt_resp = (row.get("gpt_response") or "").strip()
    claude_resp = (row.get("claude_response") or "").strip()

    if not GENERATE_IF_MISSING:
        return gpt_resp, claude_resp

    # Build context once if either is missing
    ctx = None
    if not gpt_resp or not claude_resp:
        docs = retrieve_docs(row["question"])
        ctx = build_context(docs) if docs else "(No retrieved context.)"

    if not gpt_resp:
        gpt_resp, _ = ask_gpt(SYSTEM, row["question"], ctx)
        time.sleep(PAUSE)
    if not claude_resp:
        claude_resp, _ = ask_claude(SYSTEM, row["question"], ctx)
        time.sleep(PAUSE)

    return gpt_resp, claude_resp


# ---------- Metric helpers ----------
def compute_rouge(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    R = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for h, r in zip(hyps, refs):
        s = R.score(r, h)
        for k in sums:
            sums[k] += s[k].fmeasure
    n = max(1, len(hyps))
    return {k: v / n for k, v in sums.items()}


def compute_bleu(hyps: list[str], refs: list[str]) -> dict[str, float]:
    # BLEU-4 (default max_ngram_order=4)
    bleu4_metric = BLEU(effective_order=True, smooth_method="exp")
    bleu4 = bleu4_metric.corpus_score(hyps, [refs]).score

    # BLEU-1 (set max_ngram_order=1)
    bleu1_metric = BLEU(effective_order=True, smooth_method="exp", max_ngram_order=1)
    bleu1 = bleu1_metric.corpus_score(hyps, [refs]).score

    return {"bleu1": bleu1, "bleu4": bleu4}



def compute_bertscore(hyps: List[str], refs: List[str]) -> float:
    # Returns average F1
    _, _, F = bertscore(hyps, refs, lang="en")
    return float(F.mean())


def pretty_print_block(title: str, metrics: Dict[str, float]):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}")


# ---------- Main ----------
def main():
    rows = load_rows(INPUT_CSV)

    refs_clean: List[str] = []
    gpt_clean: List[str] = []
    claude_clean: List[str] = []

    # Optional: write per-question metrics
    perq_rows: List[Dict[str, str]] = []

    for i, row in enumerate(rows, 1):
        q = row["question"]
        ref = clean_text(row["ideal_answer"])
        gpt_resp, claude_resp = generate_if_missing(row)

        gpt = clean_text(gpt_resp)
        cld = clean_text(claude_resp)

        refs_clean.append(ref)
        gpt_clean.append(gpt)
        claude_clean.append(cld)

    # ---- Aggregate metrics (GPT) ----
    gpt_rouge = compute_rouge(gpt_clean, refs_clean)
    gpt_bleu = compute_bleu(gpt_clean, refs_clean)
    gpt_bert = compute_bertscore(gpt_clean, refs_clean)
    gpt_all = {**gpt_rouge, **gpt_bleu, "bertscore_f1": gpt_bert}
    pretty_print_block("GPT", gpt_all)

    # ---- Aggregate metrics (Claude) ----
    cld_rouge = compute_rouge(claude_clean, refs_clean)
    cld_bleu = compute_bleu(claude_clean, refs_clean)
    cld_bert = compute_bertscore(claude_clean, refs_clean)
    cld_all = {**cld_rouge, **cld_bleu, "bertscore_f1": cld_bert}
    pretty_print_block("Claude", cld_all)

    # ---- Per-question metrics CSV (optional but useful for analysis) ----
    # We’ll compute per-question ROUGE-L and BERTScore F1 (most interpretable),
    # and store raw cleaned strings for auditability.
    R = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    perq_header = [
        "question", "gpt_response_clean", "claude_response_clean", "ideal_answer_clean",
        "rougeL_gpt", "rougeL_claude", "bertscore_f1_gpt", "bertscore_f1_claude"
    ]

    # To avoid many tiny BERTScore calls, we compute once per model for all rows:
    # (We already computed aggregate above, but we also need per-question.)
    # Run BERTScore again with rescale=True? Typically not needed; keep consistent with aggregate.
    P_g, R_g, F_g = bertscore(gpt_clean, refs_clean, lang="en")
    P_c, R_c, F_c = bertscore(claude_clean, refs_clean, lang="en")

    for idx, row in enumerate(rows):
        q = row["question"]
        ref = refs_clean[idx]
        gpt = gpt_clean[idx]
        cld = claude_clean[idx]
        rougeL_g = R.score(ref, gpt)["rougeL"].fmeasure
        rougeL_c = R.score(ref, cld)["rougeL"].fmeasure
        perq_rows.append({
            "question": q,
            "gpt_response_clean": gpt,
            "claude_response_clean": cld,
            "ideal_answer_clean": ref,
            "rougeL_gpt": f"{rougeL_g:.4f}",
            "rougeL_claude": f"{rougeL_c:.4f}",
            "bertscore_f1_gpt": f"{float(F_g[idx]):.4f}",
            "bertscore_f1_claude": f"{float(F_c[idx]):.4f}",
        })

    os.makedirs(os.path.dirname(OUTPUT_PER_Q), exist_ok=True)
    with open(OUTPUT_PER_Q, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=perq_header)
        w.writeheader()
        for r in perq_rows:
            w.writerow(r)
    print(f"\nPer-question metrics written → {OUTPUT_PER_Q}")


if __name__ == "__main__":
    main()
