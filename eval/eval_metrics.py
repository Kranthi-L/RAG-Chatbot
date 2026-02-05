#!/usr/bin/env python
"""
Aggregate metrics for judged_* CSVs.

Scans eval/ for files named judged_{course}_temp{TEMP}_topk{K}.csv,
computes per-model metrics, and writes eval/summary_metrics.csv.
Run:
    python eval/eval_metrics.py
"""
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "summary_metrics.csv"


def parse_filename(path: Path) -> Tuple[str, float, int]:
    """
    Parse course, temperature, top_k from judged_{course}_temp{TEMP}_topk{K}.csv
    TEMP uses 'p' instead of '.'.
    """
    m = re.match(r"judged_(.+)_temp([^_]+)_topk(\d+)(?:_.*)?\.csv", path.name)
    if not m:
        raise ValueError(f"Filename does not match pattern: {path.name}")
    course = m.group(1)
    temp_raw = m.group(2).replace("p", ".")
    try:
        temp = float(temp_raw)
    except ValueError:
        temp = 0.0
    top_k = int(m.group(3))
    return course, temp, top_k


def tokenize(text: str) -> List[str]:
    return text.strip().split()


def rouge_l_f1(ref: str, hyp: str) -> float:
    """
    Compute ROUGE-L F1 between two strings using LCS.
    """
    r_tokens = tokenize(ref)
    h_tokens = tokenize(hyp)
    if not r_tokens or not h_tokens:
        return 0.0

    # LCS dynamic programming
    dp = [[0] * (len(h_tokens) + 1) for _ in range(len(r_tokens) + 1)]
    for i in range(1, len(r_tokens) + 1):
        for j in range(1, len(h_tokens) + 1):
            if r_tokens[i - 1] == h_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    recall = lcs / len(r_tokens)
    precision = lcs / len(h_tokens)
    if recall + precision == 0:
        return 0.0
    return (2 * recall * precision) / (recall + precision)


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    if not hypotheses:
        return 0.0
    refs = [[tokenize(r)] for r in references]
    hyps = [tokenize(h) for h in hypotheses]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothie = SmoothingFunction().method1
        try:
            return float(corpus_bleu(refs, hyps, smoothing_function=smoothie))
        except Exception:
            return 0.0


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    if not hypotheses:
        return 0.0
    scores = [rouge_l_f1(r, h) for r, h in zip(references, hypotheses)]
    return float(np.mean(scores)) if scores else 0.0


def compute_bert_sim(
    model: SentenceTransformer, references: List[str], hypotheses: List[str], batch_size: int = 32
) -> float:
    if not hypotheses:
        return 0.0
    ref_emb = model.encode(references, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    hyp_emb = model.encode(hypotheses, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    sims = np.sum(ref_emb * hyp_emb, axis=1)  # cosine sim due to normalization
    return float(np.mean(sims)) if sims.size else 0.0


def process_file(path: Path, model: SentenceTransformer, rows_out: List[Dict[str, object]]) -> None:
    try:
        course, temp, top_k = parse_filename(path)
    except ValueError as e:
        print(f"Skipping {path.name}: {e}", file=sys.stderr)
        return

    stem = path.stem  # e.g., judged_networking_temp0p2_topk8_hybrid
    retriever_type = "dense"
    if "_hybrid" in stem:
        retriever_type = "hybrid"
    elif "_bm25" in stem:
        retriever_type = "bm25"
    elif "_section_aware" in stem:
        retriever_type = "section_aware"
    elif "_dense" in stem:
        retriever_type = "dense"

    try:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        print(f"Skipping {path.name}: failed to read ({e})", file=sys.stderr)
        return

    required = {"question", "ideal_answer", "gpt_response", "claude_response", "gpt_score", "claude_score"}
    if not required.issubset(df.columns):
        print(f"Skipping {path.name}: missing columns {required - set(df.columns)}", file=sys.stderr)
        return

    if df.empty:
        print(f"Skipping {path.name}: empty file", file=sys.stderr)
        return

    for model_name, resp_col, score_col in [
        ("gpt", "gpt_response", "gpt_score"),
        ("claude", "claude_response", "claude_score"),
    ]:
        try:
            answers = df[resp_col].fillna("").astype(str).tolist()
            ideals = df["ideal_answer"].fillna("").astype(str).tolist()

            mask = [(bool(a.strip()) and bool(b.strip())) for a, b in zip(answers, ideals)]
            answers_f = [a for a, m in zip(answers, mask) if m]
            ideals_f = [b for b, m in zip(ideals, mask) if m]

            if not answers_f:
                print(f"{path.name} [{model_name}]: no answers to score, skipping", file=sys.stderr)
                continue

            num_questions = len(answers_f)
            avg_answer_length = float(np.mean([len(a) for a in answers_f])) if answers_f else 0.0
            bleu = compute_bleu(list(ideals_f), list(answers_f))
            rouge_l = compute_rouge_l(list(ideals_f), list(answers_f))
            bert_sim = compute_bert_sim(model, list(ideals_f), list(answers_f))

            judge_scores = df.loc[mask, score_col].astype(float) if any(mask) else pd.Series(dtype=float)
            avg_judge_score = float(judge_scores.mean()) if not judge_scores.empty else 0.0

            rows_out.append(
                {
                    "course": course,
                    "temperature": temp,
                    "top_k": top_k,
                    "retriever_type": retriever_type,
                    "model": model_name,
                    "num_questions": num_questions,
                    "avg_answer_length": avg_answer_length,
                    "bleu": bleu,
                    "rouge_l": rouge_l,
                    "bert_sim": bert_sim,
                    "avg_judge_score": avg_judge_score,
                }
            )
        except Exception as e:
            print(f"Error processing {path.name} [{model_name}]: {e}", file=sys.stderr)


def main() -> None:
    judged_files = sorted(BASE_DIR.glob("judged_*_temp*_topk*.csv"))
    if not judged_files:
        print("No judged_* files found in eval/", file=sys.stderr)
        return

    print(f"Found {len(judged_files)} judged files.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rows_out: List[Dict[str, object]] = []

    for f in judged_files:
        print(f"Processing {f.name} ...")
        process_file(f, model, rows_out)

    if not rows_out:
        print("No metrics generated; nothing to write.", file=sys.stderr)
        return

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"Wrote summary for {len(rows_out)} rows across {len(judged_files)} files → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
