#!/usr/bin/env python
"""
ablation_generate_answers.py

Generate answers for the accessibility prompt ablation study.

For each question in ablation_questions.csv, this script:
  - Retrieves context from the RAG vector store (Chroma) using rag_core.py
  - Generates answers for:
        model in {gpt, claude}
        condition in {baseline, accessible}
  - Uses per-course, per-model best (temperature, top_k) configs
  - Writes all outputs to ablation_answers.csv

Each question is thus answered 4 times:
  (gpt, baseline), (gpt, accessible), (claude, baseline), (claude, accessible)
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# Ensure we can import rag_core from project root
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PARENT_DIR.parent

for p in [SCRIPT_DIR, PARENT_DIR, ROOT_DIR]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from rag_core import retrieve_docs, build_context, ask_gpt, ask_claude
except ImportError as e:
    raise ImportError(
        "Could not import rag_core. Make sure this script is located in "
        "eval/ablation/ and rag_core.py is in the project root."
    ) from e


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ABLATION_Q_PATH = SCRIPT_DIR / "ablation_questions.csv"
OUTPUT_PATH = SCRIPT_DIR / "ablation_answers.csv"


# ----------------------------------------------------------------------
# Best configs per (course, model)
# These are taken from your earlier select_best_configs.py output.
# ----------------------------------------------------------------------

# (course, model) -> (temperature, top_k)
BEST_CONFIGS: Dict[Tuple[str, str], Tuple[float, int]] = {
    ("architecture", "claude"): (1.0, 6),
    ("architecture", "gpt"): (0.5, 8),
    ("machine_learning", "claude"): (0.2, 8),
    ("machine_learning", "gpt"): (1.0, 8),
    ("networking", "claude"): (0.0, 8),
    ("networking", "gpt"): (0.2, 8),
}


# ----------------------------------------------------------------------
# System prompts
# ----------------------------------------------------------------------

SYSTEM_BASELINE = """You are a course Q&A assistant for university-level students.
Answer the user's question using ONLY the provided context.
If the answer is not clearly supported by the context, say "I don't know" and do not guess.
Always cite the source filename and page number when possible.
"""

SYSTEM_ACCESSIBLE = """You are a course Q&A assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don’t know.
Always cite the source filename and page if available.
Be concise and structured.

When you need to mention any formulas, equations, or math expressions:
- Do NOT use LaTeX, TeX, or math markup of any kind.
- Do NOT use $, $$, backslash notation, subscripts, or superscripts.
- Instead, always describe formulas in plain English words.

Example:
- Instead of: “The transmission delay is d_trans equals L over R.”
- Say: “The transmission delay equals the packet length L divided by the link rate R.”
"""


def get_system_prompt(condition: str) -> str:
    if condition == "baseline":
        return SYSTEM_BASELINE
    if condition == "accessible":
        return SYSTEM_ACCESSIBLE
    raise ValueError(f"Unknown condition: {condition}")


# ----------------------------------------------------------------------
# Core generation logic
# ----------------------------------------------------------------------

def load_questions() -> pd.DataFrame:
    if not ABLATION_Q_PATH.exists():
        raise FileNotFoundError(f"Missing ablation_questions.csv at {ABLATION_Q_PATH}")
    df = pd.read_csv(ABLATION_Q_PATH)
    # Normalize basic fields
    if "course" in df.columns:
        df["course"] = df["course"].astype(str).str.strip().str.lower()
    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].astype(str).str.strip()
    return df


def main():
    df = load_questions()
    print(f"Loaded {len(df)} questions from {ABLATION_Q_PATH}")

    # Prepare output CSV with header
    fieldnames = [
        "course",
        "question_id",
        "question",
        "ideal_answer",
        "question_type",
        "model",
        "condition",
        "temperature",
        "top_k",
        "latency_ms",
        "response",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = len(df)
        row_idx = 0

        for _, row in df.iterrows():
            row_idx += 1
            course = str(row["course"]).strip().lower()
            qid = str(row["question_id"]).strip()
            question = str(row["question"])
            ideal_answer = str(row.get("ideal_answer", ""))
            qtype = str(row.get("question_type", ""))

            print(f"\n=== [{row_idx}/{total_rows}] course={course}, question_id={qid} ===")
            print(f"Question: {question}")

            for model in ["gpt", "claude"]:
                key = (course, model)
                if key not in BEST_CONFIGS:
                    print(f"  [WARN] No BEST_CONFIGS entry for (course={course}, model={model}); skipping.")
                    continue

                temperature, top_k = BEST_CONFIGS[key]
                print(f"  -> Model={model.upper()}, temperature={temperature}, top_k={top_k}")

                # Retrieve context ONCE per (question, course, model)
                docs = retrieve_docs(query=question, course=course, top_k=top_k, last_assistant=None)
                context = build_context(docs)

                for condition in ["baseline", "accessible"]:
                    system_prompt = get_system_prompt(condition)

                    # Call the appropriate model function
                    t0 = time.time()
                    if model == "gpt":
                        answer, _ = ask_gpt(
                            system=system_prompt,
                            question=question,
                            context=context,
                            summary="",
                            temperature=temperature,
                        )
                    else:
                        answer, _ = ask_claude(
                            system=system_prompt,
                            question=question,
                            context=context,
                            summary="",
                            temperature=temperature,
                        )
                    latency_ms = (time.time() - t0) * 1000.0

                    print(f"    [{model}/{condition}] Answer length={len(answer)} chars, latency={latency_ms:.1f} ms")

                    writer.writerow(
                        {
                            "course": course,
                            "question_id": qid,
                            "question": question,
                            "ideal_answer": ideal_answer,
                            "question_type": qtype,
                            "model": model,
                            "condition": condition,
                            "temperature": temperature,
                            "top_k": top_k,
                            "latency_ms": f"{latency_ms:.2f}",
                            "response": answer,
                        }
                    )
                    f_out.flush()

    print(f"\nDone. Saved answers to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
