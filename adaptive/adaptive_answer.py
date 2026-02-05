from __future__ import annotations

"""
NOTE: This module implements a prototype adaptive learning simulation.
It is NOT used in the main experiments for the paper and is kept as
future work / exploratory code.
"""

# Allow this script to be run both as a module and as a standalone script.
import os
import sys

if __package__ is None or __package__ == "":
    # When run as: python adaptive/adaptive_answer.py
    # we need to add the project root to sys.path so that
    # imports like `from adaptive.learner_model import ...` work.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import csv
import os
import time
from typing import Any, Dict, Optional

from adaptive.learner_model import get_state, update_state
from rag_core import retrieve_docs, build_context, answer_with_model
from eval.llm_judge_core import judge_answer


def _append_log_row(path: str, row_dict: Dict[str, Any]) -> None:
    """Append a single row to CSV, creating header if missing."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def adaptive_answer(
    user_id: str,
    course: str,
    question: str,
    model: str = "gpt",
    learner_profile: str | None = None,
    top_k: int = 6,
    temperature: float = 0.0,
    ideal_answer: Optional[str] = None,
    log_path: str = "adaptive/adaptive_logs.csv",
) -> Dict[str, Any]:
    """
    Run RAG with adaptive learner level and log results.
    """
    # Load learner state
    state = get_state(user_id, course)
    level_before = state.level

    # RAG retrieval
    docs = retrieve_docs(question, course, top_k=top_k)
    context = build_context(docs)

    # Generate answer
    start = time.time()
    try:
        answer, _lat_ms = answer_with_model(
            model=model,
            question=question,
            context=context,
            summary="",
            temperature=temperature,
            learner_level=level_before,  # for future use if supported
        )
    except TypeError:
        # fallback if learner_level not supported
        answer, _lat_ms = answer_with_model(
            model=model,
            question=question,
            context=context,
            summary="",
            temperature=temperature,
        )
    latency = time.time() - start

    # Judge correctness
    correctness_score = judge_answer(question, ideal_answer or "", answer)

    # Adjust performance based on learner_profile
    perf = correctness_score
    if learner_profile == "weak":
        perf = max(0.0, min(1.0, correctness_score * 0.6))
    elif learner_profile == "strong":
        perf = max(0.0, min(1.0, correctness_score * 1.1))

    # Update learner state
    updated_state = update_state(state, perf)

    # Log
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    row = {
        "user_id": user_id,
        "course": course,
        "question": question,
        "model": model,
        "level_before": level_before,
        "level_after": updated_state.level,
        "correctness_score": correctness_score,
        "profile_performance": perf,
        "learner_profile": learner_profile or "",
        "latency": latency,
        "top_k": top_k,
        "temperature": temperature,
        "timestamp": timestamp,
    }
    _append_log_row(log_path, row)

    return {
        "answer": answer,
        "correctness": correctness_score,
        "profile_performance": perf,
        "learner_profile": learner_profile,
        "level_before": level_before,
        "level_after": updated_state.level,
        "latency": latency,
        "state": updated_state.__dict__,
    }


if __name__ == "__main__":
    result = adaptive_answer(
        user_id="demo_user",
        course="architecture",
        question="Explain Amdahl's Law.",
        model="gpt",
        top_k=6,
        temperature=0.0,
        ideal_answer="Amdahl's law defines speedup limits under partial parallelizability.",
    )
    print(result)
