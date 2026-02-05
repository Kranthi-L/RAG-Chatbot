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
    # When run as: python adaptive/simulate_learners.py
    # we need to add the project root to sys.path so that
    # imports like `from adaptive.adaptive_answer import ...` work.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import csv
from datetime import datetime
from typing import Any, Dict, List, Literal

from rag_core import retrieve_docs, build_context, answer_with_model
from eval.llm_judge_core import judge_answer
from adaptive.adaptive_answer import adaptive_answer
from adaptive.learner_model import reset_state

Strategy = Literal["adaptive", "fixed_beginner", "fixed_intermediate", "fixed_advanced"]

# Hard-coded best configs
BEST_CONFIGS: Dict[str, Dict[str, Dict[str, float]]] = {
    "architecture": {
        "gpt": {"top_k": 8, "temperature": 0.5},
        "claude": {"top_k": 6, "temperature": 1.0},
    },
    "machine_learning": {
        "gpt": {"top_k": 8, "temperature": 1.0},
        "claude": {"top_k": 8, "temperature": 0.2},
    },
    "networking": {
        "gpt": {"top_k": 8, "temperature": 0.2},
        "claude": {"top_k": 8, "temperature": 0.0},
    },
}


def get_best_config(course: str, model: str) -> Dict[str, float]:
    course = course.lower()
    if course not in BEST_CONFIGS or model not in BEST_CONFIGS[course]:
        raise ValueError(f"No best config for course={course}, model={model}")
    return BEST_CONFIGS[course][model]


def load_questions_for_course(course: str) -> List[Dict[str, Any]]:
    """
    Load questions and ideal answers for a given course.

    Assumes CSVs are stored in the eval/ folder with names like:
        eval/architecture_eval.csv
        eval/machine_learning_eval.csv
        eval/networking_eval.csv

    and have at least the following columns:
        question, gpt_response, claude_response, ideal_answer

    We only use 'question' and 'ideal_answer' here.
    """
    base_dir = "eval"
    filename = f"{course}_eval.csv"
    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Question file not found for course={course}: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            ideal = (row.get("ideal_answer") or "").strip()

            # Skip empty rows or chapter headers without an ideal answer
            if not q or not ideal:
                continue

            diff_raw = (row.get("difficulty") or "").strip()
            try:
                difficulty = int(diff_raw) if diff_raw else 2
            except ValueError:
                difficulty = 2

            rows.append({
                "question": q,
                "ideal_answer": ideal,
                "difficulty": difficulty,
            })

    if not rows:
        raise ValueError(f"No valid question/ideal_answer rows found in {path}")

    return rows


def run_simulation(
    course: str,
    model: str,
    strategy: Strategy,
    max_questions: int | None = None,
    difficulty_filter: List[int] | None = None,
    learner_profile: str | None = None,
    reset_before_run: bool = True,
    log_dir: str = os.path.join("adaptive", "simulations"),
) -> str:
    """
    Run a sequence of questions for a given (course, model, strategy).
    Returns the path to the simulation log CSV.
    """
    if strategy == "adaptive":
        profile_suffix = learner_profile or "default"
        user_id = f"sim_user_{strategy}_{profile_suffix}"
    else:
        user_id = f"sim_user_{strategy}"

    if reset_before_run and strategy == "adaptive":
        reset_state(user_id, course)

    config = get_best_config(course, model)
    top_k = int(config["top_k"])
    temperature = float(config["temperature"])

    questions = load_questions_for_course(course)
    if difficulty_filter is not None:
        questions = [q for q in questions if q.get("difficulty") in difficulty_filter]
    if max_questions is not None:
        questions = questions[:max_questions]

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{course}_{model}_{strategy}_{user_id}_{timestamp}"
    log_path = os.path.join(log_dir, f"sim_{run_id}.csv")

    fieldnames = [
        "timestamp",
        "run_id",
        "strategy",
        "user_id",
        "course",
        "model",
        "question_idx",
        "question",
        "ideal_answer",
        "learner_profile",
        "level_used",
        "level_before",
        "level_after",
        "correctness_score",
        "raw_correctness_score",
        "latency_sec",
        "top_k",
        "temperature",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for idx, row in enumerate(questions):
        q_text = row["question"]
        ideal = row.get("ideal_answer", "")

        if strategy == "adaptive":
            result = adaptive_answer(
                user_id=user_id,
                course=course,
                question=q_text,
                model=model,
                top_k=top_k,
                temperature=temperature,
                ideal_answer=ideal,
                log_path=os.path.join(log_dir, "adaptive_logs.csv"),
                learner_profile=learner_profile,
            )
            level_used = result["level_before"]
            level_before = result["level_before"]
            level_after = result["level_after"]
            correctness = result.get("profile_performance", result.get("correctness", 0.0))
            latency = result["latency"]
        else:
            if strategy == "fixed_beginner":
                level_used = "beginner"
            elif strategy == "fixed_intermediate":
                level_used = "intermediate"
            elif strategy == "fixed_advanced":
                level_used = "advanced"
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            docs = retrieve_docs(q_text, course, top_k=top_k)
            context = build_context(docs)
            t0 = time_now = datetime.utcnow().timestamp()
            try:
                answer = answer_with_model(
                    model=model,
                    question=q_text,
                    context=context,
                    summary="",
                    temperature=temperature,
                    learner_level=level_used,
                )
                if isinstance(answer, tuple):
                    answer_text = answer[0]
                else:
                    answer_text = answer
            except TypeError:
                answer_text, _lat_ms = answer_with_model(
                    model=model,
                    question=q_text,
                    context=context,
                    summary="",
                    temperature=temperature,
                )
            latency = datetime.utcnow().timestamp() - t0
            correctness = judge_answer(q_text, ideal, answer_text)
            level_before = level_used
            level_after = level_used

        row_out = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "strategy": strategy,
            "user_id": user_id,
            "course": course,
            "model": model,
            "question_idx": idx,
            "question": q_text,
            "ideal_answer": ideal,
            "learner_profile": learner_profile or "",
            "level_used": level_used,
            "level_before": level_before,
            "level_after": level_after,
            "correctness_score": correctness,
            "raw_correctness_score": result.get("correctness", correctness) if strategy == "adaptive" else correctness,
            "latency_sec": latency,
            "top_k": top_k,
            "temperature": temperature,
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_out)

    return log_path


if __name__ == "__main__":
    courses = ["architecture"]
    models = ["gpt"]
    strategies: List[Strategy] = [
        "adaptive",
        "fixed_beginner",
        "fixed_intermediate",
        "fixed_advanced",
    ]

    max_questions = 30
    for course in courses:
        for model in models:
            for strategy in strategies:
                print(f"Running simulation: {course}, {model}, {strategy}")
                log_path = run_simulation(
                    course=course,
                    model=model,
                    strategy=strategy,
                    max_questions=max_questions,
                )
                print(f"  -> log written to {log_path}")

    # Example: run only on hard questions for architecture + gpt + adaptive
    # run_simulation(
    #     course="architecture",
    #     model="gpt",
    #     strategy="adaptive",
    #     max_questions=20,
    #     difficulty_filter=[3],
    # )

    # Optional extra experiment: only hard questions (difficulty=3) for architecture + gpt.
    # This is useful to see how adaptive vs fixed behave on challenging items.
    #
    # Uncomment this block if you want to run it.
    #
    # hard_strategies = ["adaptive", "fixed_beginner", "fixed_intermediate", "fixed_advanced"]
    # for strategy in hard_strategies:
    #     print(f"Running HARD-ONLY simulation: architecture, gpt, {strategy}")
    #     run_simulation(
    #         course="architecture",
    #         model="gpt",
    #         strategy=strategy,
    #         max_questions=20,
    #         difficulty_filter=[3],
    #     )
