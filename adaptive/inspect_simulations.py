# Allow this script to be run as: python adaptive/inspect_simulations.py
from __future__ import annotations

"""
NOTE: This module implements a prototype adaptive learning simulation.
It is NOT used in the main experiments for the paper and is kept as
future work / exploratory code.
"""
import os
import sys

if __package__ is None or __package__ == "":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import csv
import glob
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RunSummary:
    path: str
    strategy: str
    course: str
    model: str
    top_k: int
    temperature: float
    num_rows: int
    avg_correctness: float
    min_correctness: float
    max_correctness: float
    level_counts: Dict[str, int]
    transition_counts: Dict[str, int]


def load_run(path: str) -> RunSummary:
    """
    Load a single simulation CSV and compute basic summary statistics.
    Assumes the CSV has columns:
        strategy, course, model, top_k, temperature,
        correctness_score, level_used, level_before, level_after
    """
    level_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    scores: List[float] = []

    strategy = None
    course = None
    model = None
    top_k = None
    temperature = None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if strategy is None:
                strategy = row.get("strategy", "").strip()
            if course is None:
                course = row.get("course", "").strip()
            if model is None:
                model = row.get("model", "").strip()
            if top_k is None:
                try:
                    top_k = int(row.get("top_k", "0"))
                except ValueError:
                    top_k = 0
            if temperature is None:
                try:
                    temperature = float(row.get("temperature", "0.0"))
                except ValueError:
                    temperature = 0.0

            try:
                score = float(row.get("correctness_score", "nan"))
            except ValueError:
                continue

            scores.append(score)

            level_used = (row.get("level_used") or "").strip()
            if level_used:
                level_counts[level_used] += 1

            lb = (row.get("level_before") or "").strip()
            la = (row.get("level_after") or "").strip()
            if lb and la:
                key = f"{lb}->{la}"
                transition_counts[key] += 1

    if not scores:
        raise ValueError(f"No valid correctness_score values found in {path}")

    avg_correctness = sum(scores) / len(scores)
    min_correctness = min(scores)
    max_correctness = max(scores)

    _strategy = strategy or "unknown"
    _course = course or "unknown"
    _model = model or "unknown"
    _top_k = int(top_k or 0)
    _temperature = float(temperature or 0.0)

    return RunSummary(
        path=path,
        strategy=_strategy,
        course=_course,
        model=_model,
        top_k=_top_k,
        temperature=_temperature,
        num_rows=len(scores),
        avg_correctness=avg_correctness,
        min_correctness=min_correctness,
        max_correctness=max_correctness,
        level_counts=dict(level_counts),
        transition_counts=dict(transition_counts),
    )


def summarize_all(sim_dir: str = None) -> None:
    if sim_dir is None:
        sim_dir = os.path.join("adaptive", "simulations")

    pattern = os.path.join(sim_dir, "sim_*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No simulation logs found matching {pattern}")
        return

    print(f"Found {len(paths)} simulation log(s) in {sim_dir}.\n")

    grouped: Dict[tuple, List[RunSummary]] = defaultdict(list)

    for path in paths:
        try:
            summary = load_run(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        key = (summary.course, summary.model, summary.strategy)
        grouped[key].append(summary)

    for (course, model, strategy), runs in grouped.items():
        print("=" * 80)
        print(f"Course: {course} | Model: {model} | Strategy: {strategy}")
        for run in runs:
            print(f"  File: {os.path.basename(run.path)}")
            print(f"    top_k={run.top_k}, temperature={run.temperature}")
            print(f"    num_rows={run.num_rows}")
            print(
                f"    correctness: avg={run.avg_correctness:.3f}, "
                f"min={run.min_correctness:.3f}, max={run.max_correctness:.3f}"
            )
            if run.level_counts:
                levels_str = ", ".join(
                    f"{lvl}: {cnt}" for lvl, cnt in sorted(run.level_counts.items())
                )
                print(f"    level_used counts: {levels_str}")
            if run.transition_counts:
                trans_str = ", ".join(
                    f"{k}: {v}" for k, v in sorted(run.transition_counts.items())
                )
                print(f"    transitions: {trans_str}")
            print()


if __name__ == "__main__":
    summarize_all()
