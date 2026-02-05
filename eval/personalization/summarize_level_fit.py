#!/usr/bin/env python
"""
Summarize level-fit judge scores for networking personalization.

Defaults to the three files produced by run_level_fit_judge:
  - eval/level_fit_networking_temp0p5_topk8_hybrid_beginner.csv
  - eval/level_fit_networking_temp0p5_topk8_hybrid_intermediate.csv
  - eval/level_fit_networking_temp0p5_topk8_hybrid_advanced.csv

Outputs a console summary matrix:
  learner_level x model -> avg level_fit_score, clarity_for_level, fraction too_advanced/too_simple.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_FILES = {
    "beginner": "eval/level_fit_networking_temp0p5_topk8_hybrid_beg.csv",
    "intermediate": "eval/level_fit_networking_temp0p5_topk8_hybrid_int.csv",
    "advanced": "eval/level_fit_networking_temp0p5_topk8_hybrid_adv.csv",
}


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"== {name} ==")
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        print(f" model: {model}")
        print(f"  avg level_fit_score: {sub['level_fit_score'].mean():.3f}")
        print(f"  avg clarity_for_level: {sub['clarity_for_level'].mean():.3f}")
        print(f"  frac too_advanced: {sub['too_advanced'].mean():.3f}")
        print(f"  frac too_simple: {sub['too_simple'].mean():.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Summarize level-fit judge scores.")
    parser.add_argument(
        "--beginner_file",
        default=DEFAULT_FILES["beginner"],
        help="Path to beginner level_fit CSV",
    )
    parser.add_argument(
        "--intermediate_file",
        default=DEFAULT_FILES["intermediate"],
        help="Path to intermediate level_fit CSV",
    )
    parser.add_argument(
        "--advanced_file",
        default=DEFAULT_FILES["advanced"],
        help="Path to advanced level_fit CSV",
    )
    args = parser.parse_args()

    files = [
        ("beginner", Path(args.beginner_file)),
        ("intermediate", Path(args.intermediate_file)),
        ("advanced", Path(args.advanced_file)),
    ]

    for name, path in files:
        if not path.exists():
            print(f"[WARN] Missing file for {name}: {path}")
            continue
        df = pd.read_csv(path)
        summarize(df, name)


if __name__ == "__main__":
    main()
