#!/usr/bin/env python
"""
Summarize retrieval depth by learner level using num_chunks/context_chars.

Defaults to the networking level_fit files:
  - eval/level_fit_networking_temp0p5_topk8_hybrid_beg.csv
  - eval/level_fit_networking_temp0p5_topk8_hybrid_int.csv
  - eval/level_fit_networking_temp0p5_topk8_hybrid_adv.csv

Run:
    python -m eval.personalization.summarize_context_depth
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_FILES = {
    "beginner": "eval/filled_networking_temp0p5_topk8_hybrid_beg.csv",
    "intermediate": "eval/filled_networking_temp0p5_topk8_hybrid_int.csv",
    "advanced": "eval/filled_networking_temp0p5_topk8_hybrid_adv.csv",
}


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"=== {name} ===")
    if "num_chunks" in df and "context_chars" in df:
        print("avg num_chunks    :", df["num_chunks"].mean())
        print("avg context_chars :", df["context_chars"].mean())
        print("min/max num_chunks:", df["num_chunks"].min(), "/", df["num_chunks"].max())
    else:
        print("Missing num_chunks/context_chars columns.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize context depth by learner level.")
    parser.add_argument("--beginner_file", default=DEFAULT_FILES["beginner"])
    parser.add_argument("--intermediate_file", default=DEFAULT_FILES["intermediate"])
    parser.add_argument("--advanced_file", default=DEFAULT_FILES["advanced"])
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
