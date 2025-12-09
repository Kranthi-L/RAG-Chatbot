#!/usr/bin/env python

import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("summary_metrics.csv")  # adjust if needed
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize
    df["course"] = df["course"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    # Just to be safe
    df["avg_judge_score"] = pd.to_numeric(df["avg_judge_score"], errors="coerce")
    df = df.dropna(subset=["avg_judge_score"])

    # For each (course, model), get row with max avg_judge_score
    idx = df.groupby(["course", "model"])["avg_judge_score"].idxmax()
    best = df.loc[idx].copy()

    # Sort for readability
    best = best.sort_values(["course", "model"])

    print("Best configuration per (course, model):\n")
    for _, row in best.iterrows():
        course = row["course"]
        model = row["model"]
        temp = row["temperature"]
        top_k = row["top_k"]
        score = row["avg_judge_score"]
        print(
            f"course={course:16s}  model={model:7s}  "
            f"temperature={temp}  top_k={top_k}  avg_judge_score={score:.3f}"
        )

    # Also save a small CSV for reference
    out_path = Path("best_configs_for_accessibility.csv")
    best.to_csv(out_path, index=False)
    print(f"\nSaved best configs to {out_path.resolve()}")

if __name__ == "__main__":
    main()
