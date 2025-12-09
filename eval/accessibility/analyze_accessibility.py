#!/usr/bin/env python

"""
analyze_accessibility.py

Reads accessibility_results.csv and computes:
  - Overall accessibility averages per model
  - Per-course accessibility averages per model
  - Correlation between overall_accessibility and correctness score per model

Also generates basic plots using matplotlib (no seaborn) into figures_accessibility/.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

INPUT_CSV = Path("accessibility_results.csv")
FIG_DIR = Path("figures_accessibility")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}, expected at {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    # Normalize model + course names a bit
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip().str.lower()
    if "course" in df.columns:
        df["course"] = df["course"].astype(str).str.strip().str.lower()

    # Ensure numeric types for scores
    numeric_cols = [
        "clarity_score",
        "structure_score",
        "tts_friendly_score",
        "cognitive_load_score",
        "learning_difficulties_score",
        "overall_accessibility",
        "score",  # correctness judge score from original eval
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without overall_accessibility
    if "overall_accessibility" in df.columns:
        df = df.dropna(subset=["overall_accessibility"])

    return df


def title_case_course(c: str) -> str:
    return c.replace("_", " ").title()


# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------

def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall averages per model for accessibility dimensions.
    """
    metrics = [
        "clarity_score",
        "structure_score",
        "tts_friendly_score",
        "cognitive_load_score",
        "learning_difficulties_score",
        "overall_accessibility",
    ]
    available_metrics = [m for m in metrics if m in df.columns]

    summary = (
        df.groupby("model")[available_metrics]
        .mean()
        .reset_index()
        .sort_values("model")
    )

    return summary


def summarize_by_course_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-course averages per model for accessibility dimensions.
    """
    metrics = [
        "clarity_score",
        "structure_score",
        "tts_friendly_score",
        "cognitive_load_score",
        "learning_difficulties_score",
        "overall_accessibility",
    ]
    available_metrics = [m for m in metrics if m in df.columns]

    summary = (
        df.groupby(["course", "model"])[available_metrics]
        .mean()
        .reset_index()
        .sort_values(["course", "model"])
    )

    return summary


def compute_correlations(df: pd.DataFrame):
    """
    Compute correlation between overall_accessibility and correctness score (score)
    per model.
    """
    if "overall_accessibility" not in df.columns or "score" not in df.columns:
        print("Correlation: missing 'overall_accessibility' or 'score' column.")
        return

    print("\nCorrelation between overall_accessibility and correctness (score):")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].dropna(subset=["overall_accessibility", "score"])
        if len(sub) < 2:
            print(f"  model={model}: not enough data")
            continue
        r = np.corrcoef(sub["overall_accessibility"], sub["score"])[0, 1]
        print(f"  model={model}: r = {r:.3f}")


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_overall_accessibility_by_model(df_model: pd.DataFrame):
    """
    Bar chart: overall_accessibility averaged over all courses, per model.
    """
    if "overall_accessibility" not in df_model.columns:
        print("Skipping plot_overall_accessibility_by_model: no overall_accessibility column.")
        return

    models = list(df_model["model"])
    vals = list(df_model["overall_accessibility"])

    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x, vals, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overall accessibility (mean)")
    ax.set_title("Overall accessibility by model")

    fig.tight_layout()
    out_path = FIG_DIR / "figA_overall_accessibility_by_model.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_overall_accessibility_by_course_and_model(df_course_model: pd.DataFrame):
    """
    Grouped bar chart: overall_accessibility by course and model.
    """
    if "overall_accessibility" not in df_course_model.columns:
        print("Skipping plot_overall_accessibility_by_course_and_model: no overall_accessibility column.")
        return

    courses = sorted(df_course_model["course"].unique())
    models = sorted(df_course_model["model"].unique())

    # Build data matrix: rows=courses, cols=models
    vals = []
    for course in courses:
        row_vals = []
        for model in models:
            sub = df_course_model[
                (df_course_model["course"] == course) &
                (df_course_model["model"] == model)
            ]
            v = sub["overall_accessibility"].iloc[0] if not sub.empty else np.nan
            row_vals.append(v)
        vals.append(row_vals)
    vals = np.array(vals)  # shape (n_courses, n_models)

    x = np.arange(len(courses))
    width = 0.35 if len(models) == 2 else 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for j, model in enumerate(models):
        offsets = x + (j - (len(models) - 1) / 2) * width
        ax.bar(offsets, vals[:, j], width=width, label=model.upper())

    ax.set_xticks(x)
    ax.set_xticklabels([title_case_course(c) for c in courses])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overall accessibility (mean)")
    ax.set_xlabel("Course")
    ax.set_title("Overall accessibility by course and model")
    ax.legend()

    fig.tight_layout()
    out_path = FIG_DIR / "figB_overall_accessibility_by_course_and_model.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_accessibility_vs_correctness_scatter(df: pd.DataFrame):
    """
    Scatter plot: overall_accessibility vs correctness score (score), colored by model.
    """
    if "overall_accessibility" not in df.columns or "score" not in df.columns:
        print("Skipping plot_accessibility_vs_correctness_scatter: missing columns.")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    models = sorted(df["model"].unique())
    markers = {"gpt": "o", "claude": "^"}

    for model in models:
        sub = df[df["model"] == model].dropna(subset=["overall_accessibility", "score"])
        if sub.empty:
            continue
        ax.scatter(
            sub["overall_accessibility"],
            sub["score"],
            label=model.upper(),
            marker=markers.get(model, "o"),
            alpha=0.7,
            s=30,
        )

    ax.set_xlabel("Overall accessibility score")
    ax.set_ylabel("Correctness score (LLM judge)")
    ax.set_title("Accessibility vs correctness")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    out_path = FIG_DIR / "figC_accessibility_vs_correctness_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ensure_fig_dir()

    df = load_data()
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Overall by model
    by_model = summarize_by_model(df)
    print("\n=== Overall accessibility by model ===")
    print(by_model)
    by_model_out = Path("accessibility_summary_by_model.csv")
    by_model.to_csv(by_model_out, index=False)
    print(f"Saved {by_model_out.resolve()}")

    # Per-course by model
    by_course_model = summarize_by_course_model(df)
    print("\n=== Accessibility by course and model ===")
    print(by_course_model)
    by_course_model_out = Path("accessibility_summary_by_course_model.csv")
    by_course_model.to_csv(by_course_model_out, index=False)
    print(f"Saved {by_course_model_out.resolve()}")

    # Correlations
    compute_correlations(df)

    # Plots
    plot_overall_accessibility_by_model(by_model)
    plot_overall_accessibility_by_course_and_model(by_course_model)
    plot_accessibility_vs_correctness_scatter(df)

    print("\nAnalysis complete. Figures saved in:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
