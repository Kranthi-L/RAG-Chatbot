#!/usr/bin/env python
"""
ablation_analyze.py

Analyze the ablation study comparing:
  - baseline vs accessible prompts
  - GPT vs Claude

Input:  ablation_accessibility_results.csv
Output:
  - Printed summaries
  - CSVs with grouped stats
  - A couple of plots in figures_ablation/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "ablation_accessibility_results.csv"
FIG_DIR = SCRIPT_DIR / "figures_ablation"


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    num_cols = [
        "clarity_score",
        "structure_score",
        "tts_friendly_score",
        "cognitive_load_score",
        "learning_difficulties_score",
        "overall_accessibility",
        "correctness_score",
        "latency_ms",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["course"] = df["course"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    return df


def ensure_figdir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def summarize_by_model_condition(df: pd.DataFrame):
    """
    Average accessibility + correctness by (model, condition).
    """
    cols = [
        "clarity_score",
        "structure_score",
        "tts_friendly_score",
        "cognitive_load_score",
        "learning_difficulties_score",
        "overall_accessibility",
        "correctness_score",
    ]
    summary = (
        df.groupby(["model", "condition"])[cols]
        .mean()
        .reset_index()
        .sort_values(["model", "condition"])
    )
    print("\n=== Mean scores by (model, condition) ===")
    print(summary)
    out_path = SCRIPT_DIR / "ablation_summary_by_model_condition.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return summary


def paired_deltas(df: pd.DataFrame):
    """
    For each (course, question_id, model), compute:
      accessible - baseline
    for overall_accessibility and correctness_score.
    """
    # Pivot to have baseline and accessible on same row
    key_cols = ["course", "question_id", "model"]
    value_cols = ["overall_accessibility", "correctness_score"]

    pivot = (
        df.pivot_table(
            index=key_cols,
            columns="condition",
            values=value_cols,
        )
    )

    # The pivot has a MultiIndex on columns: (metric, condition)
    # e.g., ("overall_accessibility", "baseline")
    def get_col(metric, cond):
        return pivot[(metric, cond)] if (metric, cond) in pivot.columns else None

    results = []
    for model in ["gpt", "claude"]:
        sub = pivot[pivot.index.get_level_values("model") == model]
        base_acc = get_col("overall_accessibility", "baseline")
        acc_acc = get_col("overall_accessibility", "accessible")
        base_cor = get_col("correctness_score", "baseline")
        acc_cor = get_col("correctness_score", "accessible")

        if base_acc is None or acc_acc is None:
            print(f"[WARN] Missing accessibility columns for model={model}")
            continue

        # Align them
        delta_acc = acc_acc.loc[sub.index] - base_acc.loc[sub.index]
        delta_cor = None
        if base_cor is not None and acc_cor is not None:
            delta_cor = acc_cor.loc[sub.index] - base_cor.loc[sub.index]

        mean_delta_acc = float(delta_acc.mean())
        std_delta_acc = float(delta_acc.std(ddof=1) if len(delta_acc) > 1 else 0.0)

        print(f"\n=== Paired deltas for model={model} ===")
        print(f"Mean Δ overall_accessibility (accessible - baseline): {mean_delta_acc:.3f}")
        print(f"Std  Δ overall_accessibility: {std_delta_acc:.3f}")
        if delta_cor is not None:
            mean_delta_cor = float(delta_cor.mean())
            std_delta_cor = float(delta_cor.std(ddof=1) if len(delta_cor) > 1 else 0.0)
            print(f"Mean Δ correctness_score (accessible - baseline): {mean_delta_cor:.3f}")
            print(f"Std  Δ correctness_score: {std_delta_cor:.3f}")

        # Save per-question deltas for further analysis
        df_out = pd.DataFrame({
            "course": [idx[0] for idx in sub.index],
            "question_id": [idx[1] for idx in sub.index],
            "model": [idx[2] for idx in sub.index],
            "delta_overall_accessibility": delta_acc.values,
        })
        if delta_cor is not None:
            df_out["delta_correctness_score"] = delta_cor.values

        out_path = SCRIPT_DIR / f"ablation_deltas_{model}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved per-question deltas for {model} to {out_path}")


def plot_overall_accessibility(summary: pd.DataFrame):
    """
    Bar plot: overall_accessibility by (model, condition).
    """

    ensure_figdir()

    # Order conditions: baseline then accessible
    models = ["gpt", "claude"]
    conditions = ["baseline", "accessible"]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(models))
    width = 0.35

    for i, cond in enumerate(conditions):
        vals = []
        for m in models:
            row = summary[(summary["model"] == m) & (summary["condition"] == cond)]
            if not row.empty:
                vals.append(row["overall_accessibility"].iloc[0])
            else:
                vals.append(0.0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=cond.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Overall accessibility (0–1)")
    ax.set_title("Ablation: Baseline vs Accessible Prompt")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "fig_ablation_overall_accessibility.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    summary = summarize_by_model_condition(df)
    paired_deltas(df)
    plot_overall_accessibility(summary)

    print("\nAnalysis complete. Figures in:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
