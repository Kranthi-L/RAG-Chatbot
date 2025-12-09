#!/usr/bin/env python
"""
Generate IEEE-style figures from eval/summary_metrics.csv.

Uses pandas and matplotlib only.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_figures_dir(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.csv at {csv_path}")
    df = pd.read_csv(csv_path)
    # normalize strings
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip().str.lower()
    if "course" in df.columns:
        df["course"] = df["course"].astype(str).str.strip().str.lower()

    # numeric conversions
    num_cols = [
        "temperature",
        "top_k",
        "num_questions",
        "avg_answer_length",
        "bleu",
        "rouge_l",
        "bert_sim",
        "avg_judge_score",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows without judge score
    df = df.dropna(subset=["avg_judge_score"])
    return df


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def title_case_course(course: str) -> str:
    return course.replace("_", " ").title()


def plot_heatmaps_per_course(df: pd.DataFrame, fig_dir: Path):
    courses = ["architecture", "machine_learning", "networking"]
    models = ["gpt", "claude"]

    # global vmin/vmax for consistency
    vmin = df["avg_judge_score"].min()
    vmax = df["avg_judge_score"].max()

    for idx, course in enumerate(courses):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.4), sharey=True)
        ims = []
        for j, model in enumerate(models):
            ax = axes[j]
            sub = df[(df["course"] == course) & (df["model"] == model)]
            if sub.empty:
                ax.axis("off")
                continue
            pivot = sub.pivot_table(
                index="temperature",
                columns="top_k",
                values="avg_judge_score",
                aggfunc="mean",
            )
            pivot = pivot.sort_index().sort_index(axis=1)
            temps = list(pivot.index)
            ks = list(pivot.columns)
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            ims.append(im)

            # annotations
            for i_t, t in enumerate(temps):
                for i_k, k in enumerate(ks):
                    val = pivot.loc[t, k]
                    if not math.isnan(val):
                        ax.text(i_k, i_t, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([str(int(k)) for k in ks])
            ax.set_xlabel("top_k")
            if j == 0:
                ax.set_ylabel("Temperature")
                ax.set_yticks(range(len(temps)))
                ax.set_yticklabels([f"{t:.1f}" for t in temps])
            else:
                ax.set_yticks(range(len(temps)))
                ax.set_yticklabels([])
            ax.set_title(f"{model.upper()}")

        label_map = {
            "architecture": "Fig. 2(a). Architecture – GPT vs Claude",
            "machine_learning": "Fig. 2(b). Machine Learning – GPT vs Claude",
            "networking": "Fig. 2(c). Networking – GPT vs Claude",
        }
        fig.suptitle(label_map.get(course, title_case_course(course)), fontsize=10)
        fig.tight_layout()
        fig.subplots_adjust(top=0.82, bottom=0.24)
        # colorbar placed below x-labels
        if ims:
            cbar_ax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
            cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
            cbar.set_label("avg_judge_score")
        out_name = {
            "architecture": "fig2a_heatmap_architecture.png",
            "machine_learning": "fig2b_heatmap_machine_learning.png",
            "networking": "fig2c_heatmap_networking.png",
        }.get(course, f"fig2_heatmap_{course}.png")
        out_path = fig_dir / out_name
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_fig3_best_per_course(df: pd.DataFrame, fig_dir: Path):
    courses = ["architecture", "machine_learning", "networking"]
    models = ["gpt", "claude"]
    records = []
    for course in courses:
        for model in models:
            sub = df[(df["course"] == course) & (df["model"] == model)]
            if sub.empty:
                continue
            best_row = sub.loc[sub["avg_judge_score"].idxmax()]
            records.append(best_row)

    if not records:
        print("No data for Fig. 3")
        return

    best_df = pd.DataFrame(records)
    # save table
    table_path = fig_dir / "best_judge_per_course_table.csv"
    best_df[
        ["course", "model", "temperature", "top_k", "avg_judge_score", "bleu", "rouge_l", "bert_sim", "avg_answer_length"]
    ].to_csv(table_path, index=False)
    print(f"Saved {table_path}")

    x = np.arange(len(courses))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, model in enumerate(models):
        vals = []
        for course in courses:
            sub = best_df[(best_df["course"] == course) & (best_df["model"] == model)]
            vals.append(sub["avg_judge_score"].iloc[0] if not sub.empty else 0.0)
        offsets = x + (idx - 0.5) * width
        ax.bar(offsets, vals, width=width, label=model.upper())

    ax.set_xticks(x)
    ax.set_xticklabels([title_case_course(c) for c in courses], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best average judge score")
    ax.set_xlabel("Course")
    ax.set_title("Fig. 3. Best judge score per course and model")
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "fig3_best_judge_per_course.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig4_judge_vs_topk(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["gpt", "claude"]
    for model in models:
        sub = df[df["model"] == model]
        grp = sub.groupby("top_k")["avg_judge_score"].mean().reset_index()
        grp = grp.sort_values("top_k")
        ax.plot(grp["top_k"], grp["avg_judge_score"], marker="o", label=model.upper())

    ax.set_xlabel("top_k (retrieval depth)")
    ax.set_ylabel("Mean judge score (across courses, temperatures)")
    ax.set_title("Fig. 4. Effect of retrieval depth on judge score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "fig4_judge_vs_topk.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig5_judge_vs_temperature(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["gpt", "claude"]
    for model in models:
        sub = df[df["model"] == model]
        grp = sub.groupby("temperature")["avg_judge_score"].mean().reset_index()
        grp = grp.sort_values("temperature")
        ax.plot(grp["temperature"], grp["avg_judge_score"], marker="o", label=model.upper())

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean judge score (across courses, top_k)")
    ax.set_title("Fig. 5. Effect of temperature on judge score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "fig5_judge_vs_temperature.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig6_length_vs_judge(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"gpt": "o", "claude": "^"}
    colors = {"gpt": "C0", "claude": "C1"}
    models = ["gpt", "claude"]

    for model in models:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["avg_answer_length"],
            sub["avg_judge_score"],
            label=model.upper(),
            marker=markers.get(model, "o"),
            color=colors.get(model, None),
            alpha=0.8,
            s=25,
        )
        if len(sub) > 1:
            corr = np.corrcoef(sub["avg_answer_length"], sub["avg_judge_score"])[0, 1]
            print(f"Fig.6 correlation (length vs judge) for {model}: {corr:.3f}")

    ax.set_xlabel("Average answer length (tokens or characters)")
    ax.set_ylabel("Average judge score")
    ax.set_title("Fig. 6. Relationship between answer length and correctness")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "fig6_length_vs_judge.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig7_bert_vs_judge(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"gpt": "o", "claude": "^"}
    colors = {"gpt": "C0", "claude": "C1"}
    models = ["gpt", "claude"]

    for model in models:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["bert_sim"],
            sub["avg_judge_score"],
            label=model.upper(),
            marker=markers.get(model, "o"),
            color=colors.get(model, None),
            alpha=0.8,
            s=25,
        )
        if len(sub) > 1:
            corr = np.corrcoef(sub["bert_sim"], sub["avg_judge_score"])[0, 1]
            print(f"Fig.7 correlation (bert_sim vs judge) for {model}: {corr:.3f}")

    ax.set_xlabel("Semantic similarity (BERT / MiniLM cosine)")
    ax.set_ylabel("Average judge score")
    ax.set_title("Fig. 7. Semantic similarity vs judge score")
    ax.set_ylim(0, 1.05)
    xmin = df["bert_sim"].min() if not df.empty else 0
    xmax = df["bert_sim"].max() if not df.empty else 1
    padding = 0.01
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "fig7_bert_vs_judge.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from summary_metrics.csv")
    parser.add_argument(
        "--input",
        type=str,
        default="summary_metrics.csv",
        help="Path to summary_metrics.csv (default: summary_metrics.csv in script directory or CWD)",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="figures",
        help="Directory to save figures (default: figures/)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    script_dir = Path(__file__).resolve().parent

    # Try CWD path, then script-dir-relative
    if not csv_path.exists():
        alt = script_dir / csv_path
        if alt.exists():
            csv_path = alt

    fig_dir = Path(args.figdir)

    ensure_figures_dir(fig_dir)
    setup_style()

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_heatmaps_per_course(df, fig_dir)
    plot_fig3_best_per_course(df, fig_dir)
    plot_fig4_judge_vs_topk(df, fig_dir)
    plot_fig5_judge_vs_temperature(df, fig_dir)
    plot_fig6_length_vs_judge(df, fig_dir)
    plot_fig7_bert_vs_judge(df, fig_dir)

    print("Figures saved to:", fig_dir.resolve())


if __name__ == "__main__":
    main()
