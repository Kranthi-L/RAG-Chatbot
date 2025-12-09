#!/usr/bin/env python

import math
from pathlib import Path

import pandas as pd


# Helper: convert temperature float to the filename token, e.g. 0.2 -> "0p2"
def temp_to_token(t: float) -> str:
    # Handle simple cases like 0.0, 0.2, 0.5, 0.7, 1.0
    # Convert to string with one decimal, then replace '.' with 'p'
    s = f"{float(t):.1f}"  # e.g. "0.2", "1.0"
    return s.replace(".", "p")  # -> "0p2", "1p0"


def main():
    base_dir = Path(".")  # current directory
    best_cfg_path = base_dir / "best_configs_for_accessibility.csv"

    if not best_cfg_path.exists():
        raise FileNotFoundError(f"Could not find {best_cfg_path}")

    best_cfg = pd.read_csv(best_cfg_path)

    # Normalize a bit
    best_cfg["course"] = best_cfg["course"].astype(str).str.strip().str.lower()
    best_cfg["model"] = best_cfg["model"].astype(str).str.strip().str.lower()
    best_cfg["temperature"] = pd.to_numeric(best_cfg["temperature"], errors="coerce")
    best_cfg["top_k"] = pd.to_numeric(best_cfg["top_k"], errors="coerce")

    # Just in case, drop bad rows
    best_cfg = best_cfg.dropna(subset=["temperature", "top_k", "avg_judge_score"])

    print("Using best configurations:")
    print(best_cfg[["course", "model", "temperature", "top_k", "avg_judge_score"]])

    # We will build:
    # 1) long_df: one row per (course, question, model)
    long_rows = []

    # Cache loaded CSVs by (course, temp_token, top_k)
    file_cache = {}

    for _, row in best_cfg.iterrows():
        course = row["course"]
        model = row["model"]  # "gpt" or "claude"
        temp = float(row["temperature"])
        top_k = int(row["top_k"])

        temp_token = temp_to_token(temp)  # e.g. 0.2 -> "0p2"
        # filenames follow pattern: judged_{course}_temp{temp_token}_topk{top_k}.csv
        fname = f"judged_{course}_temp{temp_token}_topk{top_k}.csv"
        fpath = base_dir / fname

        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected file {fname} for course={course}, model={model}, "
                f"temperature={temp}, top_k={top_k} but it does not exist in {base_dir}"
            )

        print(f"\nLoading {fpath} for course={course}, model={model}...")

        # Load or reuse cached
        key = (course, temp_token, top_k)
        if key in file_cache:
            df = file_cache[key]
        else:
            df = pd.read_csv(fpath)
            file_cache[key] = df

        # Sanity check columns
        required_cols = [
            "question",
            "gpt_response",
            "claude_response",
            "ideal_answer",
            "gpt_score",
            "claude_score",
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {fpath}")

        # Decide which columns to use based on model
        if model == "gpt":
            answer_col = "gpt_response"
            score_col = "gpt_score"
            ms_col = "gpt_ms" if "gpt_ms" in df.columns else None
        elif model == "claude":
            answer_col = "claude_response"
            score_col = "claude_score"
            ms_col = "claude_ms" if "claude_ms" in df.columns else None
        else:
            raise ValueError(f"Unexpected model name: {model}")

        # Add one row per question for this (course, model, temp, top_k)
        for i, qrow in df.iterrows():
            record = {
                "course": course,
                "model": model,
                "temperature": temp,
                "top_k": top_k,
                "question_index": i + 1,  # 1-based index within this course
                "question": qrow["question"],
                "ideal_answer": qrow["ideal_answer"],
                "answer": qrow[answer_col],
                "score": qrow[score_col],
            }
            if ms_col is not None:
                record["latency_ms"] = qrow[ms_col]

            long_rows.append(record)

    long_df = pd.DataFrame(long_rows)

    # Now we can assign a consistent question_id per (course, question) if we want
    # (using question text as key)
    long_df["question_id"] = (
        long_df.groupby("course")["question"]
        .transform(lambda s: pd.factorize(s)[0]) + 1
    )

    # Reorder columns
    long_cols = [
        "course",
        "model",
        "question_id",
        "question_index",
        "question",
        "ideal_answer",
        "temperature",
        "top_k",
        "answer",
        "score",
    ]
    if "latency_ms" in long_df.columns:
        long_cols.append("latency_ms")

    long_df = long_df[long_cols]

    # Save long-format file
    out_long = base_dir / "accessibility_input_long.csv"
    long_df.to_csv(out_long, index=False)
    print(f"\nSaved long-format accessibility input to: {out_long.resolve()}")
    print(f"Rows in long-format: {len(long_df)}")

    # Optional: build a wide-format version with one row per (course, question_id),
    # containing best GPT and best Claude answers side by side.
    print("\nBuilding wide-format file (one row per course+question)...")
    wide_rows = []
    for (course, question_id), sub in long_df.groupby(["course", "question_id"]):
        # We expect at most one GPT row and one Claude row per (course, question_id)
        gpt_row = sub[sub["model"] == "gpt"].iloc[0] if (sub["model"] == "gpt").any() else None
        claude_row = sub[sub["model"] == "claude"].iloc[0] if (sub["model"] == "claude").any() else None

        # Use any row to get question + ideal
        base_row = gpt_row if gpt_row is not None else claude_row

        record = {
            "course": course,
            "question_id": question_id,
            "question": base_row["question"],
            "ideal_answer": base_row["ideal_answer"],
        }

        # GPT fields
        if gpt_row is not None:
            record.update({
                "gpt_temperature": gpt_row["temperature"],
                "gpt_top_k": gpt_row["top_k"],
                "gpt_answer": gpt_row["answer"],
                "gpt_score": gpt_row["score"],
            })
            if "latency_ms" in gpt_row:
                record["gpt_latency_ms"] = gpt_row["latency_ms"]

        # Claude fields
        if claude_row is not None:
            record.update({
                "claude_temperature": claude_row["temperature"],
                "claude_top_k": claude_row["top_k"],
                "claude_answer": claude_row["answer"],
                "claude_score": claude_row["score"],
            })
            if "latency_ms" in claude_row:
                record["claude_latency_ms"] = claude_row["latency_ms"]

        wide_rows.append(record)

    wide_df = pd.DataFrame(wide_rows)

    # Sort for readability
    wide_df = wide_df.sort_values(["course", "question_id"]).reset_index(drop=True)

    out_wide = base_dir / "accessibility_input_wide.csv"
    wide_df.to_csv(out_wide, index=False)
    print(f"Saved wide-format accessibility input to: {out_wide.resolve()}")
    print(f"Rows in wide-format: {len(wide_df)}")


if __name__ == "__main__":
    main()
