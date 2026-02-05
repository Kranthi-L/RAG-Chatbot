#!/usr/bin/env python
"""
Level-fit judge for networking personalization.

Scans eval/ for filled_networking_*.csv files (produced by run_batch_eval),
filters to learner_level in {beginner, intermediate, advanced}, and for each
model column (default: gpt_response, claude_response) calls an LLM judge to
assess whether the answer matches the intended learner level.

Output: level_fit_<rest_of_filename>.csv in eval/ (long format: one row per model per question).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # eval/
REPO_ROOT = BASE_DIR.parent


def build_level_fit_prompt(
    question: str,
    ideal_answer: str,
    model_answer: str,
    learner_level: str,
    model_name: str,
) -> str:
    learner_block = f"""Learner levels:
BEGINNER:
  - Little to no prior background.
  - Needs simple language, terms defined in plain English.
  - Step-by-step explanations, no heavy math or dense jargon.
INTERMEDIATE:
  - Knows basic networking terminology (e.g., packets, latency, routing).
  - Can handle moderate technical detail and some math/notation.
  - Balance of intuition and precision.
ADVANCED:
  - Strong CS/math background.
  - Expects concise, technical explanations with deeper insights, trade-offs, edge cases.
  - References to protocols, algorithms, or performance analysis are fine.
"""

    return f"""
You are evaluating whether an explanation fits a target learner level in a university computer networking course.

{learner_block}

Target learner level: {learner_level.upper()}
Model: {model_name.upper()}

Question:
{question}

Ideal reference answer (expert):
{ideal_answer}

Model answer to evaluate:
{model_answer}

Think about:
- Is the answer too advanced/overloaded for the given level?
- Is it too shallow/oversimplified for the given level?
- How clear and well-structured is it for that level?

Return JSON ONLY:
{{
  "level_fit_score": float in [0,1],
  "too_advanced": boolean,
  "too_simple": boolean,
  "clarity_for_level": float in [0,1],
  "comments": "short justification"
}}
""".strip()


def call_level_fit_judge(client: OpenAI, prompt: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict evaluator of level-appropriate explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            return {
                "level_fit_score": float(data.get("level_fit_score", 0.0)),
                "too_advanced": bool(data.get("too_advanced", False)),
                "too_simple": bool(data.get("too_simple", False)),
                "clarity_for_level": float(data.get("clarity_for_level", 0.0)),
                "level_fit_comments": str(data.get("comments", "")),
            }
        except Exception as e:
            print(f"  [WARN] judge attempt failed ({attempt+1}/3): {e}", file=sys.stderr)
    return {
        "level_fit_score": 0.0,
        "too_advanced": False,
        "too_simple": False,
        "clarity_for_level": 0.0,
        "level_fit_comments": "PARSE_ERROR",
    }


def process_file(path: Path, model_cols: List[str], client: OpenAI) -> None:
    # Output path
    out_path = path.with_name("level_fit_" + path.name.replace("filled_", ""))
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Skipping {path.name} (output exists: {out_path.name})")
        return

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    if "learner_level" not in df.columns:
        print(f"Skipping {path.name}: no learner_level column", file=sys.stderr)
        return

    df["learner_level"] = df["learner_level"].astype(str).str.strip().str.lower()
    df = df[df["learner_level"].isin({"beginner", "intermediate", "advanced"})]
    if df.empty:
        print(f"Skipping {path.name}: no rows with beginner/intermediate/advanced", file=sys.stderr)
        return

    results: List[Dict[str, Any]] = []
    total_rows = len(df)
    for idx, row in df.iterrows():
        learner_level = row["learner_level"]
        if (idx + 1) % 10 == 0:
            print(f"[{path.name}] row {idx+1}/{total_rows}, level={learner_level}")

        question = str(row.get("question", ""))
        ideal = str(row.get("ideal_answer", ""))

        for col in model_cols:
            if col not in df.columns:
                continue
            ans = str(row.get(col, ""))
            if not ans.strip():
                continue
            model_name = "gpt" if "gpt" in col else "claude" if "claude" in col else col
            prompt = build_level_fit_prompt(
                question=question,
                ideal_answer=ideal,
                model_answer=ans,
                learner_level=learner_level,
                model_name=model_name,
            )
            judge = call_level_fit_judge(client, prompt)

            out_row = {
                "course": row.get("course", "networking"),
                "question": question,
                "ideal_answer": ideal,
                "learner_level": learner_level,
                "model": model_name,
                "model_column": col,
                "model_answer": ans,
                "level_fit_score": judge["level_fit_score"],
                "too_advanced": judge["too_advanced"],
                "too_simple": judge["too_simple"],
                "clarity_for_level": judge["clarity_for_level"],
                "level_fit_comments": judge["level_fit_comments"],
            }
            # carry over a few useful fields if present
            for extra in ["temperature", "top_k", "retriever_type"]:
                if extra in row:
                    out_row[extra] = row[extra]
            results.append(out_row)

    if not results:
        print(f"No results for {path.name}")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path.name} with {len(out_df)} rows")


def main():
    parser = argparse.ArgumentParser(description="Judge level-fit for networking personalization runs.")
    parser.add_argument(
        "--model_columns",
        nargs="+",
        default=["gpt_response", "claude_response"],
        help="Response columns to evaluate (default: gpt_response claude_response)",
    )
    args = parser.parse_args()

    client = OpenAI()
    files = sorted(BASE_DIR.glob("filled_networking_*.csv"))
    if not files:
        print("No filled_networking_*.csv files found in eval/", file=sys.stderr)
        return

    for f in files:
        print(f"Processing {f.name} ...")
        process_file(f, args.model_columns, client)


if __name__ == "__main__":
    main()
