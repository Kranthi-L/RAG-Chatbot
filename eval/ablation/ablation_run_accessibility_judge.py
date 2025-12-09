#!/usr/bin/env python
"""
ablation_run_accessibility_judge.py

Use an LLM judge to score accessibility and correctness for the ablation study.

Input:  ablation_answers.csv  (from ablation_generate_answers.py)
Output: ablation_accessibility_results.csv

For each row (question, model, condition, response), we compute:

- clarity_score                 [0..1]
- structure_score               [0..1]
- tts_friendly_score            [0..1]
- cognitive_load_score          [0..1]
- learning_difficulties_score   [0..1]
- overall_accessibility         [0..1]  (e.g., mean of the above, but judge decides)
- correctness_score             [0..1]  (alignment with ideal_answer)
- judge_explanation             (short natural-language justification)
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "ablation_answers.csv"
OUTPUT_PATH = SCRIPT_DIR / "ablation_accessibility_results.csv"

client = OpenAI()  # uses OPENAI_API_KEY from environment


def load_answers() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    # basic normalization
    df["course"] = df["course"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    return df


JUDGE_SYSTEM_PROMPT = """You are an expert educational and accessibility evaluator.
You will assess model answers for both accessibility and correctness.

You are given:
- a question
- an ideal reference answer (for correctness)
- a model-generated answer

You must evaluate:

1) Accessibility dimensions (for a university-level learner, including those with learning difficulties):
   - clarity_score: Is the explanation clear, unambiguous, and readable?
   - structure_score: Is the answer well structured (paragraphs, lists, steps)?
   - tts_friendly_score: Is the answer friendly to text-to-speech (no LaTeX, minimal symbols, simple punctuation)?
   - cognitive_load_score: Is the answer broken into manageable chunks and not overwhelming?
   - learning_difficulties_score: Would this help a learner with dyslexia/ADHD or similar difficulties (simple language, good chunking)?

   Each should be a score between 0 and 1, where:
     0   = unusable / very poor
     0.5 = mixed / partially accessible
     1   = highly accessible

   overall_accessibility should summarize the above (e.g., an approximate average).

2) Correctness:
   - correctness_score: A score between 0 and 1 indicating how well the model answer matches
     the ideal answer in terms of correctness and completeness for the key ideas.
     0   = completely wrong or irrelevant
     0.5 = partially correct, missing or confusing important elements
     1   = essentially correct and covers the important points

Return your evaluation STRICTLY as a JSON object with the following fields:

{
  "clarity_score": float in [0,1],
  "structure_score": float in [0,1],
  "tts_friendly_score": float in [0,1],
  "cognitive_load_score": float in [0,1],
  "learning_difficulties_score": float in [0,1],
  "overall_accessibility": float in [0,1],
  "correctness_score": float in [0,1],
  "judge_explanation": "short natural-language explanation"
}

Do NOT include any extra keys, and do NOT include commentary outside the JSON.
"""


def call_judge(
    question: str,
    ideal_answer: str,
    model_answer: str,
    course: str,
    model: str,
    condition: str,
) -> Dict[str, Any]:
    """
    Call the LLM judge and parse its JSON response.
    """
    user_prompt = f"""Course: {course}
Model: {model}
Condition: {condition}

Question:
{question}

Ideal (reference) answer:
{ideal_answer}

Model-generated answer:
{model_answer}

Evaluate accessibility and correctness as per the rubric."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            # basic sanity checks
            return {
                "clarity_score": float(data.get("clarity_score", 0.0)),
                "structure_score": float(data.get("structure_score", 0.0)),
                "tts_friendly_score": float(data.get("tts_friendly_score", 0.0)),
                "cognitive_load_score": float(data.get("cognitive_load_score", 0.0)),
                "learning_difficulties_score": float(data.get("learning_difficulties_score", 0.0)),
                "overall_accessibility": float(data.get("overall_accessibility", 0.0)),
                "correctness_score": float(data.get("correctness_score", 0.0)),
                "judge_explanation": str(data.get("judge_explanation", "")),
            }
        except Exception as e:
            print(f"  [WARN] judge call failed on attempt {attempt+1}: {e}")
            time.sleep(1.5)

    # fallback if judge fails
    print("  [ERROR] judge failed after 3 attempts; returning zeros.")
    return {
        "clarity_score": 0.0,
        "structure_score": 0.0,
        "tts_friendly_score": 0.0,
        "cognitive_load_score": 0.0,
        "learning_difficulties_score": 0.0,
        "overall_accessibility": 0.0,
        "correctness_score": 0.0,
        "judge_explanation": "judge_failed",
    }


def main():
    df = load_answers()
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    out_rows = []
    total = len(df)
    for idx, row in df.iterrows():
        i = idx + 1
        course = row["course"]
        model = row["model"]
        condition = row["condition"]
        question = str(row["question"])
        ideal_answer = str(row.get("ideal_answer", ""))
        response = str(row.get("response", ""))

        print(f"\n=== [{i}/{total}] course={course}, model={model}, condition={condition}, qid={row.get('question_id')} ===")

        scores = call_judge(
            question=question,
            ideal_answer=ideal_answer,
            model_answer=response,
            course=course,
            model=model,
            condition=condition,
        )

        merged = dict(row)  # keep original columns
        merged.update(scores)  # add judge scores
        out_rows.append(merged)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved ablation accessibility results to: {OUTPUT_PATH.resolve()}")
    print(f"Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()
