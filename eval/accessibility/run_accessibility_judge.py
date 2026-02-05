#!/usr/bin/env python
"""
run_accessibility_judge.py

Reads accessibility_input_long.csv and, for each (course, model, question, answer),
calls GPT (via langchain_openai.ChatOpenAI) to evaluate accessibility on:

- clarity
- structure
- tts_friendly
- cognitive_load
- learning_difficulties

Outputs accessibility_results.csv with original fields + scores + justifications.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Input produced by build_accessibility_inputs.py
INPUT_CSV = "accessibility_input_long.csv"

# Output produced by this script
OUTPUT_CSV = "accessibility_results.csv"

# Optional checkpoint to avoid losing progress
CHECKPOINT_EVERY = 20

# Model to use as accessibility judge; can override via env if you want
ACCESSIBILITY_JUDGE_MODEL = "gpt-4o"  # or "gpt-4o-mini" if that's what your key supports

# -------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------

ACCESSIBILITY_SYSTEM_PROMPT = (
    "You are an expert in accessibility, inclusive learning, and assistive technology. "
    "You evaluate the accessibility of educational text responses according to WCAG 2.1, "
    "Universal Design for Learning (UDL), and cognitive load theory. "
    "You must return scores between 0 and 1 for each dimension."
)


def build_accessibility_user_prompt(question: str, answer: str) -> str:
    """
    Build the user-side prompt for the accessibility judge.
    """
    return f"""
Evaluate the accessibility of the following educational answer.

Question:
{question}

Answer:
{answer}

Rate the answer on the following dimensions, each from 0 (very poor) to 1 (excellent):

1. Clarity – sentence simplicity, readability, avoidance of ambiguity.
2. Structure – chunking, bullet points, transitions, logical flow.
3. TTS Friendliness – no symbols that break screen readers, no references to missing visuals, and reasonable sentence length.
4. Cognitive Load – does the explanation scaffold concepts? does it overwhelm the learner or reasonably pace information?
5. Support for Diverse Learning Needs – dyslexia-friendly wording, consistent terminology, accessible phrasing, and avoidance of confusing references.

Return JSON ONLY in the following structure:

{{
  "clarity": float,
  "structure": float,
  "tts_friendly": float,
  "cognitive_load": float,
  "learning_difficulties": float,
  "overall_accessibility": float,
  "justification": {{
      "clarity": "...",
      "structure": "...",
      "tts_friendly": "...",
      "cognitive_load": "...",
      "learning_difficulties": "..."
  }}
}}
""".strip()


# -------------------------------------------------------------------
# LLM call and JSON parsing
# -------------------------------------------------------------------

def get_llm() -> ChatOpenAI:
    """
    Create a ChatOpenAI client for the accessibility judge.
    Temperature = 0 for deterministic scoring.
    """
    return ChatOpenAI(model=ACCESSIBILITY_JUDGE_MODEL, temperature=0.0)


def parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse JSON from the model output.
    Tries a direct json.loads, and if that fails, trims to the first '{' and last '}'.
    """
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to rescue JSON if there's extra text around it
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return None


def call_accessibility_judge(llm: ChatOpenAI, question: str, answer: str) -> Optional[Dict[str, Any]]:
    """
    Call the LLM to get accessibility scores.
    Returns a dict matching the expected JSON structure, or None if parsing fails.
    """
    user_prompt = build_accessibility_user_prompt(question, answer)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ACCESSIBILITY_SYSTEM_PROMPT),
        ("user", "{user_prompt}"),
    ])
    messages = prompt.format_messages(user_prompt=user_prompt)

    resp = llm.invoke(messages)
    raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)

    data = parse_json_safely(raw_text)
    if data is None:
        print("WARNING: Failed to parse JSON from model output:")
        print(raw_text)
    return data


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():
    load_dotenv()

    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Expected at {input_path.resolve()}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Basic column checks
    required_cols = ["course", "model", "question_id", "question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {INPUT_CSV}")
    # Ensure retriever_type present (forwarded from best-configs); fallback to 'dense'
    if "retriever_type" not in df.columns:
        df["retriever_type"] = "dense"

    rows = df.to_dict(orient="records")
    results: List[Dict[str, Any]] = []

    llm = get_llm()
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        course = row.get("course")
        model = row.get("model")
        qid = row.get("question_id")
        retriever = row.get("retriever_type", "dense")
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        print(f"[{i}/{total}] course={course}, model={model}, question_id={qid}")

        judge_data = None
        try:
            judge_data = call_accessibility_judge(llm, question, answer)
        except Exception as e:
            print(f"ERROR during judge call on row {i}: {e}")

        result_row = dict(row)  # start with original columns

        if isinstance(judge_data, dict):
            result_row["clarity_score"] = judge_data.get("clarity")
            result_row["structure_score"] = judge_data.get("structure")
            result_row["tts_friendly_score"] = judge_data.get("tts_friendly")
            result_row["cognitive_load_score"] = judge_data.get("cognitive_load")
            result_row["learning_difficulties_score"] = judge_data.get("learning_difficulties")
            result_row["overall_accessibility"] = judge_data.get("overall_accessibility")

            just = judge_data.get("justification", {}) or {}
            result_row["justification_clarity"] = just.get("clarity")
            result_row["justification_structure"] = just.get("structure")
            result_row["justification_tts"] = just.get("tts_friendly")
            result_row["justification_cognitive"] = just.get("cognitive_load")
            result_row["justification_learning_difficulties"] = just.get("learning_difficulties")
        else:
            # Mark as missing if judge failed
            result_row["clarity_score"] = None
            result_row["structure_score"] = None
            result_row["tts_friendly_score"] = None
            result_row["cognitive_load_score"] = None
            result_row["learning_difficulties_score"] = None
            result_row["overall_accessibility"] = None
            result_row["justification_clarity"] = None
            result_row["justification_structure"] = None
            result_row["justification_tts"] = None
            result_row["justification_cognitive"] = None
            result_row["justification_learning_difficulties"] = None
        # Ensure retriever_type is preserved
        result_row["retriever_type"] = retriever

        results.append(result_row)

        # Optional: checkpoint
        if i % CHECKPOINT_EVERY == 0:
            checkpoint_path = Path(OUTPUT_CSV).with_suffix(".checkpoint.csv")
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Small delay if you want to be gentle with the API
        # time.sleep(0.1)

    out_df = pd.DataFrame(results)
    out_path = Path(OUTPUT_CSV)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved accessibility results to: {out_path.resolve()}")
    print(f"Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()
