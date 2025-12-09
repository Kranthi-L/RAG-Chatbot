#!/usr/bin/env python
"""
LLM judge for filled_* evaluation CSVs.

Examples:
  python eval/run_llm_judge.py               # judge all filled_*.csv under eval/
  python eval/run_llm_judge.py eval/filled_networking_temp0p2_topk4.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def get_judge_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Reuse the same OpenAI client pattern as other scripts (gpt-4o-mini here).
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


JUDGE_SYSTEM = (
    "You are a strict but fair university instructor. You grade short free-text answers "
    "on correctness and completeness only. You must output valid JSON and nothing else."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Ideal reference answer:\n{ideal_answer}\n\n"
    "Student answer:\n{model_answer}\n\n"
    "Compare the student answer to the ideal answer.\n"
    "- Ignore wording differences and focus on whether the important ideas are present and correct.\n"
    "- If the answer is fully correct and complete, score it 1.0.\n"
    "- If it is completely wrong or irrelevant, score it 0.0.\n"
    "- Use intermediate values for partially correct answers.\n\n"
    'Respond with a single JSON object with keys "score" (float between 0 and 1) and "justification" (short string).'
)


def judge_answer(llm: ChatOpenAI, question: str, ideal: str, answer: str) -> Dict[str, Any]:
    """
    Call the judge LLM and return parsed score/justification.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM),
            ("user", JUDGE_USER_TEMPLATE),
        ]
    )
    msgs = prompt.format_messages(
        question=question.strip(),
        ideal_answer=ideal.strip(),
        model_answer=answer.strip(),
    )
    try:
        raw = llm.invoke(msgs).content
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        # clamp score to [0,1]
        score = max(0.0, min(1.0, score))
        justification = str(parsed.get("justification", "")).strip()
        return {"score": score, "justification": justification}
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "justification": f"Judge error: {e}"}


def process_file(path: Path, llm: ChatOpenAI, progress_every: int = 20, sleep_between: float = 0.1) -> None:
    # Use encoding_errors to safely replace bad bytes
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    if not {"question", "ideal_answer", "gpt_response", "claude_response"}.issubset(df.columns):
        print(f"Skipping {path.name}: missing required columns.", file=sys.stderr)
        return

    gpt_scores: List[float] = []
    gpt_justs: List[str] = []
    claude_scores: List[float] = []
    claude_justs: List[str] = []

    for idx, row in df.iterrows():
        if idx % progress_every == 0:
            print(f"[{path.name}] judging row {idx+1}/{len(df)}")

        question = str(row.get("question", ""))
        ideal = str(row.get("ideal_answer", ""))

        gpt_ans = str(row.get("gpt_response", ""))
        gpt_res = judge_answer(llm, question, ideal, gpt_ans)
        gpt_scores.append(gpt_res["score"])
        gpt_justs.append(gpt_res["justification"])

        claude_ans = str(row.get("claude_response", ""))
        claude_res = judge_answer(llm, question, ideal, claude_ans)
        claude_scores.append(claude_res["score"])
        claude_justs.append(claude_res["justification"])

        if sleep_between > 0:
            time.sleep(sleep_between)

    df["gpt_score"] = gpt_scores
    df["gpt_justification"] = gpt_justs
    df["claude_score"] = claude_scores
    df["claude_justification"] = claude_justs

    out_name = path.name.replace("filled_", "judged_", 1)
    out_path = path.parent / out_name
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote judged file: {out_path}")


def find_input_files(args: List[str]) -> List[Path]:
    if args:
        return [Path(a).resolve() for a in args]
    # No args: process all filled_*.csv in eval/
    return sorted((BASE_DIR).glob("filled_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for filled evaluation CSVs.")
    parser.add_argument("files", nargs="*", help="Optional CSV files to judge. Defaults to all filled_*.csv in eval/")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge model temperature (default: 0.0)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between judge calls (seconds). Use 0 to disable.")
    args = parser.parse_args()

    files = find_input_files(args.files)
    if not files:
        print("No input files found to judge.", file=sys.stderr)
        sys.exit(1)

    llm = get_judge_llm(temperature=args.temperature)

    for f in files:
        if not f.exists():
            print(f"Skipping missing file: {f}", file=sys.stderr)
            continue
        process_file(f, llm, sleep_between=args.sleep)


if __name__ == "__main__":
    main()
