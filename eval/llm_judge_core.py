"""
Lightweight LLM judge utility.

Expose a simple judge_answer(question, ideal_answer, model_answer) -> float in [0,1].
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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


def _get_judge_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def judge_answer(question: str, ideal_answer: str, model_answer: str) -> float:
    """
    Return a score in [0,1] judging how close model_answer is to ideal_answer.
    If parsing fails, returns 0.0.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM),
            ("user", JUDGE_USER_TEMPLATE),
        ]
    )
    msgs = prompt.format_messages(
        question=question.strip(),
        ideal_answer=ideal_answer.strip(),
        model_answer=model_answer.strip(),
    )
    try:
        llm = _get_judge_llm()
        raw = llm.invoke(msgs).content
        parsed: Any = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0
