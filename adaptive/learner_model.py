from __future__ import annotations

"""
NOTE: This module implements a prototype adaptive learning simulation.
It is NOT used in the main experiments for the paper and is kept as
future work / exploratory code.
"""

import json
import os
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal

LearnerLevel = Literal["beginner", "intermediate", "advanced"]

STATE_FILE = os.path.join(os.path.dirname(__file__), "learner_states.json")
MAX_RECENT = 5


@dataclass
class LearnerState:
    user_id: str
    course: str
    level: LearnerLevel
    num_questions: int = 0
    recent_scores: List[float] | None = None
    rolling_avg: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to plain dict for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnerState":
        """Create a LearnerState from a dict, filling defaults."""
        return cls(
            user_id=data["user_id"],
            course=data["course"],
            level=data["level"],
            num_questions=data.get("num_questions", 0),
            recent_scores=data.get("recent_scores", []),
            rolling_avg=data.get("rolling_avg", 0.0),
        )


def _load_all_states() -> Dict[str, Dict]:
    """Load all learner states from disk."""
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_all_states(states: Dict[str, Dict]) -> None:
    """Persist all learner states to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(states, f, indent=2)


def get_state(user_id: str, course: str) -> LearnerState:
    """
    Retrieve the learner state for a given user and course.
    Creates a new beginner state if none exists yet.
    """
    key = f"{user_id}|{course}"
    states = _load_all_states()
    if key in states:
        return LearnerState.from_dict(states[key])

    # create default beginner state
    state = LearnerState(
        user_id=user_id,
        course=course,
        level="beginner",
        num_questions=0,
        recent_scores=[],
        rolling_avg=0.0,
    )
    save_state(state)
    return state


def decide_level(state: LearnerState) -> LearnerLevel:
    """
    Decide the learner level based on recent scores.
    If fewer than 3 scores, do not change level.
    """
    scores = state.recent_scores or []
    if len(scores) < 3:
        return state.level

    avg = statistics.mean(scores)

    if avg >= 0.8:
        if state.level == "beginner":
            return "intermediate"
        if state.level == "intermediate":
            return "advanced"
    if avg <= 0.4:
        if state.level == "advanced":
            return "intermediate"
        if state.level == "intermediate":
            return "beginner"
    return state.level


def save_state(state: LearnerState) -> None:
    """Save a single learner state to disk."""
    states = _load_all_states()
    key = f"{state.user_id}|{state.course}"
    states[key] = state.to_dict()
    _save_all_states(states)


def reset_state(user_id: str, course: str) -> None:
    """
    Remove the stored learner state for this (user_id, course).
    Next get_state call will recreate a fresh beginner record.
    """
    states = _load_all_states()
    key = f"{user_id}|{course}"
    if key in states:
        del states[key]
        _save_all_states(states)


def reset_all_states() -> None:
    """Clear all stored learner states (used for simulations or resets)."""
    _save_all_states({})


def update_state(state: LearnerState, correctness_score: float) -> LearnerState:
    """
    Update learner state with a new correctness score.
    Maintains rolling average and adapts level based on recent scores.
    """
    scores = state.recent_scores or []
    scores.append(correctness_score)
    # keep only last MAX_RECENT scores
    scores = scores[-MAX_RECENT:]
    state.recent_scores = scores
    state.num_questions += 1
    if scores:
        state.rolling_avg = statistics.mean(scores)
    state.level = decide_level(state)
    save_state(state)
    return state


if __name__ == "__main__":
    # Simple demo
    s = get_state("demo", "architecture")
    for sc in [0.6, 0.7, 0.9, 0.85]:
        s = update_state(s, sc)
    print("Final state:", s)
