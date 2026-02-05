"""
personalization.py

Learner profiles and instructions for personalized RAG answers.

We currently support three profiles:
  - beginner
  - intermediate
  - advanced

These affect only HOW explanations are presented:
  * Beginner  -> simpler language, step-by-step, examples.
  * Intermediate -> mix of intuition + technical detail.
  * Advanced  -> concise, technical, focus on trade-offs/limits.

This is profile-based personalization, not full adaptive learning.
"""

from typing import Literal, Optional

LearnerLevel = Literal["beginner", "intermediate", "advanced"]


def normalize_learner_level(level: Optional[str]) -> Optional[str]:
    """Normalize learner level to a known string or None."""
    if level is None:
        return None
    lvl = str(level).strip().lower()
    if lvl in {"beginner", "intermediate", "advanced"}:
        return lvl
    return None


def get_retrieval_config_for_level(base_top_k: int, learner_level: Optional[str]) -> dict:
    """
    Return retrieval configuration for a learner level.
    - effective_top_k adjusts breadth of context.
    - use_level_reranker signals whether to rerank for level.
    """
    lvl = normalize_learner_level(learner_level)
    if lvl is None:
        return {"effective_top_k": base_top_k, "use_level_reranker": False}
    if lvl == "beginner":
        return {"effective_top_k": 12, "use_level_reranker": True}
    if lvl == "intermediate":
        return {"effective_top_k": base_top_k, "use_level_reranker": True}
    if lvl == "advanced":
        return {"effective_top_k": 4, "use_level_reranker": True}
    return {"effective_top_k": base_top_k, "use_level_reranker": False}


def get_generation_style_instructions(learner_level: Optional[str]) -> str:
    """
    Style guidance to append to system prompt based on learner level.
    """
    lvl = normalize_learner_level(learner_level)
    if lvl is None:
        return ""
    if lvl == "beginner":
        return (
            "You are explaining to a BEGINNER.\n"
            "Use simple, everyday language.\n"
            "Explain every technical term or complex formula in simpler steps.\n"
            "Break your explanation into steps and make it easily understandable.\n"
            "Include one simple analogy to illustrate the concept."
        )
    if lvl == "intermediate":
        return (
            "You are explaining to an INTERMEDIATE learner.\n"
            "Assume they know basic terminology.\n"
            "Provide a mix of intuition and accurate technical details."
        )
    if lvl == "advanced":
        return (
            "You are explaining to an ADVANCED learner.\n"
            "Be concise, precise, and technical.\n"
            "Use domain-specific terminology freely."
        )
    return ""


def render_profile_instructions(level: Optional[LearnerLevel]) -> str:
    """
    Return natural-language instructions describing how the model
    should adapt explanations for this learner profile.

    This text will be appended to the normal system prompt.
    """
    if level is None:
        return ""

    if level == "beginner":
        return (
            "You are helping a BEGINNER student.\n"
            "- Assume limited prior knowledge of the course.\n"
            "- Start with a one-sentence high-level idea.\n"
            "- Use simple language and avoid jargon unless you define it.\n"
            "- Prefer short paragraphs and bulleted lists.\n"
            "- Explain formulas in plain English and walk through them step by step.\n"
            "- Include a small, concrete example when helpful.\n"
            "- Explicitly connect the explanation back to the question so the student\n"
            "  understands why each part of the answer matters."
        )

    if level == "intermediate":
        return (
            "You are helping an INTERMEDIATE student.\n"
            "- Assume they know the basic terminology of the course.\n"
            "- Use a mix of intuition and moderate technical detail.\n"
            "- Organize the answer into clearly labeled parts (e.g., Idea, Formula, Example).\n"
            "- When using formulas, state them in plain English and relate each term to\n"
            "  the scenario in the question.\n"
            "- Highlight common pitfalls or misconceptions.\n"
            "- Keep the answer focused and avoid unnecessary digressions."
        )

    if level == "advanced":
        return (
            "You are helping an ADVANCED student.\n"
            "- Assume solid background in the course prerequisites.\n"
            "- Be concise but precise; prioritize the core mechanism and key trade-offs.\n"
            "- You may use technical terms, but briefly justify any non-obvious step.\n"
            "- Emphasize edge cases, limitations, and how this concept interacts with\n"
            "  related topics in the course.\n"
            "- When relevant, connect the explanation to performance, scalability,\n"
            "  or generalization issues rather than only basic definitions."
        )

    # Unknown string -> no extra instructions
    return ""
