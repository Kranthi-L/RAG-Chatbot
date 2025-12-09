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
