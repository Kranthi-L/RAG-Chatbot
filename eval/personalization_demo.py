#!/usr/bin/env python
"""
personalization_demo.py

Quick manual test of personalized RAG behavior.

It runs the SAME question for:
  - no profile (None)
  - beginner
  - intermediate
  - advanced

and for BOTH:
  - GPT
  - Claude

so you can visually inspect how explanation style changes.

Run from the project root:
    python eval/personalization_demo.py
"""

import sys
from pathlib import Path

# Ensure project root (directory containing rag_core.py) is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from rag_core import retrieve_docs, build_context, answer_with_model
from personalization import LearnerLevel


# You can change this to test other domains/courses
QUESTION = "Explain Amdahl's Law and why it limits the maximum speedup of a system."
COURSE = "architecture"  # must match your Chroma 'course' metadata


def run_for_profile(level: LearnerLevel | None):
    label = level or "none"
    print("\n" + "=" * 80)
    print(f"PROFILE: {label.upper()}")
    print("=" * 80)

    # Retrieve context from your existing vector store
    docs = retrieve_docs(query=QUESTION, course=COURSE, top_k=6)
    ctx = build_context(docs)

    if not docs:
        print("[WARN] No documents retrieved. Check COURSE name or Chroma DB.")
        return

    for model in ["gpt", "claude"]:
        print("\n" + "-" * 40)
        print(f"MODEL: {model.upper()}")
        print("-" * 40)

        answer, ms = answer_with_model(
            model=model,
            question=QUESTION,
            context=ctx,
            summary="",          # no prior conversation summary
            temperature=0.2,     # low temp for consistency
            learner_level=level, # <-- THIS is what we’re testing
        )

        print(f"[latency: {ms:.0f} ms]")
        print(answer)
        print()


def main():
    # Run with:
    #   - no personalization (None)
    #   - beginner
    #   - intermediate
    #   - advanced
    for level in [None, "beginner", "intermediate", "advanced"]:
        run_for_profile(level)


if __name__ == "__main__":
    main()
