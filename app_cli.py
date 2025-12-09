# app_cli.py
import os, sys, time, argparse
from dotenv import load_dotenv
from rich import print as rprint

# Vector store
from langchain_core.prompts import ChatPromptTemplate

from rag_core import (
    answer_with_model,
    ask_claude,
    ask_gpt,
    build_context,
    get_gpt_llm,
    get_vector_store,
    retrieve_docs,
)

# Session memory
from memory import load_history, save_turn, new_session, reset_session

load_dotenv()

SYSTEM = open("prompts/qa_system.md").read()

# ---------------------------
# LLM helpers (cheap & tight)
# ---------------------------
def _cheap_llm():
    # same small, fast model for meta-tasks (summary / rewrite)
    # Now uses cached instance instead of creating new one each time
    return get_gpt_llm(temperature=0)

def summarize_recent(turns, max_chars=1200) -> str:
    """Summarize last few turns into a tiny context for follow-ups."""
    if not turns:
        return ""
    # format a small window: last 6 turns max
    window = turns[-6:]
    convo = []
    for role, text in window:
        tag = "User" if role == "user" else "Assistant"
        convo.append(f"{tag}: {text}")
    convo_text = "\n".join(convo)[-max_chars:]

    llm = _cheap_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You produce a compact conversation summary for grounding follow-up questions."
                   " Keep it under 80 tokens, no markdown, no lists. Mention key entities and topics only."),
        ("user", f"Conversation excerpt:\n{convo_text}\n\nReturn a one-sentence summary:")
    ])
    msg = prompt.format_messages()
    try:
        out = llm.invoke(msg).content.strip()
        return out
    except Exception:
        # fail-safe: return last user question only
        for role, text in reversed(window):
            if role == "user":
                return text
        return ""
    
def is_standalone_question(question: str) -> bool:
    """
    Optimization 4: Detect if a question is standalone (doesn't need follow-up processing).
    Questions starting with common question words are usually standalone.
    """
    q_lower = question.strip().lower()
    standalone_starters = (
        'what', 'who', 'where', 'when', 'why', 'how', 'explain', 'describe', 
        'list', 'define', 'tell me about', 'what is', 'what are', 'what does',
        'compare', 'difference between', 'advantages', 'disadvantages'
    )
    return q_lower.startswith(standalone_starters)

def rewrite_question(user_q: str, summary: str) -> str:
    """Rewrite follow-up into standalone question using summary context."""
    if not summary:
        return user_q
    llm = _cheap_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You rewrite the user's question into a standalone query using the provided summary."
                   " Keep it concise (<30 words). If it is already standalone, return it unchanged."
                   " No markdown, no prefices like 'Rewritten:'."),
        ("user", f"Summary: {summary}\nQuestion: {user_q}\nStandalone question:")
    ])
    msg = prompt.format_messages()
    try:
        return llm.invoke(msg).content.strip()
    except Exception:
        return user_q

# ---------------------------
# Retrieval & generation
# ---------------------------
# ---------------------------
# CLI plumbing
# ---------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="RAG Chatbot CLI with follow-up support")
    p.add_argument("backend", nargs="?", default="gpt", choices=["gpt","claude","both"],
                   help="LLM backend")
    p.add_argument("target", nargs="?", default="all", choices=["all","networking","architecture"],
                   help="Filter retrieval by course")
    p.add_argument("--session", type=str, default=None, help="Session ID to load/save turns")
    p.add_argument("--reset", action="store_true", help="Reset session history before starting")
    p.add_argument("--primary", choices=["gpt","claude"], default="gpt",
                   help="When backend=both, which answer to store in the session log")
    p.add_argument("--topk", type=int, default=6, help="retrieval depth (k)")
    return p.parse_args()

def main():
    # vector store
    vs = get_vector_store()

    args = parse_args()
    backend = args.backend
    target  = args.target
    session_id = args.session
    primary = args.primary
    topk = args.topk

    # Prepare session
    if session_id:
        if args.reset:
            reset_session(session_id)
            rprint(f"[green]Session '{session_id}' reset.[/green]")
        else:
            if not os.path.exists(os.path.join("sessions", f"{session_id}.json")):
                new_session(session_id)
        history = load_history(session_id)
        if history:
            rprint(f"[bold]Loaded session[/bold] '{session_id}' with {len(history)} turn(s).")
        else:
            rprint(f"[bold]New session[/bold] '{session_id}'.")
    else:
        history = []

    rprint(f"[bold]RAG Chatbot[/bold] — backend: {backend.upper()}  |  course: {target}"
           + (f"  |  session: {session_id}" if session_id else ""))

    while True:
        q_raw = input("\nAsk a question (or 'q' to quit): ").strip()
        if q_raw.lower() == "q":
            break
        if not q_raw:
            continue

        # Save user turn immediately
        if session_id:
            save_turn(session_id, "user", q_raw)

        # ---- follow-up smarts (Optimization 4: Skip for standalone questions) ----
        # Check if question is standalone - if so, skip expensive follow-up processing
        if history and not is_standalone_question(q_raw):
            # This is likely a follow-up question, so we need context
            summary = summarize_recent(history)
            # get the last assistant message (if any)
            last_assistant = ""
            for role, text in reversed(history[-6:]):
                if role == "assistant":
                    last_assistant = text
                    break
            # include the last assistant answer as context for rewriting
            q_input_for_rewriter = f"{q_raw}\n(Reference: last assistant answer: {last_assistant})" if last_assistant else q_raw
            q_standalone = rewrite_question(q_input_for_rewriter, summary)
        else:
            # Standalone question - use it directly, no expensive API calls needed
            summary = ""
            q_standalone = q_raw
            last_assistant = ""
            # Still try to get last assistant for potential retrieval boost
            for role, text in reversed(history[-6:]):
                if role == "assistant":
                    last_assistant = text
                    break


        # ---- retrieval using the standalone form ----
        docs = retrieve_docs(
            query=q_standalone,
            course=target if target != "all" else None,
            top_k=topk,
            last_assistant=last_assistant
        )


        if not docs:
            rprint("[yellow]No relevant chunks found. Try a simpler query or expand your index.[/yellow]")
            if session_id:
                save_turn(session_id, "assistant", "[No relevant chunks found]")
            # update history with this turn so next summary has it
            history.append(("user", q_raw))
            history.append(("assistant", "[No relevant chunks found]"))
            continue

        ctx = build_context(docs)

        # ---- generation with summary + context ----
        gpt_ans = claude_ans = None
        if backend in {"gpt", "both"}:
            ans, ms = answer_with_model("gpt", q_standalone, ctx, summary, temperature=0.0)
            gpt_ans = ans
            rprint(f"\n[bold cyan]GPT[/bold cyan]  ({ms:.0f} ms):\n{ans}")

        if backend in {"claude", "both"}:
            ans, ms = answer_with_model("claude", q_standalone, ctx, summary, temperature=0.0)
            claude_ans = ans
            rprint(f"\n[bold magenta]Claude[/bold magenta]  ({ms:.0f} ms):\n{ans}")

        # Decide which answer to store in session
        if session_id:
            if backend == "gpt":
                save_turn(session_id, "assistant", gpt_ans or "")
                history.append(("user", q_raw)); history.append(("assistant", gpt_ans or ""))
            elif backend == "claude":
                save_turn(session_id, "assistant", claude_ans or "")
                history.append(("user", q_raw)); history.append(("assistant", claude_ans or ""))
            else:
                chosen = gpt_ans if primary == "gpt" else claude_ans
                tag = "GPT" if primary == "gpt" else "Claude"
                save_turn(session_id, "assistant", f"[{tag}] {chosen or ''}")
                history.append(("user", q_raw)); history.append(("assistant", f"[{tag}] {chosen or ''}"))

if __name__ == "__main__":
    main()
