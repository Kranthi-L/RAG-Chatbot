# app_cli.py
import os, sys, time, argparse
from dotenv import load_dotenv
from rich import print as rprint

# Vector store
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# LLMs
from langchain_openai import ChatOpenAI
import anthropic

# Session memory
from memory import load_history, save_turn, new_session, reset_session

load_dotenv()

DB_DIR = "chroma_db"
SYSTEM = open("prompts/qa_system.md").read()

# ---------------------------
# LLM helpers (cheap & tight)
# ---------------------------
def _cheap_llm():
    # same small, fast model for meta-tasks (summary / rewrite)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
def build_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        blocks.append(f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})")
    return "\n\n".join(blocks)

def ask_gpt(system: str, question: str, context: str, summary: str) -> tuple[str, float]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user",   "Conversation summary (for pronouns, references): {summary}"),
        ("user",   "Question: {q}\n\nContext:\n{ctx}")
    ])
    msgs = prompt.format_messages(summary=summary, q=question, ctx=context)
    t0 = time.time()
    out = llm.invoke(msgs).content
    ms = (time.time() - t0) * 1000
    return out, ms

def claude_candidates():
    model = os.getenv("CLAUDE_MODEL")
    if model:
        return [model]
    return ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet", "claude-3-haiku-20240307"]

def ask_claude(system: str, question: str, context: str, summary: str) -> tuple[str, float]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    text = (f"{system}\n\n"
            f"Conversation summary (for pronouns, references): {summary}\n\n"
            f"Question: {question}\n\nContext:\n{context}")
    last_err = None
    for m in claude_candidates():
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=m, max_tokens=700, temperature=0,
                messages=[{"role": "user", "content": text}]
            )
            ms = (time.time() - t0) * 1000
            parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
            return ("\n".join(parts).strip(), ms)
        except anthropic.NotFoundError as e:
            last_err = e
        except Exception as e:
            raise
    raise last_err or RuntimeError("No working Claude model ID found")

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
    vs = Chroma(persist_directory=DB_DIR)

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

    # Retrieval filter
    filt = None if target == "all" else {"course": {"$eq": target}}
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

        # ---- follow-up smarts ----
        # ---- follow-up smarts ----
        summary = summarize_recent(history) if history else ""

        # get the last assistant message (if any)
        last_assistant = ""
        for role, text in reversed(history[-6:]):
            if role == "assistant":
                last_assistant = text
                break

        # include the last assistant answer as context for rewriting
        q_input_for_rewriter = f"{q_raw}\n(Reference: last assistant answer: {last_assistant})" if last_assistant else q_raw
        q_standalone = rewrite_question(q_input_for_rewriter, summary)


        # ---- retrieval using the standalone form ----
        try:
            if filt is None:
                docs = vs.similarity_search(q_standalone, k=topk)
            else:
                docs = vs.similarity_search(q_standalone, k=topk, filter=filt)
        except TypeError:
            docs = vs.similarity_search(q_standalone, k=topk)

        # If retrieval returned nothing, retry with assistant answer as booster
        if not docs and last_assistant:
            boosted_query = f"{q_standalone}\nDetails mentioned previously: {last_assistant}"
            try:
                if filt is None:
                    docs = vs.similarity_search(boosted_query, k=topk)
                else:
                    docs = vs.similarity_search(boosted_query, k=topk, filter=filt)
            except TypeError:
                docs = vs.similarity_search(boosted_query, k=topk)


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
            ans, ms = ask_gpt(SYSTEM, q_standalone, ctx, summary)
            gpt_ans = ans
            rprint(f"\n[bold cyan]GPT[/bold cyan]  ({ms:.0f} ms):\n{ans}")

        if backend in {"claude", "both"}:
            ans, ms = ask_claude(SYSTEM, q_standalone, ctx, summary)
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
