# app_web.py
import os, time
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple

# RAG bits
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# LLMs
from langchain_openai import ChatOpenAI
import anthropic

# session persistence (same as CLI)
from memory import load_history, save_turn, new_session, reset_session

load_dotenv()

DB_DIR   = "chroma_db"
SYSTEM   = open("prompts/qa_system.md").read()
COURSES  = ["all", "networking", "architecture"]

# ---------------------------
# LLM client caching (Optimization 1: Reuse instances instead of creating new ones)
# ---------------------------
# Cache GPT LLM instances by temperature to avoid recreating connections
_gpt_llm_cache: dict[float, ChatOpenAI] = {}
_claude_client: anthropic.Anthropic | None = None

def get_gpt_llm(temperature: float = 0) -> ChatOpenAI:
    """Get or create a cached GPT LLM instance for the given temperature."""
    if temperature not in _gpt_llm_cache:
        _gpt_llm_cache[temperature] = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    return _gpt_llm_cache[temperature]

def get_claude_client() -> anthropic.Anthropic:
    """Get or create a cached Claude client instance."""
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _claude_client

# ---------------------------
# Helpers from CLI version
# ---------------------------
def build_context(docs) -> str:
    """
    Optimization 5: Build context with length limits to prevent token limit issues.
    Limits total context to ~8000 characters (roughly 2000 tokens).
    """
    blocks = []
    total_chars = 0
    max_context_chars = 8000  # Conservative limit to stay well under token limits
    
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        block = f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})"
        
        # Check if adding this block would exceed limit
        block_size = len(block) + 2  # +2 for "\n\n" separator
        if total_chars + block_size > max_context_chars and blocks:
            # Stop adding chunks if we'd exceed the limit
            break
        
        blocks.append(block)
        total_chars += block_size
    
    return "\n\n".join(blocks)

def _cheap_llm(temp=0.0):
    # Now uses cached instance instead of creating new one each time
    return get_gpt_llm(temperature=temp)

def summarize_recent(turns: List[Tuple[str,str]], max_chars=1200) -> str:
    if not turns:
        return ""
    window = turns[-6:]
    convo = []
    for role, text in window:
        tag = "User" if role == "user" else "Assistant"
        convo.append(f"{tag}: {text}")
    convo_text = "\n".join(convo)[-max_chars:]
    llm = _cheap_llm(0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You produce a compact conversation summary for grounding follow-up questions. "
                   "Keep it under ~80 tokens, no markdown, no lists. Mention key entities and topics only."),
        ("user", f"Conversation excerpt:\n{convo_text}\n\nReturn a one-sentence summary:")
    ])
    msg = prompt.format_messages()
    try:
        return llm.invoke(msg).content.strip()
    except Exception:
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
    if not summary:
        return user_q
    llm = _cheap_llm(0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question into a standalone query using the provided summary. "
                   "Keep it concise (<30 words). If it's already standalone, return it unchanged. "
                   "No markdown, no prefixes."),
        ("user", f"Summary: {summary}\nQuestion: {user_q}\nStandalone question:")
    ])
    msg = prompt.format_messages()
    try:
        return llm.invoke(msg).content.strip()
    except Exception:
        return user_q

def ask_gpt(system: str, question: str, context: str, summary: str, temperature: float) -> Tuple[str, float]:
    # Use cached LLM instance instead of creating new one
    llm = get_gpt_llm(temperature=temperature)
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
    return [model] if model else ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet", "claude-3-haiku-20240307"]

def ask_claude(system: str, question: str, context: str, summary: str, temperature: float) -> Tuple[str, float]:
    # Use cached Claude client instead of creating new one
    client = get_claude_client()
    text = (f"{system}\n\n"
            f"Conversation summary (for pronouns, references): {summary}\n\n"
            f"Question: {question}\n\nContext:\n{context}")
    last_err = None
    for m in claude_candidates():
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=m, max_tokens=700, temperature=temperature,
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG Chatbot (Follow-ups)", page_icon="💬", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")

    session_id = st.text_input("Session ID", value=st.session_state.get("session_id", "study1"))
    colA, colB = st.columns(2)
    with colA:
        if st.button("New/Reset Session"):
            reset_session(session_id)
            new_session(session_id)
            st.session_state.history = []
            st.success(f"Session '{session_id}' reset.")
    st.session_state.session_id = session_id

    backend = st.selectbox("Backend", ["gpt", "claude", "both"], index=0)
    course  = st.selectbox("Course filter", COURSES, index=1)  # default networking
    topk    = st.slider("Top-K (retrieval)", min_value=2, max_value=12, value=6, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                            help="Higher = more creative; lower = more factual and concise.")
    primary = st.radio("Primary answer to store (when Both)", ["gpt","claude"], index=0)

st.title("💬 RAG Chatbot with Follow-ups")
st.caption("Uses your Chroma DB + session memory + follow-up rewrite")

# Vector store (cache in session)
if "vs" not in st.session_state:
    st.session_state.vs = Chroma(persist_directory=DB_DIR)

# Load history (once per session id)
if "history" not in st.session_state or st.session_state.get("loaded_for") != session_id:
    # ensure session file exists
    if not os.path.exists(os.path.join("sessions", f"{session_id}.json")):
        new_session(session_id)
    st.session_state.history = load_history(session_id)
    st.session_state.loaded_for = session_id

history = st.session_state.history
vs      = st.session_state.vs

# Input (check for new question first)
q_raw = st.chat_input("Ask a question…")

# Chat UI (show all past history - keep all previous responses visible normally)
for role, text in history[-12:]:
    if role == "user":
        st.chat_message("user").markdown(text)
    else:
        st.chat_message("assistant").markdown(text)

# Process new question if any
if q_raw:
    # save user turn
    save_turn(session_id, "user", q_raw)
    history.append(("user", q_raw))
    st.chat_message("user").markdown(q_raw)
    
    # Show a placeholder assistant message that will be replaced with the actual response
    response_placeholder = st.empty()
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown("_Thinking..._")
    
    # follow-up helpers (Optimization 4: Skip for standalone questions)
    # Check if question is standalone - if so, skip expensive follow-up processing
    if history[:-1] and not is_standalone_question(q_raw):
        # This is likely a follow-up question, so we need context
        summary = summarize_recent(history[:-1])
        # include last assistant message to help rewrite pronouns
        last_assistant = ""
        for role, text in reversed(history[-6:]):
            if role == "assistant":
                last_assistant = text
                break
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

    # retrieval
    filt = None if course == "all" else {"course": {"$eq": course}}
    try:
        docs = vs.similarity_search(q_standalone, k=topk) if filt is None else vs.similarity_search(q_standalone, k=topk, filter=filt)
    except TypeError:
        docs = vs.similarity_search(q_standalone, k=topk)

    # fallback if empty
    if not docs and last_assistant:
        boosted = f"{q_standalone}\nDetails mentioned previously: {last_assistant}"
        try:
            docs = vs.similarity_search(boosted, k=topk) if filt is None else vs.similarity_search(boosted, k=topk, filter=filt)
        except TypeError:
            docs = vs.similarity_search(boosted, k=topk)

    if not docs:
        answer = "[No relevant chunks found. Try rephrasing or broaden your query.]"
        response_placeholder.empty()
        st.chat_message("assistant").markdown(answer)
        save_turn(session_id, "assistant", answer)
        history.append(("assistant", answer))
        st.stop()

    ctx = build_context(docs)

    # generate
    answer_gpt = answer_claude = None
    lat_gpt = lat_claude = None

    if backend in {"gpt","both"}:
        answer_gpt, lat_gpt = ask_gpt(SYSTEM, q_standalone, ctx, summary, temperature)

    if backend in {"claude","both"}:
        answer_claude, lat_claude = ask_claude(SYSTEM, q_standalone, ctx, summary, temperature)

    # Clear placeholder and show response(s)
    response_placeholder.empty()
    if backend == "both":
        # Show both responses together
        st.chat_message("assistant").markdown(f"**GPT** ({lat_gpt:.0f} ms):\n\n{answer_gpt}\n\n---\n\n**Claude** ({lat_claude:.0f} ms):\n\n{answer_claude}")
    elif backend == "gpt":
        st.chat_message("assistant").markdown(f"**GPT** ({lat_gpt:.0f} ms):\n\n{answer_gpt}")
    elif backend == "claude":
        st.chat_message("assistant").markdown(f"**Claude** ({lat_claude:.0f} ms):\n\n{answer_claude}")

    # store a single "primary" answer in session history file
    if backend == "gpt":
        save_turn(session_id, "assistant", answer_gpt or "")
        history.append(("assistant", answer_gpt or ""))
    elif backend == "claude":
        save_turn(session_id, "assistant", answer_claude or "")
        history.append(("assistant", answer_claude or ""))
    else:
        chosen = (answer_gpt if primary=="gpt" else answer_claude) or ""
        tag = "GPT" if primary=="gpt" else "Claude"
        save_turn(session_id, "assistant", f"[{tag}] {chosen}")
        history.append(("assistant", f"[{tag}] {chosen}"))

    # Expandable sources
    with st.expander("📚 Retrieved sources"):
        for d in docs:
            src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source",""))
            page = d.metadata.get("page","?")
            st.markdown(f"- **{src}**, p.{page}")
