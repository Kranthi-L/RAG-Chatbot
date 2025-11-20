# app_streamlit.py
import os
import time
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from rich import print as rprint  # dev logs to console if you want

# LangChain (modern imports)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# GPT (OpenAI)
from langchain_openai import ChatOpenAI

# Claude (Anthropic)
import anthropic
from anthropic import NotFoundError

# Quiet HF tokenizers fork warning (prevents a noisy warning on macOS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv()
DB_DIR = "chroma_db"
SYSTEM_PROMPT_PATH = "prompts/qa_system.md"

def read_system_prompt() -> str:
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            return f.read()
    return (
        "You are a course Q&A assistant. Answer ONLY using the provided context.\n"
        "If the answer is not in the context, reply exactly: \"I don’t know based on the provided material.\"\n"
        "Always cite the source filename and page if available.\n"
        "Be concise and structured."
    )

# -----------------------------------------------------------------------------
# Caching: load vector store once
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vs() -> Chroma:
    return Chroma(persist_directory=DB_DIR)

# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------
def ask_gpt(system: str, question: str, context: str, temperature: float) -> Tuple[str, float]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {q}\n\nContext:\n{ctx}")
    ])
    msgs = prompt.format_messages(q=question, ctx=context)
    t0 = time.time()
    out = llm.invoke(msgs).content
    ms = (time.time() - t0) * 1000
    return out, ms

def get_claude_candidates() -> List[str]:
    env_model = os.getenv("CLAUDE_MODEL")
    if env_model:
        return [env_model]
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet",
        "claude-3-haiku-20240307",
    ]

def ask_claude(system: str, question: str, context: str, temperature: float) -> Tuple[str, float]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_text = f"{system}\n\nQuestion: {question}\n\nContext:\n{context}"
    last_err = None
    for model_id in get_claude_candidates():
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=model_id,
                max_tokens=700,
                temperature=temperature,
                messages=[{"role": "user", "content": user_text}],
            )
            ms = (time.time() - t0) * 1000
            parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
            return ("\n".join(parts).strip(), ms)
        except NotFoundError as e:
            last_err = e
        except Exception as e:
            raise
    raise last_err or RuntimeError("No working Claude model ID found.")

# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------
def build_filter(selected_courses: List[str]) -> Optional[dict]:
    if not selected_courses or "all" in selected_courses:
        return None
    if len(selected_courses) == 1:
        return {"course": {"$eq": selected_courses[0]}}
    return {"course": {"$in": selected_courses}}

def build_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        blocks.append(f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})")
    return "\n\n".join(blocks)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG Course Chatbot", page_icon="📚", layout="wide")
st.title("📚 RAG Course Chatbot")

with st.sidebar:
    st.header("Settings")
    # Courses (auto add here if/when you add folders later)
    courses = st.multiselect(
        "Course(s)",
        options=["all", "networking", "architecture"],
        default=["networking"],
        help="Choose one or many. 'all' searches everything.",
    )
    model_pick = st.selectbox("Model", ["GPT", "Claude", "Both"], index=2)
    k = st.slider("Top-K (retrieval)", 1, 12, 4)
    temperature = st.slider("Temperature", 0.0, 0.7, 0.0, step=0.1)
    show_context = st.checkbox("Show retrieved chunks", value=True)
    system_prompt = read_system_prompt()

    st.divider()
    st.caption("API keys come from your .env")
    ok_openai = bool(os.getenv("OPENAI_API_KEY"))
    ok_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    st.write(f"OpenAI key: {'✅' if ok_openai else '❌'}")
    st.write(f"Anthropic key: {'✅' if ok_claude else '❌'}")

vs = get_vs()

q = st.text_input("Ask a question", placeholder="e.g., Explain pipeline hazards.")
go = st.button("Ask")

if go and q.strip():
    filt = build_filter(courses)
    # Retrieve
    with st.spinner("Retrieving…"):
        try:
            if filt is None:
                docs = vs.similarity_search(q, k=k)
            else:
                docs = vs.similarity_search(q, k=k, filter=filt)
        except TypeError:
            docs = vs.similarity_search(q, k=k)

    if not docs:
        st.warning("No relevant chunks found. Try a simpler query or index more content.")
    else:
        ctx = build_context(docs)

        # Show retrieved chunks
        if show_context:
            with st.expander("Retrieved chunks (context sent to the model)"):
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
                    page = d.metadata.get("page", "?")
                    course = d.metadata.get("course") or d.metadata.get("book", "unknown")
                    st.markdown(f"**[{i}] {src} — p.{page} — course: {course}**")
                    st.code(d.page_content[:1500] + ("…" if len(d.page_content) > 1500 else ""))

        cols = st.columns(2) if model_pick == "Both" else [st.container()]
        # GPT
        if model_pick in ("GPT", "Both"):
            with cols[0]:
                try:
                    ans, ms = ask_gpt(system_prompt, q, ctx, temperature)
                    st.subheader(f"GPT ({ms:.0f} ms)")
                    st.write(ans)
                except Exception as e:
                    st.error(f"GPT error: {e}")

        # Claude
        if model_pick in ("Claude", "Both"):
            with (cols[1] if model_pick == "Both" else cols[0]):
                try:
                    ans, ms = ask_claude(system_prompt, q, ctx, temperature)
                    st.subheader(f"Claude ({ms:.0f} ms)")
                    st.write(ans)
                except NotFoundError as e:
                    st.error("Claude model not found. Set CLAUDE_MODEL in .env or edit candidates in app.")
                except Exception as e:
                    st.error(f"Claude error: {e}")
