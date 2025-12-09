"""
Shared RAG utilities: vector store loader, retrieval, context building, and model answering.
Centralizes the core logic so UI/CLI/evaluation code use the same pipeline.
"""
import os
import time
from typing import Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic
from personalization import render_profile_instructions, LearnerLevel


load_dotenv()

# DB_DIR = os.getenv("DB_DIR", "chroma_db")
# SYSTEM_PATH = os.getenv("SYSTEM_PROMPT", "prompts/qa_system.md")
# SYSTEM = open(SYSTEM_PATH, "r").read()


# Resolve paths relative to this file's directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- DB_DIR: point to the same Chroma DB regardless of cwd ---
raw_db_dir = os.getenv("DB_DIR", "chroma_db")
if not os.path.isabs(raw_db_dir):
    DB_DIR = os.path.join(BASE_DIR, raw_db_dir)
else:
    DB_DIR = raw_db_dir

# --- SYSTEM prompt path ---
default_system_path = os.path.join(BASE_DIR, "prompts", "qa_system.md")
SYSTEM_PATH = os.getenv("SYSTEM_PROMPT", default_system_path)

with open(SYSTEM_PATH, "r", encoding="utf-8") as f:
    SYSTEM = f.read()


_vector_store: Optional[Chroma] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_gpt_llm_cache: Dict[float, ChatOpenAI] = {}
_claude_client: Optional[anthropic.Anthropic] = None


def get_vector_store() -> Chroma:
    """
    Load (and cache) the persisted Chroma vector store built by ingest.py.
    Returns the shared instance so callers reuse the same index.
    """
    global _vector_store
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
    if _vector_store is None:
        _vector_store = Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)
    return _vector_store


def _get_gpt_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a cached GPT client for the requested temperature."""
    if temperature not in _gpt_llm_cache:
        _gpt_llm_cache[temperature] = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    return _gpt_llm_cache[temperature]


def get_gpt_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Public helper to reuse cached GPT clients."""
    return _get_gpt_llm(temperature)


def _get_claude_client() -> anthropic.Anthropic:
    """Return a cached Claude client."""
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _claude_client


def retrieve_docs(
    query: str,
    course: Optional[str],
    top_k: int,
    last_assistant: Optional[str] = None,
) -> List[Document]:
    """
    Run similarity search with optional course filter and boosted fallback.
    Mirrors the retrieval flow used in the Streamlit app.
    """
    vs = get_vector_store()
    filt = None if not course or course == "all" else {"course": {"$eq": course}}

    try:
        docs = vs.similarity_search(query, k=top_k) if filt is None else vs.similarity_search(query, k=top_k, filter=filt)
    except TypeError:
        docs = vs.similarity_search(query, k=top_k)

    if not docs and last_assistant:
        boosted = f"{query}\nDetails mentioned previously: {last_assistant}"
        try:
            docs = vs.similarity_search(boosted, k=top_k) if filt is None else vs.similarity_search(boosted, k=top_k, filter=filt)
        except TypeError:
            docs = vs.similarity_search(boosted, k=top_k)

    return docs


def build_context(docs: List[Document]) -> str:
    """
    Construct a bounded context string from retrieved documents (~8000 chars max).
    Keeps ordering and includes filename/page/course metadata for citation clarity.
    """
    blocks: List[str] = []
    total_chars = 0
    max_context_chars = 8000

    for i, d in enumerate(docs, 1):
        src = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        block = f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})"

        block_size = len(block) + 2  # account for separator
        if total_chars + block_size > max_context_chars and blocks:
            break

        blocks.append(block)
        total_chars += block_size

    return "\n\n".join(blocks)


def _claude_candidates() -> List[str]:
    """Return preferred Claude model IDs (env override first)."""
    model = os.getenv("CLAUDE_MODEL")
    return [model] if model else ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet", "claude-3-haiku-20240307"]


# def ask_gpt(system: str, question: str, context: str, summary: str = "", temperature: float = 0.0) -> Tuple[str, float]:
#     """
#     Query GPT with the given system prompt, context, and optional summary.
#     Returns (answer, latency_ms).
#     """
#     llm = _get_gpt_llm(temperature=temperature)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         ("user", "Conversation summary (for pronouns, references): {summary}"),
#         ("user", "Question: {q}\n\nContext:\n{ctx}"),
#     ])
#     msgs = prompt.format_messages(summary=summary, q=question, ctx=context)
#     t0 = time.time()
#     out = llm.invoke(msgs).content
#     ms = (time.time() - t0) * 1000
#     return out, ms

def ask_gpt(
    system: str,
    question: str,
    context: str,
    summary: str = "",
    temperature: float = 0.0,
    learner_level: Optional[LearnerLevel] = None,
) -> Tuple[str, float]:
    """
    Query GPT with the given system prompt, context, and optional summary.

    learner_level:
        - None          -> no personalization (original behavior)
        - "beginner"    -> simpler, more scaffolded explanations
        - "intermediate"
        - "advanced"
    Returns (answer, latency_ms).
    """
    # Append learner profile instructions to the system prompt if provided
    if learner_level is not None:
        system_text = system + "\n\n" + render_profile_instructions(learner_level)
    else:
        system_text = system

    llm = _get_gpt_llm(temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("user", "Conversation summary (for pronouns, references): {summary}"),
            ("user", "Question: {q}\n\nContext:\n{ctx}"),
        ]
    )
    msgs = prompt.format_messages(summary=summary, q=question, ctx=context)
    t0 = time.time()
    out = llm.invoke(msgs).content
    ms = (time.time() - t0) * 1000
    return out, ms


# def ask_claude(system: str, question: str, context: str, summary: str = "", temperature: float = 0.0) -> Tuple[str, float]:
#     """
#     Query Claude with the given system prompt, context, and optional summary.
#     Returns (answer, latency_ms).
#     """
#     client = _get_claude_client()
#     text = (
#         f"{system}\n\n"
#         f"Conversation summary (for pronouns, references): {summary}\n\n"
#         f"Question: {question}\n\nContext:\n{context}"
#     )
#     last_err: Exception | None = None
#     for model in _claude_candidates():
#         try:
#             t0 = time.time()
#             resp = client.messages.create(
#                 model=model,
#                 max_tokens=700,
#                 temperature=temperature,
#                 messages=[{"role": "user", "content": text}],
#             )
#             ms = (time.time() - t0) * 1000
#             parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
#             return "\n".join(parts).strip(), ms
#         except anthropic.NotFoundError as e:
#             last_err = e
#         except Exception:
#             raise
#     raise last_err or RuntimeError("No working Claude model ID found")

def ask_claude(
    system: str,
    question: str,
    context: str,
    summary: str = "",
    temperature: float = 0.0,
    learner_level: Optional[LearnerLevel] = None,
) -> Tuple[str, float]:
    """
    Query Claude with the given system prompt, context, and optional summary.

    learner_level:
        - None          -> no personalization (original behavior)
        - "beginner"    -> simpler, more scaffolded explanations
        - "intermediate"
        - "advanced"
    Returns (answer, latency_ms).
    """
    # Append learner profile instructions to the system prompt if provided
    if learner_level is not None:
        system_text = system + "\n\n" + render_profile_instructions(learner_level)
    else:
        system_text = system

    client = _get_claude_client()
    text = (
        f"{system_text}\n\n"
        f"Conversation summary (for pronouns, references): {summary}\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    last_err: Exception | None = None
    for model in _claude_candidates():
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=model,
                max_tokens=700,
                temperature=temperature,
                messages=[{"role": "user", "content": text}],
            )
            ms = (time.time() - t0) * 1000
            parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
            return "\n".join(parts).strip(), ms
        except anthropic.NotFoundError as e:
            last_err = e
        except Exception:
            raise
    raise last_err or RuntimeError("No working Claude model ID found")


# def answer_with_model(
#     model: Literal["gpt", "claude"],
#     question: str,
#     context: str,
#     summary: str,
#     temperature: float,
# ) -> Tuple[str, float]:
#     """
#     Dispatch to GPT or Claude using the shared prompt format and return (answer, latency_ms).
#     """
#     if model == "gpt":
#         return ask_gpt(SYSTEM, question, context, summary, temperature)
#     if model == "claude":
#         return ask_claude(SYSTEM, question, context, summary, temperature)
#     raise ValueError(f"Unsupported model '{model}'")

def answer_with_model(
    model: Literal["gpt", "claude"],
    question: str,
    context: str,
    summary: str,
    temperature: float,
    learner_level: Optional[LearnerLevel] = None,
) -> Tuple[str, float]:
    """
    Dispatch to GPT or Claude using the shared prompt format and return (answer, latency_ms).

    learner_level:
        - None          -> no personalization (current default)
        - "beginner"    -> simpler, more scaffolded explanations
        - "intermediate"
        - "advanced"
    """
    if model == "gpt":
        return ask_gpt(SYSTEM, question, context, summary, temperature, learner_level)
    if model == "claude":
        return ask_claude(SYSTEM, question, context, summary, temperature, learner_level)
    raise ValueError(f"Unsupported model '{model}'")

