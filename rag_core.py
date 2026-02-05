"""
Shared RAG utilities: vector store loader, retrieval, context building, and model answering.
Centralizes the core logic so UI/CLI/evaluation code use the same pipeline.
"""
import os
import pickle
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic
from rank_bm25 import BM25Okapi
from personalization import (
    normalize_learner_level,
    get_retrieval_config_for_level,
    get_generation_style_instructions,
)


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
DEBUG_RETRIEVAL_LOG = os.getenv("DEBUG_RETRIEVAL_LOG", "false").lower() == "true"


class RetrieverType(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    SECTION_AWARE = "section_aware"


DEFAULT_RETRIEVER = RetrieverType.DENSE

# Hybrid retrieval tuning
HYBRID_DENSE_K = 30
HYBRID_BM25_K = 30
HYBRID_ALPHA = 0.5  # weight for dense; (1-alpha) for BM25

# Section-aware tuning
SECTION_CANDIDATE_K = 50

# BM25 cache: course -> (bm25, docs)
_bm25_cache: Dict[str, Tuple[BM25Okapi, List[Dict]]] = {}


def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


def _doc_id(meta: Dict) -> str:
    course = meta.get("course", "unknown")
    filename = meta.get("filename", meta.get("source", "unknown"))
    page = meta.get("page", "0")
    idx = meta.get("chunk_index", meta.get("idx", "0"))
    return f"{course}|{filename}|{page}|{idx}"

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


def get_bm25_index(course: str) -> Tuple[BM25Okapi, List[Dict]]:
    """
    Lazy-load BM25 index and doc records for a course.
    Files are expected at {DB_DIR}/bm25_{course}.pkl and bm25_docs_{course}.pkl.
    """
    key = (course or "unknown").lower()
    if key in _bm25_cache:
        return _bm25_cache[key]

    idx_path = os.path.join(DB_DIR, f"bm25_{key}.pkl")
    docs_path = os.path.join(DB_DIR, f"bm25_docs_{key}.pkl")
    if not (os.path.exists(idx_path) and os.path.exists(docs_path)):
        raise FileNotFoundError(f"BM25 index for course '{course}' not found at {idx_path}")

    with open(idx_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    _bm25_cache[key] = (bm25, docs)
    return _bm25_cache[key]


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


def _retrieve_dense(query: str, course: Optional[str], top_k: int, last_assistant: Optional[str] = None) -> List[Document]:
    vs = get_vector_store()
    filt = None if not course or course == "all" else {"course": {"$eq": course}}
    try:
        docs = vs.similarity_search(query, k=top_k) if filt is None else vs.similarity_search(query, k=top_k, filter=filt)
    except Exception:
        docs = []

    if not docs and last_assistant:
        boosted = f"{query}\nDetails mentioned previously: {last_assistant}"
        try:
            docs = vs.similarity_search(boosted, k=top_k) if filt is None else vs.similarity_search(boosted, k=top_k, filter=filt)
        except Exception:
            docs = []
    for idx, d in enumerate(docs):
        d.metadata.setdefault("chunk_index", idx)
        d.metadata.setdefault("doc_id", _doc_id(d.metadata))
    return docs


def _retrieve_bm25(query: str, course: Optional[str], top_k: int) -> List[Document]:
    if not course:
        return []
    bm25, bm25_docs = get_bm25_index(course)
    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results: List[Document] = []
    for i in top_indices:
        rec = bm25_docs[i]
        meta = rec.get("metadata", {}).copy()
        meta.setdefault("doc_id", rec.get("doc_id"))
        meta.setdefault("score_bm25", scores[i])
        results.append(Document(page_content=rec.get("text", ""), metadata=meta))
    return results


def _retrieve_hybrid(query: str, course: Optional[str], top_k: int, last_assistant: Optional[str] = None) -> List[Document]:
    dense_candidates = _retrieve_dense(query, course, HYBRID_DENSE_K, last_assistant)
    bm25_candidates = _retrieve_bm25(query, course, HYBRID_BM25_K) if course else []

    merged: Dict[str, Dict] = {}
    # Collect dense scores using relative order as a proxy if scores missing
    for idx, d in enumerate(dense_candidates):
        key = _doc_id(d.metadata)
        if key not in merged:
            merged[key] = {"doc": d, "dense_rank": idx, "bm25_score": 0.0, "dense_score": float(HYBRID_DENSE_K - idx)}
        else:
            merged[key]["dense_score"] = float(HYBRID_DENSE_K - idx)

    for idx, d in enumerate(bm25_candidates):
        key = _doc_id(d.metadata)
        if key not in merged:
            merged[key] = {"doc": d, "dense_score": 0.0, "bm25_score": float(HYBRID_BM25_K - idx)}
        else:
            merged[key]["bm25_score"] = float(HYBRID_BM25_K - idx)

    if not merged:
        return []

    dense_vals = [v["dense_score"] for v in merged.values()]
    bm25_vals = [v["bm25_score"] for v in merged.values()]
    d_min, d_max = min(dense_vals), max(dense_vals)
    b_min, b_max = min(bm25_vals), max(bm25_vals)

    def norm(val: float, vmin: float, vmax: float) -> float:
        if vmax == vmin:
            return 0.0
        return (val - vmin) / (vmax - vmin)

    for rec in merged.values():
        d_norm = norm(rec["dense_score"], d_min, d_max) if dense_vals else 0.0
        b_norm = norm(rec["bm25_score"], b_min, b_max) if bm25_vals else 0.0
        rec["hybrid_score"] = HYBRID_ALPHA * d_norm + (1 - HYBRID_ALPHA) * b_norm

    ordered = sorted(merged.values(), key=lambda r: r["hybrid_score"], reverse=True)
    return [r["doc"] for r in ordered[:top_k]]


def _retrieve_section_aware(query: str, course: Optional[str], top_k: int, last_assistant: Optional[str] = None) -> List[Document]:
    # Start from hybrid candidates for better coverage
    candidates = _retrieve_hybrid(query, course, SECTION_CANDIDATE_K, last_assistant)
    if DEBUG_RETRIEVAL_LOG:
        print(f"[section_aware] query='{query[:60]}' candidates={len(candidates)}")
    if not candidates:
        return []

    # Approximate scores: preserve order weight
    scored = []
    for idx, d in enumerate(candidates):
        scored.append((d, float(SECTION_CANDIDATE_K - idx)))

    sections: Dict[str, Dict] = defaultdict(lambda: {"docs": [], "score": 0.0})
    for doc, sc in scored:
        meta = doc.metadata
        section_id = f"{meta.get('filename')}:{meta.get('page')}"
        sections[section_id]["docs"].append((doc, sc))
        sections[section_id]["score"] = max(sections[section_id]["score"], sc)

    ordered_sections = sorted(sections.items(), key=lambda kv: kv[1]["score"], reverse=True)
    if DEBUG_RETRIEVAL_LOG:
        top_sections = [sid for sid, _ in ordered_sections[:5]]
        print(f"[section_aware] unique_sections={len(sections)} top_sections={top_sections}")
    output: List[Document] = []
    for sec_id, payload in ordered_sections:
        # keep original order within section
        for doc, _s in payload["docs"]:
            doc.metadata["section_id"] = sec_id
            output.append(doc)
            if len(output) >= top_k:
                return output
    return output


def rerank_docs_for_level(
    query: str,
    docs: List[Document],
    learner_level: Optional[str],
    max_docs_for_rerank: int = 20,
) -> List[Document]:
    """
    Level-aware reranker using GPT to reorder top-N docs for the target learner level.
    Falls back to original order on any error.
    """
    if not docs:
        return docs

    lvl = normalize_learner_level(learner_level)
    if lvl is None:
        return docs

    subset = docs[:max_docs_for_rerank]
    # Ensure doc_ids
    for idx, d in enumerate(subset):
        d.metadata.setdefault("chunk_index", idx)
        d.metadata.setdefault("doc_id", _doc_id(d.metadata))

    snippets = []
    for i, d in enumerate(subset, 1):
        meta = d.metadata
        snippet = d.page_content[:400].replace("\n", " ")
        snippets.append(
            f"{i}. id={meta.get('doc_id')} | file={meta.get('filename')} p={meta.get('page')} | text: {snippet}"
        )

    level_norm = (lvl or "general").lower()
    if level_norm not in {"beginner", "intermediate", "advanced"}:
        level_norm = "general"
    level_text = {
        "beginner": "Beginner (prefers simpler, high-level explanations).",
        "intermediate": "Intermediate (comfortable with some detail and formalism).",
        "advanced": "Advanced (prefers depth, rigor, and technical detail).",
        "general": "General audience.",
    }.get(level_norm, "General audience.")

    prompt_text = (
        "You are reranking retrieved chunks for a learner.\n"
        f"Learner level: {level_text}\n"
        f"Question: {query}\n\n"
        "Chunks:\n"
        + "\n".join(snippets)
        + "\n\nReturn a single JSON array of doc_ids ordered from MOST to LEAST useful "
        "for answering the question for this learner level. Example: [\"id1\",\"id3\",\"id2\"]"
    )

    try:
        llm = _get_gpt_llm(temperature=0.0)
        resp = llm.invoke([("user", prompt_text)]).content
        import json

        parsed = json.loads(resp)
        if not isinstance(parsed, list):
            raise ValueError("LLM response not a list")
        id_to_doc = {d.metadata.get("doc_id"): d for d in subset}
        ordered = [id_to_doc[i] for i in parsed if i in id_to_doc]
        # Append any missing docs in original order
        for d in subset:
            if d not in ordered:
                ordered.append(d)
        # Preserve untouched docs beyond rerank window
        if len(docs) > len(subset):
            ordered.extend(docs[len(subset):])
        return ordered
    except Exception:
        return docs


def retrieve_docs(
    query: str,
    course: Optional[str],
    top_k: int,
    retriever_type: RetrieverType = DEFAULT_RETRIEVER,
    last_assistant: Optional[str] = None,
    learner_level: Optional[str] = None,
) -> List[Document]:
    """
    Pluggable retrieval entry point. Defaults to dense retrieval (original behavior).
    """
    lvl = normalize_learner_level(learner_level)
    cfg = get_retrieval_config_for_level(top_k, lvl)
    effective_top_k = cfg["effective_top_k"]
    use_reranker = cfg["use_level_reranker"]

    if retriever_type == RetrieverType.DENSE:
        docs = _retrieve_dense(query, course, effective_top_k, last_assistant)
    elif retriever_type == RetrieverType.BM25:
        docs = _retrieve_bm25(query, course, effective_top_k)
    elif retriever_type == RetrieverType.HYBRID:
        docs = _retrieve_hybrid(query, course, effective_top_k, last_assistant)
    elif retriever_type == RetrieverType.SECTION_AWARE:
        docs = _retrieve_section_aware(query, course, effective_top_k, last_assistant)
    else:
        docs = _retrieve_dense(query, course, effective_top_k, last_assistant)

    if use_reranker and lvl is not None and docs:
        docs = rerank_docs_for_level(query, docs, lvl)

    if DEBUG_RETRIEVAL_LOG:
        summary = f"[retrieval] type={retriever_type.value} course={course} q='{query[:60]}' k={len(docs)}"
        if docs:
            d0 = docs[0]
            meta = d0.metadata
            snippet = d0.page_content[:80].replace("\n", " ")
            summary += f" first=({meta.get('filename')} p{meta.get('page')}): {snippet}"
        print(summary)

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
    learner_level: Optional[str] = None,
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
    style_text = get_generation_style_instructions(learner_level)
    constraint = ""
    lvl_norm = normalize_learner_level(learner_level)
    if lvl_norm == "beginner":
        constraint = "Keep the answer concise (under 180 tokens)."
    elif lvl_norm == "advanced":
        constraint = "Ensure technical depth (minimum 120 tokens)."
    style_block = ""
    if style_text:
        style_block = style_text + "\nFollow these instructions exactly."
        if constraint:
            style_block += "\n" + constraint
    system_text = system if not style_block else system + "\n\n" + style_block

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
    learner_level: Optional[str] = None,
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
    style_text = get_generation_style_instructions(learner_level)
    constraint = ""
    lvl_norm = normalize_learner_level(learner_level)
    if lvl_norm == "beginner":
        constraint = "Keep the answer concise (under 180 tokens)."
    elif lvl_norm == "advanced":
        constraint = "Ensure technical depth (minimum 120 tokens)."
    style_block = ""
    if style_text:
        style_block = style_text + "\nFollow these instructions exactly."
        if constraint:
            style_block += "\n" + constraint
    system_text = system if not style_block else system + "\n\n" + style_block

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
    learner_level: Optional[str] = None,
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
