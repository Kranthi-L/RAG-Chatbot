# eval/run_batch_networking.py
import os, csv, time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Vector store
from langchain_community.vectorstores import Chroma

# Prompting
from langchain_core.prompts import ChatPromptTemplate

# LLMs
from langchain_openai import ChatOpenAI
import anthropic
from anthropic import NotFoundError

load_dotenv()

DB_DIR = os.getenv("DB_DIR", "chroma_db")
SYSTEM_PATH = os.getenv("SYSTEM_PROMPT", "prompts/qa_system.md")
INPUT_CSV = os.getenv("INPUT_CSV", "eval/networking_eval.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "eval/networking_eval_filled.csv")

# retrieval params
COURSE = os.getenv("COURSE", "networking")   # filter to networking
TOP_K = int(os.getenv("TOP_K", "6"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Which models to run
RUN_GPT = os.getenv("RUN_GPT", "1") == "1"
RUN_CLAUDE = os.getenv("RUN_CLAUDE", "1") == "1"

# Simple backoff between API calls (seconds)
PAUSE = float(os.getenv("PAUSE", "0.3"))

def read_system() -> str:
    try:
        with open(SYSTEM_PATH, "r") as f:
            return f.read()
    except:
        return ("You are a course Q&A assistant. Answer ONLY using the provided context.\n"
                "If the answer is not in the context, reply exactly: \"I don’t know based on the provided material.\"")

def build_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        blocks.append(f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})")
    return "\n\n".join(blocks)

def get_vs():
    return Chroma(persist_directory=DB_DIR)

def retrieve(vs, q: str, k: int, course: Optional[str]):
    filt = None if not course or course == "all" else {"course": {"$eq": course}}
    try:
        # try with relevance scores
        pairs = vs.similarity_search_with_relevance_scores(q, k=k, filter=filt) if filt else \
                vs.similarity_search_with_relevance_scores(q, k=k)
        docs = [p[0] for p in pairs]
    except Exception:
        docs = vs.similarity_search(q, k=k) if filt is None else vs.similarity_search(q, k=k, filter=filt)
    return docs

def ask_gpt(system: str, q: str, ctx: str) -> Tuple[str, float]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {q}\n\nContext:\n{ctx}")
    ])
    msgs = prompt.format_messages(q=q, ctx=ctx)
    t0 = time.time()
    out = llm.invoke(msgs).content
    ms = (time.time() - t0) * 1000
    return out, ms

def claude_candidates() -> List[str]:
    model = os.getenv("CLAUDE_MODEL")
    if model:
        return [model]
    return ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet", "claude-3-haiku-20240307"]

def ask_claude(system: str, q: str, ctx: str) -> Tuple[str, float]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    text = f"{system}\n\nQuestion: {q}\n\nContext:\n{ctx}"
    last_err = None
    for m in claude_candidates():
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=m, max_tokens=700, temperature=TEMPERATURE,
                messages=[{"role": "user", "content": text}]
            )
            parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
            return "\n".join(parts).strip(), (time.time() - t0) * 1000
        except NotFoundError as e:
            last_err = e
        except Exception as e:
            raise
    raise last_err or RuntimeError("No working Claude model ID found")

def load_rows(path: str):
    # Skip chapter header comment lines starting with '#'
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            rows.append(line)
    # re-parse as CSV
    reader = csv.DictReader(rows)
    # Ensure required columns exist
    needed = {"question", "gpt_response", "claude_response", "ideal_answer"}
    if not needed.issubset(reader.fieldnames or set()):
        raise ValueError(f"CSV must contain columns: {needed}. Found: {reader.fieldnames}")
    return list(reader)

def save_rows(path: str, rows: List[Dict[str, str]]):
    fieldnames = ["question", "gpt_response", "claude_response", "ideal_answer", "gpt_ms", "claude_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    system = read_system()
    vs = get_vs()
    rows = load_rows(INPUT_CSV)
    out = []
    for i, r in enumerate(rows, 1):
        q = r["question"].strip()
        ideal = r["ideal_answer"].strip()
        gpt_text = r.get("gpt_response", "").strip()
        claude_text = r.get("claude_response", "").strip()

        # retrieve context
        docs = retrieve(vs, q, TOP_K, COURSE)
        if not docs:
            ctx = "(No retrieved context.)"
        else:
            ctx = build_context(docs)

        # fill responses
        gpt_ms = ""
        claude_ms = ""
        if RUN_GPT:
            try:
                gpt_text, gpt_latency = ask_gpt(system, q, ctx)
                gpt_ms = f"{gpt_latency:.0f}"
            except Exception as e:
                gpt_text = f"[GPT ERROR] {e}"
        if RUN_CLAUDE:
            # polite pacing
            time.sleep(PAUSE)
            try:
                claude_text, claude_latency = ask_claude(system, q, ctx)
                claude_ms = f"{claude_latency:.0f}"
            except Exception as e:
                claude_text = f"[CLAUDE ERROR] {e}"

        out.append({
            "question": q,
            "gpt_response": gpt_text,
            "claude_response": claude_text,
            "ideal_answer": ideal,
            "gpt_ms": gpt_ms,
            "claude_ms": claude_ms
        })

        # light progress note
        print(f"[{i}/{len(rows)}] done")

        # optional: gentle pacing between rows
        time.sleep(PAUSE)

    save_rows(OUTPUT_CSV, out)
    print(f"\nWrote {OUTPUT_CSV} with {len(out)} rows.")

if __name__ == "__main__":
    main()
