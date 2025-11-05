# app_cli.py
import os, sys, time
from dotenv import load_dotenv
from rich import print as rprint

# Vector store + prompts (modern LangChain imports)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# GPT (OpenAI)
from langchain_openai import ChatOpenAI

# Claude (Anthropic)
import anthropic
from anthropic import NotFoundError

# Quiet HF tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

DB_DIR = "chroma_db"
SYSTEM = open("prompts/qa_system.md").read()

# Open the vector store (already built with local embeddings via ingest.py)
vs = Chroma(persist_directory=DB_DIR)

def ask_gpt(system: str, question: str, context: str) -> tuple[str, float]:
    """Return (answer, latency_ms) from GPT."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {q}\n\nContext:\n{ctx}")
    ])
    msgs = prompt.format_messages(q=question, ctx=context)
    t0 = time.time()
    out = llm.invoke(msgs).content
    ms = (time.time() - t0) * 1000
    return out, ms

def get_claude_model() -> list[str] | str:
    """Pick a Claude model; allow override via env; try a few common IDs."""
    env_model = os.getenv("CLAUDE_MODEL")
    if env_model:
        return env_model
    # Try newest → generic alias → cheap fallback
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet",
        "claude-3-haiku-20240307",
    ]

def ask_claude(system: str, question: str, context: str) -> tuple[str, float]:
    """Return (answer, latency_ms) from Claude with model fallback."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_text = f"{system}\n\nQuestion: {question}\n\nContext:\n{context}"

    candidates = get_claude_model()
    if isinstance(candidates, str):
        candidates = [candidates]

    last_err = None
    for model_id in candidates:
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=model_id,
                max_tokens=700,
                temperature=0,
                messages=[{"role": "user", "content": user_text}],
            )
            ms = (time.time() - t0) * 1000
            parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
            return ("\n".join(parts).strip(), ms)
        except NotFoundError as e:
            last_err = e  # try next candidate
        except Exception as e:
            raise  # surface non-404 errors immediately
    raise last_err or RuntimeError("No working Claude model ID found.")

def build_context(docs) -> str:
    """Format retrieved chunks as context with simple citations."""
    blocks = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        course = d.metadata.get("course") or d.metadata.get("book", "unknown")
        blocks.append(f"[{i}] {d.page_content}\n(Source: {src}, p.{page}, course:{course})")
    return "\n\n".join(blocks)

def build_filter(targets: list[str] | None):
    """Return a Chroma filter dict or None."""
    if not targets or "all" in targets:
        return None
    if len(targets) == 1:
        return {"course": {"$eq": targets[0]}}
    return {"course": {"$in": targets}}

def main():
    # Usage: python app_cli.py [gpt|claude|both] [all|networking|architecture|comma,list]
    backend = (sys.argv[1] if len(sys.argv) > 1 else "gpt").lower()
    raw_target = (sys.argv[2] if len(sys.argv) > 2 else "all").lower()

    if backend not in {"gpt", "claude", "both"}:
        print("Usage: python app_cli.py [gpt|claude|both] [all|networking|architecture|comma,list]")
        return

    # Allow comma-separated list: "networking,architecture"
    targets = [t.strip() for t in raw_target.split(",")] if raw_target else ["all"]

    # (Optional) validate known courses; you can skip this block if you want to allow any folder name
    known = {"all", "networking", "architecture"}
    if not all(t in known for t in targets):
        print("Second arg must be: all | networking | architecture | comma-separated list (e.g., networking,architecture)")
        return

    filt = build_filter(targets)
    rprint(f"[bold]RAG Chatbot[/bold] — backend: {backend.upper()}  |  course(s): {','.join(targets)}")

    while True:
        q = input("\nAsk a question (or 'q' to quit): ").strip()
        if q.lower() == "q":
            break
        if not q:
            continue

        # Retrieve with/without filter
        try:
            docs = vs.similarity_search(q, k=4) if filt is None else vs.similarity_search(q, k=4, filter=filt)
        except TypeError:
            # Fallback for older adapters that don't accept 'filter'
            docs = vs.similarity_search(q, k=4)

        if not docs:
            rprint("[yellow]No relevant chunks found. Try a simpler query or expand your index.[/yellow]")
            continue

        ctx = build_context(docs)

        # Generate
        if backend in {"gpt", "both"}:
            ans, ms = ask_gpt(SYSTEM, q, ctx)
            rprint(f"\n[bold cyan]GPT[/bold cyan]  ({ms:.0f} ms):\n{ans}")

        if backend in {"claude", "both"}:
            try:
                ans, ms = ask_claude(SYSTEM, q, ctx)
                rprint(f"\n[bold magenta]Claude[/bold magenta]  ({ms:.0f} ms):\n{ans}")
            except NotFoundError as e:
                rprint(f"\n[bold magenta]Claude[/bold magenta]: [red]Model not found[/red]. "
                       f"Set CLAUDE_MODEL in .env or edit app_cli.py candidates. Details: {e}")

if __name__ == "__main__":
    main()
