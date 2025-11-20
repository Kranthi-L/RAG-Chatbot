# memory.py
import json, os, time
from typing import List, Dict, Tuple, Optional

MEM_DIR = os.getenv("MEM_DIR", "sessions")
os.makedirs(MEM_DIR, exist_ok=True)

# A turn is (role, text). role in {"user","assistant"}
Turn = Tuple[str, str]

def _path(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in ("-","_"))
    return os.path.join(MEM_DIR, f"{safe}.json")

def new_session(session_id: str) -> None:
    """Create or reset a session file."""
    data = {
        "session_id": session_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "turns": []  # list of {"role":"user"|"assistant", "text":"..."}
    }
    with open(_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history(session_id: str) -> List[Turn]:
    """Load turns; if file missing, return empty."""
    p = _path(session_id)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    turns = data.get("turns", [])
    return [(t.get("role","user"), t.get("text","")) for t in turns]

def save_turn(session_id: str, role: str, text: str) -> None:
    """Append one turn and persist."""
    p = _path(session_id)
    if not os.path.exists(p):
        new_session(session_id)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("turns", []).append({"role": role, "text": text})
    data["updated_at"] = time.time()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def reset_session(session_id: str) -> None:
    """Delete and recreate session file."""
    p = _path(session_id)
    if os.path.exists(p):
        os.remove(p)
    new_session(session_id)
