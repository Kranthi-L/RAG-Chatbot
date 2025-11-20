# memory.py
import json, os, time, atexit
from typing import List, Dict, Tuple, Optional

MEM_DIR = os.getenv("MEM_DIR", "sessions")
os.makedirs(MEM_DIR, exist_ok=True)

# A turn is (role, text). role in {"user","assistant"}
Turn = Tuple[str, str]

# ---------------------------
# Session caching (Optimization 3: Keep sessions in memory, flush periodically)
# ---------------------------
# Cache active sessions in memory to avoid reading/writing files on every turn
_session_cache: Dict[str, Dict] = {}
_FLUSH_INTERVAL = 5  # Flush to disk every N turns

def _path(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in ("-","_"))
    return os.path.join(MEM_DIR, f"{safe}.json")

def _flush_session(session_id: str) -> None:
    """Write cached session data to disk."""
    if session_id not in _session_cache:
        return
    data = _session_cache[session_id]
    with open(_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _flush_all_sessions() -> None:
    """Flush all cached sessions to disk (called on program exit)."""
    for session_id in list(_session_cache.keys()):
        _flush_session(session_id)

# Register flush on exit
atexit.register(_flush_all_sessions)

def new_session(session_id: str) -> None:
    """Create or reset a session file."""
    data = {
        "session_id": session_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "turns": []  # list of {"role":"user"|"assistant", "text":"..."}
    }
    # Update cache
    _session_cache[session_id] = data
    # Write to disk immediately for new sessions
    with open(_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history(session_id: str) -> List[Turn]:
    """Load turns; if file missing, return empty. Checks cache first."""
    # Check cache first
    if session_id in _session_cache:
        turns = _session_cache[session_id].get("turns", [])
        return [(t.get("role","user"), t.get("text","")) for t in turns]
    
    # Cache miss: load from disk
    p = _path(session_id)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Store in cache for future use
    _session_cache[session_id] = data
    turns = data.get("turns", [])
    return [(t.get("role","user"), t.get("text","")) for t in turns]

def save_turn(session_id: str, role: str, text: str) -> None:
    """Append one turn and persist. Uses cache with periodic flushing."""
    # Ensure session exists in cache
    if session_id not in _session_cache:
        p = _path(session_id)
        if not os.path.exists(p):
            new_session(session_id)
        else:
            # Load existing session into cache
            with open(p, "r", encoding="utf-8") as f:
                _session_cache[session_id] = json.load(f)
    
    # Update cache
    data = _session_cache[session_id]
    data.setdefault("turns", []).append({"role": role, "text": text})
    data["updated_at"] = time.time()
    
    # Flush to disk every N turns to balance performance and persistence
    turn_count = len(data.get("turns", []))
    if turn_count % _FLUSH_INTERVAL == 0:
        _flush_session(session_id)

def reset_session(session_id: str) -> None:
    """Delete and recreate session file. Also clears cache."""
    # Clear from cache
    if session_id in _session_cache:
        del _session_cache[session_id]
    # Delete file
    p = _path(session_id)
    if os.path.exists(p):
        os.remove(p)
    # Create new session (which will also update cache)
    new_session(session_id)
