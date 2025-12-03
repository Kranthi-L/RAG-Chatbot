# app_web.py
import os, time, json
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
# Note: Using client-side session state only (Option 3) - no file persistence
# from memory import load_history, save_turn, new_session, reset_session

# Phase 2: Speech-to-Text and Text-to-Speech
from audio_utils import record_and_transcribe, speak_text

# Phase 2: Modality management (modular)
from modality_utils import get_input_modality, should_auto_speak, get_auto_speak_help_text

# Phase 2: Continuous listening (modular)
from continuous_listening import (
    start_continuous_listening,
    stop_continuous_listening,
    is_continuous_listening_active,
    should_start_listening,
    get_listening_state,
    transition_to_processing,
    transition_to_speaking,
    transition_back_to_listening
)

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
            st.session_state.history = []
            st.session_state.loaded_for = None
            st.success(f"Session '{session_id}' reset.")
    
    
    st.session_state.session_id = session_id

    backend = st.selectbox("Backend", ["gpt", "claude", "both"], index=0)
    course  = st.selectbox("Course filter", COURSES, index=1)  # default networking
    topk    = st.slider("Top-K (retrieval)", min_value=2, max_value=12, value=6, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                            help="Higher = more creative; lower = more factual and concise.")
    primary = st.radio("Primary answer to store (when Both)", ["gpt","claude"], index=0)
    
    st.divider()
    st.subheader("📝 Input Modality")
    st.caption("Choose how you want to interact with the chatbot")
    
    # Modality selector - controls auto-speak behavior, not input availability
    current_modality = get_input_modality(st.session_state, default="Text")
    modality_options = ["Text", "Speech"]
    current_modality_idx = modality_options.index(current_modality) if current_modality in modality_options else 0
    
    modality = st.radio(
        "Input Modality",
        modality_options,
        index=current_modality_idx,
        key="modality_selector",
        help="Speech mode automatically enables auto-speak. Both input methods remain available regardless of selection."
    )
    
    # Store modality in session state and manage continuous listening
    if st.session_state.get("input_modality") != modality:
        st.session_state["input_modality"] = modality
        # Start continuous listening if Speech mode, stop if Text mode
        if modality == "Speech":
            start_continuous_listening(st.session_state)
        else:
            stop_continuous_listening(st.session_state)
        st.rerun()
    
    # Show continuous listening status and controls (only in Speech mode)
    if modality == "Speech":
        is_active = is_continuous_listening_active(st.session_state)
        current_state = get_listening_state(st.session_state)
        
        if is_active:
            state_emoji = {
                "listening": "🎤",
                "processing": "⚙️",
                "speaking": "🔊",
                "idle": "⏸️"
            }.get(current_state, "🎤")
            st.caption(f"{state_emoji} Continuous listening: {current_state.capitalize()}")
            
            if st.button("⏸️ Stop Listening", use_container_width=True):
                stop_continuous_listening(st.session_state)
                # Rerun to update UI, but history is preserved in session_state
                # History won't disappear because it's stored in st.session_state.history
                st.rerun()
        else:
            st.caption("⏸️ Continuous listening stopped")
            if st.button("▶️ Start Listening", use_container_width=True, type="primary"):
                start_continuous_listening(st.session_state)
                st.rerun()
    
    st.divider()
    st.subheader("🎤 Speech Input")
    st.caption("Click to record. Stops automatically when you finish speaking.")
    
    # Optional advanced settings in expander
    with st.expander("⚙️ Recording Settings (Advanced)"):
        silence_threshold = st.slider(
            "Sensitivity", 
            min_value=0.001, 
            max_value=0.05, 
            value=0.01, 
            step=0.001,
            help="Lower = more sensitive (detects quieter speech). Higher = less sensitive (requires louder speech, filters background noise better). Default: 0.01"
        )
        silence_duration = st.slider(
            "Silence duration (seconds)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.5, 
            step=0.5,
            help="How long to wait after you stop speaking before ending recording."
        )
        use_fixed_duration = st.checkbox(
            "Use fixed duration (fallback)",
            value=False,
            help="If automatic detection doesn't work, use fixed 5-second recording"
        )
    
    # Microphone button for speech-to-text with automatic silence detection
    if st.button("🎤 Record Question", use_container_width=True, type="primary"):
        st.session_state["recording"] = True
        st.session_state["transcribed_text"] = None
    
    # Show recording status
    if st.session_state.get("recording", False):
        if use_fixed_duration:
            spinner_msg = "🎤 Recording... Speak your question now! (5 seconds)"
        else:
            spinner_msg = "🎤 Listening... Speak your question (stops automatically when you finish)"
        
        with st.spinner(spinner_msg):
            try:
                if use_fixed_duration:
                    # Fallback to fixed duration recording
                    transcribed = record_and_transcribe(use_silence_detection=False, duration=5.0)
                else:
                    # Use automatic silence detection
                    transcribed = record_and_transcribe(
                        use_silence_detection=True,
                        silence_threshold=silence_threshold,
                        silence_duration=silence_duration
                    )
                
                if not transcribed or len(transcribed.strip()) == 0:
                    st.warning("⚠️ Transcription is empty. Try speaking louder or use fixed duration mode.")
                    st.session_state["recording"] = False
                    st.session_state["transcribed_text"] = None
                    # Return to listening state in continuous mode
                    if is_continuous_listening_active(st.session_state):
                        transition_back_to_listening(st.session_state)
                else:
                    st.session_state["transcribed_text"] = transcribed
                    st.session_state["recording"] = False
                    st.success(f"✅ Transcribed: {transcribed[:50]}..." if len(transcribed) > 50 else f"✅ Transcribed: {transcribed}")
            except Exception as e:
                error_msg = str(e)
                st.error(f"Recording failed: {error_msg}")
                st.info("💡 Tip: Try enabling 'Use fixed duration' in Advanced Settings, or lower the sensitivity threshold.")
                st.session_state["recording"] = False
                st.session_state["transcribed_text"] = None
                # Return to listening state in continuous mode
                if is_continuous_listening_active(st.session_state):
                    transition_back_to_listening(st.session_state)
    
    st.divider()
    st.subheader("🔊 Text-to-Speech")
    st.caption("Read responses aloud")
    
    # Get current modality for dynamic help text
    current_modality = get_input_modality(st.session_state, default="Text")
    
    # TTS settings
    auto_speak = st.checkbox(
        "Auto-speak responses",
        value=False,
        help=get_auto_speak_help_text(current_modality)
    )
    
    with st.expander("⚙️ TTS Settings (Advanced)"):
        tts_rate = st.slider(
            "Speech rate (words/min)",
            min_value=100,
            max_value=300,
            value=200,
            step=10,
            help="How fast the text is spoken"
        )
    
    st.session_state["auto_speak"] = auto_speak
    st.session_state["tts_rate"] = tts_rate
    # Voice is always None (use Edge TTS default voice)
    st.session_state["tts_voice"] = None
    
    st.divider()
    st.subheader("🎨 Visual Accessibility")
    st.caption("Customize appearance")
    
    # Theme selector
    # Get current theme from session state or default to Dark
    current_theme_val = st.session_state.get("theme", "Dark")
    current_theme_idx = 1  # Default to Dark (index 1)
    if current_theme_val == "Light":
        current_theme_idx = 0
    elif current_theme_val == "High Contrast":
        current_theme_idx = 2
    
    theme = st.radio(
        "Theme",
        ["Light", "Dark", "High Contrast"],
        index=current_theme_idx,
        key="theme_selector",
        help="Choose a theme that works best for you"
    )
    # Update session state immediately
    if theme != st.session_state.get("theme"):
        st.session_state["theme"] = theme
        st.rerun()  # Force rerun to apply new theme immediately
    else:
        st.session_state["theme"] = theme
    
    # Font size slider
    current_font_size = st.session_state.get("font_size", 16)
    font_size = st.slider(
        "Font Size",
        min_value=12,
        max_value=24,
        value=current_font_size,
        key="font_size_slider",
        step=1,
        help="Adjust text size for better readability (12-24px)"
    )
    # Update session state and rerun if changed
    if font_size != st.session_state.get("font_size"):
        st.session_state["font_size"] = font_size
        st.rerun()  # Force rerun to apply new font size immediately
    else:
        st.session_state["font_size"] = font_size

# Apply theme CSS and font size
def apply_theme_css(theme: str, font_size: int = 16):
    """Inject CSS based on selected theme and font size."""
    css = ""
    
    # Font size CSS (applies to all themes)
    font_css = f"""
    <style>
    /* Font Size Adjustment */
    .stApp,
    .main .block-container,
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown *,
    .stChatMessage,
    .stChatMessage *,
    .stSidebar,
    .stSidebar *,
    p, span, label, li, td, th,
    .stTextInput input,
    .stSelectbox select,
    .stSlider label,
    .stCheckbox label,
    .stRadio label,
    .stRadio p,
    .stButton button {{
        font-size: {font_size}px !important;
    }}
    /* Headings scale proportionally */
    h1 {{ font-size: {font_size * 2}px !important; }}
    h2 {{ font-size: {font_size * 1.75}px !important; }}
    h3 {{ font-size: {font_size * 1.5}px !important; }}
    h4 {{ font-size: {font_size * 1.25}px !important; }}
    h5 {{ font-size: {font_size * 1.1}px !important; }}
    h6 {{ font-size: {font_size}px !important; }}
    </style>
    """
    
    # Now build theme-specific CSS
    
    if theme == "Light":
        # Explicitly force light theme (override browser/system dark mode)
        css = """
        <style>
        /* Light Mode - Force light theme even if browser is in dark mode */
        .stApp {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        .main .block-container {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        /* All text elements - ensure dark text on light background */
        h1, h2, h3, h4, h5, h6 {
            color: #262730 !important;
        }
        p, span, label, li, td, th {
            color: #262730 !important;
        }
        .stMarkdown {
            color: #262730 !important;
        }
        .stMarkdown p {
            color: #262730 !important;
        }
        .stMarkdown * {
            color: #262730 !important;
        }
        /* Sidebar text - comprehensive coverage with higher specificity */
        .stSidebar,
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background-color: #ffffff !important;
        }
        .stSidebar *,
        [data-testid="stSidebar"] * {
            color: #262730 !important;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #262730 !important;
        }
        .stSidebar p, .stSidebar span, .stSidebar label,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #262730 !important;
        }
        .stSidebar .stCaption,
        [data-testid="stSidebar"] .stCaption {
            color: #666666 !important;
        }
        /* Sidebar specific elements */
        .stSidebar .stSubheader,
        [data-testid="stSidebar"] .stSubheader {
            color: #262730 !important;
        }
        .stSidebar .stTitle,
        [data-testid="stSidebar"] .stTitle {
            color: #262730 !important;
        }
        .stSidebar hr,
        [data-testid="stSidebar"] hr {
            border-color: #cccccc !important;
        }
        .stSidebar .stDivider,
        [data-testid="stSidebar"] .stDivider {
            border-color: #cccccc !important;
        }
        /* Sidebar buttons */
        .stSidebar .stButton > button,
        [data-testid="stSidebar"] .stButton > button {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
        }
        .stSidebar .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #ff3333 !important;
        }
        /* Sidebar checkboxes */
        .stSidebar .stCheckbox label,
        [data-testid="stSidebar"] .stCheckbox label {
            color: #262730 !important;
        }
        /* Sidebar expanders */
        .stSidebar .streamlit-expanderHeader,
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #262730 !important;
        }
        .stSidebar .streamlit-expanderContent,
        [data-testid="stSidebar"] .streamlit-expanderContent {
            color: #262730 !important;
        }
        .stSidebar .streamlit-expanderContent *,
        [data-testid="stSidebar"] .streamlit-expanderContent * {
            color: #262730 !important;
        }
        /* Sidebar sliders */
        .stSidebar .stSlider label,
        [data-testid="stSidebar"] .stSlider label {
            color: #262730 !important;
        }
        /* Sidebar text inputs */
        .stSidebar .stTextInput label,
        [data-testid="stSidebar"] .stTextInput label {
            color: #262730 !important;
        }
        .stSidebar .stTextInput > div > div > input,
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #cccccc !important;
        }
        /* Sidebar selectboxes */
        .stSidebar .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #262730 !important;
        }
        .stSidebar .stSelectbox > div > div > select,
        [data-testid="stSidebar"] .stSelectbox > div > div > select {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        /* Chat messages */
        .stChatMessage {
            background-color: #f0f2f6 !important;
        }
        .stChatMessage * {
            color: #262730 !important;
        }
        .stChatMessage[data-testid="user"] {
            background-color: #d1ecf1 !important;
        }
        .stChatMessage[data-testid="user"] * {
            color: #262730 !important;
        }
        .stChatMessage[data-testid="assistant"] {
            background-color: #f0f2f6 !important;
        }
        .stChatMessage[data-testid="assistant"] * {
            color: #262730 !important;
        }
        /* Form elements */
        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #cccccc !important;
        }
        .stTextInput label {
            color: #262730 !important;
        }
        /* Chat input field - ULTRA comprehensive styling for bottom input */
        /* Target all possible chat input containers and nested divs */
        .stChatInput,
        [class*="stChatInput"],
        [class*="ChatInput"],
        [data-testid="stChatInputContainer"],
        [data-testid="stChatInputContainer"] > div,
        [data-testid="stChatInputContainer"] > div > div,
        [data-testid="stChatInputContainer"] > div > div > div,
        [data-testid="stChatInputContainer"] > div > div > div > div {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        .stChatInput > div,
        .stChatInput > div > div,
        .stChatInput > div > div > div,
        .stChatInput > div > div > div > div,
        .stChatInput > div > div > div > div > div {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        /* Chat input field itself - target every possible selector */
        .stChatInput input,
        .stChatInput input[type="text"],
        .stChatInput > div > div > input,
        .stChatInput > div > div > div > input,
        .stChatInput > div > div > div > div > input,
        [data-testid="stChatInputContainer"] input,
        [data-testid="stChatInputContainer"] > div > div > input,
        [data-testid="stChatInputContainer"] > div > div > div > input,
        [class*="stChatInput"] input,
        [class*="ChatInput"] input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #cccccc !important;
        }
        .stChatInput input::placeholder,
        .stChatInput > div > div > input::placeholder,
        [data-testid="stChatInputContainer"] input::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        /* Main area chat input - more specific */
        .main .stChatInput,
        .main [data-testid="stChatInputContainer"],
        .block-container .stChatInput,
        [data-testid="stAppViewContainer"] .stChatInput {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        .main .stChatInput input,
        .main [data-testid="stChatInputContainer"] input,
        .block-container .stChatInput input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #cccccc !important;
        }
        /* Override any dark theme styles on chat input - BaseWeb components */
        [data-baseweb="input"],
        [data-baseweb="input"] > div,
        [data-baseweb="input"] > div > div {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="input"] > div > input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Target any input at the bottom of the page */
        .main > div:last-child input[type="text"],
        .block-container > div:last-child input[type="text"],
        [data-testid="stAppViewContainer"] > div:last-child input[type="text"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Radio buttons - make labels very visible */
        .stRadio {
            color: #262730 !important;
        }
        .stRadio > label {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        .stRadio label {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        .stRadio label > div {
            color: #262730 !important;
        }
        .stRadio label > div > p {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        .stRadio [data-baseweb="radio"] {
            color: #262730 !important;
        }
        .stRadio [data-baseweb="radio"] label {
            color: #262730 !important;
        }
        .stRadio p {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        .stRadio span {
            color: #262730 !important;
        }
        .stSelectbox > div > div > select {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        .stSelectbox label,
        .stSelectbox > label,
        .stSelectbox [data-testid="stSelectboxLabel"],
        .stSelectbox p {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        /* Selectbox label specifically - Streamlit uses p tags for labels */
        .stSelectbox > p,
        .stSelectbox > div > p,
        label[for*="stSelectbox"] {
            color: #262730 !important;
            font-weight: 500 !important;
        }
        .stSlider label {
            color: #262730 !important;
        }
        .stSlider > div > div > div {
            background-color: #ffffff !important;
        }
        .stCheckbox label {
            color: #262730 !important;
        }
        /* Buttons */
        .stButton > button {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
        }
        .stButton > button:hover {
            background-color: #ff3333 !important;
        }
        /* Code blocks */
        code {
            background-color: #f0f2f6 !important;
            color: #e83e8c !important;
        }
        pre {
            background-color: #f0f2f6 !important;
            color: #262730 !important;
        }
        pre code {
            color: #262730 !important;
        }
        /* Links */
        a {
            color: #ff4b4b !important;
        }
        a:hover {
            color: #ff3333 !important;
        }
        /* Expander */
        .streamlit-expanderHeader {
            color: #262730 !important;
        }
        .streamlit-expanderContent {
            color: #262730 !important;
        }
        /* Override Streamlit's dark mode variables */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }
        [data-testid="stToolbar"] {
            background-color: #ffffff !important;
        }
        /* Force all input backgrounds to white */
        input[type="text"],
        input[type="text"]:focus,
        textarea,
        textarea:focus {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        /* Icons and symbols - ensure they're visible with appropriate colors */
        /* User message icons - keep original color (usually blue/red) */
        .stChatMessage[data-testid="user"] .stChatMessageIcon svg,
        .stChatMessage[data-testid="user"] .stChatMessageIcon * {
            color: inherit !important;
            fill: inherit !important;
        }
        /* Assistant message icons - keep original color (usually orange/yellow) */
        .stChatMessage[data-testid="assistant"] .stChatMessageIcon svg,
        .stChatMessage[data-testid="assistant"] .stChatMessageIcon * {
            color: inherit !important;
            fill: inherit !important;
        }
        /* General icons - dark but not pure black for better visibility */
        svg:not(.stChatMessageIcon svg),
        [class*="icon"]:not(.stChatMessageIcon) {
            color: #4a4a4a !important;
            fill: #4a4a4a !important;
        }
        /* Dividers and separators */
        hr,
        .stDivider,
        [class*="divider"],
        [class*="Divider"],
        .stHorizontal {
            border-color: #cccccc !important;
            background-color: #cccccc !important;
            color: #cccccc !important;
        }
        /* All buttons - ensure icons are visible */
        button svg,
        button [class*="icon"],
        .stButton svg,
        .stButton [class*="icon"] {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        /* Chat input button (send button) */
        .stChatInput button,
        .stChatInput button svg,
        [data-testid="stChatInputContainer"] button,
        [data-testid="stChatInputContainer"] button svg {
            color: #262730 !important;
            fill: #262730 !important;
        }
        /* More aggressive chat input targeting - catch ALL possible selectors */
        /* Target the chat input form and ALL nested containers */
        form,
        form[data-testid="stChatInputForm"],
        form[data-testid="stChatInputForm"] > div,
        form[data-testid="stChatInputForm"] > div > div,
        form[data-testid="stChatInputForm"] > div > div > div,
        form[data-testid="stChatInputForm"] > div > div > div > div,
        form[data-testid="stChatInputForm"] > div > div > div > div > div,
        form > div,
        form > div > div,
        form > div > div > div,
        form > div > div > div > div {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        form input,
        form[data-testid="stChatInputForm"] input,
        form[data-testid="stChatInputForm"] > div input,
        form[data-testid="stChatInputForm"] > div > div input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        input,
        input[type="text"],
        input[type="text"]:not([class*="hidden"]),
        textarea {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #cccccc !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        /* Target ALL divs that might contain the chat input */
        div[class*="ChatInput"],
        div[class*="chatInput"],
        div[class*="input"],
        div[data-baseweb="input"],
        form[class*="ChatInput"],
        form[class*="chatInput"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        div[class*="ChatInput"] input,
        div[class*="chatInput"] input,
        div[class*="input"] input,
        div[data-baseweb="input"] input,
        form[class*="ChatInput"] input,
        form[class*="chatInput"] input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Ultra-aggressive: target by position (last input in main area) */
        .main form:last-child,
        .main form:last-child > div,
        .main form:last-child > div > div,
        .main form:last-child > div > div > div,
        .main form:last-child > div > div > div > div,
        .block-container form:last-child,
        .block-container form:last-child > div,
        .block-container form:last-child > div > div,
        .block-container form:last-child > div > div > div {
            background-color: #ffffff !important;
            background: #ffffff !important;
        }
        .main form:last-child input,
        .main form:last-child textarea,
        .block-container form:last-child input,
        .block-container form:last-child textarea {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Target ALL forms in the main area - chat input is usually the last one */
        .main form,
        .main form *,
        .block-container form,
        .block-container form * {
            background-color: #ffffff !important;
        }
        .main form input,
        .main form textarea,
        .block-container form input,
        .block-container form textarea {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Even more specific: target the actual input element by attribute */
        input[placeholder*="question"],
        input[placeholder*="Question"],
        input[placeholder*="ask"],
        input[placeholder*="Ask"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Target by data attributes that Streamlit might use */
        [data-baseweb="form-control"],
        [data-baseweb="form-control"] > div,
        [data-baseweb="form-control"] input {
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #262730 !important;
        }
        /* Override Streamlit's default dark theme completely */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > div {
            background-color: #ffffff !important;
        }
        /* Override any inline styles or theme variables */
        :root {
            --background-color: #ffffff !important;
            --text-color: #262730 !important;
            --primary-color: #ff4b4b !important;
        }
        /* Ensure all text elements have dark text (but not buttons which should have white text) */
        body, .stApp, .main, .block-container,
        p, span, div:not(.stButton):not(button), label, h1, h2, h3, h4, h5, h6 {
            color: #262730 !important;
        }
        /* But keep button text white */
        .stButton, button, .stButton *, button * {
            color: #ffffff !important;
        }
        .stButton svg, button svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }
        </style>
        """
    elif theme == "Dark":
        css = """
        <style>
        /* Dark Mode - Force dark theme with visible text */
        .stApp {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        .main .block-container {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        /* All text elements - comprehensive coverage */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        p, span, label, li, td, th {
            color: #ffffff !important;
        }
        div[class*="st"] {
            color: #ffffff !important;
        }
        .stMarkdown {
            color: #ffffff !important;
        }
        .stMarkdown p {
            color: #ffffff !important;
        }
        .stMarkdown * {
            color: #ffffff !important;
        }
        /* Sidebar text */
        .stSidebar {
            background-color: #262626 !important;
        }
        .stSidebar * {
            color: #ffffff !important;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #ffffff !important;
        }
        .stSidebar p, .stSidebar span, .stSidebar label {
            color: #ffffff !important;
        }
        .stSidebar .stCaption {
            color: #cccccc !important;
        }
        /* Chat messages */
        .stChatMessage {
            background-color: #2d2d2d !important;
        }
        .stChatMessage * {
            color: #ffffff !important;
        }
        .stChatMessage[data-testid="user"] {
            background-color: #2d4a5e !important;
        }
        .stChatMessage[data-testid="user"] * {
            color: #ffffff !important;
        }
        .stChatMessage[data-testid="assistant"] {
            background-color: #2d2d2d !important;
        }
        .stChatMessage[data-testid="assistant"] * {
            color: #ffffff !important;
        }
        /* Form elements */
        .stTextInput > div > div > input {
            background-color: #3d3d3d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        .stTextInput label {
            color: #ffffff !important;
        }
        /* Chat input field - comprehensive styling */
        .stChatInput {
            background-color: #3d3d3d !important;
        }
        .stChatInput > div {
            background-color: #3d3d3d !important;
        }
        .stChatInput > div > div {
            background-color: #3d3d3d !important;
        }
        .stChatInput > div > div > input {
            background-color: #3d3d3d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        .stChatInput > div > div > input::placeholder {
            color: #aaaaaa !important;
            opacity: 1 !important;
        }
        /* Radio buttons - make labels very visible */
        .stRadio {
            color: #ffffff !important;
        }
        .stRadio > label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        .stRadio label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        .stRadio label > div {
            color: #ffffff !important;
        }
        .stRadio label > div > p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        .stRadio [data-baseweb="radio"] {
            color: #ffffff !important;
        }
        .stRadio [data-baseweb="radio"] label {
            color: #ffffff !important;
        }
        /* Radio button text specifically */
        .stRadio p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        .stRadio span {
            color: #ffffff !important;
        }
        .stSelectbox > div > div > select {
            background-color: #3d3d3d !important;
            color: #ffffff !important;
        }
        .stSelectbox label {
            color: #ffffff !important;
        }
        .stSlider label {
            color: #ffffff !important;
        }
        .stSlider > div > div > div {
            background-color: #2d2d2d !important;
        }
        .stCheckbox label {
            color: #ffffff !important;
        }
        /* Buttons */
        .stButton > button {
            background-color: #4a9eff !important;
            color: #ffffff !important;
        }
        .stButton > button:hover {
            background-color: #3a8eef !important;
        }
        /* Code blocks */
        code {
            background-color: #3d3d3d !important;
            color: #ffd700 !important;
        }
        pre {
            background-color: #3d3d3d !important;
            color: #ffffff !important;
        }
        pre code {
            color: #ffffff !important;
        }
        /* Links */
        a {
            color: #4a9eff !important;
        }
        a:hover {
            color: #6ab8ff !important;
        }
        /* Override Streamlit's theme variables */
        [data-testid="stAppViewContainer"] {
            background-color: #1e1e1e !important;
        }
        [data-testid="stHeader"] {
            background-color: #1e1e1e !important;
        }
        [data-testid="stToolbar"] {
            background-color: #1e1e1e !important;
        }
        /* Expander */
        .streamlit-expanderHeader {
            color: #ffffff !important;
        }
        .streamlit-expanderContent {
            color: #ffffff !important;
        }
        /* Toast notifications */
        .stToast {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        </style>
        """
    elif theme == "High Contrast":
        css = """
        <style>
        /* High Contrast Mode - Maximum contrast for accessibility */
        .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .main .block-container {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: bold !important;
        }
        .stMarkdown {
            color: #ffffff !important;
        }
        .stMarkdown p {
            color: #ffffff !important;
        }
        .stChatMessage {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #ffffff !important;
        }
        .stChatMessage[data-testid="user"] {
            background-color: #000000 !important;
            border: 2px solid #00ffff !important;
        }
        .stChatMessage[data-testid="assistant"] {
            background-color: #000000 !important;
            border: 2px solid #ffff00 !important;
        }
        .stSidebar {
            background-color: #000000 !important;
            border-right: 2px solid #ffffff !important;
            color: #ffffff !important;
        }
        .stTextInput > div > div > input {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #ffffff !important;
        }
        .stSelectbox > div > div > select {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #ffffff !important;
        }
        .stSlider > div > div > div {
            background-color: #000000 !important;
        }
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #ffffff !important;
            font-weight: bold !important;
        }
        .stButton > button:hover {
            background-color: #ffff00 !important;
            color: #000000 !important;
        }
        code {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ffffff !important;
            font-weight: bold !important;
        }
        pre {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #ffffff !important;
        }
        a {
            color: #00ffff !important;
            text-decoration: underline !important;
        }
        a:hover {
            color: #ffff00 !important;
        }
        /* Override Streamlit's theme variables */
        [data-testid="stAppViewContainer"] {
            background-color: #000000 !important;
        }
        </style>
        """
    
    # Combine theme CSS with font size CSS
    combined_css = font_css + css if css else font_css
    
    if combined_css:
        # Extract CSS content (remove style tags if present)
        css_content = combined_css.replace("<style>", "").replace("</style>", "").strip()
        # Inject CSS using BOTH methods for maximum reliability
        # Method 1: Direct st.markdown (most reliable for Streamlit)
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        
        # Method 2: JavaScript injection as backup to ensure it's in head and updates
        style_id = f"rag-chatbot-theme-{theme}-{font_size}"
        js_injection = f"""
        <script>
        (function() {{
            // Remove ALL existing theme styles (in case of multiple)
            const existingStyles = document.querySelectorAll('[id^="rag-chatbot-theme"]');
            existingStyles.forEach(style => style.remove());
            
            // Create new style element with unique ID as backup
            const style = document.createElement('style');
            style.id = '{style_id}';
            style.textContent = {json.dumps(css_content)};
            document.head.appendChild(style);
        }})();
        </script>
        """
        st.markdown(js_injection, unsafe_allow_html=True)
        # Also add aggressive JavaScript to force styles
        if theme == "Light":
            js = """
            <script>
            (function() {
                function forceLightMode() {
                    // Force ALL sidebar text to be dark - no exceptions
                    const sidebar = document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        // Force ALL text elements in sidebar
                        const allText = sidebar.querySelectorAll('p, label, span, div, h1, h2, h3, h4, h5, h6');
                        allText.forEach(el => {
                            const style = window.getComputedStyle(el);
                            const color = style.color;
                            const bgColor = style.backgroundColor;
                            // Force dark text
                            if (color && (color.includes('255') || color.includes('250') || 
                                color === 'rgb(255, 255, 255)' || color === 'rgb(250, 250, 250)')) {
                                el.style.setProperty('color', '#262730', 'important');
                            }
                            // Also check computed color value
                            const rgb = style.color.match(/\\d+/g);
                            if (rgb && rgb.length >= 3 && parseInt(rgb[0]) > 200) {
                                el.style.setProperty('color', '#262730', 'important');
                            }
                        });
                        
                        // Specifically target selectbox labels (they're usually p tags before the select)
                        const selectboxes = sidebar.querySelectorAll('.stSelectbox');
                        selectboxes.forEach(box => {
                            const pTags = box.querySelectorAll('p');
                            pTags.forEach(p => {
                                p.style.setProperty('color', '#262730', 'important');
                                p.style.setProperty('font-weight', '500', 'important');
                            });
                        });
                    }
                    
                    // Force chat input background - EXTREMELY aggressive
                    // Target all forms (chat input is usually in a form)
                    const allForms = document.querySelectorAll('form');
                    allForms.forEach(form => {
                        form.style.setProperty('background-color', '#ffffff', 'important');
                        form.style.setProperty('background', '#ffffff', 'important');
                        // Force all nested divs, spans, and other elements
                        const allNested = form.querySelectorAll('div, span, section, article');
                        allNested.forEach(el => {
                            el.style.setProperty('background-color', '#ffffff', 'important');
                            el.style.setProperty('background', '#ffffff', 'important');
                        });
                        // Force all inputs and textareas in form
                        const inputs = form.querySelectorAll('input, textarea');
                        inputs.forEach(input => {
                            input.style.setProperty('background-color', '#ffffff', 'important');
                            input.style.setProperty('background', '#ffffff', 'important');
                            input.style.setProperty('color', '#262730', 'important');
                            // Also force parent elements
                            let parent = input.parentElement;
                            let depth = 0;
                            while (parent && depth < 20) {
                                parent.style.setProperty('background-color', '#ffffff', 'important');
                                parent.style.setProperty('background', '#ffffff', 'important');
                                parent = parent.parentElement;
                                depth++;
                            }
                        });
                    });
                    
                    // Target inputs by placeholder text
                    const inputsByPlaceholder = document.querySelectorAll('input[placeholder*="question"], input[placeholder*="Question"], input[placeholder*="ask"], input[placeholder*="Ask"]');
                    inputsByPlaceholder.forEach(input => {
                        input.style.setProperty('background-color', '#ffffff', 'important');
                        input.style.setProperty('background', '#ffffff', 'important');
                        input.style.setProperty('color', '#262730', 'important');
                        // Force all parent elements up to body
                        let parent = input.parentElement;
                        let depth = 0;
                        while (parent && parent !== document.body && depth < 25) {
                            parent.style.setProperty('background-color', '#ffffff', 'important');
                            parent.style.setProperty('background', '#ffffff', 'important');
                            parent = parent.parentElement;
                            depth++;
                        }
                    });
                    
                    // Also target by data attributes and class names
                    const chatContainers = document.querySelectorAll(
                        '[data-testid="stChatInputContainer"], ' +
                        '[data-testid="stChatInputForm"], ' +
                        '[data-testid*="ChatInput"], ' +
                        '.stChatInput, ' +
                        '[class*="ChatInput"], ' +
                        '[class*="chatInput"], ' +
                        '[data-baseweb="input"], ' +
                        '[data-baseweb="form-control"]'
                    );
                    chatContainers.forEach(container => {
                        container.style.setProperty('background-color', '#ffffff', 'important');
                        container.style.setProperty('background', '#ffffff', 'important');
                        // Force all parent elements
                        let parent = container.parentElement;
                        let depth = 0;
                        while (parent && parent !== document.body && depth < 20) {
                            parent.style.setProperty('background-color', '#ffffff', 'important');
                            parent.style.setProperty('background', '#ffffff', 'important');
                            parent = parent.parentElement;
                            depth++;
                        }
                        // Force all nested elements
                        const allNested = container.querySelectorAll('div, span, input, textarea');
                        allNested.forEach(el => {
                            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                                el.style.setProperty('background-color', '#ffffff', 'important');
                                el.style.setProperty('background', '#ffffff', 'important');
                                el.style.setProperty('color', '#262730', 'important');
                            } else {
                                el.style.setProperty('background-color', '#ffffff', 'important');
                                el.style.setProperty('background', '#ffffff', 'important');
                            }
                        });
                    });
                }
                
                // Run immediately
                forceLightMode();
                
                // Run multiple times with delays
                [100, 300, 500, 1000, 2000].forEach(delay => {
                    setTimeout(forceLightMode, delay);
                });
                
                // Continuous monitoring
                const observer = new MutationObserver(function(mutations) {
                    forceLightMode();
                });
                observer.observe(document.body, { 
                    childList: true, 
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['style', 'class']
                });
                
                // Also run on interval as backup
                setInterval(forceLightMode, 2000);
            })();
            </script>
            """
            st.markdown(js, unsafe_allow_html=True)

# Apply selected theme and font size - do this AFTER sidebar sets the theme
# Get theme from session state (set by sidebar) or default to Dark
current_theme = st.session_state.get("theme", "Dark")
current_font_size = st.session_state.get("font_size", 16)
# Always apply theme CSS to ensure it updates when changed
apply_theme_css(current_theme, current_font_size)


st.title("💬 RAG Chatbot with Follow-ups")
st.caption(" Chroma DB + session memory + follow-up rewrite")

# Vector store (cache in session)
if "vs" not in st.session_state:
    st.session_state.vs = Chroma(persist_directory=DB_DIR)

# Initialize history in session state (client-side only)
if "history" not in st.session_state or st.session_state.get("loaded_for") != session_id:
    st.session_state.history = []
    st.session_state.loaded_for = session_id

history = st.session_state.history
vs      = st.session_state.vs

# Input (check for new question first)
# Always show chat input, but prioritize transcribed text if available
transcribed_text = st.session_state.get("transcribed_text")
chat_input_value = st.chat_input("Ask a question…")

# Use transcribed text if available, otherwise use chat input
# IMPORTANT: Process transcribed_text FIRST before checking for new listening
if transcribed_text:
    # Pre-fill the input with transcribed text
    q_raw = transcribed_text
    # Clear transcribed text after using it
    st.session_state["transcribed_text"] = None
    # Stop any ongoing recording
    st.session_state["recording"] = False
    # Transition to processing state for continuous listening
    if is_continuous_listening_active(st.session_state):
        transition_to_processing(st.session_state)
elif chat_input_value:
    q_raw = chat_input_value
else:
    q_raw = None

# Continuous listening: Auto-start recording if in Speech mode and listening state
# ONLY check this AFTER processing any transcribed_text (to avoid restarting listening immediately)
current_modality = get_input_modality(st.session_state, default="Text")
current_state = get_listening_state(st.session_state)

# Only start listening if:
# 1. We're in Speech modality
# 2. Continuous listening is active
# 3. We're in "listening" state (not processing or speaking)
# 4. We don't have a question to process (q_raw is None)
# 5. We're not already recording
if (should_start_listening(st.session_state, current_modality) and 
    current_state == "listening" and 
    q_raw is None and 
    not st.session_state.get("recording", False)):
    # Automatically start recording
    st.session_state["recording"] = True
    st.rerun()

# Chat UI (show all past history - keep all previous responses visible normally)
for idx, (role, text) in enumerate(history):
    if role == "user":
        st.chat_message("user").markdown(text)
    else:
        # Assistant message with speaker button
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            st.chat_message("assistant").markdown(text)
        with col2:
            # Speaker button for this response
            button_key = f"speak_{idx}"
            if st.button("🔊", key=button_key, help="Read this response aloud"):
                try:
                    current_modality_btn = get_input_modality(st.session_state, default="Text")
                    btn_rate = st.session_state.get("tts_rate", 200)
                    print(f"🔊 Manual TTS: Speaking response with rate={btn_rate}, modality={current_modality_btn}")
                    speak_text(
                        text,
                        rate=btn_rate,
                        voice=st.session_state.get("tts_voice")
                    )
                    st.toast("✅ Finished speaking", icon="✅")
                except Exception as e:
                    print(f"❌ Manual TTS failed: {str(e)}")
                    st.toast(f"❌ Failed to speak: {str(e)}", icon="❌")

# Process new question if any
if q_raw:
    # Add user turn to history (client-side only)
    history.append(("user", q_raw))
    # Update session state
    st.session_state.history = history
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
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            st.chat_message("assistant").markdown(answer)
        with col2:
            button_key = f"speak_no_docs_{len(history)}"
            if st.button("🔊", key=button_key, help="Read this response aloud"):
                try:
                    speak_text(
                        answer,
                        rate=st.session_state.get("tts_rate", 200),
                        voice=st.session_state.get("tts_voice")
                    )
                    st.toast("✅ Finished speaking", icon="✅")
                except Exception as e:
                    st.toast(f"❌ Failed to speak: {str(e)}", icon="❌")
        history.append(("assistant", answer))
        # Persist history before any auto-speak / rerun
        st.session_state.history = history

        # Handle continuous listening for no-docs case
        current_modality = get_input_modality(st.session_state, default="Text")
        auto_speak_checkbox = st.session_state.get("auto_speak", False)
        
        if should_auto_speak(current_modality, auto_speak_checkbox):
            if not answer or len(answer.strip()) == 0:
                print("⚠️ Auto-speak (no-docs): No text to speak")
            else:
                print(f"🔊 Auto-speak (no-docs): Speaking response")
                if is_continuous_listening_active(st.session_state):
                    transition_to_speaking(st.session_state)
                
                try:
                    speak_text(
                        answer,
                        rate=st.session_state.get("tts_rate", 200),
                        voice=st.session_state.get("tts_voice")
                    )
                    print("✅ Auto-speak (no-docs): Successfully completed")
                    if is_continuous_listening_active(st.session_state):
                        transition_back_to_listening(st.session_state)
                        # History already updated above; just rerun
                        st.rerun()
                except Exception as e:
                    print(f"❌ Auto-speak (no-docs) failed: {str(e)}")
                    if is_continuous_listening_active(st.session_state):
                        transition_back_to_listening(st.session_state)
                        # History already updated above; just rerun
                        st.rerun()
        
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
    
    # Determine which answer to display and speak
    if backend == "both":
        # Show both responses together
        display_text = f"**GPT** ({lat_gpt:.0f} ms):\n\n{answer_gpt}\n\n---\n\n**Claude** ({lat_claude:.0f} ms):\n\n{answer_claude}"
        # For auto-speak, speak the primary answer
        text_to_speak = answer_gpt if primary == "gpt" else answer_claude
    elif backend == "gpt":
        display_text = f"**GPT** ({lat_gpt:.0f} ms):\n\n{answer_gpt}"
        text_to_speak = answer_gpt
    elif backend == "claude":
        display_text = f"**Claude** ({lat_claude:.0f} ms):\n\n{answer_claude}"
        text_to_speak = answer_claude
    
    # Display response with speaker button
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.chat_message("assistant").markdown(display_text)
    with col2:
        # Speaker button for new response
        button_key = f"speak_new_{len(history)}"
        if st.button("🔊", key=button_key, help="Read this response aloud"):
            try:
                speak_text(
                    text_to_speak,
                    rate=st.session_state.get("tts_rate", 200),
                    voice=st.session_state.get("tts_voice")
                )
                st.toast("✅ Finished speaking", icon="✅")
            except Exception as e:
                st.toast(f"❌ Failed to speak: {str(e)}", icon="❌")

    # Store answer in session state history (client-side only)
    # IMPORTANT: Do this before any auto-speak / rerun so the response is not lost.
    if backend == "gpt":
        history.append(("assistant", answer_gpt or ""))
    elif backend == "claude":
        history.append(("assistant", answer_claude or ""))
    else:
        chosen = (answer_gpt if primary == "gpt" else answer_claude) or ""
        tag = "GPT" if primary == "gpt" else "Claude"
        history.append(("assistant", f"[{tag}] {chosen}"))
    # Update session state with the new history immediately
    st.session_state.history = history

    # Auto-speak if enabled (check modality and checkbox)
    current_modality = get_input_modality(st.session_state, default="Text")
    auto_speak_checkbox = st.session_state.get("auto_speak", False)
    
    should_speak = should_auto_speak(current_modality, auto_speak_checkbox)
    
    if should_speak:
        # Verify we have text to speak
        if not text_to_speak or len(text_to_speak.strip()) == 0:
            print("⚠️ Auto-speak: No text to speak (text_to_speak is empty)")
        else:
            print(f"🔊 Auto-speak: Speaking response (modality: {current_modality}, checkbox: {auto_speak_checkbox})")
            # Transition to speaking state for continuous listening
            if is_continuous_listening_active(st.session_state):
                transition_to_speaking(st.session_state)
            
            try:
                speak_text(
                    text_to_speak,
                    rate=st.session_state.get("tts_rate", 200),
                    voice=st.session_state.get("tts_voice")
                )
                print("✅ Auto-speak: Successfully completed")
                
                # After speaking completes, transition back to listening in continuous mode
                if is_continuous_listening_active(st.session_state):
                    transition_back_to_listening(st.session_state)
                    # Ensure history is saved before rerun
                    st.session_state.history = history
                    # Trigger rerun to start next listening cycle
                    # Use a small delay to ensure UI updates are complete
                    import time
                    time.sleep(0.1)  # Small delay to ensure history is rendered
                    st.rerun()
            except Exception as e:
                # Log error for debugging (but don't show to user to avoid spam)
                print(f"❌ Auto-speak failed: {str(e)}")
                # But still transition back to listening in continuous mode
                if is_continuous_listening_active(st.session_state):
                    transition_back_to_listening(st.session_state)
                    # History already updated above; just rerun
                    st.rerun()
    else:
        print(f"⏸️ Auto-speak: Disabled (modality: {current_modality}, checkbox: {auto_speak_checkbox})")

    # Expandable sources
    with st.expander("📚 Retrieved sources"):
        for d in docs:
            src  = d.metadata.get("filename") or os.path.basename(d.metadata.get("source",""))
            page = d.metadata.get("page","?")
            st.markdown(f"- **{src}**, p.{page}")
