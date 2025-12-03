"""
Continuous listening state management for Speech modality.

This module provides modular functions for managing continuous listening mode,
keeping the state machine logic separate from the main application.
"""

from typing import Literal

# Listening states
ListeningState = Literal["idle", "listening", "processing", "speaking"]


def get_listening_state(session_state) -> ListeningState:
    """
    Get current listening state from session state.
    
    Args:
        session_state: Streamlit session state object
    
    Returns:
        Current listening state
    """
    return session_state.get("listening_state", "idle")


def set_listening_state(session_state, state: ListeningState) -> None:
    """
    Set listening state in session state.
    
    Args:
        session_state: Streamlit session state object
        state: New listening state
    """
    session_state["listening_state"] = state


def is_continuous_listening_active(session_state) -> bool:
    """
    Check if continuous listening mode is active.
    
    Args:
        session_state: Streamlit session state object
    
    Returns:
        True if continuous listening is active
    """
    return session_state.get("continuous_listening_active", False)


def start_continuous_listening(session_state) -> None:
    """
    Start continuous listening mode.
    
    Args:
        session_state: Streamlit session state object
    """
    session_state["continuous_listening_active"] = True
    session_state["listening_state"] = "listening"


def stop_continuous_listening(session_state) -> None:
    """
    Stop continuous listening mode.
    
    Args:
        session_state: Streamlit session state object
    """
    session_state["continuous_listening_active"] = False
    session_state["listening_state"] = "idle"
    # Clear any ongoing recording state
    session_state["recording"] = False


def should_start_listening(session_state, modality: str) -> bool:
    """
    Determine if listening should start based on modality and state.
    
    Args:
        session_state: Streamlit session state object
        modality: Current input modality ("Speech" or "Text")
    
    Returns:
        True if listening should start
    """
    if modality != "Speech":
        return False
    
    if not is_continuous_listening_active(session_state):
        return False
    
    current_state = get_listening_state(session_state)
    return current_state == "listening"


def transition_to_processing(session_state) -> None:
    """
    Transition state to processing (after transcription).
    
    Args:
        session_state: Streamlit session state object
    """
    set_listening_state(session_state, "processing")


def transition_to_speaking(session_state) -> None:
    """
    Transition state to speaking (after answer generation).
    
    Args:
        session_state: Streamlit session state object
    """
    set_listening_state(session_state, "speaking")


def transition_back_to_listening(session_state) -> None:
    """
    Transition state back to listening (after TTS completes).
    
    Args:
        session_state: Streamlit session state object
    """
    if is_continuous_listening_active(session_state):
        set_listening_state(session_state, "listening")

