"""
Modality utilities for handling input modality selection and auto-speak behavior.

This module provides modular functions for managing input modality (Speech vs Text)
and determining when auto-speak should be enabled. This keeps modality logic
separate from the main application, making it easier to maintain and test.
"""


def get_input_modality(session_state, default="Text"):
    """
    Get the current input modality from session state.
    
    Args:
        session_state: Streamlit session state object
        default: Default modality if not set (default: "Text")
    
    Returns:
        str: Current modality ("Speech" or "Text")
    """
    return session_state.get("input_modality", default)


def should_auto_speak(modality, auto_speak_checkbox):
    """
    Determine if auto-speak should be enabled based on modality and checkbox.
    
    When modality is "Speech", auto-speak is automatically enabled.
    When modality is "Text", auto-speak respects the checkbox setting.
    
    Args:
        modality: Current input modality ("Speech" or "Text")
        auto_speak_checkbox: Boolean value from auto-speak checkbox
    
    Returns:
        bool: True if auto-speak should be enabled, False otherwise
    """
    if modality == "Speech":
        return True
    else:
        return auto_speak_checkbox


def get_auto_speak_help_text(modality):
    """
    Get help text for the auto-speak checkbox based on current modality.
    
    Args:
        modality: Current input modality ("Speech" or "Text")
    
    Returns:
        str: Help text for the auto-speak checkbox
    """
    if modality == "Speech":
        return "Auto-enabled for Speech mode. Responses will be read aloud automatically."
    else:
        return "Automatically read responses when they are generated"

