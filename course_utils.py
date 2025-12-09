"""
Utilities for discovering course folders and presenting friendly course names.
"""
import os
from typing import List

def discover_courses(data_root: str = "data") -> List[str]:
    """
    Return course IDs based on immediate subfolders of `data_root`.
    Uses folder names directly (e.g., data/networking -> "networking").
    """
    if not os.path.isdir(data_root):
        return []
    courses = []
    with os.scandir(data_root) as entries:
        for entry in entries:
            if entry.is_dir():
                courses.append(entry.name)
    return sorted(courses)


def get_course_display_name(course_id: str) -> str:
    """
    Human-friendly course name, e.g., "machine_learning" -> "Machine Learning".
    """
    if course_id.lower() == "all":
        return "All Courses"
    cleaned = course_id.replace("_", " ").replace("-", " ")
    return cleaned.title() if cleaned else course_id
