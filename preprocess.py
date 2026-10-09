"""Minimal text cleaning helpers for social-media research."""

import re
import html


def clean_text(text: str) -> str:
    """Normalize whitespace and HTML entities while retaining useful text."""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
