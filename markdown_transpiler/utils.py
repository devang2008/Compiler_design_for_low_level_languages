"""
utils.py — Shared Helper Functions
====================================
Utility functions used across the markdown transpiler pipeline.
"""


def escape_html(text: str) -> str:
    """
    Escape characters that have special meaning in HTML.

    This prevents raw user content from being interpreted as HTML tags
    or entities, which is critical for fenced code blocks and inline code.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text
