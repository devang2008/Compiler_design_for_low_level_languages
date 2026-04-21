"""
lexer.py — Markdown Tokenizer
=============================
Reads raw markdown text and produces a flat list of tokens.
Each token is a dict with at least a 'type' key and a 'value' key.

Pipeline position: SOURCE TEXT → [LEXER] → Token Stream → Parser → Generator
"""

import re

# ──────────────────────────────────────────────
# Token type constants
# ──────────────────────────────────────────────
HEADING       = "HEADING"
BOLD_ITALIC   = "BOLD_ITALIC"
BOLD          = "BOLD"
ITALIC        = "ITALIC"
INLINE_CODE   = "INLINE_CODE"
FENCE_OPEN    = "FENCE_OPEN"
FENCE_CLOSE   = "FENCE_CLOSE"
CODE_LINE     = "CODE_LINE"
UNORDERED_LI  = "UNORDERED_LI"
ORDERED_LI    = "ORDERED_LI"
BLOCKQUOTE    = "BLOCKQUOTE"
HORIZONTAL    = "HORIZONTAL"
LINK          = "LINK"
IMAGE         = "IMAGE"
TEXT          = "TEXT"
BLANK         = "BLANK"

# ──────────────────────────────────────────────
# Regex patterns (compiled once for performance)
# ──────────────────────────────────────────────
_HEADING_RE      = re.compile(r'^(#{1,6})\s+(.*)')
_FENCE_RE        = re.compile(r'^```(\w*)\s*$')
_HR_RE           = re.compile(r'^-{3,}\s*$')
_UL_RE           = re.compile(r'^[-*+]\s+(.*)')
_OL_RE           = re.compile(r'^\d+\.\s+(.*)')
_BLOCKQUOTE_RE   = re.compile(r'^>\s?(.*)')
_BLANK_RE        = re.compile(r'^\s*$')


def tokenize(source: str) -> list[dict]:
    """
    Tokenize a markdown source string into a list of token dicts.

    Each token has the form:
        {"type": <TOKEN_TYPE>, "value": <str>, ...extra keys}

    Fenced code blocks are tracked with a state flag so that lines
    inside a fence are emitted as CODE_LINE tokens rather than being
    parsed for markdown syntax.
    """
    tokens: list[dict] = []
    lines = source.split('\n')
    in_fence = False
    fence_lang = ""

    for line in lines:

        # ── Fenced code block handling ──────────────────────
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            if not in_fence:
                # Opening fence
                fence_lang = fence_match.group(1)
                tokens.append({"type": FENCE_OPEN, "value": fence_lang})
                in_fence = True
            else:
                # Closing fence
                tokens.append({"type": FENCE_CLOSE, "value": ""})
                in_fence = False
                fence_lang = ""
            continue

        if in_fence:
            tokens.append({"type": CODE_LINE, "value": line})
            continue

        # ── Blank line ──────────────────────────────────────
        if _BLANK_RE.match(line):
            tokens.append({"type": BLANK, "value": ""})
            continue

        # ── Horizontal rule (must come before UL check) ─────
        if _HR_RE.match(line):
            tokens.append({"type": HORIZONTAL, "value": "---"})
            continue

        # ── Heading ─────────────────────────────────────────
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            tokens.append({"type": HEADING, "value": m.group(2), "level": level})
            continue

        # ── Blockquote ──────────────────────────────────────
        m = _BLOCKQUOTE_RE.match(line)
        if m:
            tokens.append({"type": BLOCKQUOTE, "value": m.group(1)})
            continue

        # ── Unordered list item ─────────────────────────────
        m = _UL_RE.match(line)
        if m:
            tokens.append({"type": UNORDERED_LI, "value": m.group(1)})
            continue

        # ── Ordered list item ───────────────────────────────
        m = _OL_RE.match(line)
        if m:
            tokens.append({"type": ORDERED_LI, "value": m.group(1)})
            continue

        # ── Plain text (paragraph line) ─────────────────────
        tokens.append({"type": TEXT, "value": line})

    return tokens
