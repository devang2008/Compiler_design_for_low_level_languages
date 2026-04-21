"""
parser.py — Markdown Block Parser
==================================
Takes the flat token stream from the lexer and groups tokens into
block-level elements (paragraphs, lists, code blocks, etc.).

Pipeline position: Source Text → Lexer → Token Stream → [PARSER] → Generator
"""

from lexer import (
    HEADING, BOLD_ITALIC, BOLD, ITALIC, INLINE_CODE,
    FENCE_OPEN, FENCE_CLOSE, CODE_LINE,
    UNORDERED_LI, ORDERED_LI, BLOCKQUOTE,
    HORIZONTAL, LINK, IMAGE, TEXT, BLANK,
)

# ──────────────────────────────────────────────
# Block type constants
# ──────────────────────────────────────────────
BLOCK_HEADING     = "BLOCK_HEADING"
BLOCK_PARAGRAPH   = "BLOCK_PARAGRAPH"
BLOCK_CODE        = "BLOCK_CODE"
BLOCK_UL          = "BLOCK_UL"
BLOCK_OL          = "BLOCK_OL"
BLOCK_BLOCKQUOTE  = "BLOCK_BLOCKQUOTE"
BLOCK_HR          = "BLOCK_HR"


def parse(tokens: list[dict]) -> list[dict]:
    """
    Group a flat token list into a list of block-level dicts.

    Each block has:
        {"type": BLOCK_*, ...payload}

    Multi-line constructs (lists, fenced code, blockquotes) are merged
    into a single block so the generator can emit one HTML element.
    """
    blocks: list[dict] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        # ── Heading ─────────────────────────────────────────
        if tok["type"] == HEADING:
            blocks.append({
                "type": BLOCK_HEADING,
                "level": tok["level"],
                "text": tok["value"],
            })
            i += 1
            continue

        # ── Horizontal rule ─────────────────────────────────
        if tok["type"] == HORIZONTAL:
            blocks.append({"type": BLOCK_HR})
            i += 1
            continue

        # ── Fenced code block ──────────────────────────────
        if tok["type"] == FENCE_OPEN:
            lang = tok["value"]
            code_lines: list[str] = []
            i += 1
            while i < n and tokens[i]["type"] != FENCE_CLOSE:
                code_lines.append(tokens[i]["value"])
                i += 1
            # Skip past the FENCE_CLOSE token
            if i < n:
                i += 1
            blocks.append({
                "type": BLOCK_CODE,
                "lang": lang,
                "code": "\n".join(code_lines),
            })
            continue

        # ── Unordered list ─────────────────────────────────
        if tok["type"] == UNORDERED_LI:
            items: list[str] = []
            while i < n and tokens[i]["type"] == UNORDERED_LI:
                items.append(tokens[i]["value"])
                i += 1
            blocks.append({"type": BLOCK_UL, "items": items})
            continue

        # ── Ordered list ───────────────────────────────────
        if tok["type"] == ORDERED_LI:
            items = []
            while i < n and tokens[i]["type"] == ORDERED_LI:
                items.append(tokens[i]["value"])
                i += 1
            blocks.append({"type": BLOCK_OL, "items": items})
            continue

        # ── Blockquote (consecutive lines merged) ──────────
        if tok["type"] == BLOCKQUOTE:
            quote_lines: list[str] = []
            while i < n and tokens[i]["type"] == BLOCKQUOTE:
                quote_lines.append(tokens[i]["value"])
                i += 1
            blocks.append({
                "type": BLOCK_BLOCKQUOTE,
                "text": "\n".join(quote_lines),
            })
            continue

        # ── Blank line (skip) ──────────────────────────────
        if tok["type"] == BLANK:
            i += 1
            continue

        # ── Paragraph (consecutive TEXT tokens) ────────────
        if tok["type"] == TEXT:
            para_lines: list[str] = []
            while i < n and tokens[i]["type"] == TEXT:
                para_lines.append(tokens[i]["value"])
                i += 1
            blocks.append({
                "type": BLOCK_PARAGRAPH,
                "text": " ".join(para_lines),
            })
            continue

        # ── Fallback: skip unknown tokens ──────────────────
        i += 1

    return blocks
