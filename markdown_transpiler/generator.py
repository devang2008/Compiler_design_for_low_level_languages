"""
generator.py — HTML Code Generator
====================================
Converts the list of parsed block objects into a complete, styled HTML page.

Pipeline position: Source → Lexer → Tokens → Parser → [GENERATOR] → Output
"""

import re
from parser import (
    BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_CODE,
    BLOCK_UL, BLOCK_OL, BLOCK_BLOCKQUOTE, BLOCK_HR,
)
from utils import escape_html

# ──────────────────────────────────────────────
# Inline element patterns (applied inside text)
# ──────────────────────────────────────────────
_IMAGE_RE      = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_LINK_RE       = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_BOLD_ITALIC_RE = re.compile(r'\*\*\*(.+?)\*\*\*')
_BOLD_RE       = re.compile(r'\*\*(.+?)\*\*')
_ITALIC_RE     = re.compile(r'\*(.+?)\*')
_CODE_RE       = re.compile(r'`([^`]+)`')


def _process_inline(text: str) -> str:
    """
    Convert inline markdown syntax to HTML within a text string.

    Order matters: bold+italic before bold before italic, and
    images before links (since image syntax is a superset of link syntax).
    """
    # Escape HTML entities first, then apply markdown → HTML conversions
    text = escape_html(text)

    # Images (must come before links)
    text = _IMAGE_RE.sub(r'<img src="\2" alt="\1" />', text)

    # Links
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)

    # Bold + Italic (triple asterisk)
    text = _BOLD_ITALIC_RE.sub(r'<strong><em>\1</em></strong>', text)

    # Bold
    text = _BOLD_RE.sub(r'<strong>\1</strong>', text)

    # Italic
    text = _ITALIC_RE.sub(r'<em>\1</em>', text)

    # Inline code
    text = _CODE_RE.sub(r'<code>\1</code>', text)

    return text


# ──────────────────────────────────────────────
# Minimal inline CSS for clean, readable output
# ──────────────────────────────────────────────
_CSS = """\
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.7;
        color: #1a1a2e;
        background: #f8f9fc;
        max-width: 820px;
        margin: 3rem auto;
        padding: 2.5rem 2rem;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 1.8rem; margin-bottom: 0.6rem;
        color: #16213e;
    }
    h1 { font-size: 2rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }
    h2 { font-size: 1.6rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem; }
    h3 { font-size: 1.3rem; }
    p  { margin-bottom: 1rem; }
    a  { color: #0f3460; text-decoration: none; border-bottom: 1px solid #0f3460; }
    a:hover { color: #e94560; border-color: #e94560; }
    code {
        background: #eef1f8; padding: 2px 6px; border-radius: 4px;
        font-family: "Fira Code", "Consolas", monospace; font-size: 0.9em;
    }
    pre {
        background: #1a1a2e; color: #e2e8f0; padding: 1.2rem;
        border-radius: 8px; overflow-x: auto; margin-bottom: 1.2rem;
        font-size: 0.88em; line-height: 1.6;
    }
    pre code { background: none; padding: 0; color: inherit; }
    blockquote {
        border-left: 4px solid #e94560; padding: 0.6rem 1rem;
        margin: 1rem 0; background: #fff5f7; border-radius: 0 6px 6px 0;
        color: #533a4a;
    }
    ul, ol { margin: 0.8rem 0 1rem 1.8rem; }
    li { margin-bottom: 0.35rem; }
    hr {
        border: none; height: 2px; background: linear-gradient(90deg, #e94560, #0f3460);
        margin: 2rem 0; border-radius: 2px;
    }
    img { max-width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; }
"""


def generate(blocks: list[dict], title: str = "Document") -> str:
    """
    Convert a list of block dicts into a complete HTML string.
    """
    body_parts: list[str] = []

    for block in blocks:
        btype = block["type"]

        # ── Heading ────────────────────────────────────────
        if btype == BLOCK_HEADING:
            lvl = block["level"]
            inner = _process_inline(block["text"])
            body_parts.append(f"<h{lvl}>{inner}</h{lvl}>")

        # ── Paragraph ──────────────────────────────────────
        elif btype == BLOCK_PARAGRAPH:
            inner = _process_inline(block["text"])
            body_parts.append(f"<p>{inner}</p>")

        # ── Fenced code block ──────────────────────────────
        elif btype == BLOCK_CODE:
            lang = block.get("lang", "")
            code = escape_html(block["code"])
            lang_attr = f' class="language-{lang}"' if lang else ""
            body_parts.append(f"<pre><code{lang_attr}>{code}</code></pre>")

        # ── Unordered list ─────────────────────────────────
        elif btype == BLOCK_UL:
            items_html = "\n".join(
                f"  <li>{_process_inline(item)}</li>"
                for item in block["items"]
            )
            body_parts.append(f"<ul>\n{items_html}\n</ul>")

        # ── Ordered list ───────────────────────────────────
        elif btype == BLOCK_OL:
            items_html = "\n".join(
                f"  <li>{_process_inline(item)}</li>"
                for item in block["items"]
            )
            body_parts.append(f"<ol>\n{items_html}\n</ol>")

        # ── Blockquote ─────────────────────────────────────
        elif btype == BLOCK_BLOCKQUOTE:
            inner = _process_inline(block["text"])
            # Support multi-line blockquotes with <br>
            inner = inner.replace("\n", "<br>\n")
            body_parts.append(f"<blockquote>{inner}</blockquote>")

        # ── Horizontal rule ────────────────────────────────
        elif btype == BLOCK_HR:
            body_parts.append("<hr />")

    body_html = "\n\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Generated from Markdown source" />
  <title>{escape_html(title)}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""
