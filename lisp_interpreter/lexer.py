"""
lexer.py — Lisp Tokenizer
===========================
Splits a raw Lisp source string into a flat list of tokens.

Handles: parentheses, symbols, integers, floats, strings,
booleans (#t #f), and strips comments (;) and whitespace.

Pipeline position: SOURCE STRING → [LEXER] → Token List → Parser → AST → Evaluator
"""

import re


class LispString(str):
    """
    Marker subclass of str to distinguish Lisp string literals from symbols.
    Both are Python str objects; this tag lets the evaluator tell them apart.
    """
    pass


# ──────────────────────────────────────────────
# Token patterns (order matters — checked top to bottom)
# ──────────────────────────────────────────────
_TOKEN_PATTERNS = [
    ("COMMENT",   r';[^\n]*'),          # ; line comments (discarded)
    ("WHITESPACE", r'\s+'),              # whitespace (discarded)
    ("BOOLEAN",   r'#t|#f'),             # boolean literals
    ("STRING",    r'"(?:[^"\\]|\\.)*"'), # double-quoted strings
    ("NUMBER",    r'-?\d+\.\d+'),        # float literals (must precede int)
    ("INTEGER",   r'-?\d+'),             # integer literals
    ("LPAREN",    r'\('),                # opening paren
    ("RPAREN",    r'\)'),                # closing paren
    ("SYMBOL",    r"""[^\s()"';]+"""),    # symbols / identifiers
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS)
)


def tokenize(source: str) -> list:
    """
    Tokenize a Lisp source string into a flat list of token values.

    Returns a list where each element is one of:
      - '(' or ')'              (parentheses as strings)
      - int or float            (numeric literals)
      - True or False           (boolean literals)
      - str starting with '"'   (string literals, kept with quotes)
      - str                     (symbol / identifier)

    Comments and whitespace are silently discarded.
    """
    tokens = []

    for match in _MASTER_RE.finditer(source):
        kind = match.lastgroup
        value = match.group()

        # Skip comments and whitespace
        if kind in ("COMMENT", "WHITESPACE"):
            continue

        if kind == "LPAREN":
            tokens.append("(")
        elif kind == "RPAREN":
            tokens.append(")")
        elif kind == "BOOLEAN":
            tokens.append(True if value == "#t" else False)
        elif kind == "STRING":
            # Strip surrounding quotes, unescape basic sequences
            inner = value[1:-1]
            inner = inner.replace('\\"', '"')
            inner = inner.replace("\\n", "\n")
            inner = inner.replace("\\t", "\t")
            inner = inner.replace("\\\\", "\\")
            tokens.append(LispString(inner))
        elif kind == "NUMBER":
            tokens.append(float(value))
        elif kind == "INTEGER":
            tokens.append(int(value))
        elif kind == "SYMBOL":
            tokens.append(value)

    return tokens
