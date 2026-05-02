"""
lexer.py — Hand-Written Tokenizer for Small-C
===============================================
Reads a source string character-by-character and produces a flat
list of Token objects. No regex, no third-party tools.

Pipeline position: Source String → [Lexer] → Token List → Parser
"""

from __future__ import annotations
from typing import List
from errors import LexerError


# ──────────────────────────────────────────────────────────────────
# Token class
# ──────────────────────────────────────────────────────────────────

class Token:
    """A single lexical token with its type, raw value, and source line."""

    __slots__ = ("type", "value", "line")

    def __init__(self, type: str, value: str, line: int):
        self.type = type
        self.value = value
        self.line = line

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, line={self.line})"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.type == other.type and self.value == other.value


# ──────────────────────────────────────────────────────────────────
# Token type constants
# ──────────────────────────────────────────────────────────────────

# Keywords
T_INT     = "INT"
T_CHAR    = "CHAR"
T_VOID    = "VOID"
T_IF      = "IF"
T_ELSE    = "ELSE"
T_WHILE   = "WHILE"
T_FOR     = "FOR"
T_RETURN  = "RETURN"
T_BREAK   = "BREAK"

# Literals
T_INTEGER_LITERAL = "INTEGER_LITERAL"
T_CHAR_LITERAL    = "CHAR_LITERAL"
T_STRING_LITERAL  = "STRING_LITERAL"

# Identifier
T_IDENTIFIER = "IDENTIFIER"

# Operators
T_PLUS    = "PLUS"
T_MINUS   = "MINUS"
T_STAR    = "STAR"
T_SLASH   = "SLASH"
T_PERCENT = "PERCENT"
T_EQ      = "EQ"
T_NEQ     = "NEQ"
T_LT      = "LT"
T_GT      = "GT"
T_LEQ     = "LEQ"
T_GEQ     = "GEQ"
T_ASSIGN  = "ASSIGN"
T_AND     = "AND"
T_OR      = "OR"
T_NOT     = "NOT"
T_INC     = "INC"
T_DEC     = "DEC"

# Delimiters
T_LPAREN   = "LPAREN"
T_RPAREN   = "RPAREN"
T_LBRACE   = "LBRACE"
T_RBRACE   = "RBRACE"
T_LBRACKET = "LBRACKET"
T_RBRACKET = "RBRACKET"
T_SEMICOLON = "SEMICOLON"
T_COMMA    = "COMMA"

# Special
T_EOF = "EOF"

# Map of keyword strings to their token types.
# After reading an identifier we check this table to decide
# whether to emit a keyword token or a generic IDENTIFIER.
KEYWORDS = {
    "int":    T_INT,
    "char":   T_CHAR,
    "void":   T_VOID,
    "if":     T_IF,
    "else":   T_ELSE,
    "while":  T_WHILE,
    "for":    T_FOR,
    "return": T_RETURN,
    "break":  T_BREAK,
}

# Escape sequence map used by both string and char literal readers.
ESCAPE_MAP = {
    "n":  "\n",
    "t":  "\t",
    "0":  "\0",
    "\\": "\\",
    "\"": "\"",
    "'":  "'",
}


# ──────────────────────────────────────────────────────────────────
# Lexer class
# ──────────────────────────────────────────────────────────────────

class Lexer:
    """Character-by-character scanner that produces tokens.

    Usage:
        lexer = Lexer(source_text, "factorial.c")
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.pos = 0               # current index into source
        self.line = 1              # current line number (1-based)
        self.length = len(source)

    # ── character helpers ─────────────────────────────────────────

    def _current(self) -> str:
        """Return current character or empty string at EOF."""
        if self.pos < self.length:
            return self.source[self.pos]
        return ""

    def _peek_next(self) -> str:
        """Return the character after the current one, or empty string."""
        nxt = self.pos + 1
        if nxt < self.length:
            return self.source[nxt]
        return ""

    def _advance(self) -> str:
        """Consume and return the current character, tracking newlines."""
        ch = self.source[self.pos]
        if ch == "\n":
            self.line += 1
        self.pos += 1
        return ch

    def _error(self, msg: str):
        """Convenience: raise LexerError at the current position."""
        raise LexerError(msg, self.filename, self.line)

    # ── whitespace and comments ───────────────────────────────────

    def _skip_whitespace_and_comments(self):
        """Advance past spaces, tabs, newlines, // comments, and /* */ comments."""
        while self.pos < self.length:
            ch = self._current()

            # Plain whitespace
            if ch in " \t\r\n":
                self._advance()
                continue

            # Possible comment start
            if ch == "/" and self.pos + 1 < self.length:
                nxt = self._peek_next()

                # Single-line comment: skip to end of line
                if nxt == "/":
                    self._advance()  # consume first /
                    self._advance()  # consume second /
                    while self.pos < self.length and self._current() != "\n":
                        self._advance()
                    continue

                # Block comment: skip to matching */
                if nxt == "*":
                    start_line = self.line
                    self._advance()  # consume /
                    self._advance()  # consume *
                    while self.pos < self.length:
                        if self._current() == "*" and self._peek_next() == "/":
                            self._advance()  # consume *
                            self._advance()  # consume /
                            break
                        self._advance()
                    else:
                        # Reached EOF without closing */
                        raise LexerError(
                            "unterminated block comment",
                            self.filename, start_line
                        )
                    continue

            # Not whitespace or comment — stop
            break

    # ── literal readers ───────────────────────────────────────────

    def _read_string_literal(self) -> Token:
        """Read a double-quoted string, handling escape sequences."""
        start_line = self.line
        self._advance()  # consume opening "
        chars: list[str] = []

        while self.pos < self.length:
            ch = self._current()

            if ch == "\"":
                self._advance()  # consume closing "
                return Token(T_STRING_LITERAL, "".join(chars), start_line)

            if ch == "\\":
                self._advance()  # consume backslash
                esc = self._current()
                if esc in ESCAPE_MAP:
                    chars.append(ESCAPE_MAP[esc])
                    self._advance()
                else:
                    self._error(f"invalid escape sequence '\\{esc}' in string")
            elif ch == "\n":
                # Newlines inside strings are allowed — they're literal
                chars.append(ch)
                self._advance()
            else:
                chars.append(ch)
                self._advance()

        self._error("unterminated string literal")

    def _read_char_literal(self) -> Token:
        """Read a single-quoted character literal like 'a' or '\\n'."""
        start_line = self.line
        self._advance()  # consume opening '

        if self.pos >= self.length:
            self._error("unterminated character literal")

        ch = self._current()

        if ch == "\\":
            # Escape sequence
            self._advance()  # consume backslash
            if self.pos >= self.length:
                self._error("unterminated escape in character literal")
            esc = self._current()
            if esc in ESCAPE_MAP:
                value = ESCAPE_MAP[esc]
                self._advance()
            else:
                self._error(f"invalid escape sequence '\\{esc}' in char literal")
        elif ch == "'":
            self._error("empty character literal")
        else:
            value = ch
            self._advance()

        # Expect closing quote
        if self.pos >= self.length or self._current() != "'":
            self._error("unterminated character literal — expected closing '")
        self._advance()  # consume closing '

        return Token(T_CHAR_LITERAL, value, start_line)

    def _read_integer_literal(self) -> Token:
        """Read a sequence of digits as an integer literal."""
        start_line = self.line
        start_pos = self.pos
        while self.pos < self.length and self.source[self.pos].isdigit():
            self.pos += 1
        value = self.source[start_pos:self.pos]
        return Token(T_INTEGER_LITERAL, value, start_line)

    def _read_identifier_or_keyword(self) -> Token:
        """Read [a-zA-Z_][a-zA-Z0-9_]* and classify as keyword or identifier."""
        start_line = self.line
        start_pos = self.pos
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch.isalnum() or ch == "_":
                self.pos += 1
            else:
                break
        word = self.source[start_pos:self.pos]

        # Check keyword table — keywords have their own token type
        token_type = KEYWORDS.get(word, T_IDENTIFIER)
        return Token(token_type, word, start_line)

    # ── main tokenize loop ────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        """Scan the entire source and return a list of tokens ending with EOF."""
        tokens: List[Token] = []

        while True:
            self._skip_whitespace_and_comments()

            if self.pos >= self.length:
                tokens.append(Token(T_EOF, "", self.line))
                break

            ch = self._current()
            start_line = self.line

            # ── String literal ────────────────────────────────────
            if ch == "\"":
                tokens.append(self._read_string_literal())
                continue

            # ── Char literal ──────────────────────────────────────
            if ch == "'":
                tokens.append(self._read_char_literal())
                continue

            # ── Integer literal ───────────────────────────────────
            if ch.isdigit():
                tokens.append(self._read_integer_literal())
                continue

            # ── Identifier or keyword ─────────────────────────────
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier_or_keyword())
                continue

            # ── Two-character operators ───────────────────────────
            # Must check these BEFORE single-character operators
            nxt = self._peek_next()

            if ch == "=" and nxt == "=":
                self._advance(); self._advance()
                tokens.append(Token(T_EQ, "==", start_line))
                continue

            if ch == "!" and nxt == "=":
                self._advance(); self._advance()
                tokens.append(Token(T_NEQ, "!=", start_line))
                continue

            if ch == "<" and nxt == "=":
                self._advance(); self._advance()
                tokens.append(Token(T_LEQ, "<=", start_line))
                continue

            if ch == ">" and nxt == "=":
                self._advance(); self._advance()
                tokens.append(Token(T_GEQ, ">=", start_line))
                continue

            if ch == "&" and nxt == "&":
                self._advance(); self._advance()
                tokens.append(Token(T_AND, "&&", start_line))
                continue

            if ch == "|" and nxt == "|":
                self._advance(); self._advance()
                tokens.append(Token(T_OR, "||", start_line))
                continue

            if ch == "+" and nxt == "+":
                self._advance(); self._advance()
                tokens.append(Token(T_INC, "++", start_line))
                continue

            if ch == "-" and nxt == "-":
                self._advance(); self._advance()
                tokens.append(Token(T_DEC, "--", start_line))
                continue

            # ── Single-character operators and delimiters ─────────
            SINGLE_CHAR = {
                "+": T_PLUS,
                "-": T_MINUS,
                "*": T_STAR,
                "/": T_SLASH,
                "%": T_PERCENT,
                "<": T_LT,
                ">": T_GT,
                "=": T_ASSIGN,
                "!": T_NOT,
                "(": T_LPAREN,
                ")": T_RPAREN,
                "{": T_LBRACE,
                "}": T_RBRACE,
                "[": T_LBRACKET,
                "]": T_RBRACKET,
                ";": T_SEMICOLON,
                ",": T_COMMA,
            }

            if ch in SINGLE_CHAR:
                self._advance()
                tokens.append(Token(SINGLE_CHAR[ch], ch, start_line))
                continue

            # ── Unexpected character ──────────────────────────────
            self._error(f"unexpected character '{ch}'")

        return tokens


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def tokenize(source: str, filename: str = "<stdin>") -> List[Token]:
    """Tokenize a source string and return the token list.

    This is the public entry point that the rest of the compiler
    imports and calls.
    """
    return Lexer(source, filename).tokenize()
