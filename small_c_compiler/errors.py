"""
errors.py — Custom Exception Classes for the Small-C Compiler
==============================================================
Every compiler error flows through one of these four exception classes.
Each carries the source filename and line number so the user always
knows exactly where the problem is in their code.
"""


class LexerError(Exception):
    """Raised when the lexer encounters an unexpected character or
    malformed token (e.g. unterminated string, invalid escape)."""

    def __init__(self, message: str, filename: str, line: int):
        self.filename = filename
        self.line = line
        self.raw_message = message
        full = f"[{filename}:{line}] LexerError: {message}"
        super().__init__(full)


class ParseError(Exception):
    """Raised when the parser encounters a token that doesn't match
    the expected grammar rule (e.g. missing semicolon, unmatched brace)."""

    def __init__(self, message: str, filename: str, line: int):
        self.filename = filename
        self.line = line
        self.raw_message = message
        full = f"[{filename}:{line}] ParseError: {message}"
        super().__init__(full)


class SemanticError(Exception):
    """Raised when the semantic analyzer detects a logic error
    (e.g. undeclared variable, type mismatch, missing main)."""

    def __init__(self, message: str, filename: str, line: int):
        self.filename = filename
        self.line = line
        self.raw_message = message
        full = f"[{filename}:{line}] SemanticError: {message}"
        super().__init__(full)


class CodeGenError(Exception):
    """Raised when code generation hits an impossible state
    (e.g. unsupported TAC instruction, register overflow)."""

    def __init__(self, message: str):
        self.raw_message = message
        full = f"CodeGenError: {message}"
        super().__init__(full)
