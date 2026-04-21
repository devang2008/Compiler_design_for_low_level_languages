"""
parser.py — Lisp Parser (Token List → AST)
============================================
Converts a flat token list into a nested Python list structure (the AST).

Example:
    "(+ 1 (* 2 3))"  →  tokens  →  ['+', 1, ['*', 2, 3]]

Pipeline position: Source → Lexer → Token List → [PARSER] → AST → Evaluator
"""


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""
    pass


def parse(tokens: list):
    """
    Parse a flat token list into one or more AST expressions.

    If the token list contains multiple top-level expressions,
    returns a list of all of them. If it contains exactly one,
    returns that single expression.
    """
    if not tokens:
        raise ParseError("Unexpected end of input: no tokens to parse")

    results = []
    pos = [0]  # Using list so nested function can mutate it

    while pos[0] < len(tokens):
        results.append(_read_expr(tokens, pos))

    if len(results) == 1:
        return results[0]
    return results


def _read_expr(tokens: list, pos: list):
    """
    Read a single expression starting at tokens[pos[0]].

    An expression is either:
      - An atom (number, string, boolean, symbol)
      - A list  ( expr ... )
    """
    if pos[0] >= len(tokens):
        raise ParseError("Unexpected end of input while reading expression")

    token = tokens[pos[0]]

    if token == "(":
        return _read_list(tokens, pos)
    elif token == ")":
        raise ParseError("Unexpected closing parenthesis ')'")
    else:
        # Atom — advance position and return the value directly
        pos[0] += 1
        return token


def _read_list(tokens: list, pos: list) -> list:
    """
    Read a parenthesised list: ( expr expr ... )

    Consumes the opening '(', reads expressions until ')',
    and consumes the closing ')'.
    """
    pos[0] += 1  # skip '('
    elements = []

    while True:
        if pos[0] >= len(tokens):
            raise ParseError("Unexpected end of input: missing closing ')'")

        if tokens[pos[0]] == ")":
            pos[0] += 1  # skip ')'
            return elements

        elements.append(_read_expr(tokens, pos))
