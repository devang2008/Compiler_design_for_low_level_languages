"""
main.py — Lisp Interpreter Entry Point
========================================
Supports two modes:
    REPL mode : python main.py         → interactive prompt
    File mode : python main.py prog.lisp → run a file top to bottom

Pipeline: Source String → Lexer → Token List → Parser → AST → Evaluator → Result
"""

import sys
import os
from lexer import tokenize
from parser import parse, ParseError
from evaluator import lisp_eval, EvalError, Closure
from stdlib import create_global_env


def format_result(value) -> str:
    """Format a Lisp value for REPL display."""
    if value is None:
        return ""
    if value is True:
        return "#t"
    if value is False:
        return "#f"
    if isinstance(value, Closure):
        return f"<closure params={value.params}>"
    if isinstance(value, list):
        inner = " ".join(format_result(v) for v in value)
        return f"({inner})"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _count_top_level_groups(tokens) -> int:
    """Count balanced top-level parenthesised groups in the token stream."""
    depth = 0
    groups = 0
    for tok in tokens:
        if tok == "(":
            if depth == 0:
                groups += 1
            depth += 1
        elif tok == ")":
            depth -= 1
    return groups


def run_source(source: str, env):
    """
    Run a complete source string (may contain multiple expressions).

    Returns the value of the last expression evaluated.
    """
    tokens = tokenize(source)
    if not tokens:
        return None

    ast = parse(tokens)

    # Use paren counting to detect multiple top-level expressions
    if _count_top_level_groups(tokens) > 1 and isinstance(ast, list):
        result = None
        for expr in ast:
            result = lisp_eval(expr, env)
        return result
    else:
        return lisp_eval(ast, env)


def repl():
    """
    Interactive Read-Eval-Print Loop.

    Supports multi-line input: keeps reading until parentheses balance.
    Handles errors gracefully without crashing.
    """
    env = create_global_env()
    print("Lisp Interpreter v1.0")
    print('Type (quit) or Ctrl+C to exit.\n')

    while True:
        try:
            # Read input, supporting multi-line for unbalanced parens
            source = _read_input()
            if source is None:
                break

            stripped = source.strip()
            if not stripped:
                continue
            if stripped == "(quit)":
                print("Goodbye!")
                break

            result = run_source(source, env)
            output = format_result(result)
            if output:
                print(output)

        except (ParseError, EvalError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def _read_input() -> str | None:
    """
    Read input from the user, handling multi-line expressions.

    If parentheses are unbalanced after the first line, continue
    prompting with '  ... ' until they balance.
    """
    try:
        line = input("lisp> ")
    except EOFError:
        return None

    source = line
    open_count = source.count("(") - source.count(")")

    while open_count > 0:
        try:
            continuation = input("  ... ")
        except EOFError:
            break
        source += "\n" + continuation
        open_count = source.count("(") - source.count(")")

    return source


def run_file(filepath: str):
    """
    Run a Lisp source file top-to-bottom.

    Each top-level expression is evaluated in order.
    """
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    env = create_global_env()

    tokens = tokenize(source)
    if not tokens:
        return

    ast = parse(tokens)

    # Use paren counting to detect multiple top-level expressions
    if _count_top_level_groups(tokens) > 1 and isinstance(ast, list):
        for expr in ast:
            result = lisp_eval(expr, env)
            output = format_result(result)
            if output:
                print(output)
    else:
        result = lisp_eval(ast, env)
        output = format_result(result)
        if output:
            print(output)


def main():
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        repl()


if __name__ == "__main__":
    main()
