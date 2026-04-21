"""
builtins.py — Built-in Operations
===================================
Defines all built-in functions for the Lisp interpreter and provides
a factory function to create a fully-loaded global environment.

Includes: arithmetic, comparisons, boolean ops, list ops, type checks,
           and I/O primitives.

Pipeline position: Loaded into the global Environment at startup.
"""

import operator
from environment import Environment


def create_global_env() -> Environment:
    """
    Build and return the global environment pre-loaded with all builtins.
    """
    env = Environment()

    # ── Arithmetic ─────────────────────────────────────────
    env.set("+", lambda *args: sum(args))
    env.set("-", lambda *args: -args[0] if len(args) == 1 else args[0] - sum(args[1:]))
    env.set("*", lambda *args: _reduce_mul(args))
    env.set("/", lambda *args: args[0] / args[1] if len(args) == 2 else _reduce_div(args))
    env.set("modulo", lambda a, b: a % b)
    env.set("abs", lambda x: abs(x))
    env.set("min", lambda *args: min(args))
    env.set("max", lambda *args: max(args))

    # ── Comparisons ────────────────────────────────────────
    env.set("=",  lambda a, b: a == b)
    env.set(">",  lambda a, b: a > b)
    env.set("<",  lambda a, b: a < b)
    env.set(">=", lambda a, b: a >= b)
    env.set("<=", lambda a, b: a <= b)
    env.set("equal?", lambda a, b: a == b)

    # ── Boolean logic ──────────────────────────────────────
    env.set("and", lambda *args: all(args))
    env.set("or",  lambda *args: any(args))
    env.set("not", lambda x: not x)

    # ── List operations ────────────────────────────────────
    env.set("list",   lambda *args: list(args))
    env.set("car",    lambda lst: lst[0])
    env.set("cdr",    lambda lst: lst[1:])
    env.set("cons",   lambda a, b: [a] + (b if isinstance(b, list) else [b]))
    env.set("append", lambda *lsts: sum((list(l) for l in lsts), []))
    env.set("length", lambda lst: len(lst))
    env.set("null?",  lambda lst: lst == [] or lst is None)

    # ── Type checks ────────────────────────────────────────
    env.set("number?",  lambda x: isinstance(x, (int, float)))
    env.set("string?",  lambda x: isinstance(x, str))
    env.set("list?",    lambda x: isinstance(x, list))
    env.set("boolean?", lambda x: isinstance(x, bool))
    env.set("symbol?",  lambda x: isinstance(x, str))
    env.set("pair?",    lambda x: isinstance(x, list) and len(x) > 0)
    env.set("zero?",    lambda x: x == 0)

    # ── String operations ──────────────────────────────────
    env.set("string-length",  lambda s: len(s))
    env.set("string-append",  lambda *args: "".join(args))
    env.set("number->string", lambda x: str(x))
    env.set("string->number", lambda s: float(s) if "." in s else int(s))

    # ── I/O primitives ─────────────────────────────────────
    env.set("display", lambda x: _display(x))
    env.set("newline", lambda: print())
    env.set("print",   lambda x: print(_format_value(x)))

    # ── Misc ───────────────────────────────────────────────
    env.set("apply", lambda fn, args: fn(*args))
    env.set("map",   lambda fn, lst: list(map(fn, lst)))
    env.set("filter", lambda fn, lst: list(filter(fn, lst)))

    return env


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
def _reduce_mul(args):
    """Multiply all arguments together."""
    result = 1
    for a in args:
        result *= a
    return result


def _reduce_div(args):
    """Divide first argument by all subsequent arguments."""
    result = args[0]
    for a in args[1:]:
        result /= a
    return result


def _format_value(val) -> str:
    """Format a Lisp value for display."""
    if val is True:
        return "#t"
    if val is False:
        return "#f"
    if isinstance(val, list):
        inner = " ".join(_format_value(v) for v in val)
        return f"({inner})"
    return str(val)


def _display(val):
    """Display a value without a trailing newline (Scheme semantics)."""
    print(_format_value(val), end="")
