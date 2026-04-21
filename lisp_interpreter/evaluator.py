"""
evaluator.py — Core Recursive Evaluator
=========================================
The heart of the Lisp interpreter. Implements a recursive eval()
function that walks the AST and produces results.

Handles: literals, symbols, define, if, lambda, let, begin,
         and general function application (including closures).

Pipeline position: Source → Lexer → Tokens → Parser → AST → [EVALUATOR] → Result
"""

from environment import Environment, EnvironmentError
from lexer import LispString


class EvalError(Exception):
    """Raised when the evaluator encounters a semantic error."""
    pass


class Closure:
    """
    Represents a user-defined function (lambda).

    Stores the parameter list, body expression, and the environment
    in which the lambda was *defined* (for lexical scoping / closures).
    """

    def __init__(self, params: list, body, env: Environment):
        self.params = params   # list of parameter name strings
        self.body = body       # the body expression (AST)
        self.env = env         # the defining environment (captured scope)

    def __repr__(self):
        return f"<closure params={self.params}>"

    def __call__(self, *args):
        """
        Call this closure: extend defining env with args, then eval body.
        """
        child_env = self.env.extend(self.params, list(args))
        return lisp_eval(self.body, child_env)


def lisp_eval(expr, env: Environment):
    """
    Recursively evaluate a Lisp expression in the given environment.

    Dispatch order:
      1. Number or string literal → return as-is
      2. Boolean                  → return True/False
      3. Symbol                   → look up in environment
      4. List: (define ...)       → bind in current env
      5. List: (if ...)           → conditional branch
      6. List: (lambda ...)       → return Closure object
      7. List: (let ...)          → create new scope, eval body
      8. List: (begin ...)        → eval each expr, return last
      9. List: (quote ...)        → return unevaluated data
     10. List: (set! ...)         → mutate existing binding
     11. List: (cond ...)         → multi-branch conditional
     12. Any other list           → function application
    """

    # ── 1. Numeric / boolean literal ─────────────────────
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, (int, float)):
        return expr

    # ── 2. String literal (LispString from lexer) ─────────
    if isinstance(expr, LispString):
        return str(expr)

    # ── 3. Symbol lookup ───────────────────────────────────
    if isinstance(expr, str):
        try:
            return env.get(expr)
        except EnvironmentError as e:
            raise EvalError(str(e))

    # ── 3. List forms (special forms + application) ────────
    if not isinstance(expr, list):
        raise EvalError(f"Cannot evaluate: {expr!r}")

    if len(expr) == 0:
        raise EvalError("Cannot evaluate empty list ()")

    head = expr[0]

    # ── define ─────────────────────────────────────────────
    if head == "define":
        return _eval_define(expr, env)

    # ── if ─────────────────────────────────────────────────
    if head == "if":
        return _eval_if(expr, env)

    # ── lambda ─────────────────────────────────────────────
    if head == "lambda":
        return _eval_lambda(expr, env)

    # ── let ────────────────────────────────────────────────
    if head == "let":
        return _eval_let(expr, env)

    # ── begin ──────────────────────────────────────────────
    if head == "begin":
        return _eval_begin(expr, env)

    # ── quote ──────────────────────────────────────────────
    if head == "quote":
        if len(expr) != 2:
            raise EvalError("quote requires exactly 1 argument")
        return expr[1]

    # ── set! ───────────────────────────────────────────────
    if head == "set!":
        return _eval_set(expr, env)

    # ── cond ───────────────────────────────────────────────
    if head == "cond":
        return _eval_cond(expr, env)

    # ── Function application ───────────────────────────────
    return _eval_apply(expr, env)


# ──────────────────────────────────────────────
# Special form handlers
# ──────────────────────────────────────────────

def _eval_define(expr, env):
    """
    Handle (define name value) and (define (name params...) body).

    The shorthand (define (f x y) body) is sugar for
    (define f (lambda (x y) body)).
    """
    if len(expr) < 3:
        raise EvalError("define requires at least 2 arguments")

    target = expr[1]

    # Shorthand: (define (f x y) body)
    if isinstance(target, list):
        name = target[0]
        params = target[1:]
        body = expr[2]
        closure = Closure(params, body, env)
        env.set(name, closure)
        return None

    # Standard: (define name value)
    if not isinstance(target, str):
        raise EvalError(f"define: expected symbol, got {target!r}")

    value = lisp_eval(expr[2], env)
    env.set(target, value)
    return None


def _eval_if(expr, env):
    """
    Handle (if condition then-branch else-branch).

    The else-branch is optional; if absent and condition is false,
    returns None.
    """
    if len(expr) < 3:
        raise EvalError("if requires at least 2 arguments (condition, then)")

    condition = lisp_eval(expr[1], env)

    # In Scheme, only #f is falsy; everything else (including 0) is truthy.
    # We'll follow Python truthiness for practicality.
    if condition and condition is not False:
        return lisp_eval(expr[2], env)
    elif len(expr) > 3:
        return lisp_eval(expr[3], env)
    else:
        return None


def _eval_lambda(expr, env):
    """
    Handle (lambda (params...) body).

    Returns a Closure that captures the current environment.
    """
    if len(expr) < 3:
        raise EvalError("lambda requires params and body")

    params = expr[1]
    if not isinstance(params, list):
        raise EvalError(f"lambda: params must be a list, got {params!r}")

    # Validate all params are symbols
    for p in params:
        if not isinstance(p, str):
            raise EvalError(f"lambda: parameter must be a symbol, got {p!r}")

    body = expr[2]
    return Closure(params, body, env)


def _eval_let(expr, env):
    """
    Handle (let ((x 5) (y 3)) body).

    Creates a new child environment with the bindings, then evaluates
    the body in that environment.
    """
    if len(expr) < 3:
        raise EvalError("let requires bindings and body")

    bindings_list = expr[1]
    body = expr[2]

    if not isinstance(bindings_list, list):
        raise EvalError("let: bindings must be a list")

    # Evaluate each binding value in the OUTER env
    child_env = Environment(parent=env)
    for binding in bindings_list:
        if not isinstance(binding, list) or len(binding) != 2:
            raise EvalError(f"let: each binding must be (name value), got {binding!r}")
        name, val_expr = binding
        child_env.set(name, lisp_eval(val_expr, env))

    return lisp_eval(body, child_env)


def _eval_begin(expr, env):
    """
    Handle (begin expr1 expr2 ... exprN).

    Evaluates each expression in order, returns the value of the last one.
    """
    result = None
    for sub_expr in expr[1:]:
        result = lisp_eval(sub_expr, env)
    return result


def _eval_set(expr, env):
    """
    Handle (set! name value).

    Updates an *existing* binding (walks up scope chain).
    """
    if len(expr) != 3:
        raise EvalError("set! requires exactly 2 arguments")
    name = expr[1]
    value = lisp_eval(expr[2], env)
    # Walk up the chain to find and update
    _set_in_chain(env, name, value)
    return None


def _set_in_chain(env, name, value):
    """Walk up the environment chain to find and update a binding."""
    if name in env._store:
        env._store[name] = value
    elif env._parent is not None:
        _set_in_chain(env._parent, name, value)
    else:
        raise EvalError(f"set!: unbound variable '{name}'")


def _eval_cond(expr, env):
    """
    Handle (cond (test1 expr1) (test2 expr2) ... (else exprN)).
    """
    for clause in expr[1:]:
        if not isinstance(clause, list) or len(clause) < 2:
            raise EvalError(f"cond: invalid clause {clause!r}")
        test = clause[0]
        if test == "else":
            return lisp_eval(clause[1], env)
        if lisp_eval(test, env):
            return lisp_eval(clause[1], env)
    return None


def _eval_apply(expr, env):
    """
    General function application: (func arg1 arg2 ...)

    Evaluates all positions, then calls the function with the arguments.
    """
    evaluated = [lisp_eval(e, env) for e in expr]
    func = evaluated[0]
    args = evaluated[1:]

    if callable(func):
        try:
            return func(*args)
        except TypeError as e:
            raise EvalError(f"Error calling {expr[0]}: {e}")
    else:
        raise EvalError(f"'{expr[0]}' is not a callable function (got {type(func).__name__})")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_symbol(s: str) -> bool:
    """Check if a string looks like a Lisp symbol (not a parsed string literal)."""
    # After lexing, string literals have their quotes stripped,
    # so anything reaching eval as a str is a symbol.
    return True
