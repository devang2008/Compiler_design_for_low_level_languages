"""
test_evaluator.py — Unit Tests for the Lisp Evaluator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize
from parser import parse
from evaluator import lisp_eval, EvalError
from stdlib import create_global_env


def _eval_str(s, env=None):
    """Helper: lex -> parse -> eval a Lisp string."""
    if env is None:
        env = create_global_env()
    tokens = tokenize(s)
    ast = parse(tokens)
    # parse() returns a single expr for one top-level form,
    # or a list of exprs for multiple. We detect multiple by
    # checking if it's a list AND every element is also a list
    # AND the token stream has multiple balanced top-level parens.
    if _is_multi_expr(tokens, ast):
        result = None
        for expr in ast:
            result = lisp_eval(expr, env)
        return result
    return lisp_eval(ast, env)


def _is_multi_expr(tokens, ast):
    """
    Check if the parsed AST represents multiple top-level expressions.
    We count balanced top-level parenthesised groups in the token stream.
    """
    if not isinstance(ast, list) or len(ast) == 0:
        return False
    # Count top-level groups by tracking paren depth
    depth = 0
    groups = 0
    for tok in tokens:
        if tok == "(":
            if depth == 0:
                groups += 1
            depth += 1
        elif tok == ")":
            depth -= 1
    return groups > 1


# ── Arithmetic ─────────────────────────────────────────────

def test_addition():
    assert _eval_str("(+ 1 2)") == 3
    print("[PASS] test_addition")


def test_subtraction():
    assert _eval_str("(- 10 3)") == 7
    print("[PASS] test_subtraction")


def test_multiplication():
    assert _eval_str("(* 4 5)") == 20
    print("[PASS] test_multiplication")


def test_division():
    assert _eval_str("(/ 10 2)") == 5.0
    print("[PASS] test_division")


def test_nested_arithmetic():
    assert _eval_str("(- 10 (* 2 3))") == 4
    print("[PASS] test_nested_arithmetic")


# ── Variables ──────────────────────────────────────────────

def test_define_variable():
    env = create_global_env()
    _eval_str("(define x 42)", env)
    assert _eval_str("x", env) == 42
    print("[PASS] test_define_variable")


def test_define_function():
    env = create_global_env()
    _eval_str("(define (square x) (* x x))", env)
    assert _eval_str("(square 5)", env) == 25
    print("[PASS] test_define_function")


# ── Conditionals ──────────────────────────────────────────

def test_if_true():
    assert _eval_str('(if (> 10 5) "big" "small")') == "big"
    print("[PASS] test_if_true")


def test_if_false():
    assert _eval_str('(if (< 10 5) "big" "small")') == "small"
    print("[PASS] test_if_false")


# ── Lambda & Closures ─────────────────────────────────────

def test_lambda_call():
    assert _eval_str("((lambda (x) (* x x)) 7)") == 49
    print("[PASS] test_lambda_call")


def test_closure():
    env = create_global_env()
    source = """
    (define (make-adder n)
        (lambda (x) (+ n x)))
    """
    _eval_str(source.strip(), env)
    _eval_str("(define add5 (make-adder 5))", env)
    result = _eval_str("(add5 10)", env)
    assert result == 15, f"Got: {result}"
    print("[PASS] test_closure")


# ── Recursion ──────────────────────────────────────────────

def test_factorial():
    env = create_global_env()
    _eval_str("""
    (define (factorial n)
        (if (= n 0) 1 (* n (factorial (- n 1)))))
    """.strip(), env)
    assert _eval_str("(factorial 5)", env) == 120
    print("[PASS] test_factorial")


def test_fibonacci():
    env = create_global_env()
    _eval_str("""
    (define (fibonacci n)
        (if (< n 2)
            n
            (+ (fibonacci (- n 1)) (fibonacci (- n 2)))))
    """.strip(), env)
    assert _eval_str("(fibonacci 10)", env) == 55
    print("[PASS] test_fibonacci")


# ── Booleans ───────────────────────────────────────────────

def test_and():
    assert _eval_str("(and #t #t)") == True
    assert _eval_str("(and #t #f)") == False
    print("[PASS] test_and")


def test_or():
    assert _eval_str("(or #f #t)") == True
    assert _eval_str("(or #f #f)") == False
    print("[PASS] test_or")


def test_not():
    assert _eval_str("(not #t)") == False
    assert _eval_str("(not #f)") == True
    print("[PASS] test_not")


# ── Lists ──────────────────────────────────────────────────

def test_list():
    result = _eval_str("(list 1 2 3)")
    assert result == [1, 2, 3], f"Got: {result}"
    print("[PASS] test_list")


def test_car():
    result = _eval_str("(car (list 1 2 3))")
    assert result == 1, f"Got: {result}"
    print("[PASS] test_car")


def test_cdr():
    result = _eval_str("(cdr (list 1 2 3))")
    assert result == [2, 3], f"Got: {result}"
    print("[PASS] test_cdr")


def test_cons():
    result = _eval_str("(cons 0 (list 1 2))")
    assert result == [0, 1, 2], f"Got: {result}"
    print("[PASS] test_cons")


# ── Let bindings ───────────────────────────────────────────

def test_let():
    result = _eval_str("(let ((x 5) (y 3)) (+ x y))")
    assert result == 8, f"Got: {result}"
    print("[PASS] test_let")


# ── Begin blocks ───────────────────────────────────────────

def test_begin():
    env = create_global_env()
    result = _eval_str("(begin (define x 10) (+ x 5))", env)
    assert result == 15, f"Got: {result}"
    print("[PASS] test_begin")


# ── Comparisons ───────────────────────────────────────────

def test_comparisons():
    assert _eval_str("(> 5 3)") == True
    assert _eval_str("(< 5 3)") == False
    assert _eval_str("(= 5 5)") == True
    assert _eval_str("(>= 5 5)") == True
    assert _eval_str("(<= 3 5)") == True
    print("[PASS] test_comparisons")


# ── Error handling ──────────────────────────────────────────

def test_unbound_variable():
    try:
        _eval_str("undefined_var")
        assert False, "Should have raised EvalError"
    except EvalError:
        pass
    print("[PASS] test_unbound_variable")


if __name__ == "__main__":
    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_nested_arithmetic()
    test_define_variable()
    test_define_function()
    test_if_true()
    test_if_false()
    test_lambda_call()
    test_closure()
    test_factorial()
    test_fibonacci()
    test_and()
    test_or()
    test_not()
    test_list()
    test_car()
    test_cdr()
    test_cons()
    test_let()
    test_begin()
    test_comparisons()
    test_unbound_variable()
    print("\n--- All evaluator tests passed! ---")
