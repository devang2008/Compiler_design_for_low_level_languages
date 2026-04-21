"""
test_parser.py — Unit Tests for the Lisp Parser
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize
from parser import parse, ParseError


def _parse_str(s):
    """Helper: tokenize then parse a string."""
    return parse(tokenize(s))


def test_atom():
    assert _parse_str("42") == 42
    print("[PASS] test_atom")


def test_symbol():
    assert _parse_str("hello") == "hello"
    print("[PASS] test_symbol")


def test_simple_list():
    result = _parse_str("(+ 1 2)")
    assert result == ["+", 1, 2], f"Got: {result}"
    print("[PASS] test_simple_list")


def test_nested_list():
    result = _parse_str("(+ 1 (* 2 3))")
    assert result == ["+", 1, ["*", 2, 3]], f"Got: {result}"
    print("[PASS] test_nested_list")


def test_deeply_nested():
    result = _parse_str("(+ (* 2 (- 5 1)) (/ 10 2))")
    expected = ["+", ["*", 2, ["-", 5, 1]], ["/", 10, 2]]
    assert result == expected, f"Got: {result}"
    print("[PASS] test_deeply_nested")


def test_define_function():
    result = _parse_str("(define (square x) (* x x))")
    expected = ["define", ["square", "x"], ["*", "x", "x"]]
    assert result == expected, f"Got: {result}"
    print("[PASS] test_define_function")


def test_lambda():
    result = _parse_str("(lambda (x y) (+ x y))")
    expected = ["lambda", ["x", "y"], ["+", "x", "y"]]
    assert result == expected, f"Got: {result}"
    print("[PASS] test_lambda")


def test_if_expression():
    result = _parse_str('(if (> x 5) "big" "small")')
    expected = ["if", [">", "x", 5], "big", "small"]
    assert result == expected, f"Got: {result}"
    print("[PASS] test_if_expression")


def test_let_binding():
    result = _parse_str("(let ((x 5) (y 3)) (+ x y))")
    expected = ["let", [["x", 5], ["y", 3]], ["+", "x", "y"]]
    assert result == expected, f"Got: {result}"
    print("[PASS] test_let_binding")


def test_empty_list():
    result = _parse_str("()")
    assert result == [], f"Got: {result}"
    print("[PASS] test_empty_list")


def test_boolean_in_expr():
    result = _parse_str("(and #t #f)")
    assert result == ["and", True, False], f"Got: {result}"
    print("[PASS] test_boolean_in_expr")


def test_multiple_top_level():
    result = _parse_str("(define x 1) (define y 2)")
    assert result == [["define", "x", 1], ["define", "y", 2]], f"Got: {result}"
    print("[PASS] test_multiple_top_level")


def test_unbalanced_paren():
    try:
        _parse_str("(+ 1 2")
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    print("[PASS] test_unbalanced_paren")


if __name__ == "__main__":
    test_atom()
    test_symbol()
    test_simple_list()
    test_nested_list()
    test_deeply_nested()
    test_define_function()
    test_lambda()
    test_if_expression()
    test_let_binding()
    test_empty_list()
    test_boolean_in_expr()
    test_multiple_top_level()
    test_unbalanced_paren()
    print("\n--- All parser tests passed! ---")
