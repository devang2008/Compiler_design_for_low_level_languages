"""
test_lexer.py — Unit Tests for the Lisp Lexer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize


def test_simple_expression():
    tokens = tokenize("(+ 1 2)")
    assert tokens == ["(", "+", 1, 2, ")"], f"Got: {tokens}"
    print("[PASS] test_simple_expression")


def test_nested_expression():
    tokens = tokenize("(+ 1 (* 2 3))")
    assert tokens == ["(", "+", 1, "(", "*", 2, 3, ")", ")"], f"Got: {tokens}"
    print("[PASS] test_nested_expression")


def test_booleans():
    tokens = tokenize("#t #f")
    assert tokens == [True, False], f"Got: {tokens}"
    print("[PASS] test_booleans")


def test_string_literal():
    tokens = tokenize('"hello world"')
    assert tokens == ["hello world"], f"Got: {tokens}"
    print("[PASS] test_string_literal")


def test_float_number():
    tokens = tokenize("3.14")
    assert tokens == [3.14], f"Got: {tokens}"
    print("[PASS] test_float_number")


def test_negative_number():
    tokens = tokenize("(- -5 3)")
    assert tokens == ["(", "-", -5, 3, ")"], f"Got: {tokens}"
    print("[PASS] test_negative_number")


def test_comments_stripped():
    tokens = tokenize("; this is a comment\n(+ 1 2)")
    assert tokens == ["(", "+", 1, 2, ")"], f"Got: {tokens}"
    print("[PASS] test_comments_stripped")


def test_define():
    tokens = tokenize("(define x 42)")
    assert tokens == ["(", "define", "x", 42, ")"], f"Got: {tokens}"
    print("[PASS] test_define")


def test_lambda_tokens():
    tokens = tokenize("(lambda (x) (* x x))")
    expected = ["(", "lambda", "(", "x", ")", "(", "*", "x", "x", ")", ")"]
    assert tokens == expected, f"Got: {tokens}"
    print("[PASS] test_lambda_tokens")


def test_empty_input():
    tokens = tokenize("")
    assert tokens == [], f"Got: {tokens}"
    print("[PASS] test_empty_input")


def test_whitespace_only():
    tokens = tokenize("   \n\t  ")
    assert tokens == [], f"Got: {tokens}"
    print("[PASS] test_whitespace_only")


def test_string_with_escapes():
    tokens = tokenize(r'"hello \"world\""')
    # After unescaping: hello "world"
    assert tokens == ['hello "world"'], f"Got: {tokens}"
    print("[PASS] test_string_with_escapes")


if __name__ == "__main__":
    test_simple_expression()
    test_nested_expression()
    test_booleans()
    test_string_literal()
    test_float_number()
    test_negative_number()
    test_comments_stripped()
    test_define()
    test_lambda_tokens()
    test_empty_input()
    test_whitespace_only()
    test_string_with_escapes()
    print("\n--- All lexer tests passed! ---")
