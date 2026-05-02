"""
test_lexer.py — Unit tests for the Small-C lexer.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from lexer import tokenize, Token
from errors import LexerError


class TestLexerKeywords(unittest.TestCase):
    """Keywords should be tokenised as their own type, not IDENTIFIER."""

    def test_all_keywords(self):
        src = "int char void if else while for return break"
        tokens = tokenize(src, "test.c")
        types = [t.type for t in tokens if t.type != "EOF"]
        self.assertEqual(types, [
            "INT", "CHAR", "VOID", "IF", "ELSE",
            "WHILE", "FOR", "RETURN", "BREAK",
        ])


class TestLexerLiterals(unittest.TestCase):
    """Integer, char, and string literals."""

    def test_integer(self):
        tokens = tokenize("42 0 99999", "t.c")
        vals = [t.value for t in tokens if t.type == "INTEGER_LITERAL"]
        self.assertEqual(vals, ["42", "0", "99999"])

    def test_char_literal(self):
        tokens = tokenize("'a' 'Z'", "t.c")
        vals = [t.value for t in tokens if t.type == "CHAR_LITERAL"]
        self.assertEqual(vals, ["a", "Z"])

    def test_char_escape(self):
        tokens = tokenize(r"'\n' '\t' '\0' '\\'", "t.c")
        vals = [t.value for t in tokens if t.type == "CHAR_LITERAL"]
        self.assertEqual(vals, ["\n", "\t", "\0", "\\"])

    def test_string_literal(self):
        tokens = tokenize('"hello world"', "t.c")
        vals = [t.value for t in tokens if t.type == "STRING_LITERAL"]
        self.assertEqual(vals, ["hello world"])

    def test_string_escape(self):
        tokens = tokenize(r'"line1\nline2"', "t.c")
        vals = [t.value for t in tokens if t.type == "STRING_LITERAL"]
        self.assertEqual(vals, ["line1\nline2"])


class TestLexerOperators(unittest.TestCase):
    """All single- and two-character operators."""

    def test_two_char_ops(self):
        src = "== != <= >= && || ++ --"
        tokens = tokenize(src, "t.c")
        types = [t.type for t in tokens if t.type != "EOF"]
        self.assertEqual(types, [
            "EQ", "NEQ", "LEQ", "GEQ", "AND", "OR", "INC", "DEC",
        ])

    def test_single_char_ops(self):
        src = "+ - * / % < > = ! ( ) { } [ ] ; ,"
        tokens = tokenize(src, "t.c")
        types = [t.type for t in tokens if t.type != "EOF"]
        self.assertEqual(types, [
            "PLUS", "MINUS", "STAR", "SLASH", "PERCENT",
            "LT", "GT", "ASSIGN", "NOT",
            "LPAREN", "RPAREN", "LBRACE", "RBRACE",
            "LBRACKET", "RBRACKET", "SEMICOLON", "COMMA",
        ])


class TestLexerComments(unittest.TestCase):
    """Comment skipping."""

    def test_single_line_comment(self):
        src = "int x; // this is a comment\nchar c;"
        tokens = tokenize(src, "t.c")
        types = [t.type for t in tokens if t.type != "EOF"]
        self.assertIn("INT", types)
        self.assertIn("CHAR", types)
        self.assertNotIn("IDENTIFIER", [t.type for t in tokens if "comment" in t.value])

    def test_block_comment(self):
        src = "int /* skip all\nof this */ x;"
        tokens = tokenize(src, "t.c")
        types = [t.type for t in tokens if t.type != "EOF"]
        self.assertEqual(types, ["INT", "IDENTIFIER", "SEMICOLON"])


class TestLexerErrors(unittest.TestCase):
    """LexerError on invalid characters."""

    def test_invalid_char(self):
        with self.assertRaises(LexerError):
            tokenize("int x = @;", "t.c")


if __name__ == "__main__":
    unittest.main()
