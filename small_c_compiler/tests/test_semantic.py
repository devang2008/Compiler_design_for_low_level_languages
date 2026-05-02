"""
test_semantic.py — Unit tests for the semantic analyzer.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from lexer import tokenize
from parser import parse
from semantic import analyze
from errors import SemanticError


def quick_analyze(src: str):
    tokens = tokenize(src, "test.c")
    ast = parse(tokens, "test.c")
    return analyze(ast, "test.c")


class TestSemanticValid(unittest.TestCase):

    def test_valid_program(self):
        src = """
        int main() {
            int x;
            x = 5;
            print_int(x);
            return 0;
        }
        """
        analyzer, frame_sizes = quick_analyze(src)
        self.assertIn("main", frame_sizes)


class TestSemanticUndeclared(unittest.TestCase):

    def test_undeclared_variable(self):
        src = "int main() { x = 5; return 0; }"
        with self.assertRaises(SemanticError):
            quick_analyze(src)


class TestSemanticDuplicate(unittest.TestCase):

    def test_duplicate_variable(self):
        src = "int main() { int x; int x; return 0; }"
        with self.assertRaises(SemanticError):
            quick_analyze(src)


class TestSemanticArgCount(unittest.TestCase):

    def test_wrong_arg_count(self):
        src = """
        int add(int a, int b) { return a; }
        int main() { add(1); return 0; }
        """
        with self.assertRaises(SemanticError):
            quick_analyze(src)


class TestSemanticBreak(unittest.TestCase):

    def test_break_outside_loop(self):
        src = "int main() { break; return 0; }"
        with self.assertRaises(SemanticError):
            quick_analyze(src)

    def test_break_inside_loop(self):
        src = "int main() { while (1) { break; } return 0; }"
        # Should not raise
        quick_analyze(src)


class TestSemanticMain(unittest.TestCase):

    def test_missing_main(self):
        src = "int foo() { return 0; }"
        with self.assertRaises(SemanticError):
            quick_analyze(src)


if __name__ == "__main__":
    unittest.main()
