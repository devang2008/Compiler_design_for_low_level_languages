"""
test_parser.py — Unit tests for the Small-C parser.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from lexer import tokenize
from parser import parse
from errors import ParseError
from ast_nodes import (
    Program, FunctionDecl, Block, VarDecl, ArrayDecl,
    IfStmt, WhileStmt, ForStmt, ReturnStmt,
    BinaryExpr, CallExpr, ArrayAccessExpr, IdentifierExpr,
    IntLiteral, AssignExpr, ExprStmt,
)


def quick_parse(src: str) -> Program:
    tokens = tokenize(src, "test.c")
    return parse(tokens, "test.c")


class TestParserFunctions(unittest.TestCase):

    def test_simple_function(self):
        ast = quick_parse("int main() { return 0; }")
        self.assertEqual(len(ast.functions), 1)
        self.assertEqual(ast.functions[0].name, "main")
        self.assertEqual(ast.functions[0].return_type, "int")

    def test_function_with_params(self):
        ast = quick_parse("int add(int a, int b) { return a; }")
        self.assertEqual(len(ast.functions[0].params), 2)
        self.assertEqual(ast.functions[0].params[0].name, "a")


class TestParserControlFlow(unittest.TestCase):

    def test_if_else_if_else(self):
        src = """
        int main() {
            if (x == 1) { return 1; }
            else if (x == 2) { return 2; }
            else { return 3; }
        }
        """
        ast = quick_parse(src)
        body = ast.functions[0].body.statements
        self.assertIsInstance(body[0], IfStmt)
        self.assertEqual(len(body[0].elif_clauses), 1)
        self.assertIsNotNone(body[0].else_block)

    def test_while_loop(self):
        src = "int main() { while (x < 10) { x = x + 1; } return 0; }"
        ast = quick_parse(src)
        self.assertIsInstance(ast.functions[0].body.statements[0], WhileStmt)

    def test_for_loop(self):
        src = "int main() { for (i = 0; i < 5; i++) { x = 1; } return 0; }"
        ast = quick_parse(src)
        self.assertIsInstance(ast.functions[0].body.statements[0], ForStmt)


class TestParserDeclarations(unittest.TestCase):

    def test_var_decl(self):
        ast = quick_parse("int main() { int x = 5; return 0; }")
        stmt = ast.functions[0].body.statements[0]
        self.assertIsInstance(stmt, VarDecl)
        self.assertEqual(stmt.name, "x")

    def test_array_decl(self):
        ast = quick_parse("int main() { int arr[10]; return 0; }")
        stmt = ast.functions[0].body.statements[0]
        self.assertIsInstance(stmt, ArrayDecl)
        self.assertEqual(stmt.size, 10)


class TestParserExpressions(unittest.TestCase):

    def test_precedence_mul_before_add(self):
        """2 + 3 * 4 should parse as 2 + (3 * 4)."""
        src = "int main() { x = 2 + 3 * 4; return 0; }"
        ast = quick_parse(src)
        stmt = ast.functions[0].body.statements[0]
        # stmt is ExprStmt containing AssignExpr
        assign = stmt.expr
        self.assertIsInstance(assign, AssignExpr)
        rhs = assign.value
        self.assertIsInstance(rhs, BinaryExpr)
        self.assertEqual(rhs.op, "+")
        self.assertIsInstance(rhs.right, BinaryExpr)
        self.assertEqual(rhs.right.op, "*")

    def test_function_call(self):
        src = "int main() { print_int(42); return 0; }"
        ast = quick_parse(src)
        stmt = ast.functions[0].body.statements[0]
        self.assertIsInstance(stmt.expr, CallExpr)
        self.assertEqual(stmt.expr.name, "print_int")

    def test_array_access(self):
        src = "int main() { x = arr[0]; return 0; }"
        ast = quick_parse(src)
        assign = ast.functions[0].body.statements[0].expr
        self.assertIsInstance(assign.value, ArrayAccessExpr)


class TestParserErrors(unittest.TestCase):

    def test_mismatched_paren(self):
        with self.assertRaises(ParseError):
            quick_parse("int main() { x = (1 + 2; return 0; }")

    def test_missing_semicolon(self):
        with self.assertRaises(ParseError):
            quick_parse("int main() { return 0 }")


if __name__ == "__main__":
    unittest.main()
