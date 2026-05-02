"""
test_ir.py — Unit tests for the IR (TAC) generator.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from lexer import tokenize
from parser import parse
from semantic import analyze
from ir_generator import (
    generate_ir,
    TACCopy, TACBinOp, TACLoadImm, TACLabel,
    TACJumpIfNot, TACJump, TACParam, TACCall, TACReturn,
)


def quick_ir(src: str):
    tokens = tokenize(src, "test.c")
    ast = parse(tokens, "test.c")
    analyze(ast, "test.c")
    return generate_ir(ast)


class TestIRAssignment(unittest.TestCase):

    def test_simple_assignment(self):
        src = "int main() { int x; x = 5; return 0; }"
        ir = quick_ir(src)
        instrs = ir["functions"]["main"]
        # Should contain TACLoadImm(_, 5) and TACCopy(x, _)
        has_copy = any(isinstance(i, TACCopy) and i.result == "x" for i in instrs)
        self.assertTrue(has_copy)


class TestIRBinaryExpr(unittest.TestCase):

    def test_addition(self):
        src = "int main() { int x; x = 3 + 4; return 0; }"
        ir = quick_ir(src)
        instrs = ir["functions"]["main"]
        has_binop = any(isinstance(i, TACBinOp) and i.op == "+" for i in instrs)
        self.assertTrue(has_binop)


class TestIRIfStmt(unittest.TestCase):

    def test_if_generates_jump(self):
        src = "int main() { int x; x = 1; if (x == 1) { x = 2; } return 0; }"
        ir = quick_ir(src)
        instrs = ir["functions"]["main"]
        has_jumpifnot = any(isinstance(i, TACJumpIfNot) for i in instrs)
        self.assertTrue(has_jumpifnot)


class TestIRWhileStmt(unittest.TestCase):

    def test_while_generates_labels_and_jumps(self):
        src = "int main() { int x; x = 0; while (x < 5) { x = x + 1; } return 0; }"
        ir = quick_ir(src)
        instrs = ir["functions"]["main"]
        labels = [i for i in instrs if isinstance(i, TACLabel)]
        jumps = [i for i in instrs if isinstance(i, TACJump)]
        self.assertGreaterEqual(len(labels), 2)
        self.assertGreaterEqual(len(jumps), 1)


class TestIRFunctionCall(unittest.TestCase):

    def test_call_generates_param_and_call(self):
        src = """
        int add(int a, int b) { return a; }
        int main() { int r; r = add(1, 2); return 0; }
        """
        ir = quick_ir(src)
        instrs = ir["functions"]["main"]
        has_param = any(isinstance(i, TACParam) for i in instrs)
        has_call = any(isinstance(i, TACCall) and i.function == "add" for i in instrs)
        self.assertTrue(has_param)
        self.assertTrue(has_call)


if __name__ == "__main__":
    unittest.main()
