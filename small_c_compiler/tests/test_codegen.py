"""
test_codegen.py — Unit tests for the MIPS code generator.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from lexer import tokenize
from parser import parse
from semantic import analyze
from ir_generator import generate_ir
from register_allocator import allocate_registers
from code_generator import generate_code


def quick_compile(src: str) -> str:
    tokens = tokenize(src, "test.c")
    ast = parse(tokens, "test.c")
    analyzer, frame_sizes = analyze(ast, "test.c")
    ir_output = generate_ir(ast)
    reg_maps = {}
    for fname, tac_list in ir_output["functions"].items():
        fs = frame_sizes.get(fname, 0)
        reg_maps[fname] = allocate_registers(fname, tac_list, fs)
    return generate_code(ir_output, reg_maps, reg_maps, frame_sizes, analyzer.var_info)


class TestCodeGenPrologue(unittest.TestCase):

    def test_prologue_present(self):
        asm = quick_compile("int main() { return 0; }")
        self.assertIn("subu  $sp, $sp", asm)
        self.assertIn("sw    $ra", asm)
        self.assertIn("sw    $fp", asm)

    def test_epilogue_not_needed_for_main(self):
        """main uses exit syscall, not jr $ra."""
        asm = quick_compile("int main() { return 0; }")
        self.assertIn("li    $v0, 10", asm)
        self.assertIn("syscall", asm)


class TestCodeGenArithmetic(unittest.TestCase):

    def test_addition(self):
        asm = quick_compile("int main() { int x; x = 3 + 4; return 0; }")
        self.assertIn("add", asm)


class TestCodeGenSyscall(unittest.TestCase):

    def test_print_int(self):
        asm = quick_compile("int main() { print_int(42); return 0; }")
        self.assertIn("li    $v0, 1", asm)
        self.assertIn("syscall", asm)

    def test_print_string(self):
        asm = quick_compile('int main() { print_string("hi"); return 0; }')
        self.assertIn("li    $v0, 4", asm)
        self.assertIn(".asciiz", asm)


class TestCodeGenReturn(unittest.TestCase):

    def test_function_return(self):
        src = """
        int double_it(int n) { return n * 2; }
        int main() { print_int(double_it(5)); return 0; }
        """
        asm = quick_compile(src)
        self.assertIn("jr    $ra", asm)


class TestCodeGenStructure(unittest.TestCase):

    def test_data_and_text_sections(self):
        asm = quick_compile('int main() { print_string("hello"); return 0; }')
        self.assertIn(".data", asm)
        self.assertIn(".text", asm)
        self.assertIn(".globl main", asm)


if __name__ == "__main__":
    unittest.main()
