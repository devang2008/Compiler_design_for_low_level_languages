"""
main.py — Small-C Compiler Entry Point
========================================
Wires the full compilation pipeline and provides the CLI interface.

Usage:
    python main.py program.c                        # compile to program.asm
    python main.py program.c --ast                   # also dump AST
    python main.py program.c --ir                    # also dump TAC
    python main.py program.c --symbols               # also dump symbol table
    python main.py program.c --ast --ir --symbols    # all debug info

Pipeline:
    Source → Lexer → Parser → Semantic → IR → RegAlloc → CodeGen → .asm
"""

import sys
import os
import argparse
from dataclasses import fields, asdict

from errors import LexerError, ParseError, SemanticError, CodeGenError
from lexer import tokenize
from parser import parse
from semantic import analyze
from ir_generator import generate_ir
from register_allocator import allocate_registers
from code_generator import generate_code


# ──────────────────────────────────────────────────────────────────
# Debug printers
# ──────────────────────────────────────────────────────────────────

def dump_ast(node, indent: int = 0) -> None:
    """Pretty-print an AST node tree recursively."""
    prefix = "  " * indent
    name = type(node).__name__

    if hasattr(node, "__dataclass_fields__"):
        print(f"{prefix}{name}(")
        for f in fields(node):
            value = getattr(node, f.name)
            if isinstance(value, list):
                print(f"{prefix}  {f.name}=[")
                for item in value:
                    if isinstance(item, tuple):
                        # elif clause tuple: (condition, block)
                        print(f"{prefix}    (")
                        dump_ast(item[0], indent + 3)
                        dump_ast(item[1], indent + 3)
                        print(f"{prefix}    )")
                    elif hasattr(item, "__dataclass_fields__"):
                        dump_ast(item, indent + 2)
                    else:
                        print(f"{prefix}    {item!r}")
                print(f"{prefix}  ]")
            elif hasattr(value, "__dataclass_fields__"):
                print(f"{prefix}  {f.name}=")
                dump_ast(value, indent + 2)
            else:
                print(f"{prefix}  {f.name}={value!r}")
        print(f"{prefix})")
    else:
        print(f"{prefix}{node!r}")


def dump_ir(ir_output: dict) -> None:
    """Print TAC instructions grouped by function."""
    for func_name, instrs in ir_output["functions"].items():
        print(f"\n--- TAC for {func_name} ---")
        for i, instr in enumerate(instrs):
            print(f"  {i:3d}: {instr}")

    if ir_output["strings"]:
        print("\n--- String Literals ---")
        for label, value in ir_output["strings"].items():
            print(f"  {label}: {value!r}")


def dump_symbols(analyzer) -> None:
    """Print the symbol table state."""
    print("\n--- Symbol Table ---")
    print(analyzer.symbol_table.dump())
    print("\n--- Frame Sizes ---")
    for name, size in analyzer.frame_sizes.items():
        print(f"  {name}: {size} bytes")


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

def compile_file(source_path: str, dump_ast_flag: bool,
                 dump_ir_flag: bool, dump_symbols_flag: bool) -> None:
    """Run the full compilation pipeline on a single source file."""
    filename = os.path.basename(source_path)

    # 1. Read source
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    # 2. Lexer
    tokens = tokenize(source, filename)

    # 3. Parser
    ast = parse(tokens, filename)

    if dump_ast_flag:
        print("\n========== AST ==========")
        dump_ast(ast)

    # 4. Semantic Analysis
    analyzer, frame_sizes = analyze(ast, filename)

    if dump_symbols_flag:
        dump_symbols(analyzer)

    # 5. IR Generation
    ir_output = generate_ir(ast)

    if dump_ir_flag:
        dump_ir(ir_output)

    # 6. Register Allocation (per function)
    reg_maps = {}
    for func_name, tac_list in ir_output["functions"].items():
        fs = frame_sizes.get(func_name, 0)
        reg_maps[func_name] = allocate_registers(func_name, tac_list, fs)

    # 7. Code Generation
    asm_text = generate_code(ir_output, reg_maps, reg_maps, frame_sizes, analyzer.var_info)

    # 8. Write output file
    out_path = source_path.rsplit(".", 1)[0] + ".asm"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(asm_text)

    print(f"Compiled successfully: {out_path}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Small-C to MIPS Assembly Compiler"
    )
    ap.add_argument("source", help="Path to .c source file")
    ap.add_argument("--ast", action="store_true",
                    help="Dump the AST after parsing")
    ap.add_argument("--ir", action="store_true",
                    help="Dump the TAC after IR generation")
    ap.add_argument("--symbols", action="store_true",
                    help="Dump the symbol table after semantic analysis")

    args = ap.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: File '{args.source}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        compile_file(args.source, args.ast, args.ir, args.symbols)
    except (LexerError, ParseError, SemanticError, CodeGenError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
