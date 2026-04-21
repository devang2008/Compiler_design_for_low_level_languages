"""
main.py — Markdown Transpiler Entry Point
==========================================
Usage:  python main.py <input.md> [output.html]

Runs the full compiler pipeline:
    Source Text → Lexer → Token Stream → Parser → Blocks → Generator → HTML
"""

import sys
import os
from lexer import tokenize
from parser import parse
from generator import generate


def main() -> None:
    # ── Validate arguments ─────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python main.py <input.md> [output.html]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "output.html"

    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    # ── Read source ────────────────────────────────────────
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    # ── Pipeline: Lex → Parse → Generate ──────────────────
    print(f"[1/3] Lexing  '{input_path}' ...")
    tokens = tokenize(source)
    print(f"      -> {len(tokens)} tokens produced")

    print(f"[2/3] Parsing tokens into blocks ...")
    blocks = parse(tokens)
    print(f"      -> {len(blocks)} blocks produced")

    print(f"[3/3] Generating HTML ...")
    title = os.path.splitext(os.path.basename(input_path))[0].replace("_", " ").title()
    html = generate(blocks, title=title)

    # ── Write output ───────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = len(html) / 1024
    print(f"\n[OK] Success! Written {size_kb:.1f} KB to '{output_path}'")
    print(f"  Open in a browser to view the result.")


if __name__ == "__main__":
    main()
