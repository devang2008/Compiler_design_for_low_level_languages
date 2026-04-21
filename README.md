# Compiler Projects

> A collection of compiler and interpreter projects built from scratch in Python.

This repository contains **two complete projects** that demonstrate core
compiler/interpreter concepts using proper multi-stage pipelines.

---

## Project 1: Markdown to HTML Transpiler

A command-line tool that reads a `.md` file and outputs a fully formatted
`.html` file using a proper compiler pipeline:

```
Source Text → Lexer → Token Stream → Parser → HTML Generator → Output File
```

### Features

- Headings (H1–H6)
- **Bold**, *Italic*, ***Bold+Italic***
- Inline `code` and fenced code blocks
- Unordered and ordered lists
- Blockquotes, horizontal rules
- Links and images
- Paragraphs

### Usage

```bash
cd markdown_transpiler
python main.py test.md
```

---

## Project 2: Lisp Interpreter

A fully working interpreter for a Scheme-like Lisp that supports
recursion, closures, and a REPL:

```
Source String → Lexer → Token List → Parser → AST → Evaluator → Result
```

### Features

- Arithmetic, comparisons, boolean logic
- Variable definitions and lambda expressions
- Recursion (fibonacci, factorial)
- Closures with proper lexical scoping
- List operations (car, cdr, cons)
- Let bindings and begin blocks

### Usage

```bash
cd lisp_interpreter
python main.py              # REPL mode
python main.py program.lisp # File mode
```

---

## Requirements

- **Python 3.10+**
- **No third-party libraries** — everything is built with the standard library

## License

MIT
