# Python Lisp Interpreter

A complete, dependency-free Lisp interpreter written from scratch in Python. It features a robust Lexer, a recursive descent Parser, and a tree-walking Evaluator with lexical scoping capabilities.

## Features
- **Data Types**: Integers, Floats, Strings, Booleans (`#t`, `#f`), Symbols, Lists, and Closures.
- **Arithmetic**: `+`, `-`, `*`, `/`, `modulo`.
- **Variables**: `define` to bind variables in the environment.
- **Conditionals**: `if`, `cond`, and logical operators (`and`, `or`, `not`).
- **Comparisons**: `=`, `<`, `>`, `<=`, `>=`.
- **Functions**: Anonymous functions via `lambda` and named functions via `define`. Fully supports lexical scoping and closures.
- **List Operations**: `car`, `cdr`, `cons`, `list`, `length`, `null?`, `map`, `filter`.
- **Control Flow**: `let` bindings for local scope, `begin` blocks for sequential execution.
- **Recursion**: Perfectly handles deep recursive algorithms like Fibonacci and Factorials.

## Architecture
1. **Lexer (`lexer.py`)**: Tokenizes the raw text into a flat list of strings, identifying specific types like strings and booleans.
2. **Parser (`parser.py`)**: Converts the flat token list into a nested Python list (Abstract Syntax Tree), representing nested s-expressions.
3. **Evaluator (`evaluator.py` & `stdlib.py`)**: Evaluates the AST recursively. Variables and functions are tracked within an `Environment` class, which handles nested scopes perfectly for closures.

## How to Run

### 1. REPL Mode (Interactive)
To launch the Read-Eval-Print Loop and type Lisp code interactively:
```bash
python main.py
```
*Note: The REPL intelligently waits for you to close all open parentheses before evaluating.*

### 2. File Mode
To run a complete Lisp script:
```bash
python main.py path/to/script.lisp
```

## Examples
Check the `/examples` folder for various Lisp implementations evaluated by this engine:
- `01_arithmetic.lisp`
- `02_variables.lisp`
- `03_conditionals.lisp`
- `04_functions.lisp`
- `05_recursion.lisp` (Fibonacci & Factorial)
- `06_lists.lisp`
- `07_let_begin.lisp`
- `08_closures.lisp`

## Running Tests
An extensive suite of 49 unit tests ensures the interpreter's stability. Run them individually via:
```bash
cd tests
python test_lexer.py
python test_parser.py
python test_evaluator.py
```
