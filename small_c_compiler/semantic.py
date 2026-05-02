"""
semantic.py — Semantic Analyzer for Small-C
=============================================
Walks the AST produced by the parser and enforces every language rule
that cannot be expressed in the grammar alone:
  - Type checking
  - Scope checking (undeclared / duplicate names)
  - Break-outside-loop detection
  - main() existence check
  - Stack frame size computation

After analysis, every IdentifierExpr and CallExpr node is annotated
with a .symbol attribute pointing to the resolved Symbol entry.

Pipeline position: AST → [Semantic Analyzer] → annotated AST + frame sizes
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from errors import SemanticError
from symbol_table import SymbolTable, Symbol, SymbolKind
from ast_nodes import (
    Program, FunctionDecl, ParamDecl, Block,
    VarDecl, ArrayDecl,
    IfStmt, WhileStmt, ForStmt, ReturnStmt, BreakStmt, ExprStmt,
    AssignExpr, BinaryExpr, UnaryExpr, PostfixExpr,
    CallExpr, ArrayAccessExpr, IdentifierExpr,
    IntLiteral, CharLiteral, StringLiteral,
)


# Built-in I/O functions — the semantic analyzer pre-registers these
# so user code can call them without forward declarations.
BUILTINS = {
    "print_int":    {"params": ["int"],  "return": "void"},
    "print_char":   {"params": ["int"],  "return": "void"},
    "print_string": {"params": ["int"],  "return": "void"},  # accepts string addr
    "read_int":     {"params": [],       "return": "int"},
    "exit":         {"params": [],       "return": "void"},
}


class SemanticAnalyzer:
    """Walks the AST and enforces Small-C semantic rules.

    Usage:
        analyzer = SemanticAnalyzer("factorial.c")
        frame_sizes = analyzer.analyze(ast)
        # frame_sizes is { "factorial": 8, "main": 12, ... }
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.symbol_table = SymbolTable()
        self.current_function: Optional[FunctionDecl] = None
        self.loop_depth: int = 0
        self.frame_sizes: Dict[str, int] = {}
        # Per-function variable metadata: { func_name: { var_name: {offset, kind, size, type} } }
        self.var_info: Dict[str, Dict[str, Dict]] = {}

    # ── public entry point ────────────────────────────────────────

    def analyze(self, program: Program) -> Dict[str, int]:
        """Run semantic analysis on the full program.

        Returns a dict mapping function names to their stack frame
        sizes in bytes.
        """
        self._register_builtins()
        self._register_all_functions(program)
        self._check_main_exists(program)

        for func in program.functions:
            self.visit_FunctionDecl(func)

        return self.frame_sizes

    # ── pre-registration passes ───────────────────────────────────

    def _register_builtins(self) -> None:
        """Add built-in I/O functions to the global scope so user
        code can call them without a forward declaration."""
        for name, info in BUILTINS.items():
            sym = Symbol(
                name=name, type="function", kind=SymbolKind.FUNCTION,
                param_types=info["params"], return_type=info["return"],
            )
            self.symbol_table.declare(sym)

    def _register_all_functions(self, program: Program) -> None:
        """Pre-register every user-defined function in the global scope
        before analysing bodies.  This allows mutual recursion and
        forward calls without requiring forward declarations."""
        for func in program.functions:
            if self.symbol_table.lookup_local(func.name):
                raise SemanticError(
                    f"duplicate function definition '{func.name}'",
                    self.filename, func.line
                )
            param_types = [p.type for p in func.params]
            sym = Symbol(
                name=func.name, type="function", kind=SymbolKind.FUNCTION,
                param_types=param_types, return_type=func.return_type,
            )
            self.symbol_table.declare(sym)

    def _check_main_exists(self, program: Program) -> None:
        """Verify that a main() function exists and returns int."""
        main_sym = self.symbol_table.lookup("main")
        if main_sym is None or main_sym.kind != SymbolKind.FUNCTION:
            # Use line 1 as a fallback since there's no node to point at
            line = program.functions[0].line if program.functions else 1
            raise SemanticError("program must define a main() function",
                                self.filename, line)
        if main_sym.return_type != "int":
            # Find the main function node for its line number
            for f in program.functions:
                if f.name == "main":
                    raise SemanticError("main() must return int",
                                        self.filename, f.line)

    # ── error helper ──────────────────────────────────────────────

    def _error(self, msg: str, line: int) -> None:
        raise SemanticError(msg, self.filename, line)

    # ── visitor dispatch ──────────────────────────────────────────

    def visit(self, node: Any) -> Optional[str]:
        """Dispatch to the appropriate visit_* method based on node type.
        Returns a type string when visiting expressions, None for statements."""
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is None:
            self._error(f"internal: no visitor for {type(node).__name__}", 0)
        return method(node)

    # ── declarations ──────────────────────────────────────────────

    def visit_FunctionDecl(self, node: FunctionDecl) -> None:
        """Analyze a function body: open scope, declare params, visit body."""
        self.current_function = node
        self.symbol_table.enter_scope()
        self.symbol_table.reset_offsets()
        self._current_var_info: Dict[str, Dict] = {}

        # Declare parameters as local symbols
        for param in node.params:
            if param.type == "void":
                self._error("parameter cannot have void type", param.line)
            offset = self.symbol_table.allocate_local(4)
            sym = Symbol(
                name=param.name, type=param.type,
                kind=SymbolKind.PARAMETER,
                memory_offset=offset, size=1,
            )
            try:
                self.symbol_table.declare(sym)
            except KeyError:
                self._error(f"duplicate parameter '{param.name}'", param.line)
            self._current_var_info[param.name] = {
                "offset": offset, "kind": "parameter", "size": 1, "type": param.type
            }

        # Visit the body
        self.visit_Block(node.body)

        # Record frame size and variable info for this function
        self.frame_sizes[node.name] = self.symbol_table.offset_counter
        self.var_info[node.name] = dict(self._current_var_info)

        self.symbol_table.exit_scope()
        self.current_function = None

    def visit_Block(self, node: Block) -> None:
        """Visit every statement in a block."""
        for stmt in node.statements:
            self.visit(stmt)

    def visit_VarDecl(self, node: VarDecl) -> None:
        """Declare a scalar variable, check for void type, visit initialiser."""
        if node.type == "void":
            self._error("cannot declare variable with void type", node.line)

        offset = self.symbol_table.allocate_local(4)
        sym = Symbol(
            name=node.name, type=node.type,
            kind=SymbolKind.VARIABLE,
            memory_offset=offset, size=1,
        )
        try:
            self.symbol_table.declare(sym)
        except KeyError:
            self._error(f"duplicate variable '{node.name}' in this scope", node.line)
        self._current_var_info[node.name] = {
            "offset": offset, "kind": "variable", "size": 1, "type": node.type
        }

        if node.init_expr is not None:
            self.visit(node.init_expr)

    def visit_ArrayDecl(self, node: ArrayDecl) -> None:
        """Declare an array, check size > 0."""
        if node.size <= 0:
            self._error(f"array '{node.name}' must have size > 0", node.line)
        if node.type == "void":
            self._error("cannot declare array with void type", node.line)

        # Each int element is 4 bytes
        element_size = 4 if node.type == "int" else 1
        total_bytes = node.size * element_size
        # Round up to word boundary
        total_bytes = ((total_bytes + 3) // 4) * 4
        offset = self.symbol_table.allocate_local(total_bytes)

        sym = Symbol(
            name=node.name, type=node.type,
            kind=SymbolKind.ARRAY,
            memory_offset=offset, size=node.size,
        )
        try:
            self.symbol_table.declare(sym)
        except KeyError:
            self._error(f"duplicate variable '{node.name}' in this scope", node.line)
        self._current_var_info[node.name] = {
            "offset": offset, "kind": "array", "size": node.size, "type": node.type
        }

    # ── statements ────────────────────────────────────────────────

    def visit_IfStmt(self, node: IfStmt) -> None:
        self.visit(node.condition)
        self.symbol_table.enter_scope()
        self.visit_Block(node.then_block)
        self.symbol_table.exit_scope()

        for elif_cond, elif_block in node.elif_clauses:
            self.visit(elif_cond)
            self.symbol_table.enter_scope()
            self.visit_Block(elif_block)
            self.symbol_table.exit_scope()

        if node.else_block is not None:
            self.symbol_table.enter_scope()
            self.visit_Block(node.else_block)
            self.symbol_table.exit_scope()

    def visit_WhileStmt(self, node: WhileStmt) -> None:
        self.visit(node.condition)
        self.loop_depth += 1
        self.symbol_table.enter_scope()
        self.visit_Block(node.body)
        self.symbol_table.exit_scope()
        self.loop_depth -= 1

    def visit_ForStmt(self, node: ForStmt) -> None:
        # The init may declare a variable, so wrap in its own scope
        self.symbol_table.enter_scope()
        if node.init is not None:
            self.visit(node.init)
        if node.condition is not None:
            self.visit(node.condition)

        self.loop_depth += 1
        self.symbol_table.enter_scope()
        self.visit_Block(node.body)
        self.symbol_table.exit_scope()
        self.loop_depth -= 1

        if node.update is not None:
            self.visit(node.update)
        self.symbol_table.exit_scope()

    def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        func = self.current_function
        if func is None:
            self._error("return statement outside function", node.line)

        if func.return_type == "void" and node.expr is not None:
            self._error(
                f"void function '{func.name}' cannot return a value",
                node.line
            )

        if node.expr is not None:
            self.visit(node.expr)

    def visit_BreakStmt(self, node: BreakStmt) -> None:
        if self.loop_depth == 0:
            self._error("break statement outside of loop", node.line)

    def visit_ExprStmt(self, node: ExprStmt) -> None:
        self.visit(node.expr)

    # ── expressions ───────────────────────────────────────────────

    def visit_AssignExpr(self, node: AssignExpr) -> str:
        self.visit(node.target)
        right_type = self.visit(node.value)
        return right_type or "int"

    def visit_BinaryExpr(self, node: BinaryExpr) -> str:
        self.visit(node.left)
        self.visit(node.right)
        # All binary ops produce int in Small-C
        return "int"

    def visit_UnaryExpr(self, node: UnaryExpr) -> str:
        self.visit(node.operand)
        return "int"

    def visit_PostfixExpr(self, node: PostfixExpr) -> str:
        self.visit(node.operand)
        return "int"

    def visit_CallExpr(self, node: CallExpr) -> str:
        """Validate function call: existence, argument count, annotate node."""
        sym = self.symbol_table.lookup(node.name)
        if sym is None or sym.kind != SymbolKind.FUNCTION:
            self._error(f"call to undeclared function '{node.name}'", node.line)

        # Check argument count (skip check for built-in print_string
        # because it accepts a string literal which has a special type flow)
        expected = len(sym.param_types)
        got = len(node.args)
        if got != expected:
            self._error(
                f"function '{node.name}' expects {expected} argument(s), got {got}",
                node.line
            )

        for arg in node.args:
            self.visit(arg)

        # Annotate the call node with the resolved symbol
        node.symbol = sym  # type: ignore[attr-defined]
        return sym.return_type

    def visit_ArrayAccessExpr(self, node: ArrayAccessExpr) -> str:
        """Validate array access: name must be an array, index is visited."""
        sym = self.symbol_table.lookup(node.name)
        if sym is None:
            self._error(f"undeclared variable '{node.name}'", node.line)
        if sym.kind != SymbolKind.ARRAY:
            self._error(
                f"'{node.name}' is not an array — cannot use [] on it",
                node.line
            )
        self.visit(node.index)
        # Annotate
        node.symbol = sym  # type: ignore[attr-defined]
        return sym.type

    def visit_IdentifierExpr(self, node: IdentifierExpr) -> str:
        """Validate variable reference: must be declared, annotate node."""
        sym = self.symbol_table.lookup(node.name)
        if sym is None:
            self._error(f"undeclared variable '{node.name}'", node.line)
        if sym.kind == SymbolKind.FUNCTION:
            # Using a function name as a value is not supported
            self._error(
                f"'{node.name}' is a function — cannot use as a variable",
                node.line
            )
        node.symbol = sym  # type: ignore[attr-defined]
        return sym.type

    def visit_IntLiteral(self, _node: IntLiteral) -> str:
        return "int"

    def visit_CharLiteral(self, _node: CharLiteral) -> str:
        return "char"

    def visit_StringLiteral(self, _node: StringLiteral) -> str:
        return "char[]"


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def analyze(program: Program, filename: str = "<stdin>"):
    """Run semantic analysis and return (analyzer, frame_sizes).

    The analyzer object holds the symbol_table, which later passes need.
    """
    sa = SemanticAnalyzer(filename)
    frame_sizes = sa.analyze(program)
    return sa, frame_sizes
