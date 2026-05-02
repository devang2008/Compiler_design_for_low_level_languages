"""
ast_nodes.py — Abstract Syntax Tree Node Definitions
=====================================================
Every construct in Small-C is represented by one of these dataclasses.
The parser builds a tree of these nodes; later passes walk the tree.

Every node carries a `line` field so error messages can always point
back to the original source location.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────
# Top-level program
# ──────────────────────────────────────────────────────────────────

@dataclass
class Program:
    """Root of the AST — holds all top-level function declarations."""
    functions: List[FunctionDecl] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# Functions and parameters
# ──────────────────────────────────────────────────────────────────

@dataclass
class FunctionDecl:
    """A function definition: return type, name, params, body."""
    name: str
    return_type: str
    params: List[ParamDecl]
    body: Block
    line: int


@dataclass
class ParamDecl:
    """A single formal parameter in a function signature."""
    name: str
    type: str
    line: int


# ──────────────────────────────────────────────────────────────────
# Blocks and statements
# ──────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """A brace-enclosed sequence of statements."""
    statements: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class VarDecl:
    """Variable declaration, optionally with an initialiser expression."""
    name: str
    type: str
    init_expr: Optional[Any] = None
    line: int = 0


@dataclass
class ArrayDecl:
    """Fixed-size array declaration: int arr[10];"""
    name: str
    type: str
    size: int = 0
    line: int = 0


@dataclass
class IfStmt:
    """if / else-if / else chain.

    elif_clauses is a list of (condition, Block) tuples.
    else_block is None when there is no else branch.
    """
    condition: Any = None
    then_block: Optional[Block] = None
    elif_clauses: List[Tuple[Any, Block]] = field(default_factory=list)
    else_block: Optional[Block] = None
    line: int = 0


@dataclass
class WhileStmt:
    """while (condition) { body }"""
    condition: Any = None
    body: Optional[Block] = None
    line: int = 0


@dataclass
class ForStmt:
    """for (init; condition; update) { body }"""
    init: Optional[Any] = None
    condition: Optional[Any] = None
    update: Optional[Any] = None
    body: Optional[Block] = None
    line: int = 0


@dataclass
class ReturnStmt:
    """return expr; or return;"""
    expr: Optional[Any] = None
    line: int = 0


@dataclass
class BreakStmt:
    """break; — only valid inside a loop."""
    line: int = 0


@dataclass
class ExprStmt:
    """An expression used as a statement (e.g. function call)."""
    expr: Any = None
    line: int = 0


# ──────────────────────────────────────────────────────────────────
# Expressions
# ──────────────────────────────────────────────────────────────────

@dataclass
class AssignExpr:
    """target = value"""
    target: Any = None
    value: Any = None
    line: int = 0


@dataclass
class BinaryExpr:
    """left op right  (e.g. a + b, x < y, p && q)"""
    op: str = ""
    left: Any = None
    right: Any = None
    line: int = 0


@dataclass
class UnaryExpr:
    """Prefix unary: !x, -x, ++x, --x"""
    op: str = ""
    operand: Any = None
    line: int = 0


@dataclass
class PostfixExpr:
    """Postfix increment / decrement: x++, x--"""
    op: str = ""
    operand: Any = None
    line: int = 0


@dataclass
class CallExpr:
    """Function call: name(arg1, arg2, ...)"""
    name: str = ""
    args: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class ArrayAccessExpr:
    """Array subscript: name[index]"""
    name: str = ""
    index: Any = None
    line: int = 0


@dataclass
class IdentifierExpr:
    """A variable or parameter name reference."""
    name: str = ""
    line: int = 0


# ──────────────────────────────────────────────────────────────────
# Literals
# ──────────────────────────────────────────────────────────────────

@dataclass
class IntLiteral:
    """An integer constant like 42."""
    value: int = 0
    line: int = 0


@dataclass
class CharLiteral:
    """A character constant like 'a' or '\\n'."""
    value: str = ""
    line: int = 0


@dataclass
class StringLiteral:
    """A string constant like \"hello world\"."""
    value: str = ""
    line: int = 0
