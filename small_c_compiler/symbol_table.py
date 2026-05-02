"""
symbol_table.py — Lexically-Scoped Symbol Table
=================================================
Tracks every declared name (variables, parameters, functions, arrays)
across nested scopes. Used by the semantic analyzer and code generator
to validate references and compute stack frame layouts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class SymbolKind(Enum):
    """The category of a declared symbol."""
    VARIABLE  = "variable"
    PARAMETER = "parameter"
    FUNCTION  = "function"
    ARRAY     = "array"


@dataclass
class Symbol:
    """A single entry in the symbol table.

    Fields that are only meaningful for certain kinds:
      param_types / return_type  → only for FUNCTION kind
      size                       → element count; 1 for scalars, N for arrays
      memory_offset              → byte offset from $fp on the stack
    """
    name: str
    type: str                                    # "int", "char", "void"
    kind: SymbolKind
    scope_level: int       = 0
    memory_offset: int     = 0                   # offset from $fp
    size: int              = 1                    # 1 for scalar, N for array
    param_types: List[str] = field(default_factory=list)
    return_type: str       = ""

    def __repr__(self) -> str:
        parts = [
            f"Symbol({self.name!r}, type={self.type!r}, kind={self.kind.value}",
            f"scope={self.scope_level}, offset={self.memory_offset}, size={self.size}",
        ]
        if self.kind == SymbolKind.FUNCTION:
            parts.append(f"params={self.param_types}, ret={self.return_type}")
        return ", ".join(parts) + ")"


class SymbolTable:
    """A stack of scope dictionaries for lexical name resolution.

    Scopes are pushed when entering a function body or block,
    and popped when leaving.  Lookups search from the innermost
    scope outward, which gives proper shadowing semantics.
    """

    def __init__(self):
        # Start with one global scope
        self.scopes: List[Dict[str, Symbol]] = [{}]
        self.scope_level: int = 0
        self.offset_counter: int = 0    # running stack offset for locals

    # ── scope management ──────────────────────────────────────────

    def enter_scope(self) -> None:
        """Push a new empty scope onto the stack."""
        self.scopes.append({})
        self.scope_level += 1

    def exit_scope(self) -> None:
        """Pop the top scope off the stack."""
        self.scopes.pop()
        self.scope_level -= 1

    def reset_offsets(self) -> None:
        """Reset the offset counter — called at the start of each function."""
        self.offset_counter = 0

    # ── declarations ──────────────────────────────────────────────

    def declare(self, symbol: Symbol) -> None:
        """Add a symbol to the current (top) scope.

        Raises KeyError if the name is already declared in this scope —
        the caller (semantic analyzer) should catch this and raise
        SemanticError with proper context.
        """
        top = self.scopes[-1]
        if symbol.name in top:
            raise KeyError(f"'{symbol.name}' is already declared in this scope")
        symbol.scope_level = self.scope_level
        top[symbol.name] = symbol

    # ── lookups ───────────────────────────────────────────────────

    def lookup(self, name: str) -> Optional[Symbol]:
        """Search for a name from the innermost scope outward.

        Returns the Symbol if found, or None.
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Search ONLY the current (top) scope.

        Returns the Symbol if found, or None.
        """
        return self.scopes[-1].get(name)

    # ── stack allocation ──────────────────────────────────────────

    def allocate_local(self, size_bytes: int) -> int:
        """Reserve stack space for a local variable or array.

        Returns the new offset (negative displacement from $fp)
        so the code generator knows where to store this variable.
        """
        self.offset_counter += size_bytes
        return self.offset_counter

    # ── debugging ─────────────────────────────────────────────────

    def dump(self) -> str:
        """Return a human-readable representation of all scopes."""
        lines = []
        for i, scope in enumerate(self.scopes):
            lines.append(f"--- Scope {i} ---")
            for name, sym in scope.items():
                lines.append(f"  {sym}")
        return "\n".join(lines)
