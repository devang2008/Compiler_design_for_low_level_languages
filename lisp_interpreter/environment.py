"""
environment.py — Lexical Scope & Variable Binding
===================================================
Implements the Environment class that manages variable scoping
for the Lisp interpreter. Each environment has an optional parent,
enabling nested lexical scopes (closures, let-bindings, etc.).

Pipeline position: Used by the Evaluator to look up and bind names.
"""


class EnvironmentError(Exception):
    """Raised when a variable lookup fails."""
    pass


class Environment:
    """
    A dictionary-based scope with an optional parent link.

    Variable lookup walks up the parent chain until the name is found
    or the global scope is exhausted (producing an error).
    """

    def __init__(self, bindings: dict = None, parent: "Environment" = None):
        """
        Create a new environment.

        Args:
            bindings: Optional initial name→value mappings.
            parent:   Enclosing scope (None for the global env).
        """
        self._store: dict = dict(bindings) if bindings else {}
        self._parent: "Environment" | None = parent

    # ── Lookup ─────────────────────────────────────────────
    def get(self, name: str):
        """
        Look up *name* in this scope, then walk parent scopes.

        Raises EnvironmentError if the name is unbound everywhere.
        """
        if name in self._store:
            return self._store[name]
        if self._parent is not None:
            return self._parent.get(name)
        raise EnvironmentError(f"Unbound variable: '{name}'")

    # ── Mutation ───────────────────────────────────────────
    def set(self, name: str, value):
        """Bind *name* to *value* in **this** scope (define semantics)."""
        self._store[name] = value

    # ── Scope extension (used by function calls & let) ─────
    def extend(self, params: list, args: list) -> "Environment":
        """
        Create a child environment binding *params* to *args*.

        This is the core mechanism for function calls:
            extend(['x', 'y'], [10, 20])
            → new child env where x=10, y=20

        Raises EnvironmentError if the counts don't match.
        """
        if len(params) != len(args):
            raise EnvironmentError(
                f"Argument count mismatch: expected {len(params)}, "
                f"got {len(args)}"
            )
        bindings = dict(zip(params, args))
        return Environment(bindings=bindings, parent=self)

    # ── Debug helpers ──────────────────────────────────────
    def __repr__(self):
        keys = list(self._store.keys())
        parent_info = "-> parent" if self._parent else "(global)"
        return f"Environment({keys} {parent_info})"
