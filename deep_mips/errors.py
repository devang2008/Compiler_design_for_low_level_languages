"""
errors.py — Custom Exception Classes for Deep-MIPS Compiler
=============================================================
Every stage of the compilation pipeline has its own exception type
so that error messages can pinpoint exactly where a failure occurred.
"""


class ModelParseError(Exception):
    """Raised when the JSON model file is malformed or invalid."""

    def __init__(self, message: str, field: str = None):
        self.field = field
        if field:
            self.message = f"ModelParseError: {message} (field: {field})"
        else:
            self.message = f"ModelParseError: {message}"
        super().__init__(self.message)


class GraphError(Exception):
    """Raised when the computation graph has structural issues."""

    def __init__(self, message: str, node_id: str = None):
        self.node_id = node_id
        if node_id:
            self.message = f"GraphError: {message} (node: {node_id})"
        else:
            self.message = f"GraphError: {message}"
        super().__init__(self.message)


class MemoryPlannerError(Exception):
    """Raised when memory planning fails (e.g. layout conflict)."""

    def __init__(self, message: str):
        self.message = f"MemoryPlannerError: {message}"
        super().__init__(self.message)


class CodeGenError(Exception):
    """Raised when MIPS code generation encounters an unsupported construct."""

    def __init__(self, message: str, layer_id: str = None):
        self.layer_id = layer_id
        if layer_id:
            self.message = f"CodeGenError: {message} (layer: {layer_id})"
        else:
            self.message = f"CodeGenError: {message}"
        super().__init__(self.message)


class QuantizationError(Exception):
    """Raised when float-to-fixed conversion fails (NaN, Inf, overflow)."""

    def __init__(self, message: str):
        self.message = f"QuantizationError: {message}"
        super().__init__(self.message)
