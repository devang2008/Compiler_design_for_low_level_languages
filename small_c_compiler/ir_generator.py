"""
ir_generator.py — Three Address Code (TAC) Generator
=====================================================
Walks the semantically-analysed AST and emits a flat list of TAC
instructions for each function.  The TAC uses virtual temp names
(t0, t1, …) and virtual labels (L0, L1, …) which the register
allocator and code generator will later map to real MIPS resources.

Pipeline position: annotated AST → [IR Generator] → TAC + strings
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ast_nodes import (
    Program, FunctionDecl, ParamDecl, Block,
    VarDecl, ArrayDecl,
    IfStmt, WhileStmt, ForStmt, ReturnStmt, BreakStmt, ExprStmt,
    AssignExpr, BinaryExpr, UnaryExpr, PostfixExpr,
    CallExpr, ArrayAccessExpr, IdentifierExpr,
    IntLiteral, CharLiteral, StringLiteral,
)


# ──────────────────────────────────────────────────────────────────
# TAC instruction dataclasses
# ──────────────────────────────────────────────────────────────────

@dataclass
class TACBinOp:
    """result = left op right"""
    result: str
    op: str
    left: str
    right: str
    def __repr__(self): return f"{self.result} = {self.left} {self.op} {self.right}"

@dataclass
class TACUnaryOp:
    """result = op operand"""
    result: str
    op: str
    operand: str
    def __repr__(self): return f"{self.result} = {self.op}{self.operand}"

@dataclass
class TACCopy:
    """result = source"""
    result: str
    source: str
    def __repr__(self): return f"{self.result} = {self.source}"

@dataclass
class TACLoadImm:
    """result = immediate value"""
    result: str
    value: int
    def __repr__(self): return f"{self.result} = {self.value}"

@dataclass
class TACLoadStr:
    """result = address of string label in .data"""
    result: str
    label: str
    def __repr__(self): return f"{self.result} = &{self.label}"

@dataclass
class TACArrayRead:
    """result = array[index]"""
    result: str
    array: str
    index: str
    def __repr__(self): return f"{self.result} = {self.array}[{self.index}]"

@dataclass
class TACArrayWrite:
    """array[index] = source"""
    array: str
    index: str
    source: str
    def __repr__(self): return f"{self.array}[{self.index}] = {self.source}"

@dataclass
class TACLabel:
    """A named label in the instruction stream."""
    name: str
    def __repr__(self): return f"{self.name}:"

@dataclass
class TACJump:
    """Unconditional jump to label."""
    label: str
    def __repr__(self): return f"goto {self.label}"

@dataclass
class TACJumpIf:
    """Jump to label if condition is true (non-zero)."""
    condition: str
    label: str
    def __repr__(self): return f"if {self.condition} goto {self.label}"

@dataclass
class TACJumpIfNot:
    """Jump to label if condition is false (zero)."""
    condition: str
    label: str
    def __repr__(self): return f"ifnot {self.condition} goto {self.label}"

@dataclass
class TACParam:
    """Push a parameter before a function call."""
    name: str
    def __repr__(self): return f"param {self.name}"

@dataclass
class TACCall:
    """Call function with arg_count args; store return in result."""
    result: Optional[str]
    function: str
    arg_count: int
    def __repr__(self):
        r = self.result or "_"
        return f"{r} = call {self.function}, {self.arg_count}"

@dataclass
class TACReturn:
    """Return from function, optionally with a value."""
    value: Optional[str]
    def __repr__(self):
        if self.value:
            return f"return {self.value}"
        return "return"


# ──────────────────────────────────────────────────────────────────
# IR Generator
# ──────────────────────────────────────────────────────────────────

class IRGenerator:
    """Walks the AST and produces TAC instructions per function.

    Usage:
        gen = IRGenerator()
        result = gen.generate(program_ast)
        # result["functions"]  → { "main": [TAC...], ... }
        # result["strings"]    → { "str0": "Hello\\n", ... }
    """

    def __init__(self):
        self.instructions: List[Any] = []
        self.temp_count: int = 0
        self.label_count: int = 0
        self.string_literals: Dict[str, str] = {}
        self.string_count: int = 0
        self.current_function: str = ""

        # Stack of break-target labels so nested loops work correctly
        self.break_labels: List[str] = []

    # ── helpers ───────────────────────────────────────────────────

    def new_temp(self) -> str:
        """Generate a fresh temporary name like t0, t1, t2, …"""
        name = f"t{self.temp_count}"
        self.temp_count += 1
        return name

    def new_label(self) -> str:
        """Generate a fresh label name like L0, L1, L2, …"""
        name = f"L{self.label_count}"
        self.label_count += 1
        return name

    def emit(self, instr) -> None:
        """Append an instruction to the current stream."""
        self.instructions.append(instr)

    def _add_string(self, value: str) -> str:
        """Register a string literal and return its .data label."""
        label = f"str{self.string_count}"
        self.string_count += 1
        self.string_literals[label] = value
        return label

    # ── public entry ──────────────────────────────────────────────

    def generate(self, program: Program) -> Dict:
        """Generate TAC for the whole program.

        Returns:
            {
                "functions": { name: [TAC instructions], ... },
                "strings":   { "str0": "Hello\\n", ... }
            }
        """
        functions: Dict[str, List] = {}

        for func in program.functions:
            self.instructions = []
            self.temp_count = 0   # reset temps per function
            self.current_function = func.name
            self.gen_FunctionDecl(func)
            functions[func.name] = list(self.instructions)

        return {
            "functions": functions,
            "strings": dict(self.string_literals),
        }

    # ── code generation: declarations ─────────────────────────────

    def gen_FunctionDecl(self, node: FunctionDecl) -> None:
        """Emit TAC for a function: label, copy params, body."""
        self.emit(TACLabel(node.name))

        # Copy incoming parameters to their named locations.
        # The code generator will later handle the calling convention.
        for i, param in enumerate(node.params):
            self.emit(TACCopy(param.name, f"__arg{i}"))

        self.gen_Block(node.body)

        # If the function body doesn't end with a return, emit one.
        # For main() we return 0; for void functions we return nothing.
        if not self.instructions or not isinstance(self.instructions[-1], TACReturn):
            if node.return_type == "void":
                self.emit(TACReturn(None))
            else:
                self.emit(TACReturn(None))

    def gen_Block(self, node: Block) -> None:
        for stmt in node.statements:
            self.gen_stmt(stmt)

    # ── code generation: statements ───────────────────────────────

    def gen_stmt(self, node: Any) -> None:
        """Dispatch to the correct gen_* method for a statement."""
        name = type(node).__name__
        method = getattr(self, f"gen_{name}", None)
        if method:
            method(node)
        else:
            # Fallback: treat as expression statement
            self.gen_expr(node)

    def gen_VarDecl(self, node: VarDecl) -> None:
        if node.init_expr is not None:
            val = self.gen_expr(node.init_expr)
            self.emit(TACCopy(node.name, val))

    def gen_ArrayDecl(self, node: ArrayDecl) -> None:
        # Arrays are allocated on the stack — no TAC needed at declaration time
        pass

    def gen_ExprStmt(self, node: ExprStmt) -> None:
        self.gen_expr(node.expr)

    def gen_ReturnStmt(self, node: ReturnStmt) -> None:
        if node.expr is not None:
            val = self.gen_expr(node.expr)
            self.emit(TACReturn(val))
        else:
            self.emit(TACReturn(None))

    def gen_BreakStmt(self, node: BreakStmt) -> None:
        if self.break_labels:
            self.emit(TACJump(self.break_labels[-1]))

    def gen_IfStmt(self, node: IfStmt) -> None:
        """Generate chained conditional jumps for if / else-if / else."""
        end_label = self.new_label()

        # --- main if ---
        cond = self.gen_expr(node.condition)
        else_label = self.new_label()
        self.emit(TACJumpIfNot(cond, else_label))
        self.gen_Block(node.then_block)
        self.emit(TACJump(end_label))
        self.emit(TACLabel(else_label))

        # --- else-if chain ---
        for elif_cond, elif_block in node.elif_clauses:
            cond = self.gen_expr(elif_cond)
            next_label = self.new_label()
            self.emit(TACJumpIfNot(cond, next_label))
            self.gen_Block(elif_block)
            self.emit(TACJump(end_label))
            self.emit(TACLabel(next_label))

        # --- else ---
        if node.else_block is not None:
            self.gen_Block(node.else_block)

        self.emit(TACLabel(end_label))

    def gen_WhileStmt(self, node: WhileStmt) -> None:
        start_label = self.new_label()
        end_label = self.new_label()

        self.break_labels.append(end_label)

        self.emit(TACLabel(start_label))
        cond = self.gen_expr(node.condition)
        self.emit(TACJumpIfNot(cond, end_label))
        self.gen_Block(node.body)
        self.emit(TACJump(start_label))
        self.emit(TACLabel(end_label))

        self.break_labels.pop()

    def gen_ForStmt(self, node: ForStmt) -> None:
        """for(init; cond; update) body
        → init; start: if !cond goto end; body; update; goto start; end:
        """
        start_label = self.new_label()
        end_label = self.new_label()

        self.break_labels.append(end_label)

        # init
        if node.init is not None:
            self.gen_stmt(node.init)

        self.emit(TACLabel(start_label))

        # condition
        if node.condition is not None:
            cond = self.gen_expr(node.condition)
            self.emit(TACJumpIfNot(cond, end_label))

        # body
        self.gen_Block(node.body)

        # update
        if node.update is not None:
            self.gen_expr(node.update)

        self.emit(TACJump(start_label))
        self.emit(TACLabel(end_label))

        self.break_labels.pop()

    # ── code generation: expressions ──────────────────────────────

    def gen_expr(self, node: Any) -> str:
        """Generate TAC for an expression and return the temp/name
        holding the result."""
        name = type(node).__name__
        method = getattr(self, f"gen_{name}", None)
        if method:
            return method(node)
        raise RuntimeError(f"IR: no handler for {name}")

    def gen_IntLiteral(self, node: IntLiteral) -> str:
        t = self.new_temp()
        self.emit(TACLoadImm(t, node.value))
        return t

    def gen_CharLiteral(self, node: CharLiteral) -> str:
        t = self.new_temp()
        self.emit(TACLoadImm(t, ord(node.value)))
        return t

    def gen_StringLiteral(self, node: StringLiteral) -> str:
        label = self._add_string(node.value)
        t = self.new_temp()
        self.emit(TACLoadStr(t, label))
        return t

    def gen_IdentifierExpr(self, node: IdentifierExpr) -> str:
        # The variable's TAC name is just its source name.
        # The register allocator will map it to a register or stack slot.
        return node.name

    def gen_AssignExpr(self, node: AssignExpr) -> str:
        val = self.gen_expr(node.value)

        if isinstance(node.target, ArrayAccessExpr):
            idx = self.gen_expr(node.target.index)
            self.emit(TACArrayWrite(node.target.name, idx, val))
            return val

        if isinstance(node.target, IdentifierExpr):
            self.emit(TACCopy(node.target.name, val))
            return node.target.name

        return val

    def gen_BinaryExpr(self, node: BinaryExpr) -> str:
        # Short-circuit for && and ||
        if node.op == "&&":
            return self._gen_short_circuit_and(node)
        if node.op == "||":
            return self._gen_short_circuit_or(node)

        left = self.gen_expr(node.left)
        right = self.gen_expr(node.right)
        t = self.new_temp()
        self.emit(TACBinOp(t, node.op, left, right))
        return t

    def _gen_short_circuit_and(self, node: BinaryExpr) -> str:
        """a && b → evaluate a; if false, result is 0, skip b."""
        result = self.new_temp()
        false_label = self.new_label()
        end_label = self.new_label()

        left = self.gen_expr(node.left)
        self.emit(TACJumpIfNot(left, false_label))

        right = self.gen_expr(node.right)
        self.emit(TACCopy(result, right))
        self.emit(TACJump(end_label))

        self.emit(TACLabel(false_label))
        self.emit(TACLoadImm(result, 0))

        self.emit(TACLabel(end_label))
        return result

    def _gen_short_circuit_or(self, node: BinaryExpr) -> str:
        """a || b → evaluate a; if true, result is 1, skip b."""
        result = self.new_temp()
        true_label = self.new_label()
        end_label = self.new_label()

        left = self.gen_expr(node.left)
        self.emit(TACJumpIf(left, true_label))

        right = self.gen_expr(node.right)
        self.emit(TACCopy(result, right))
        self.emit(TACJump(end_label))

        self.emit(TACLabel(true_label))
        self.emit(TACLoadImm(result, 1))

        self.emit(TACLabel(end_label))
        return result

    def gen_UnaryExpr(self, node: UnaryExpr) -> str:
        if node.op in ("++", "--"):
            # Prefix increment / decrement:  ++x  →  x = x + 1; result is x
            operand_name = self.gen_expr(node.operand)
            t = self.new_temp()
            op = "+" if node.op == "++" else "-"
            one = self.new_temp()
            self.emit(TACLoadImm(one, 1))
            self.emit(TACBinOp(t, op, operand_name, one))
            self.emit(TACCopy(operand_name, t))
            return operand_name

        operand = self.gen_expr(node.operand)
        t = self.new_temp()
        self.emit(TACUnaryOp(t, node.op, operand))
        return t

    def gen_PostfixExpr(self, node: PostfixExpr) -> str:
        """x++ → old = x; x = x + 1; result is old"""
        operand_name = self.gen_expr(node.operand)
        old = self.new_temp()
        self.emit(TACCopy(old, operand_name))

        t = self.new_temp()
        one = self.new_temp()
        self.emit(TACLoadImm(one, 1))
        op = "+" if node.op == "++" else "-"
        self.emit(TACBinOp(t, op, operand_name, one))
        self.emit(TACCopy(operand_name, t))

        return old

    def gen_CallExpr(self, node: CallExpr) -> str:
        """Emit params then call.  Built-in I/O is also emitted as
        TACParam+TACCall; the code generator will translate them to syscalls."""
        arg_temps = []
        for arg in node.args:
            arg_temps.append(self.gen_expr(arg))

        for at in arg_temps:
            self.emit(TACParam(at))

        result = self.new_temp()
        self.emit(TACCall(result, node.name, len(node.args)))
        return result

    def gen_ArrayAccessExpr(self, node: ArrayAccessExpr) -> str:
        idx = self.gen_expr(node.index)
        t = self.new_temp()
        self.emit(TACArrayRead(t, node.name, idx))
        return t


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def generate_ir(program: Program) -> Dict:
    """Generate TAC for the entire program.

    Returns { "functions": {...}, "strings": {...} }.
    """
    return IRGenerator().generate(program)
