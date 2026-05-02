"""
code_generator.py — MIPS Assembly Code Generator
==================================================
Translates TAC instructions (with register assignments) into valid
MIPS assembly that runs on MARS / SPIM without modification.

Handles:
  - .data section for string literals
  - function prologues and epilogues
  - all TAC→MIPS instruction mappings
  - built-in I/O syscalls
  - array access via $fp-relative addressing
  - caller-save / callee-save register protocol

Pipeline position: TAC + reg_map → [Code Generator] → .asm file
"""

from __future__ import annotations
from typing import Any, Dict, List

from errors import CodeGenError
from ir_generator import (
    TACBinOp, TACUnaryOp, TACCopy, TACLoadImm, TACLoadStr,
    TACArrayRead, TACArrayWrite, TACLabel, TACJump,
    TACJumpIf, TACJumpIfNot, TACParam, TACCall, TACReturn,
)


# Built-in function names → their syscall numbers and argument types
BUILTIN_SYSCALLS = {
    "print_int":    {"syscall": 1,  "arg_reg": "$a0", "has_result": False},
    "print_char":   {"syscall": 11, "arg_reg": "$a0", "has_result": False},
    "print_string": {"syscall": 4,  "arg_reg": "$a0", "has_result": False},
    "read_int":     {"syscall": 5,  "arg_reg": None,  "has_result": True},
    "exit":         {"syscall": 10, "arg_reg": None,  "has_result": False},
}


class CodeGenerator:
    """Emits MIPS assembly from TAC instructions and register maps.

    Usage:
        cg = CodeGenerator()
        asm_text = cg.generate(ir_output, reg_maps, frame_infos, semantic_frame_sizes)
    """

    def __init__(self):
        self.output: List[str] = []
        self.reg_map: Dict[str, str] = {}
        self.frame_size: int = 0
        self.s_regs_used: List[str] = []
        self.spills: Dict[str, int] = {}
        self.string_literals: Dict[str, str] = {}
        self.current_function: str = ""
        # Per-function variable metadata from semantic analysis
        self.var_info: Dict[str, Dict[str, Dict]] = {}
        self.current_var_info: Dict[str, Dict] = {}
        # Temporary list to collect TACParam args before a call
        self._pending_params: List[str] = []

    # ── output helpers ────────────────────────────────────────────

    def emit(self, line: str) -> None:
        """Append a line of assembly to the output buffer."""
        self.output.append(line)

    def emit_comment(self, text: str) -> None:
        self.emit(f"    # {text}")

    def resolve(self, name: str) -> str:
        """Map a TAC name to its MIPS register.

        For spilled variables, this returns $at as a scratch register
        and the caller must emit lw/sw to move data to/from the spill slot.
        """
        if name in self.reg_map:
            r = self.reg_map[name]
            if r.startswith("spill:"):
                return "$at"
            return r
        # Numeric literal embedded directly (shouldn't normally happen)
        return name

    def _spill_offset(self, name: str) -> int:
        """Return the stack offset for a spilled variable, or -1 if not spilled."""
        r = self.reg_map.get(name, "")
        if r.startswith("spill:"):
            return int(r.split(":")[1])
        return -1

    def _load_spill(self, name: str, dest_reg: str = "$at") -> str:
        """If name is spilled, emit lw and return dest_reg.
        Otherwise return the mapped register directly."""
        off = self._spill_offset(name)
        if off >= 0:
            self.emit(f"    lw    {dest_reg}, -{off}($fp)")
            return dest_reg
        return self.resolve(name)

    def _store_spill(self, name: str, src_reg: str = "$at") -> None:
        """If name is spilled, emit sw from src_reg to spill slot."""
        off = self._spill_offset(name)
        if off >= 0:
            self.emit(f"    sw    {src_reg}, -{off}($fp)")

    # ── public entry point ────────────────────────────────────────

    def generate(
        self,
        ir_output: Dict,
        reg_maps: Dict[str, Dict],
        frame_infos: Dict[str, Dict],
        semantic_frame_sizes: Dict[str, int],
        var_info: Dict[str, Dict[str, Dict]] = None,
    ) -> str:
        """Generate the complete MIPS assembly file.

        Args:
            ir_output:    { "functions": { name: [TAC] }, "strings": { label: value } }
            reg_maps:     { func_name: { "reg_map": {}, "frame_size": int, ... } }
            frame_infos:  same as reg_maps (contains s_regs_used, spills, etc.)
            semantic_frame_sizes: { func_name: int } from semantic pass
        """
        self.string_literals = ir_output.get("strings", {})
        self.var_info = var_info or {}
        self.output = []

        # ── .data section ─────────────────────────────────────────
        self.emit(".data")
        for label, value in self.string_literals.items():
            # Escape the string value for MIPS .asciiz
            escaped = self._escape_for_mips(value)
            self.emit(f'{label}:  .asciiz  "{escaped}"')
        if not self.string_literals:
            self.emit("    # (no string literals)")
        self.emit("")

        # ── .text section ─────────────────────────────────────────
        self.emit(".text")
        self.emit(".globl main")
        self.emit("")

        # Generate each function — put main last so it appears first in execution
        func_order = sorted(ir_output["functions"].keys(),
                            key=lambda n: (n != "main", n))

        for func_name in func_order:
            tac_list = ir_output["functions"][func_name]
            info = reg_maps.get(func_name, frame_infos.get(func_name, {}))

            self.current_function = func_name
            self.reg_map = info.get("reg_map", {})
            self.frame_size = info.get("frame_size", 32)
            self.s_regs_used = info.get("s_regs_used", [])
            self.spills = info.get("spills", {})
            self.current_var_info = self.var_info.get(func_name, {})

            self._gen_function(func_name, tac_list)
            self.emit("")

        return "\n".join(self.output) + "\n"

    # ── function structure ────────────────────────────────────────

    def _gen_function(self, name: str, tac_list: List[Any]) -> None:
        """Emit prologue, body instructions, and (epilogue is emitted per return)."""
        fs = self.frame_size

        # Label (skip the TACLabel at the start if present)
        start_idx = 0
        if tac_list and isinstance(tac_list[0], TACLabel):
            start_idx = 1
        self.emit(f"{name}:")

        # ── PROLOGUE ──────────────────────────────────────────────
        self.emit(f"    subu  $sp, $sp, {fs}")
        self.emit(f"    sw    $ra, {fs - 4}($sp)")
        self.emit(f"    sw    $fp, {fs - 8}($sp)")
        self.emit(f"    addiu $fp, $sp, {fs}")

        # Save callee-saved $s registers
        for i, sr in enumerate(self.s_regs_used):
            offset = fs - 12 - (i * 4)
            self.emit(f"    sw    {sr}, {offset}($sp)")

        # Initialize array base addresses: compute $fp - offset into the
        # mapped $s register so array read/write can use it directly
        for var_name, vinfo in self.current_var_info.items():
            if vinfo["kind"] == "array" and var_name in self.reg_map:
                reg = self.reg_map[var_name]
                if reg.startswith("$s"):
                    arr_offset = vinfo["offset"]
                    # The array occupies [fp-offset .. fp-offset+size*4)
                    # Base address is fp - offset (lowest element)
                    self.emit(f"    addiu {reg}, $fp, -{arr_offset}")

        # ── BODY ──────────────────────────────────────────────────
        for instr in tac_list[start_idx:]:
            self._gen_instr(instr)

    def _emit_epilogue(self) -> None:
        """Restore saved registers, frame pointer, return address, and
        deallocate the stack frame."""
        fs = self.frame_size

        # Restore $s registers (reverse order)
        for i, sr in enumerate(reversed(self.s_regs_used)):
            idx = len(self.s_regs_used) - 1 - i
            offset = fs - 12 - (idx * 4)
            self.emit(f"    lw    {sr}, {offset}($sp)")

        self.emit(f"    lw    $ra, {fs - 4}($sp)")
        self.emit(f"    lw    $fp, {fs - 8}($sp)")
        self.emit(f"    addiu $sp, $sp, {fs}")

    # ── instruction dispatch ──────────────────────────────────────

    def _gen_instr(self, instr: Any) -> None:
        """Dispatch a single TAC instruction to its MIPS emitter."""
        name = type(instr).__name__
        method = getattr(self, f"_gen_{name}", None)
        if method:
            method(instr)
        else:
            self.emit_comment(f"unhandled: {instr}")

    # ── TAC → MIPS translations ───────────────────────────────────

    def _gen_TACLabel(self, instr: TACLabel) -> None:
        self.emit(f"{instr.name}:")

    def _gen_TACJump(self, instr: TACJump) -> None:
        self.emit(f"    j     {instr.label}")

    def _gen_TACJumpIf(self, instr: TACJumpIf) -> None:
        cond = self._load_spill(instr.condition, "$at")
        self.emit(f"    bne   {cond}, $zero, {instr.label}")

    def _gen_TACJumpIfNot(self, instr: TACJumpIfNot) -> None:
        cond = self._load_spill(instr.condition, "$at")
        self.emit(f"    beq   {cond}, $zero, {instr.label}")

    def _gen_TACLoadImm(self, instr: TACLoadImm) -> None:
        dest = self.resolve(instr.result)
        if dest == "$at":
            self.emit(f"    li    $at, {instr.value}")
            self._store_spill(instr.result, "$at")
        else:
            self.emit(f"    li    {dest}, {instr.value}")

    def _gen_TACLoadStr(self, instr: TACLoadStr) -> None:
        dest = self.resolve(instr.result)
        if dest == "$at":
            self.emit(f"    la    $at, {instr.label}")
            self._store_spill(instr.result, "$at")
        else:
            self.emit(f"    la    {dest}, {instr.label}")

    def _gen_TACCopy(self, instr: TACCopy) -> None:
        # Handle __arg* sources — arguments pushed by the caller sit
        # just above our frame pointer: __arg0 at 0($fp), __arg1 at 4($fp), etc.
        if instr.source.startswith("__arg"):
            arg_idx = int(instr.source[len("__arg"):])
            arg_offset = arg_idx * 4
            dest = self.resolve(instr.result)
            if dest == "$at":
                self.emit(f"    lw    $at, {arg_offset}($fp)")
                self._store_spill(instr.result, "$at")
            else:
                self.emit(f"    lw    {dest}, {arg_offset}($fp)")
            return

        src = self._load_spill(instr.source, "$v1")
        dest = self.resolve(instr.result)
        if dest == "$at":
            self.emit(f"    move  $at, {src}")
            self._store_spill(instr.result, "$at")
        else:
            self.emit(f"    move  {dest}, {src}")

    def _gen_TACBinOp(self, instr: TACBinOp) -> None:
        left = self._load_spill(instr.left, "$v1")
        # Use $at as second scratch if left already took $v1
        right_scratch = "$at" if left == "$v1" else "$v1"
        right = self._load_spill(instr.right, right_scratch)
        dest = self.resolve(instr.result)
        # If dest is spilled, use $at for the result
        actual_dest = "$at" if self._spill_offset(instr.result) >= 0 else dest

        op = instr.op
        if op == "+":
            self.emit(f"    add   {actual_dest}, {left}, {right}")
        elif op == "-":
            self.emit(f"    sub   {actual_dest}, {left}, {right}")
        elif op == "*":
            self.emit(f"    mul   {actual_dest}, {left}, {right}")
        elif op == "/":
            self.emit(f"    div   {left}, {right}")
            self.emit(f"    mflo  {actual_dest}")
        elif op == "%":
            self.emit(f"    div   {left}, {right}")
            self.emit(f"    mfhi  {actual_dest}")
        elif op == "<":
            self.emit(f"    slt   {actual_dest}, {left}, {right}")
        elif op == ">":
            self.emit(f"    slt   {actual_dest}, {right}, {left}")
        elif op == "<=":
            self.emit(f"    slt   {actual_dest}, {right}, {left}")
            self.emit(f"    xori  {actual_dest}, {actual_dest}, 1")
        elif op == ">=":
            self.emit(f"    slt   {actual_dest}, {left}, {right}")
            self.emit(f"    xori  {actual_dest}, {actual_dest}, 1")
        elif op == "==":
            self.emit(f"    sub   {actual_dest}, {left}, {right}")
            self.emit(f"    sltu  {actual_dest}, $zero, {actual_dest}")
            self.emit(f"    xori  {actual_dest}, {actual_dest}, 1")
        elif op == "!=":
            self.emit(f"    sub   {actual_dest}, {left}, {right}")
            self.emit(f"    sltu  {actual_dest}, $zero, {actual_dest}")
        else:
            self.emit_comment(f"unknown binop: {op}")

        if self._spill_offset(instr.result) >= 0:
            self._store_spill(instr.result, actual_dest)

    def _gen_TACUnaryOp(self, instr: TACUnaryOp) -> None:
        src = self._load_spill(instr.operand, "$v1")
        dest = self.resolve(instr.result)
        actual_dest = "$at" if self._spill_offset(instr.result) >= 0 else dest

        if instr.op == "-":
            self.emit(f"    subu  {actual_dest}, $zero, {src}")
        elif instr.op == "!":
            self.emit(f"    sltu  {actual_dest}, $zero, {src}")
            self.emit(f"    xori  {actual_dest}, {actual_dest}, 1")

        if self._spill_offset(instr.result) >= 0:
            self._store_spill(instr.result, actual_dest)

    def _gen_TACArrayRead(self, instr: TACArrayRead) -> None:
        """result = array[index]  — array is $fp-relative."""
        # Get the register or offset for the array base
        array_reg = self.resolve(instr.array)
        idx = self._load_spill(instr.index, "$v1")
        dest = self.resolve(instr.result)
        actual_dest = "$at" if self._spill_offset(instr.result) >= 0 else dest

        # Compute address: base + index * 4
        # Array base is the $s register holding the address, or $fp-relative
        self.emit(f"    sll   $a1, {idx}, 2")          # a1 = index * 4

        if self._spill_offset(instr.array) >= 0:
            # Array is spilled — its base address is at the spill offset
            arr_off = self._spill_offset(instr.array)
            self.emit(f"    addiu $a1, $a1, -{arr_off}")
            self.emit(f"    add   $a1, $fp, $a1")
        else:
            # The array base register already points to array start on stack
            # We need to compute fp - offset + index*4
            # But for simplicity, the $s register holds the logical name.
            # Arrays are stack-allocated: use $fp - array_offset as base
            self.emit(f"    add   $a1, {array_reg}, $a1")

        self.emit(f"    lw    {actual_dest}, 0($a1)")

        if self._spill_offset(instr.result) >= 0:
            self._store_spill(instr.result, actual_dest)

    def _gen_TACArrayWrite(self, instr: TACArrayWrite) -> None:
        """array[index] = source"""
        array_reg = self.resolve(instr.array)
        idx = self._load_spill(instr.index, "$v1")
        src = self._load_spill(instr.source, "$a1")

        self.emit(f"    sll   $a2, {idx}, 2")

        if self._spill_offset(instr.array) >= 0:
            arr_off = self._spill_offset(instr.array)
            self.emit(f"    addiu $a2, $a2, -{arr_off}")
            self.emit(f"    add   $a2, $fp, $a2")
        else:
            self.emit(f"    add   $a2, {array_reg}, $a2")

        self.emit(f"    sw    {src}, 0($a2)")

    def _gen_TACParam(self, instr: TACParam) -> None:
        """Collect params; they'll be pushed when TACCall is processed."""
        self._pending_params.append(instr.name)

    def _gen_TACCall(self, instr: TACCall) -> None:
        """Emit argument passing + call instruction."""
        func = instr.function

        # ── Built-in I/O functions → inline syscall ───────────────
        if func in BUILTIN_SYSCALLS:
            info = BUILTIN_SYSCALLS[func]
            sc = info["syscall"]

            if func == "print_string" and self._pending_params:
                # The argument is a TACLoadStr temp → its register holds the label address
                arg_name = self._pending_params.pop(0)
                arg = self._load_spill(arg_name, "$a0")
                if arg != "$a0":
                    self.emit(f"    move  $a0, {arg}")
                self.emit(f"    li    $v0, {sc}")
                self.emit(f"    syscall")

            elif info["arg_reg"] and self._pending_params:
                arg_name = self._pending_params.pop(0)
                arg = self._load_spill(arg_name, "$a0")
                if arg != "$a0":
                    self.emit(f"    move  $a0, {arg}")
                self.emit(f"    li    $v0, {sc}")
                self.emit(f"    syscall")

            else:
                # No args (read_int, exit)
                self._pending_params.clear()
                self.emit(f"    li    $v0, {sc}")
                self.emit(f"    syscall")

            # read_int returns in $v0 — copy to result
            if info["has_result"] and instr.result:
                dest = self.resolve(instr.result)
                if self._spill_offset(instr.result) >= 0:
                    self._store_spill(instr.result, "$v0")
                elif dest != "$v0":
                    self.emit(f"    move  {dest}, $v0")

            return

        # ── User-defined function call ────────────────────────────
        # Save caller-save $t registers that are live
        # (simplified: save all mapped $t regs)
        saved_t = []
        for name, reg in self.reg_map.items():
            if reg.startswith("$t"):
                saved_t.append(reg)
        saved_t = sorted(set(saved_t))

        for tr in saved_t:
            self.emit(f"    subu  $sp, $sp, 4")
            self.emit(f"    sw    {tr}, 0($sp)")

        # Push arguments onto stack (right to left for C convention)
        args = list(self._pending_params)
        self._pending_params.clear()

        for arg_name in reversed(args):
            arg = self._load_spill(arg_name, "$v1")
            self.emit(f"    subu  $sp, $sp, 4")
            self.emit(f"    sw    {arg}, 0($sp)")

        # Jump and link
        self.emit(f"    jal   {func}")

        # Clean up argument space
        if args:
            self.emit(f"    addiu $sp, $sp, {4 * len(args)}")

        # Restore caller-save $t registers
        for tr in reversed(saved_t):
            self.emit(f"    lw    {tr}, 0($sp)")
            self.emit(f"    addiu $sp, $sp, 4")

        # Capture return value
        if instr.result:
            dest = self.resolve(instr.result)
            if self._spill_offset(instr.result) >= 0:
                self._store_spill(instr.result, "$v0")
            elif dest != "$v0":
                self.emit(f"    move  {dest}, $v0")

    def _gen_TACReturn(self, instr: TACReturn) -> None:
        """move return value to $v0, emit epilogue, jr $ra.
        For main(), emit exit syscall instead."""
        if instr.value:
            val = self._load_spill(instr.value, "$v0")
            if val != "$v0":
                self.emit(f"    move  $v0, {val}")

        if self.current_function == "main":
            # main returns via exit syscall
            self.emit(f"    li    $v0, 10")
            self.emit(f"    syscall")
        else:
            self._emit_epilogue()
            self.emit(f"    jr    $ra")

    # ── utility ───────────────────────────────────────────────────

    @staticmethod
    def _escape_for_mips(s: str) -> str:
        """Convert Python string to MIPS .asciiz-compatible escape sequences."""
        result = []
        for ch in s:
            if ch == "\n":
                result.append("\\n")
            elif ch == "\t":
                result.append("\\t")
            elif ch == "\0":
                result.append("\\0")
            elif ch == "\\":
                result.append("\\\\")
            elif ch == "\"":
                result.append('\\"')
            else:
                result.append(ch)
        return "".join(result)


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def generate_code(
    ir_output: Dict,
    reg_maps: Dict[str, Dict],
    frame_infos: Dict[str, Dict],
    semantic_frame_sizes: Dict[str, int],
    var_info: Dict[str, Dict[str, Dict]] = None,
) -> str:
    """Generate the complete MIPS assembly string."""
    return CodeGenerator().generate(
        ir_output, reg_maps, frame_infos, semantic_frame_sizes, var_info
    )
