"""
register_allocator.py — TAC Variable → MIPS Register Mapping
==============================================================
Maps every TAC temporary and user variable to a physical MIPS
register or a spill slot on the stack.

Strategy:
  - TAC temps (t0, t1, …) → $t0-$t9  (spill if >10)
  - User vars              → $s0-$s7  (spill if >8)
  - Frame size computed from saved regs + locals + spills

Pipeline position: TAC → [Register Allocator] → register map → Code Generator
"""

from __future__ import annotations
from typing import Any, Dict, List, Set
import re

from ir_generator import (
    TACBinOp, TACUnaryOp, TACCopy, TACLoadImm, TACLoadStr,
    TACArrayRead, TACArrayWrite, TACLabel, TACJump,
    TACJumpIf, TACJumpIfNot, TACParam, TACCall, TACReturn,
)


# Pattern that identifies a compiler-generated temporary (t0, t1, …)
_TEMP_RE = re.compile(r"^t\d+$")

# Registers available for allocation
TEMP_REGS = [f"$t{i}" for i in range(10)]   # $t0-$t9
SAVED_REGS = [f"$s{i}" for i in range(8)]   # $s0-$s7


class RegisterAllocator:
    """Maps TAC names to MIPS registers or stack spill offsets.

    Usage:
        ra = RegisterAllocator()
        info = ra.allocate("main", tac_list, frame_size_from_semantic)
    """

    def allocate(
        self,
        function_name: str,
        tac_instructions: List[Any],
        semantic_frame_size: int,
    ) -> Dict:
        """Allocate registers for one function.

        Args:
            function_name: for debugging
            tac_instructions: list of TAC objects
            semantic_frame_size: bytes of locals computed by semantic pass

        Returns:
            {
                "reg_map":     { tac_name: "$t0" | "$s0" | "spill:offset", ... },
                "frame_size":  int (total bytes, multiple of 8),
                "s_regs_used": ["$s0", "$s1", ...],
                "spills":      { name: offset, ... },
            }
        """
        # Step 1: collect all variable names referenced in the TAC
        temps: List[str] = []
        user_vars: List[str] = []
        seen: Set[str] = set()

        for instr in tac_instructions:
            for name in self._names_in(instr):
                if name not in seen:
                    seen.add(name)
                    if _TEMP_RE.match(name):
                        temps.append(name)
                    elif not name.startswith("__arg"):
                        user_vars.append(name)

        # Step 2: assign temps to $t registers
        reg_map: Dict[str, str] = {}
        spills: Dict[str, int] = {}

        for i, t in enumerate(temps):
            if i < len(TEMP_REGS):
                reg_map[t] = TEMP_REGS[i]
            else:
                # Spill to stack — offset computed later
                reg_map[t] = f"spill"
                spills[t] = 0

        # Step 3: assign user vars to $s registers
        s_regs_used: List[str] = []
        s_idx = 0
        for v in user_vars:
            if s_idx < len(SAVED_REGS):
                reg_map[v] = SAVED_REGS[s_idx]
                s_regs_used.append(SAVED_REGS[s_idx])
                s_idx += 1
            else:
                reg_map[v] = f"spill"
                spills[v] = 0

        # Step 4: assign __arg* names to special handling
        # (code generator will handle loading from stack)
        for name in seen:
            if name.startswith("__arg"):
                reg_map[name] = name  # pass-through marker

        # Step 5: compute frame size
        saved_ra = 4
        saved_fp = 4
        saved_s = 4 * len(s_regs_used)
        local_bytes = semantic_frame_size
        spill_bytes = 4 * len(spills)

        raw = saved_ra + saved_fp + saved_s + local_bytes + spill_bytes
        # Round up to nearest multiple of 8 for alignment, minimum 8
        frame_size = max(((raw + 7) // 8) * 8, 8)

        # Assign concrete negative offsets for spilled vars
        # Spills live below the local variables in the stack frame
        offset = saved_ra + saved_fp + saved_s + local_bytes
        for name in spills:
            offset += 4
            spills[name] = offset
            reg_map[name] = f"spill:{offset}"

        return {
            "reg_map": reg_map,
            "frame_size": frame_size,
            "s_regs_used": s_regs_used,
            "spills": spills,
        }

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _names_in(instr) -> List[str]:
        """Extract all TAC variable/temp names from an instruction."""
        names: List[str] = []

        if isinstance(instr, TACBinOp):
            names.extend([instr.result, instr.left, instr.right])
        elif isinstance(instr, TACUnaryOp):
            names.extend([instr.result, instr.operand])
        elif isinstance(instr, TACCopy):
            names.extend([instr.result, instr.source])
        elif isinstance(instr, TACLoadImm):
            names.append(instr.result)
        elif isinstance(instr, TACLoadStr):
            names.append(instr.result)
        elif isinstance(instr, TACArrayRead):
            names.extend([instr.result, instr.index])
            # array name is a user var
            names.append(instr.array)
        elif isinstance(instr, TACArrayWrite):
            names.extend([instr.source, instr.index])
            names.append(instr.array)
        elif isinstance(instr, TACParam):
            names.append(instr.name)
        elif isinstance(instr, TACCall):
            if instr.result:
                names.append(instr.result)
        elif isinstance(instr, TACReturn):
            if instr.value:
                names.append(instr.value)
        elif isinstance(instr, TACJumpIf):
            names.append(instr.condition)
        elif isinstance(instr, TACJumpIfNot):
            names.append(instr.condition)
        # TACLabel, TACJump have no variable names

        return names


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def allocate_registers(
    function_name: str,
    tac_instructions: List[Any],
    semantic_frame_size: int,
) -> Dict:
    """Convenience wrapper around RegisterAllocator.allocate()."""
    return RegisterAllocator().allocate(
        function_name, tac_instructions, semantic_frame_size
    )
