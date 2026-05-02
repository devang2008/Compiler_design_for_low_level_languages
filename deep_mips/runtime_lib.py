"""
runtime_lib.py - Runtime library for MIPS Inference Engine
"""
from __future__ import annotations
from typing import List

class RuntimeLibrary:
    def __init__(self):
        self.output: List[str] = []

    def emit(self, instruction: str):
        self.output.append(instruction)

    def generate_all(self) -> str:
        self.emit_memcpy()
        self.emit_vec_max()
        self.emit_vec_sum()
        self.emit_fpu_vec_max()
        self.emit_fpu_vec_sum()
        self.emit_fpu_exp()
        return "\n".join(self.output)

    def emit_memcpy(self):
        self.emit("memcpy:")
        self.emit("    li    $t0, 0")
        self.emit("memcpy_loop:")
        self.emit("    bge   $t0, $a2, memcpy_end")
        self.emit("    add   $t1, $a1, $t0")
        self.emit("    lw    $t2, 0($t1)")
        self.emit("    add   $t1, $a0, $t0")
        self.emit("    sw    $t2, 0($t1)")
        self.emit("    addi  $t0, $t0, 4")
        self.emit("    j     memcpy_loop")
        self.emit("memcpy_end:")
        self.emit("    jr    $ra")
        self.emit("")

    def emit_vec_max(self):
        self.emit("vec_max:")
        self.emit("    lw    $v0, 0($a0)")
        self.emit("    li    $t0, 1")
        self.emit("vec_max_loop:")
        self.emit("    bge   $t0, $a1, vec_max_end")
        self.emit("    sll   $t1, $t0, 2")
        self.emit("    add   $t1, $a0, $t1")
        self.emit("    lw    $t2, 0($t1)")
        self.emit("    slt   $t3, $v0, $t2")
        self.emit("    beq   $t3, $zero, vec_max_skip")
        self.emit("    move  $v0, $t2")
        self.emit("vec_max_skip:")
        self.emit("    addi  $t0, $t0, 1")
        self.emit("    j     vec_max_loop")
        self.emit("vec_max_end:")
        self.emit("    jr    $ra")
        self.emit("")

    def emit_vec_sum(self):
        self.emit("vec_sum:")
        self.emit("    li    $v0, 0")
        self.emit("    li    $t0, 0")
        self.emit("vec_sum_loop:")
        self.emit("    bge   $t0, $a1, vec_sum_end")
        self.emit("    sll   $t1, $t0, 2")
        self.emit("    add   $t1, $a0, $t1")
        self.emit("    lw    $t2, 0($t1)")
        self.emit("    add   $v0, $v0, $t2")
        self.emit("    addi  $t0, $t0, 1")
        self.emit("    j     vec_sum_loop")
        self.emit("vec_sum_end:")
        self.emit("    jr    $ra")
        self.emit("")

    def emit_fpu_vec_max(self):
        self.emit("fpu_vec_max:")
        self.emit("    lwc1  $f0, 0($a0)")
        self.emit("    li    $t0, 1")
        self.emit("fpu_vec_max_loop:")
        self.emit("    bge   $t0, $a1, fpu_vec_max_end")
        self.emit("    sll   $t1, $t0, 2")
        self.emit("    add   $t1, $a0, $t1")
        self.emit("    lwc1  $f2, 0($t1)")
        self.emit("    c.lt.s $f0, $f2")
        self.emit("    bc1f  fpu_vec_max_skip")
        self.emit("    mov.s $f0, $f2")
        self.emit("fpu_vec_max_skip:")
        self.emit("    addi  $t0, $t0, 1")
        self.emit("    j     fpu_vec_max_loop")
        self.emit("fpu_vec_max_end:")
        self.emit("    jr    $ra")
        self.emit("")

    def emit_fpu_vec_sum(self):
        self.emit("fpu_vec_sum:")
        self.emit("    lwc1  $f0, fpu_zero")
        self.emit("    li    $t0, 0")
        self.emit("fpu_vec_sum_loop:")
        self.emit("    bge   $t0, $a1, fpu_vec_sum_end")
        self.emit("    sll   $t1, $t0, 2")
        self.emit("    add   $t1, $a0, $t1")
        self.emit("    lwc1  $f2, 0($t1)")
        self.emit("    add.s $f0, $f0, $f2")
        self.emit("    addi  $t0, $t0, 1")
        self.emit("    j     fpu_vec_sum_loop")
        self.emit("fpu_vec_sum_end:")
        self.emit("    jr    $ra")
        self.emit("")

    def emit_fpu_exp(self):
        self.emit("    # e^x approx (Taylor): 1 + x + x^2/2 + x^3/6 + x^4/24")
        self.emit("fpu_exp:")
        self.emit("    lwc1  $f2, exp_clamp_neg")
        self.emit("    c.lt.s $f0, $f2")
        self.emit("    bc1f  fpu_exp_check_pos")
        self.emit("    lwc1  $f0, fpu_zero")
        self.emit("    jr    $ra")
        self.emit("fpu_exp_check_pos:")
        self.emit("    lwc1  $f2, exp_clamp_pos")
        self.emit("    c.lt.s $f2, $f0")
        self.emit("    bc1f  fpu_exp_calc")
        self.emit("    mov.s $f0, $f2") # just clamp it
        self.emit("fpu_exp_calc:")
        self.emit("    mov.s $f2, $f0") # f2 = x
        self.emit("    lwc1  $f4, exp_c4")
        self.emit("    mul.s $f6, $f4, $f2") # c4 * x
        self.emit("    lwc1  $f4, exp_c3")
        self.emit("    add.s $f6, $f6, $f4") # c3 + c4*x
        self.emit("    mul.s $f6, $f6, $f2") # x*(c3 + c4*x)
        self.emit("    lwc1  $f4, exp_c2")
        self.emit("    add.s $f6, $f6, $f4") # c2 + ...
        self.emit("    mul.s $f6, $f6, $f2")
        self.emit("    lwc1  $f4, exp_c1")
        self.emit("    add.s $f6, $f6, $f4")
        self.emit("    mul.s $f6, $f6, $f2")
        self.emit("    lwc1  $f4, exp_c0")
        self.emit("    add.s $f0, $f6, $f4")
        self.emit("    jr    $ra")
        self.emit("")
