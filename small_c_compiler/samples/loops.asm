.data
    # (no string literals)

.text
.globl main

main:
    subu  $sp, $sp, 24
    sw    $ra, 20($sp)
    sw    $fp, 16($sp)
    addiu $fp, $sp, 24
    sw    $s0, 12($sp)
    sw    $s1, 8($sp)
    li    $t0, 0
    move  $s0, $t0
    li    $t1, 1
    move  $s1, $t1
L0:
    li    $t2, 10
    slt   $t3, $t2, $s1
    xori  $t3, $t3, 1
    beq   $t3, $zero, L1
    add   $t4, $s0, $s1
    move  $s0, $t4
    move  $t5, $s1
    li    $t6, 1
    add   $t7, $s1, $t6
    move  $s1, $t7
    j     L0
L1:
    move  $a0, $s0
    li    $v0, 1
    syscall
    li    $t9, 0
    move  $v0, $t9
    li    $v0, 10
    syscall

