.data
    # (no string literals)

.text
.globl main

main:
    subu  $sp, $sp, 80
    sw    $ra, 76($sp)
    sw    $fp, 72($sp)
    addiu $fp, $sp, 80
    sw    $s0, 68($sp)
    sw    $s1, 64($sp)
    addiu $s1, $fp, -20
    li    $t0, 0
    move  $s0, $t0
L0:
    li    $t1, 5
    slt   $t2, $s0, $t1
    beq   $t2, $zero, L1
    li    $t3, 2
    mul   $t4, $s0, $t3
    sll   $a2, $s0, 2
    add   $a2, $s1, $a2
    sw    $t4, 0($a2)
    move  $t5, $s0
    li    $t6, 1
    add   $t7, $s0, $t6
    move  $s0, $t7
    j     L0
L1:
    li    $t8, 0
    move  $s0, $t8
L2:
    li    $t9, 5
    slt   $at, $s0, $t9
    sw    $at, -44($fp)
    lw    $at, -44($fp)
    beq   $at, $zero, L3
    sll   $a1, $s0, 2
    add   $a1, $s1, $a1
    lw    $at, 0($a1)
    sw    $at, -48($fp)
    lw    $a0, -48($fp)
    li    $v0, 1
    syscall
    li    $at, 10
    sw    $at, -56($fp)
    lw    $a0, -56($fp)
    li    $v0, 11
    syscall
    move  $at, $s0
    sw    $at, -64($fp)
    li    $at, 1
    sw    $at, -68($fp)
    lw    $v1, -68($fp)
    add   $at, $s0, $v1
    sw    $at, -72($fp)
    lw    $v1, -72($fp)
    move  $s0, $v1
    j     L2
L3:
    li    $at, 0
    sw    $at, -76($fp)
    lw    $v0, -76($fp)
    li    $v0, 10
    syscall

