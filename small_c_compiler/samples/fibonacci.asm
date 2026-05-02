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
    li    $t0, 0
    move  $s0, $t0
L2:
    li    $t1, 10
    slt   $t2, $s0, $t1
    beq   $t2, $zero, L3
    subu  $sp, $sp, 4
    sw    $t0, 0($sp)
    subu  $sp, $sp, 4
    sw    $t1, 0($sp)
    subu  $sp, $sp, 4
    sw    $t2, 0($sp)
    subu  $sp, $sp, 4
    sw    $t3, 0($sp)
    subu  $sp, $sp, 4
    sw    $t4, 0($sp)
    subu  $sp, $sp, 4
    sw    $t5, 0($sp)
    subu  $sp, $sp, 4
    sw    $t6, 0($sp)
    subu  $sp, $sp, 4
    sw    $t7, 0($sp)
    subu  $sp, $sp, 4
    sw    $t8, 0($sp)
    subu  $sp, $sp, 4
    sw    $t9, 0($sp)
    subu  $sp, $sp, 4
    sw    $s0, 0($sp)
    jal   fibonacci
    addiu $sp, $sp, 4
    lw    $t9, 0($sp)
    addiu $sp, $sp, 4
    lw    $t8, 0($sp)
    addiu $sp, $sp, 4
    lw    $t7, 0($sp)
    addiu $sp, $sp, 4
    lw    $t6, 0($sp)
    addiu $sp, $sp, 4
    lw    $t5, 0($sp)
    addiu $sp, $sp, 4
    lw    $t4, 0($sp)
    addiu $sp, $sp, 4
    lw    $t3, 0($sp)
    addiu $sp, $sp, 4
    lw    $t2, 0($sp)
    addiu $sp, $sp, 4
    lw    $t1, 0($sp)
    addiu $sp, $sp, 4
    lw    $t0, 0($sp)
    addiu $sp, $sp, 4
    move  $t3, $v0
    move  $a0, $t3
    li    $v0, 1
    syscall
    li    $t5, 10
    move  $a0, $t5
    li    $v0, 11
    syscall
    move  $t7, $s0
    li    $t8, 1
    add   $t9, $s0, $t8
    move  $s0, $t9
    j     L2
L3:
    li    $at, 0
    sw    $at, -20($fp)
    lw    $v0, -20($fp)
    li    $v0, 10
    syscall

fibonacci:
    subu  $sp, $sp, 16
    sw    $ra, 12($sp)
    sw    $fp, 8($sp)
    addiu $fp, $sp, 16
    sw    $s0, 4($sp)
    lw    $s0, 0($fp)
    li    $t0, 2
    slt   $t1, $s0, $t0
    beq   $t1, $zero, L1
    move  $v0, $s0
    lw    $s0, 4($sp)
    lw    $ra, 12($sp)
    lw    $fp, 8($sp)
    addiu $sp, $sp, 16
    jr    $ra
    j     L0
L1:
L0:
    li    $t2, 1
    sub   $t3, $s0, $t2
    subu  $sp, $sp, 4
    sw    $t0, 0($sp)
    subu  $sp, $sp, 4
    sw    $t1, 0($sp)
    subu  $sp, $sp, 4
    sw    $t2, 0($sp)
    subu  $sp, $sp, 4
    sw    $t3, 0($sp)
    subu  $sp, $sp, 4
    sw    $t4, 0($sp)
    subu  $sp, $sp, 4
    sw    $t5, 0($sp)
    subu  $sp, $sp, 4
    sw    $t6, 0($sp)
    subu  $sp, $sp, 4
    sw    $t7, 0($sp)
    subu  $sp, $sp, 4
    sw    $t8, 0($sp)
    subu  $sp, $sp, 4
    sw    $t3, 0($sp)
    jal   fibonacci
    addiu $sp, $sp, 4
    lw    $t8, 0($sp)
    addiu $sp, $sp, 4
    lw    $t7, 0($sp)
    addiu $sp, $sp, 4
    lw    $t6, 0($sp)
    addiu $sp, $sp, 4
    lw    $t5, 0($sp)
    addiu $sp, $sp, 4
    lw    $t4, 0($sp)
    addiu $sp, $sp, 4
    lw    $t3, 0($sp)
    addiu $sp, $sp, 4
    lw    $t2, 0($sp)
    addiu $sp, $sp, 4
    lw    $t1, 0($sp)
    addiu $sp, $sp, 4
    lw    $t0, 0($sp)
    addiu $sp, $sp, 4
    move  $t4, $v0
    li    $t5, 2
    sub   $t6, $s0, $t5
    subu  $sp, $sp, 4
    sw    $t0, 0($sp)
    subu  $sp, $sp, 4
    sw    $t1, 0($sp)
    subu  $sp, $sp, 4
    sw    $t2, 0($sp)
    subu  $sp, $sp, 4
    sw    $t3, 0($sp)
    subu  $sp, $sp, 4
    sw    $t4, 0($sp)
    subu  $sp, $sp, 4
    sw    $t5, 0($sp)
    subu  $sp, $sp, 4
    sw    $t6, 0($sp)
    subu  $sp, $sp, 4
    sw    $t7, 0($sp)
    subu  $sp, $sp, 4
    sw    $t8, 0($sp)
    subu  $sp, $sp, 4
    sw    $t6, 0($sp)
    jal   fibonacci
    addiu $sp, $sp, 4
    lw    $t8, 0($sp)
    addiu $sp, $sp, 4
    lw    $t7, 0($sp)
    addiu $sp, $sp, 4
    lw    $t6, 0($sp)
    addiu $sp, $sp, 4
    lw    $t5, 0($sp)
    addiu $sp, $sp, 4
    lw    $t4, 0($sp)
    addiu $sp, $sp, 4
    lw    $t3, 0($sp)
    addiu $sp, $sp, 4
    lw    $t2, 0($sp)
    addiu $sp, $sp, 4
    lw    $t1, 0($sp)
    addiu $sp, $sp, 4
    lw    $t0, 0($sp)
    addiu $sp, $sp, 4
    move  $t7, $v0
    add   $t8, $t4, $t7
    move  $v0, $t8
    lw    $s0, 4($sp)
    lw    $ra, 12($sp)
    lw    $fp, 8($sp)
    addiu $sp, $sp, 16
    jr    $ra

