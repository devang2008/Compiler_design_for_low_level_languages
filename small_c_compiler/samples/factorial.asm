.data
    # (no string literals)

.text
.globl main

main:
    subu  $sp, $sp, 16
    sw    $ra, 12($sp)
    sw    $fp, 8($sp)
    addiu $fp, $sp, 16
    sw    $s0, 4($sp)
    li    $t0, 5
    subu  $sp, $sp, 4
    sw    $t0, 0($sp)
    subu  $sp, $sp, 4
    sw    $t1, 0($sp)
    subu  $sp, $sp, 4
    sw    $t2, 0($sp)
    subu  $sp, $sp, 4
    sw    $t3, 0($sp)
    subu  $sp, $sp, 4
    sw    $t0, 0($sp)
    jal   factorial
    addiu $sp, $sp, 4
    lw    $t3, 0($sp)
    addiu $sp, $sp, 4
    lw    $t2, 0($sp)
    addiu $sp, $sp, 4
    lw    $t1, 0($sp)
    addiu $sp, $sp, 4
    lw    $t0, 0($sp)
    addiu $sp, $sp, 4
    move  $t1, $v0
    move  $s0, $t1
    move  $a0, $s0
    li    $v0, 1
    syscall
    li    $t3, 0
    move  $v0, $t3
    li    $v0, 10
    syscall

factorial:
    subu  $sp, $sp, 16
    sw    $ra, 12($sp)
    sw    $fp, 8($sp)
    addiu $fp, $sp, 16
    sw    $s0, 4($sp)
    lw    $s0, 0($fp)
    li    $t0, 1
    slt   $t1, $t0, $s0
    xori  $t1, $t1, 1
    beq   $t1, $zero, L1
    li    $t2, 1
    move  $v0, $t2
    lw    $s0, 4($sp)
    lw    $ra, 12($sp)
    lw    $fp, 8($sp)
    addiu $sp, $sp, 16
    jr    $ra
    j     L0
L1:
L0:
    li    $t3, 1
    sub   $t4, $s0, $t3
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
    sw    $t4, 0($sp)
    jal   factorial
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
    move  $t5, $v0
    mul   $t6, $s0, $t5
    move  $v0, $t6
    lw    $s0, 4($sp)
    lw    $ra, 12($sp)
    lw    $fp, 8($sp)
    addiu $sp, $sp, 16
    jr    $ra

