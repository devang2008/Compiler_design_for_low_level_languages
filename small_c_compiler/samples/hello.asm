.data
str0:  .asciiz  "Hello, World!\n"

.text
.globl main

main:
    subu  $sp, $sp, 8
    sw    $ra, 4($sp)
    sw    $fp, 0($sp)
    addiu $fp, $sp, 8
    la    $t0, str0
    move  $a0, $t0
    li    $v0, 4
    syscall
    li    $t2, 0
    move  $v0, $t2
    li    $v0, 10
    syscall

