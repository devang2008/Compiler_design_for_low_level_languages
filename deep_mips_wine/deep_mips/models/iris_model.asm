.data
sigmoid_lut:    .word  0, 5, 12, 25, 50, 88, 128, 168, 206, 231, 244, 249, 252, 254, 255, 256
layer_0_weights:        .float  -0.00000000, 0.65461297, 0.91123301, 0.16462853, -0.00013964, -0.00013977, -0.06987828, 0.79946300
                        .float  -0.00000000, 0.74426648, -1.58569406, 0.38963158, 0.00008933, -0.00001064, -0.02954311, 0.06587052
                        .float  -0.00000000, -0.32579134, 0.91737521, -0.09875361, 0.00000000, -0.00025020, -0.01597332, -0.56135937
                        .float  0.00000000, 0.02093286, 0.64590704, 0.20111963, -0.00000000, -0.00274970, -0.00016060, -1.20919090
layer_1_weights:        .float  -0.00070313, -0.00000000, -0.00500330, 0.79627124, -0.37936287, -0.55030480, -1.47966950, 0.23010149
                        .float  1.37582271, -0.01661015, 0.48125795, 0.04885516, 0.00306075, 0.00110875, -0.00000000, 0.00211145
                        .float  -0.00166849, -0.00004746, -0.00407051, -0.00000003, -0.00000000, 1.06892520, 1.40712027, -1.52977873
layer_0_biases:         .float  -0.61510994, 0.81382777, -0.52583469, 0.40769709, -0.27631786, -0.56897755, 0.23529997, 0.78503735
layer_1_biases:         .float  0.02541565, 0.23093718, -1.33205830
input_buffer:
    .float 6.1, 2.8, 4.7, 1.2
act_buffer_a:   .space  32
act_buffer_b:   .space  32
fpu_zero:       .float  0.0
fpu_one:        .float  1.0
fpu_ln2:        .float  0.693147
fpu_inv_ln2:    .float  1.442695
exp_c0:         .float  1.0
exp_c1:         .float  1.0
exp_c2:         .float  0.5
exp_c3:         .float  0.166667
exp_c4:         .float  0.041667
exp_clamp_neg:  .float  -3.0
exp_clamp_pos:  .float  88.0

.text
.globl main

main:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    addiu $fp, $sp, 32

    # Run inference
    la    $a0, input_buffer
    la    $a1, act_buffer_a
    la    $a2, act_buffer_b

    move  $s0, $a0
    move  $s1, $a1
    jal   layer_0_fused_mbr
    move  $s0, $a1
    move  $s1, $a0
    jal   layer_1_fused_mb
    move  $s0, $a0
    move  $s1, $a1
    jal   layer_1_act

    move  $a0, $s0
    jal   print_output

    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    addiu $sp, $sp, 32
    li    $v0, 10
    syscall

layer_0_fused_mbr:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    sw    $s0, 20($sp)
    sw    $s1, 16($sp)
    sw    $s2, 12($sp)
    sw    $s3, 8($sp)
    sw    $s4, 4($sp)
    addiu $fp, $sp, 32

    # MatMul (standard loops): 8 outputs x 4 inputs
    la    $t8, layer_0_weights
    la    $t9, layer_0_biases
    move  $t7, $s0
    move  $t6, $s1
    li    $s2, 0
layer_0_fused_mbr_oloop:
    bge   $s2, 8, layer_0_fused_mbr_oend
    sll   $t0, $s2, 2
    add   $t0, $t9, $t0
    lwc1  $f0, 0($t0)
    li    $s3, 0
layer_0_fused_mbr_iloop:
    bge   $s3, 4, layer_0_fused_mbr_iend
    sll   $t1, $s3, 2
    add   $t1, $t7, $t1
    lwc1  $f2, 0($t1)
    mul   $t2, $s3, 8
    add   $t2, $t2, $s2
    sll   $t2, $t2, 2
    add   $t2, $t8, $t2
    lwc1  $f4, 0($t2)
    mul.s $f6, $f2, $f4
    add.s $f0, $f0, $f6
    addi  $s3, $s3, 1
    j     layer_0_fused_mbr_iloop
layer_0_fused_mbr_iend:
    lwc1  $f2, fpu_zero
    c.lt.s $f0, $f2
    bc1f  layer_0_fused_mbr_relu_skip
    mov.s $f0, $f2
layer_0_fused_mbr_relu_skip:
    sll   $t1, $s2, 2
    add   $t1, $t6, $t1
    swc1  $f0, 0($t1)
    addi  $s2, $s2, 1
    j     layer_0_fused_mbr_oloop
layer_0_fused_mbr_oend:


    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    lw    $s0, 20($sp)
    lw    $s1, 16($sp)
    lw    $s2, 12($sp)
    lw    $s3, 8($sp)
    lw    $s4, 4($sp)
    addiu $sp, $sp, 32
    jr    $ra

layer_1_fused_mb:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    sw    $s0, 20($sp)
    sw    $s1, 16($sp)
    sw    $s2, 12($sp)
    sw    $s3, 8($sp)
    sw    $s4, 4($sp)
    addiu $fp, $sp, 32

    # MatMul (standard loops): 3 outputs x 8 inputs
    la    $t8, layer_1_weights
    la    $t9, layer_1_biases
    move  $t7, $s0
    move  $t6, $s1
    li    $s2, 0
layer_1_fused_mb_oloop:
    bge   $s2, 3, layer_1_fused_mb_oend
    sll   $t0, $s2, 2
    add   $t0, $t9, $t0
    lwc1  $f0, 0($t0)
    li    $s3, 0
layer_1_fused_mb_iloop:
    bge   $s3, 8, layer_1_fused_mb_iend
    sll   $t1, $s3, 2
    add   $t1, $t7, $t1
    lwc1  $f2, 0($t1)
    mul   $t2, $s3, 3
    add   $t2, $t2, $s2
    sll   $t2, $t2, 2
    add   $t2, $t8, $t2
    lwc1  $f4, 0($t2)
    mul.s $f6, $f2, $f4
    add.s $f0, $f0, $f6
    addi  $s3, $s3, 1
    j     layer_1_fused_mb_iloop
layer_1_fused_mb_iend:
    sll   $t1, $s2, 2
    add   $t1, $t6, $t1
    swc1  $f0, 0($t1)
    addi  $s2, $s2, 1
    j     layer_1_fused_mb_oloop
layer_1_fused_mb_oend:


    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    lw    $s0, 20($sp)
    lw    $s1, 16($sp)
    lw    $s2, 12($sp)
    lw    $s3, 8($sp)
    lw    $s4, 4($sp)
    addiu $sp, $sp, 32
    jr    $ra

layer_1_act:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    sw    $s0, 20($sp)
    sw    $s1, 16($sp)
    sw    $s2, 12($sp)
    sw    $s3, 8($sp)
    sw    $s4, 4($sp)
    addiu $fp, $sp, 32

    # Softmax activation
    move  $a0, $s0
    li    $a1, 3
    jal   fpu_vec_max
    mov.s $f12, $f0
    li    $t4, 0
    lwc1  $f2, exp_clamp_neg
    lwc1  $f4, fpu_zero
layer_1_act_sm_loop:
    bge   $t4, 3, layer_1_act_sm_sum
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lwc1  $f0, 0($t5)
    sub.s $f0, $f0, $f12
    sw    $t4, -4($sp)
    sw    $t5, -8($sp)
    jal   fpu_exp
    lw    $t4, -4($sp)
    lw    $t5, -8($sp)
    swc1  $f0, 0($t5)
    addi  $t4, $t4, 1
    j     layer_1_act_sm_loop
layer_1_act_sm_sum:
    move  $a0, $s0
    li    $a1, 3
    jal   fpu_vec_sum
    mov.s $f12, $f0
    li    $t4, 0
layer_1_act_sm_div:
    bge   $t4, 3, layer_1_act_sm_done
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lwc1  $f0, 0($t5)
    div.s $f0, $f0, $f12
    swc1  $f0, 0($t5)
    addi  $t4, $t4, 1
    j     layer_1_act_sm_div
layer_1_act_sm_done:


    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    lw    $s0, 20($sp)
    lw    $s1, 16($sp)
    lw    $s2, 12($sp)
    lw    $s3, 8($sp)
    lw    $s4, 4($sp)
    addiu $sp, $sp, 32
    jr    $ra

print_output:
    move  $t8, $a0
    li    $t2, 0
print_output_loop:
    bge   $t2, 3, print_output_end
    sll   $t3, $t2, 2
    add   $t3, $t8, $t3
    lwc1  $f12, 0($t3)
    li    $v0, 2
    syscall
    li    $a0, 10
    li    $v0, 11
    syscall
    addi  $t2, $t2, 1
    j     print_output_loop
print_output_end:
    jr    $ra

memcpy:
    li    $t0, 0
memcpy_loop:
    bge   $t0, $a2, memcpy_end
    add   $t1, $a1, $t0
    lw    $t2, 0($t1)
    add   $t1, $a0, $t0
    sw    $t2, 0($t1)
    addi  $t0, $t0, 4
    j     memcpy_loop
memcpy_end:
    jr    $ra

vec_max:
    lw    $v0, 0($a0)
    li    $t0, 1
vec_max_loop:
    bge   $t0, $a1, vec_max_end
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lw    $t2, 0($t1)
    slt   $t3, $v0, $t2
    beq   $t3, $zero, vec_max_skip
    move  $v0, $t2
vec_max_skip:
    addi  $t0, $t0, 1
    j     vec_max_loop
vec_max_end:
    jr    $ra

vec_sum:
    li    $v0, 0
    li    $t0, 0
vec_sum_loop:
    bge   $t0, $a1, vec_sum_end
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lw    $t2, 0($t1)
    add   $v0, $v0, $t2
    addi  $t0, $t0, 1
    j     vec_sum_loop
vec_sum_end:
    jr    $ra

fpu_vec_max:
    lwc1  $f0, 0($a0)
    li    $t0, 1
fpu_vec_max_loop:
    bge   $t0, $a1, fpu_vec_max_end
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lwc1  $f2, 0($t1)
    c.lt.s $f0, $f2
    bc1f  fpu_vec_max_skip
    mov.s $f0, $f2
fpu_vec_max_skip:
    addi  $t0, $t0, 1
    j     fpu_vec_max_loop
fpu_vec_max_end:
    jr    $ra

fpu_vec_sum:
    lwc1  $f0, fpu_zero
    li    $t0, 0
fpu_vec_sum_loop:
    bge   $t0, $a1, fpu_vec_sum_end
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lwc1  $f2, 0($t1)
    add.s $f0, $f0, $f2
    addi  $t0, $t0, 1
    j     fpu_vec_sum_loop
fpu_vec_sum_end:
    jr    $ra

    # e^x approx (Taylor): 1 + x + x^2/2 + x^3/6 + x^4/24
fpu_exp:
    lwc1  $f2, exp_clamp_neg
    c.lt.s $f0, $f2
    bc1f  fpu_exp_check_pos
    lwc1  $f0, fpu_zero
    jr    $ra
fpu_exp_check_pos:
    lwc1  $f2, exp_clamp_pos
    c.lt.s $f2, $f0
    bc1f  fpu_exp_calc
    mov.s $f0, $f2
fpu_exp_calc:
    mov.s $f2, $f0
    lwc1  $f4, exp_c4
    mul.s $f6, $f4, $f2
    lwc1  $f4, exp_c3
    add.s $f6, $f6, $f4
    mul.s $f6, $f6, $f2
    lwc1  $f4, exp_c2
    add.s $f6, $f6, $f4
    mul.s $f6, $f6, $f2
    lwc1  $f4, exp_c1
    add.s $f6, $f6, $f4
    mul.s $f6, $f6, $f2
    lwc1  $f4, exp_c0
    add.s $f0, $f6, $f4
    jr    $ra
