.data
sigmoid_lut:    .word  0, 5, 12, 25, 50, 88, 128, 168, 206, 231, 244, 249, 252, 254, 255, 256
fc1_weights:            .float  -0.35227612, 0.57364523, 0.68233103, -0.03684193, -0.35590780, 0.43869177, 0.80927122, -0.41468138
                        .float  -0.63762283, 0.30948257, -0.67706990, -0.38660520, -0.04131001, 0.65481973, 1.01082659, -0.32228875
                        .float  0.81398958, -0.70945275, 0.84600991, -0.38372278, 0.35236037, -1.07283199, -1.26292241, 0.10566145
                        .float  1.55958545, -0.75636834, 0.64684528, 0.26046526, -0.25814885, -0.43529484, -0.89812165, -0.34459609
fc2_weights:            .float  0.00846712, -1.70310533, 1.20734155, 1.02386558, 0.62983966, -1.20508707, -0.89652979, 0.31803301
                        .float  0.63701856, -0.21949573, -0.14610070, 0.11554796, -0.31131315, 0.08510008, -0.06643820, 0.84384823
                        .float  -1.17739379, -0.29678378, 0.55373359, 0.46101817, -1.09094942, 0.06590465, -0.14579315, 0.18958303
fc1_biases:             .float  -1.54274476, 1.01710117, -0.50221407, -0.40554118, -0.47674477, 0.57098138, 1.74467921, 0.22489440
fc2_biases:             .float  0.07730176, 0.32914722, -0.61843055
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
    jal   fc1_fused_mbr
    move  $s0, $a1
    move  $s1, $a0
    jal   fc2_fused_mb
    move  $s0, $a0
    move  $s1, $a1
    jal   fc2_act

    move  $a0, $s0
    jal   print_output

    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    addiu $sp, $sp, 32
    li    $v0, 10
    syscall

fc1_fused_mbr:
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
    la    $t8, fc1_weights
    la    $t9, fc1_biases
    move  $t7, $s0
    move  $t6, $s1
    li    $s2, 0
fc1_fused_mbr_oloop:
    bge   $s2, 8, fc1_fused_mbr_oend
    sll   $t0, $s2, 2
    add   $t0, $t9, $t0
    lwc1  $f0, 0($t0)
    li    $s3, 0
fc1_fused_mbr_iloop:
    bge   $s3, 4, fc1_fused_mbr_iend
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
    j     fc1_fused_mbr_iloop
fc1_fused_mbr_iend:
    lwc1  $f2, fpu_zero
    c.lt.s $f0, $f2
    bc1f  fc1_fused_mbr_relu_skip
    mov.s $f0, $f2
fc1_fused_mbr_relu_skip:
    sll   $t1, $s2, 2
    add   $t1, $t6, $t1
    swc1  $f0, 0($t1)
    addi  $s2, $s2, 1
    j     fc1_fused_mbr_oloop
fc1_fused_mbr_oend:


    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    lw    $s0, 20($sp)
    lw    $s1, 16($sp)
    lw    $s2, 12($sp)
    lw    $s3, 8($sp)
    lw    $s4, 4($sp)
    addiu $sp, $sp, 32
    jr    $ra

fc2_fused_mb:
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
    la    $t8, fc2_weights
    la    $t9, fc2_biases
    move  $t7, $s0
    move  $t6, $s1
    li    $s2, 0
fc2_fused_mb_oloop:
    bge   $s2, 3, fc2_fused_mb_oend
    sll   $t0, $s2, 2
    add   $t0, $t9, $t0
    lwc1  $f0, 0($t0)
    li    $s3, 0
fc2_fused_mb_iloop:
    bge   $s3, 8, fc2_fused_mb_iend
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
    j     fc2_fused_mb_iloop
fc2_fused_mb_iend:
    sll   $t1, $s2, 2
    add   $t1, $t6, $t1
    swc1  $f0, 0($t1)
    addi  $s2, $s2, 1
    j     fc2_fused_mb_oloop
fc2_fused_mb_oend:


    lw    $ra, 28($sp)
    lw    $fp, 24($sp)
    lw    $s0, 20($sp)
    lw    $s1, 16($sp)
    lw    $s2, 12($sp)
    lw    $s3, 8($sp)
    lw    $s4, 4($sp)
    addiu $sp, $sp, 32
    jr    $ra

fc2_act:
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
fc2_act_sm_loop:
    bge   $t4, 3, fc2_act_sm_sum
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
    j     fc2_act_sm_loop
fc2_act_sm_sum:
    move  $a0, $s0
    li    $a1, 3
    jal   fpu_vec_sum
    mov.s $f12, $f0
    li    $t4, 0
fc2_act_sm_div:
    bge   $t4, 3, fc2_act_sm_done
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lwc1  $f0, 0($t5)
    div.s $f0, $f0, $f12
    swc1  $f0, 0($t5)
    addi  $t4, $t4, 1
    j     fc2_act_sm_div
fc2_act_sm_done:


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
