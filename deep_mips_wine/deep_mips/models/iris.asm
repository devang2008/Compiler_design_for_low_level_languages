# Deep-MIPS Inference Engine
# Model: iris_classifier
# Layers: 2
# Quantized: yes
# Input: [4]  Output: [3]

.data
layer_0_weights:
    .half  115, -31, 200, -87, 143, -228, 59, 172
    .half  -143, 233, -38, 123, -184, 84, -156, 215
    .half  82, -172, 138, -59, 228, -115, 182, -97
    .half  -200, 110, -236, 166, -28, 223, -138, 74
    .align 2

layer_1_weights:
    .half  315, -115, 31, -223, 399, -87, 115, -59
    .half  276, -31, 200, -143, 251, -172, 87, -87
    .half  287, -228, 143, -115, 200, -172, 228, -59
    .align 2

layer_0_biases:
    .half  26, -13, 20, 0, -31, 8, 0, 18
    .align 2

layer_1_biases:
    .half  13, -8, 3
    .align 2

act_buffer_a:  .space  16
act_buffer_b:  .space  16

input_buffer:  .space  8

sigmoid_lut:  .word  5, 12, 27, 73, 128, 183, 229, 244, 251

newline_str:  .asciiz  "\n"
result_label: .asciiz  "Output: "

test_input:   .word  0, 0, 0, 0


.text
.globl main

main:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    addiu $fp, $sp, 32

    # Load test input
    la    $a0, input_buffer
    li    $t1, 1306
    sw    $t1, 0($a0)
    li    $t1, 896
    sw    $t1, 4($a0)
    li    $t1, 358
    sw    $t1, 8($a0)
    li    $t1, 51
    sw    $t1, 12($a0)

    # Run inference
    la    $a0, input_buffer
    jal   nn_forward

    # Print output
    move  $a0, $v0
    jal   print_output

    # Exit
    li    $v0, 10
    syscall

nn_forward:
    subu  $sp, $sp, 48
    sw    $ra, 44($sp)
    sw    $fp, 40($sp)
    sw    $s0, 36($sp)
    sw    $s1, 32($sp)
    sw    $s2, 28($sp)
    sw    $s3, 24($sp)
    addiu $fp, $sp, 48

    move  $s0, $a0
    la    $s1, act_buffer_a

    # === Layer: layer_0_fused_mbr [MatMul+Bias+ReLU] ===
    # Input: 4  Output: 8
    la    $t8, layer_0_weights
    la    $t9, layer_0_biases
    move  $t7, $s0
    move  $t6, $s1
    li    $s2, 0
layer_0_fused_mbr_oloop:
    bge   $s2, 8, layer_0_fused_mbr_oend
    sll   $t0, $s2, 2
    add   $t0, $t9, $t0
    lw    $t0, 0($t0)
    li    $s3, 0
layer_0_fused_mbr_iloop:
    bge   $s3, 4, layer_0_fused_mbr_iend
    sll   $t1, $s3, 2
    add   $t1, $t7, $t1
    lw    $t1, 0($t1)
    mul   $t2, $s3, 8
    add   $t2, $t2, $s2
    sll   $t2, $t2, 2
    add   $t2, $t8, $t2
    lw    $t2, 0($t2)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    addi  $s3, $s3, 1
    j     layer_0_fused_mbr_iloop
layer_0_fused_mbr_iend:
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_0_fused_mbr_relu_skip
    li    $t0, 0
layer_0_fused_mbr_relu_skip:
    sll   $t1, $s2, 2
    add   $t1, $t6, $t1
    sw    $t0, 0($t1)
    addi  $s2, $s2, 1
    j     layer_0_fused_mbr_oloop
layer_0_fused_mbr_oend:

    # Swap ping-pong buffers
    move  $t0, $s0
    move  $s0, $s1
    move  $s1, $t0

    # === Layer: layer_1_fused_mb [MatMul+Bias] ===
    # Input: 8  Output: 3
    # Fully unrolled: 3 outputs x 8 inputs
    la    $t8, layer_1_weights
    la    $t9, layer_1_biases
    move  $t7, $s0
    move  $t6, $s1
    # Output neuron 0
    lw    $t0, 0($t9)
    lw    $t1, 0($t7)
    lw    $t2, 0($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 12($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 8($t7)
    lw    $t2, 24($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 12($t7)
    lw    $t2, 36($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 16($t7)
    lw    $t2, 48($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 20($t7)
    lw    $t2, 60($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 24($t7)
    lw    $t2, 72($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 28($t7)
    lw    $t2, 84($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    sw    $t0, 0($t6)
    # Output neuron 1
    lw    $t0, 4($t9)
    lw    $t1, 0($t7)
    lw    $t2, 4($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 16($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 8($t7)
    lw    $t2, 28($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 12($t7)
    lw    $t2, 40($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 16($t7)
    lw    $t2, 52($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 20($t7)
    lw    $t2, 64($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 24($t7)
    lw    $t2, 76($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 28($t7)
    lw    $t2, 88($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    sw    $t0, 4($t6)
    # Output neuron 2
    lw    $t0, 8($t9)
    lw    $t1, 0($t7)
    lw    $t2, 8($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 20($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 8($t7)
    lw    $t2, 32($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 12($t7)
    lw    $t2, 44($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 16($t7)
    lw    $t2, 56($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 20($t7)
    lw    $t2, 68($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 24($t7)
    lw    $t2, 80($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    lw    $t1, 28($t7)
    lw    $t2, 92($t8)
    mul   $t3, $t1, $t2
    sra   $t3, $t3, 8
    add   $t0, $t0, $t3
    sw    $t0, 8($t6)

    # Swap ping-pong buffers
    move  $t0, $s0
    move  $s0, $s1
    move  $s1, $t0

    # Softmax activation: layer_1_act
    # Step 1: find max
    move  $a0, $s0
    li    $a1, 3
    jal   vec_max
    move  $t9, $v0
    # Step 2: subtract max, compute exp approx, store
    li    $t4, 0
layer_1_act_sm_loop:
    bge   $t4, 3, layer_1_act_sm_sum
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lw    $t0, 0($t5)
    sub   $t0, $t0, $t9
    addi  $t0, $t0, 256
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_1_act_sm_clamp
    li    $t0, 1
layer_1_act_sm_clamp:
    sw    $t0, 0($t5)
    addi  $t4, $t4, 1
    j     layer_1_act_sm_loop
layer_1_act_sm_sum:
    # Step 3: sum all
    move  $a0, $s0
    li    $a1, 3
    jal   vec_sum
    move  $t9, $v0
    # Step 4: divide each by sum (scale to 256)
    li    $t4, 0
layer_1_act_sm_div:
    bge   $t4, 3, layer_1_act_sm_done
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lw    $t0, 0($t5)
    sll   $t0, $t0, 8
    div   $t0, $t9
    mflo  $t0
    sw    $t0, 0($t5)
    addi  $t4, $t4, 1
    j     layer_1_act_sm_div
layer_1_act_sm_done:


    # Return pointer to output buffer
    move  $v0, $s0

    lw    $s3, 24($sp)
    lw    $s2, 28($sp)
    lw    $s1, 32($sp)
    lw    $s0, 36($sp)
    lw    $ra, 44($sp)
    lw    $fp, 40($sp)
    addiu $sp, $sp, 48
    jr    $ra

print_output:
    move  $t8, $a0
    li    $t2, 0
print_output_loop:
    bge   $t2, 3, print_output_end
    sll   $t3, $t2, 2
    add   $t3, $t8, $t3
    lw    $a0, 0($t3)
    li    $v0, 1
    syscall
    li    $a0, 10
    li    $v0, 11
    syscall
    addi  $t2, $t2, 1
    j     print_output_loop
print_output_end:
    jr    $ra



# -- memcpy --------------------------------------------
# $a0 = destination, $a1 = source, $a2 = byte count
memcpy:
    beq   $a2, $zero, memcpy_done
    li    $t0, 0
memcpy_loop:
    bge   $t0, $a2, memcpy_done
    add   $t1, $a1, $t0
    lb    $t2, 0($t1)
    add   $t3, $a0, $t0
    sb    $t2, 0($t3)
    addi  $t0, $t0, 1
    j     memcpy_loop
memcpy_done:
    jr    $ra


# -- vec_max -------------------------------------------
# $a0 = array pointer, $a1 = length
# Returns $v0 = maximum value
vec_max:
    lw    $v0, 0($a0)
    li    $t0, 1
vec_max_loop:
    bge   $t0, $a1, vec_max_done
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lw    $t2, 0($t1)
    slt   $t3, $v0, $t2
    beq   $t3, $zero, vec_max_skip
    move  $v0, $t2
vec_max_skip:
    addi  $t0, $t0, 1
    j     vec_max_loop
vec_max_done:
    jr    $ra


# -- vec_sum -------------------------------------------
# $a0 = array pointer, $a1 = length
# Returns $v0 = sum of all elements
vec_sum:
    li    $v0, 0
    li    $t0, 0
vec_sum_loop:
    bge   $t0, $a1, vec_sum_done
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lw    $t2, 0($t1)
    add   $v0, $v0, $t2
    addi  $t0, $t0, 1
    j     vec_sum_loop
vec_sum_done:
    jr    $ra


# -- argmax --------------------------------------------
# $a0 = array pointer, $a1 = length
# Returns $v0 = index of maximum value
argmax:
    li    $v0, 0
    lw    $t4, 0($a0)
    li    $t0, 1
argmax_loop:
    bge   $t0, $a1, argmax_done
    sll   $t1, $t0, 2
    add   $t1, $a0, $t1
    lw    $t2, 0($t1)
    slt   $t3, $t4, $t2
    beq   $t3, $zero, argmax_skip
    move  $t4, $t2
    move  $v0, $t0
argmax_skip:
    addi  $t0, $t0, 1
    j     argmax_loop
argmax_done:
    jr    $ra
