# Deep-MIPS Inference Engine
# Model: xor_classifier
# Layers: 2
# Quantized: no
# Input: [2]  Output: [1]

.data
layer_0_weights:
    .word  5, 5, -5, -5, 5, -5, 5, -5

layer_1_weights:
    .word  -10, 10, 10, -10

layer_0_biases:
    .word  -7, -2, -2, -7

layer_1_biases:
    .word  -5

act_buffer_a:  .space  16
act_buffer_b:  .space  16

input_buffer:  .space  8

sigmoid_lut:  .word  5, 12, 27, 73, 128, 183, 229, 244, 251

newline_str:  .asciiz  "\n"
result_label: .asciiz  "Output: "

test_input:   .word  0, 0


.text
.globl main

main:
    subu  $sp, $sp, 32
    sw    $ra, 28($sp)
    sw    $fp, 24($sp)
    addiu $fp, $sp, 32

    # Load test input
    la    $a0, input_buffer
    li    $t1, 0
    sw    $t1, 0($a0)
    li    $t1, 1
    sw    $t1, 4($a0)

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
    # Input: 2  Output: 4
    # Fully unrolled: 4 outputs x 2 inputs
    la    $t8, layer_0_weights
    la    $t9, layer_0_biases
    move  $t7, $s0
    move  $t6, $s1
    # Output neuron 0
    lw    $t0, 0($t9)
    lw    $t1, 0($t7)
    lw    $t2, 0($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 16($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_0_fused_mbr_relu_0
    li    $t0, 0
layer_0_fused_mbr_relu_0:
    sw    $t0, 0($t6)
    # Output neuron 1
    lw    $t0, 4($t9)
    lw    $t1, 0($t7)
    lw    $t2, 4($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 20($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_0_fused_mbr_relu_1
    li    $t0, 0
layer_0_fused_mbr_relu_1:
    sw    $t0, 4($t6)
    # Output neuron 2
    lw    $t0, 8($t9)
    lw    $t1, 0($t7)
    lw    $t2, 8($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 24($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_0_fused_mbr_relu_2
    li    $t0, 0
layer_0_fused_mbr_relu_2:
    sw    $t0, 8($t6)
    # Output neuron 3
    lw    $t0, 12($t9)
    lw    $t1, 0($t7)
    lw    $t2, 12($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 28($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    slt   $t1, $t0, $zero
    beq   $t1, $zero, layer_0_fused_mbr_relu_3
    li    $t0, 0
layer_0_fused_mbr_relu_3:
    sw    $t0, 12($t6)

    # Swap ping-pong buffers
    move  $t0, $s0
    move  $s0, $s1
    move  $s1, $t0

    # === Layer: layer_1_fused_mb [MatMul+Bias] ===
    # Input: 4  Output: 1
    # Fully unrolled: 1 outputs x 4 inputs
    la    $t8, layer_1_weights
    la    $t9, layer_1_biases
    move  $t7, $s0
    move  $t6, $s1
    # Output neuron 0
    lw    $t0, 0($t9)
    lw    $t1, 0($t7)
    lw    $t2, 0($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 4($t7)
    lw    $t2, 4($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 8($t7)
    lw    $t2, 8($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    lw    $t1, 12($t7)
    lw    $t2, 12($t8)
    mul   $t3, $t1, $t2
    add   $t0, $t0, $t3
    sw    $t0, 0($t6)

    # Swap ping-pong buffers
    move  $t0, $s0
    move  $s0, $s1
    move  $s1, $t0

    # Sigmoid activation (LUT): layer_1_act
    la    $t8, sigmoid_lut
    li    $t4, 0
layer_1_act_sig_loop:
    bge   $t4, 1, layer_1_act_sig_done
    sll   $t5, $t4, 2
    add   $t5, $s0, $t5
    lw    $t0, 0($t5)
    addi  $t1, $t0, 4
    slt   $t2, $t1, $zero
    bne   $t2, $zero, layer_1_act_sig_zero
    li    $t3, 8
    slt   $t2, $t3, $t1
    bne   $t2, $zero, layer_1_act_sig_one
    sll   $t1, $t1, 2
    add   $t1, $t8, $t1
    lw    $t0, 0($t1)
    j     layer_1_act_sig_store
layer_1_act_sig_zero:
    li    $t0, 0
    j     layer_1_act_sig_store
layer_1_act_sig_one:
    li    $t0, 1
layer_1_act_sig_store:
    sw    $t0, 0($t5)
    addi  $t4, $t4, 1
    j     layer_1_act_sig_loop
layer_1_act_sig_done:


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
    bge   $t2, 1, print_output_end
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
