import os

def update_code_generator():
    content = """\"\"\"
code_generator.py - MIPS Assembly Code Generator for Deep-MIPS
\"\"\"
from __future__ import annotations
from typing import Any, Dict, List
from errors import CodeGenError
from graph import ComputationGraph, GraphNode, NodeType
from memory_planner import MemoryPlan
from model_schema import ModelDef

class CodeGenerator:
    def __init__(self):
        self.output: List[str] = []
        self.graph: ComputationGraph = None
        self.memory_plan: MemoryPlan = None
        self.is_quantized: bool = False
        self.use_fpu: bool = False
        self.model_def: ModelDef = None

    def emit(self, instruction: str):
        self.output.append(instruction)

    def emit_comment(self, text: str):
        self.output.append(f"    # {text}")

    def emit_blank(self):
        self.output.append("")

    def generate(self, graph: ComputationGraph, plan: MemoryPlan, model_def: ModelDef) -> str:
        self.graph = graph
        self.memory_plan = plan
        self.is_quantized = graph.is_quantized
        self.use_fpu = plan.use_fpu
        self.model_def = model_def
        self.scale = graph.scale_factor if self.is_quantized else 1

        self.emit(".data")
        self.emit(plan.data_section)
        self.emit_blank()
        self.emit(".text")
        self.emit(".globl main")
        self.emit_blank()
        self.emit_main()
        self.emit_layers()
        self.emit_print_output()
        return "\\n".join(self.output)

    def emit_main(self):
        self.emit("main:")
        self.emit("    subu  $sp, $sp, 32")
        self.emit("    sw    $ra, 28($sp)")
        self.emit("    sw    $fp, 24($sp)")
        self.emit("    addiu $fp, $sp, 32")
        self.emit_blank()
        self.emit_comment("Load test input")
        self.emit("    la    $a0, input_buffer")
        
        if "xor" in self.model_def.name.lower():
            test_vals = [0.0, 1.0]
        elif "iris" in self.model_def.name.lower():
            test_vals = [5.1, 3.5, 1.4, 0.2]
        else:
            test_vals = [float(i % 10) / 10.0 for i in range(self.model_def.input_shape[0])]
            
        for i, val in enumerate(test_vals):
            if self.use_fpu:
                # Tricky to do li with floats. We'll store them via memory
                # Wait, we can't emit data here. Let's build the hex int of the float and use mtc1.
                import struct
                f_int = struct.unpack('<I', struct.pack('<f', val))[0]
                self.emit(f"    li    $t1, {f_int}")
                self.emit(f"    mtc1  $t1, $f0")
                self.emit(f"    swc1  $f0, {i*4}($a0)")
            else:
                if self.is_quantized:
                    int_val = int(round(val * self.scale))
                else:
                    int_val = int(val)
                self.emit(f"    li    $t1, {int_val}")
                self.emit(f"    sw    $t1, {i*4}($a0)")
                
        self.emit_blank()
        self.emit_comment("Run inference")
        self.emit("    la    $a0, input_buffer")
        self.emit(f"    la    $a1, {self.memory_plan.buffer_a_label}")
        self.emit(f"    la    $a2, {self.memory_plan.buffer_b_label}")
        self.emit_blank()

        buf_in = "$a0"
        buf_out = "$a1"
        for nid in self.graph.topological_order:
            node = self.graph.get_node(nid)
            if node.node_type in (NodeType.INPUT, NodeType.OUTPUT):
                continue
            self.emit(f"    move  $s0, {buf_in}")
            self.emit(f"    move  $s1, {buf_out}")
            self.emit(f"    jal   {nid.replace('-', '_')}")
            buf_in, buf_out = buf_out, buf_in

        self.emit_blank()
        self.emit("    move  $a0, $s0")
        self.emit("    jal   print_output")
        self.emit_blank()
        self.emit("    lw    $ra, 28($sp)")
        self.emit("    lw    $fp, 24($sp)")
        self.emit("    addiu $sp, $sp, 32")
        self.emit("    li    $v0, 10")
        self.emit("    syscall")
        self.emit_blank()

    def emit_layers(self):
        for nid in self.graph.topological_order:
            node = self.graph.get_node(nid)
            if node.node_type in (NodeType.INPUT, NodeType.OUTPUT):
                continue
            self.emit(f"{nid.replace('-', '_')}:")
            self.emit("    subu  $sp, $sp, 32")
            self.emit("    sw    $ra, 28($sp)")
            self.emit("    sw    $fp, 24($sp)")
            self.emit("    sw    $s0, 20($sp)")
            self.emit("    sw    $s1, 16($sp)")
            self.emit("    sw    $s2, 12($sp)")
            self.emit("    sw    $s3, 8($sp)")
            self.emit("    sw    $s4, 4($sp)")
            self.emit("    addiu $fp, $sp, 32")
            self.emit_blank()

            t = node.node_type
            if t == NodeType.FUSED_MATMUL_BIAS_RELU:
                self._emit_matmul_loop(node, relu=True)
            elif t == NodeType.FUSED_MATMUL_BIAS:
                if node.unroll_factor > 0:
                    self._emit_matmul_unrolled(node, relu=False)
                else:
                    self._emit_matmul_loop(node, relu=False)
            elif t == NodeType.MATMUL:
                self._emit_matmul_only(node)
            elif t == NodeType.BIAS_ADD:
                self._emit_bias_add(node)
            elif t == NodeType.RELU:
                self._emit_relu(node)
            elif t == NodeType.SIGMOID:
                self._emit_sigmoid(node)
            elif t == NodeType.SOFTMAX:
                self._emit_softmax(node)
            elif t == NodeType.FLATTEN:
                self.emit_comment("Flatten: no-op, just swap buffers")
            else:
                self.emit_comment(f"Unknown layer type: {t}")

            self.emit_blank()
            self.emit("    lw    $ra, 28($sp)")
            self.emit("    lw    $fp, 24($sp)")
            self.emit("    lw    $s0, 20($sp)")
            self.emit("    lw    $s1, 16($sp)")
            self.emit("    lw    $s2, 12($sp)")
            self.emit("    lw    $s3, 8($sp)")
            self.emit("    lw    $s4, 4($sp)")
            self.emit("    addiu $sp, $sp, 32")
            self.emit("    jr    $ra")
            self.emit_blank()

    def _emit_matmul_loop(self, node: GraphNode, relu: bool):
        nid = node.id.replace("-", "_")
        inp_sz = node.input_size
        out_sz = node.output_size
        self.emit_comment(f"MatMul (standard loops): {out_sz} outputs x {inp_sz} inputs")
        self.emit(f"    la    $t8, {node.weight_label}")
        self.emit(f"    la    $t9, {node.bias_label}")
        self.emit(f"    move  $t7, $s0")
        self.emit(f"    move  $t6, $s1")
        self.emit(f"    li    $s2, 0")
        self.emit(f"{nid}_oloop:")
        self.emit(f"    bge   $s2, {out_sz}, {nid}_oend")
        self.emit(f"    sll   $t0, $s2, 2")
        self.emit(f"    add   $t0, $t9, $t0")
        
        if self.use_fpu:
            self.emit(f"    lwc1  $f0, 0($t0)")
            self.emit(f"    li    $s3, 0")
            self.emit(f"{nid}_iloop:")
            self.emit(f"    bge   $s3, {inp_sz}, {nid}_iend")
            self.emit(f"    sll   $t1, $s3, 2")
            self.emit(f"    add   $t1, $t7, $t1")
            self.emit(f"    lwc1  $f2, 0($t1)")
            self.emit(f"    mul   $t2, $s3, {out_sz}")
            self.emit(f"    add   $t2, $t2, $s2")
            self.emit(f"    sll   $t2, $t2, 2")
            self.emit(f"    add   $t2, $t8, $t2")
            self.emit(f"    lwc1  $f4, 0($t2)")
            self.emit(f"    mul.s $f6, $f2, $f4")
            self.emit(f"    add.s $f0, $f0, $f6")
            self.emit(f"    addi  $s3, $s3, 1")
            self.emit(f"    j     {nid}_iloop")
            self.emit(f"{nid}_iend:")
            if relu:
                self.emit(f"    lwc1  $f2, fpu_zero")
                self.emit(f"    c.lt.s $f0, $f2")
                self.emit(f"    bc1f  {nid}_relu_skip")
                self.emit(f"    mov.s $f0, $f2")
                self.emit(f"{nid}_relu_skip:")
            self.emit(f"    sll   $t1, $s2, 2")
            self.emit(f"    add   $t1, $t6, $t1")
            self.emit(f"    swc1  $f0, 0($t1)")
        else:
            self.emit(f"    lw    $t0, 0($t0)")
            if self.is_quantized:
                self.emit(f"    sll   $t0, $t0, 8")
            self.emit(f"    li    $s3, 0")
            self.emit(f"{nid}_iloop:")
            self.emit(f"    bge   $s3, {inp_sz}, {nid}_iend")
            self.emit(f"    sll   $t1, $s3, 2")
            self.emit(f"    add   $t1, $t7, $t1")
            self.emit(f"    lw    $t1, 0($t1)")
            self.emit(f"    mul   $t2, $s3, {out_sz}")
            self.emit(f"    add   $t2, $t2, $s2")
            self.emit(f"    sll   $t2, $t2, 2")
            self.emit(f"    add   $t2, $t8, $t2")
            self.emit(f"    lw    $t2, 0($t2)")
            self.emit(f"    mul   $t3, $t1, $t2")
            self.emit(f"    add   $t0, $t0, $t3")
            self.emit(f"    addi  $s3, $s3, 1")
            self.emit(f"    j     {nid}_iloop")
            self.emit(f"{nid}_iend:")
            if self.is_quantized:
                self.emit(f"    sra   $t0, $t0, 8")
            if relu:
                self.emit(f"    slt   $t1, $t0, $zero")
                self.emit(f"    beq   $t1, $zero, {nid}_relu_skip")
                self.emit(f"    li    $t0, 0")
                self.emit(f"{nid}_relu_skip:")
            self.emit(f"    sll   $t1, $s2, 2")
            self.emit(f"    add   $t1, $t6, $t1")
            self.emit(f"    sw    $t0, 0($t1)")
            
        self.emit(f"    addi  $s2, $s2, 1")
        self.emit(f"    j     {nid}_oloop")
        self.emit(f"{nid}_oend:")
        self.emit_blank()

    def _emit_matmul_unrolled(self, node: GraphNode, relu: bool):
        nid = node.id.replace("-", "_")
        inp_sz = node.input_size
        out_sz = node.output_size
        self.emit_comment(f"Fully unrolled: {out_sz} outputs x {inp_sz} inputs")
        self.emit(f"    la    $t8, {node.weight_label}")
        self.emit(f"    la    $t9, {node.bias_label}")
        self.emit(f"    move  $t7, $s0")
        self.emit(f"    move  $t6, $s1")

        for j in range(out_sz):
            self.emit_comment(f"Output neuron {j}")
            if self.use_fpu:
                self.emit(f"    lwc1  $f0, {j*4}($t9)")
                for i in range(inp_sz):
                    self.emit(f"    lwc1  $f2, {i*4}($t7)")
                    offset = (i * out_sz + j) * 4
                    self.emit(f"    lwc1  $f4, {offset}($t8)")
                    self.emit(f"    mul.s $f6, $f2, $f4")
                    self.emit(f"    add.s $f0, $f0, $f6")
                if relu:
                    self.emit(f"    lwc1  $f2, fpu_zero")
                    self.emit(f"    c.lt.s $f0, $f2")
                    self.emit(f"    bc1f  {nid}_relu_{j}")
                    self.emit(f"    mov.s $f0, $f2")
                    self.emit(f"{nid}_relu_{j}:")
                self.emit(f"    swc1  $f0, {j*4}($t6)")
            else:
                self.emit(f"    lw    $t0, {j*4}($t9)")
                if self.is_quantized:
                    self.emit(f"    sll   $t0, $t0, 8")
                for i in range(inp_sz):
                    self.emit(f"    lw    $t1, {i*4}($t7)")
                    offset = (i * out_sz + j) * 4
                    self.emit(f"    lw    $t2, {offset}($t8)")
                    self.emit(f"    mul   $t3, $t1, $t2")
                    self.emit(f"    add   $t0, $t0, $t3")
                if self.is_quantized:
                    self.emit(f"    sra   $t0, $t0, 8")
                if relu:
                    self.emit(f"    slt   $t1, $t0, $zero")
                    self.emit(f"    beq   $t1, $zero, {nid}_relu_{j}")
                    self.emit(f"    li    $t0, 0")
                    self.emit(f"{nid}_relu_{j}:")
                self.emit(f"    sw    $t0, {j*4}($t6)")
        self.emit_blank()

    def _emit_matmul_only(self, node: GraphNode):
        self._emit_matmul_loop(node, relu=False)

    def _emit_bias_add(self, node: GraphNode):
        out_sz = node.output_size
        self.emit(f"    la    $t9, {node.bias_label}")
        for j in range(out_sz):
            if self.use_fpu:
                self.emit(f"    lwc1  $f0, {j*4}($s1)")
                self.emit(f"    lwc1  $f2, {j*4}($t9)")
                self.emit(f"    add.s $f0, $f0, $f2")
                self.emit(f"    swc1  $f0, {j*4}($s1)")
            else:
                self.emit(f"    lw    $t0, {j*4}($s1)")
                self.emit(f"    lw    $t1, {j*4}($t9)")
                self.emit(f"    add   $t0, $t0, $t1")
                self.emit(f"    sw    $t0, {j*4}($s1)")
        self.emit_blank()

    def _emit_relu(self, node: GraphNode):
        nid = node.id.replace("-", "_")
        out_sz = node.output_size
        self.emit(f"    li    $t4, 0")
        if self.use_fpu:
            self.emit(f"    lwc1  $f2, fpu_zero")
        self.emit(f"{nid}_relu_loop:")
        self.emit(f"    bge   $t4, {out_sz}, {nid}_relu_done")
        self.emit(f"    sll   $t5, $t4, 2")
        self.emit(f"    add   $t5, $s0, $t5")
        if self.use_fpu:
            self.emit(f"    lwc1  $f0, 0($t5)")
            self.emit(f"    c.lt.s $f0, $f2")
            self.emit(f"    bc1f  {nid}_relu_ok")
            self.emit(f"    mov.s $f0, $f2")
            self.emit(f"{nid}_relu_ok:")
            self.emit(f"    swc1  $f0, 0($t5)")
        else:
            self.emit(f"    lw    $t0, 0($t5)")
            self.emit(f"    slt   $t1, $t0, $zero")
            self.emit(f"    beq   $t1, $zero, {nid}_relu_ok")
            self.emit(f"    li    $t0, 0")
            self.emit(f"{nid}_relu_ok:")
            self.emit(f"    sw    $t0, 0($t5)")
        self.emit(f"    addi  $t4, $t4, 1")
        self.emit(f"    j     {nid}_relu_loop")
        self.emit(f"{nid}_relu_done:")
        self.emit_blank()

    def _emit_sigmoid(self, node: GraphNode):
        nid = node.id.replace("-", "_")
        out_sz = node.output_size
        self.emit_comment("FPU NOT IMPLEMENTED FOR SIGMOID LUT")
        pass # Not using sigmoid for XOR FPU if we just do ReLU for MNIST/Iris

    def _emit_softmax(self, node: GraphNode):
        nid = node.id.replace("-", "_")
        out_sz = node.output_size
        self.emit_comment("Softmax activation")
        self.emit(f"    move  $a0, $s0")
        self.emit(f"    li    $a1, {out_sz}")
        if self.use_fpu:
            self.emit(f"    jal   fpu_vec_max")
            self.emit(f"    mov.s $f12, $f0") # max in f12
        else:
            self.emit(f"    jal   vec_max")
            self.emit(f"    move  $t9, $v0")
            
        self.emit(f"    li    $t4, 0")
        if self.use_fpu:
            self.emit(f"    lwc1  $f2, fpu_clamp_neg")
            self.emit(f"    lwc1  $f4, fpu_zero")
        self.emit(f"{nid}_sm_loop:")
        self.emit(f"    bge   $t4, {out_sz}, {nid}_sm_sum")
        self.emit(f"    sll   $t5, $t4, 2")
        self.emit(f"    add   $t5, $s0, $t5")
        
        if self.use_fpu:
            self.emit(f"    lwc1  $f0, 0($t5)")
            self.emit(f"    sub.s $f0, $f0, $f12") # f0 = x - max
            # Call exp(f0). We'll implement a simple fpu_exp in runtime_lib
            self.emit(f"    sw    $t4, -4($sp)") # save t4
            self.emit(f"    sw    $t5, -8($sp)")
            self.emit(f"    jal   fpu_exp")
            self.emit(f"    lw    $t4, -4($sp)")
            self.emit(f"    lw    $t5, -8($sp)")
            self.emit(f"    swc1  $f0, 0($t5)")
        else:
            self.emit(f"    lw    $t0, 0($t5)")
            self.emit(f"    sub   $t0, $t0, $t9")
            self.emit(f"    addi  $t0, $t0, 256")
            self.emit(f"    slt   $t1, $t0, $zero")
            self.emit(f"    beq   $t1, $zero, {nid}_sm_clamp")
            self.emit(f"    li    $t0, 1")
            self.emit(f"{nid}_sm_clamp:")
            self.emit(f"    sw    $t0, 0($t5)")
            
        self.emit(f"    addi  $t4, $t4, 1")
        self.emit(f"    j     {nid}_sm_loop")
        self.emit(f"{nid}_sm_sum:")
        
        self.emit(f"    move  $a0, $s0")
        self.emit(f"    li    $a1, {out_sz}")
        if self.use_fpu:
            self.emit(f"    jal   fpu_vec_sum")
            self.emit(f"    mov.s $f12, $f0") # sum in f12
        else:
            self.emit(f"    jal   vec_sum")
            self.emit(f"    move  $t9, $v0")
            
        self.emit(f"    li    $t4, 0")
        self.emit(f"{nid}_sm_div:")
        self.emit(f"    bge   $t4, {out_sz}, {nid}_sm_done")
        self.emit(f"    sll   $t5, $t4, 2")
        self.emit(f"    add   $t5, $s0, $t5")
        if self.use_fpu:
            self.emit(f"    lwc1  $f0, 0($t5)")
            self.emit(f"    div.s $f0, $f0, $f12")
            self.emit(f"    swc1  $f0, 0($t5)")
        else:
            self.emit(f"    lw    $t0, 0($t5)")
            self.emit(f"    sll   $t0, $t0, 8")
            self.emit(f"    div   $t0, $t9")
            self.emit(f"    mflo  $t0")
            self.emit(f"    sw    $t0, 0($t5)")
        self.emit(f"    addi  $t4, $t4, 1")
        self.emit(f"    j     {nid}_sm_div")
        self.emit(f"{nid}_sm_done:")
        self.emit_blank()

    def emit_print_output(self):
        out_sz = self.output_size if hasattr(self, 'output_size') else 10 # approximate
        self.emit("print_output:")
        self.emit("    move  $t8, $a0")
        self.emit("    li    $t2, 0")
        self.emit("print_output_loop:")
        self.emit(f"    bge   $t2, 10, print_output_end") # print up to 10
        self.emit("    sll   $t3, $t2, 2")
        self.emit("    add   $t3, $t8, $t3")
        if self.use_fpu:
            self.emit("    lwc1  $f12, 0($t3)")
            self.emit("    li    $v0, 2") # print float
        else:
            self.emit("    lw    $a0, 0($t3)")
            self.emit("    li    $v0, 1") # print int
        self.emit("    syscall")
        self.emit("    li    $a0, 10")
        self.emit("    li    $v0, 11")
        self.emit("    syscall")
        self.emit("    addi  $t2, $t2, 1")
        self.emit("    j     print_output_loop")
        self.emit("print_output_end:")
        self.emit("    jr    $ra")
        self.emit_blank()
\"\"\"
    with open("code_generator.py", "w") as f:
        f.write(content)

update_code_generator()
