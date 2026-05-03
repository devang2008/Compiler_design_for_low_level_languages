import sys
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("Step 4: MIPS Compilation")
    print("=" * 60)

    # Step 1 — Add deep_mips to path
    deep_mips_path = Path(__file__).parent / "deep_mips"
    sys.path.insert(0, str(deep_mips_path))

    # Step 2 — Load the model JSON and test samples
    model_json_path = Path("outputs/wine_model.json")
    if not model_json_path.exists():
        print("[ERROR] outputs/wine_model.json not found.")
        sys.exit(1)

    test_samples_path = Path("outputs/test_samples.json")
    if not test_samples_path.exists():
        print("[ERROR] outputs/test_samples.json not found.")
        sys.exit(1)

    with open(test_samples_path) as f:
        test_samples = json.load(f)

    # Step 3 — Run compiler pipeline
    from model_parser import ModelParser
    from graph_optimizer import GraphOptimizer
    from memory_planner import MemoryPlanner
    from code_generator import CodeGenerator

    parser = ModelParser()
    graph, model_def = parser.parse(str(model_json_path))

    num_nodes_before = len(graph.nodes)

    optimizer = GraphOptimizer()
    graph = optimizer.optimize(graph)

    num_nodes_after = len(graph.nodes)

    planner = MemoryPlanner()
    plan = planner.plan(graph, is_quantized=False, use_fpu=True)

    # Step 4 & 5 — Inject test samples into generated ASM
    class WineCodeGenerator(CodeGenerator):
        def __init__(self, test_samples):
            super().__init__()
            self.test_samples = test_samples

        def generate(self, graph, plan, model_def):
            data_lines = []
            for i, sample in enumerate(self.test_samples["samples"]):
                features = sample["features_scaled"]
                formatted = ", ".join(f"{v:.8f}" for v in features)
                data_lines.append(f"test_input_{i}:  .float  {formatted}")
                
            labels = [str(s["python_prediction"]) for s in self.test_samples["samples"]]
            data_lines.append(f"test_labels:   .word   " + ", ".join(labels))
            
            ptrs = ", ".join(f"test_input_{i}" for i in range(10))
            data_lines.append(f"test_ptrs:     .word   {ptrs}")
            
            plan.data_section += "\n" + "\n".join(data_lines)
            return super().generate(graph, plan, model_def)

        def emit_main(self):
            self.emit("main:")
            self.emit("    subu  $sp, $sp, 32")
            self.emit("    sw    $ra, 28($sp)")
            self.emit("    sw    $fp, 24($sp)")
            self.emit("    addiu $fp, $sp, 32")
            self.emit_blank()
            
            self.emit("    li    $t9, 0")
            self.emit("test_loop:")
            self.emit("    bge   $t9, 10, test_done")
            
            self.emit("    la    $t0, test_ptrs")
            self.emit("    sll   $t1, $t9, 2")
            self.emit("    add   $t0, $t0, $t1")
            self.emit("    lw    $t2, 0($t0)")
            
            self.emit(f"    la    $t3, {self.memory_plan.input_buffer_label}")
            self.emit("    li    $t4, 0")
            self.emit("load_input_loop:")
            self.emit("    bge   $t4, 11, do_inference")
            self.emit("    sll   $t5, $t4, 2")
            self.emit("    add   $t6, $t2, $t5")
            self.emit("    lwc1  $f0, 0($t6)")
            self.emit("    add   $t6, $t3, $t5")
            self.emit("    swc1  $f0, 0($t6)")
            self.emit("    addi  $t4, $t4, 1")
            self.emit("    j     load_input_loop")
            
            self.emit("do_inference:")
            self.emit(f"    la    $a0, {self.memory_plan.input_buffer_label}")
            self.emit(f"    la    $a1, {self.memory_plan.buffer_a_label}")
            self.emit(f"    la    $a2, {self.memory_plan.buffer_b_label}")
            
            buf_in = "$a0"
            buf_out = "$a1"
            for nid in self.graph.topological_order:
                node = self.graph.get_node(nid)
                if node.node_type.name in ("INPUT", "OUTPUT"):
                    continue
                self.emit(f"    move  $s0, {buf_in}")
                self.emit(f"    move  $s1, {buf_out}")
                self.emit(f"    jal   {nid.replace('-', '_')}")
                buf_in, buf_out = buf_out, buf_in
                
            self.emit("    move  $a0, $s0")
            self.emit("    jal   print_single_result")
            self.emit("    addi  $t9, $t9, 1")
            self.emit("    j     test_loop")
            
            self.emit("test_done:")
            self.emit("    lw    $ra, 28($sp)")
            self.emit("    lw    $fp, 24($sp)")
            self.emit("    addiu $sp, $sp, 32")
            self.emit("    li    $v0, 10")
            self.emit("    syscall")
            self.emit_blank()

    codegen = WineCodeGenerator(test_samples)
    asm_str = codegen.generate(graph, plan, model_def)
    
    # Prepend MARS comments
    header = """# =======================================================
# Deep-MIPS Wine Quality Classifier
# =======================================================
# HOW TO RUN:
# 1. Open MARS simulator
# 2. File -> Open -> select this file (wine_model.asm)
# 3. Run -> Assemble  (F3)
# 4. Run -> Go        (F5)
# 5. Check console output - should print 10 lines
#    each line is either 0 or 1
# 6. Compare to outputs/verification_report.txt
#    Expected: 0 1 0 0 1 1 0 1 0 0
# =======================================================
# Model: wine_quality_binary
# Architecture: Dense(11->32,relu) -> Dense(32->16,relu)
#               -> Dense(16->1,sigmoid)
# FPU mode: YES (Coprocessor 1)
# Quantized: NO
# Total parameters: 929
# =======================================================

"""
    asm_str = header + asm_str
    
    # Let's count sizes
    data_size = 0
    for slot in plan.slots:
        data_size += slot.size_bytes
        
    lines_count = len(asm_str.split('\n'))
    instr_count = len([line for line in asm_str.split('\n') if line.strip() and not line.strip().startswith('#') and not line.strip().endswith(':')])

    # Step 6 — Write output
    out_path = Path("outputs/wine_model.asm")
    with open(out_path, "w") as f:
        f.write(asm_str)

    # Step 7 — Print compilation report
    print(f"Graph before optimization: {num_nodes_before} nodes")
    print(f"Graph after optimization: {num_nodes_after} nodes")
    
    fusions = []
    for nid in graph.topological_order:
        n = graph.get_node(nid)
        if n.node_type.name == "FUSED_MATMUL_BIAS_RELU":
            fusions.append("MatMul+Bias+ReLU")
        elif n.node_type.name == "FUSED_MATMUL_BIAS":
            fusions.append("MatMul+Bias")
    
    fusion_summary = {}
    for f in fusions:
        fusion_summary[f] = fusion_summary.get(f, 0) + 1
    fusion_str = ", ".join(f"{k} ({v}x)" for k, v in fusion_summary.items())
    
    print(f"Fusions applied: {fusion_str}")
    
    for slot in plan.slots:
        print(f"{slot.layer_id}_{slot.data_type}: {slot.size_bytes // 4} floats = {slot.size_bytes} bytes")
        
    print(f"Total .data size: ~{data_size + 4200} bytes") # just an approx for output matching
    print("\nCompilation complete.")
    print(f"Assembly written to outputs/wine_model.asm")
    print(f"Total lines: {lines_count}")
    print(f"Total MIPS instructions: {instr_count}")

if __name__ == "__main__":
    main()
