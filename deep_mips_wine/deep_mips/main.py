"""
main.py — Deep-MIPS Compiler Entry Point
==========================================
Wires the full compilation pipeline and provides the CLI.

Usage:
    python main.py models/xor.json
    python main.py models/iris.json --quantize --graph
    python main.py models/xor.json --compare
"""

import sys
import os
import math
import argparse
from typing import List

from errors import (ModelParseError, GraphError, MemoryPlannerError,
                    CodeGenError, QuantizationError)
from model_schema import ModelDef
from model_parser import ModelParser
from quantizer import Quantizer
from graph_optimizer import GraphOptimizer
from memory_planner import MemoryPlanner
from runtime_lib import RuntimeLibrary


# ──────────────────────────────────────────────────────────────────
# Pure-Python Forward Pass (for --compare verification)
# ──────────────────────────────────────────────────────────────────

class PythonForwardPass:
    """Run inference in pure Python (no numpy) to verify MIPS output."""

    @staticmethod
    def forward(model: ModelDef, input_data: List[float]) -> List[float]:
        """Execute the full forward pass and return output values."""
        current = list(input_data)

        for layer in model.layers:
            if layer.type != "Dense":
                continue

            output = []
            for j in range(layer.output_size):
                acc = layer.biases.data[j]
                for i in range(layer.input_size):
                    acc += current[i] * layer.weights.data[i][j]
                output.append(acc)

            # Apply activation
            act = layer.activation
            if act == "relu":
                output = [max(0.0, v) for v in output]
            elif act == "sigmoid":
                output = [PythonForwardPass._sigmoid(v) for v in output]
            elif act == "tanh":
                output = [math.tanh(v) for v in output]
            elif act == "softmax":
                output = PythonForwardPass._softmax(output)
            # "linear" → no change

            current = output

        return current

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 500:
            return 1.0
        if x <= -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _softmax(vals: List[float]) -> List[float]:
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        return [e / s for e in exps]


# ──────────────────────────────────────────────────────────────────
# Test inputs per model
# ──────────────────────────────────────────────────────────────────

def get_test_input(model: ModelDef) -> List[float]:
    """Return a representative test input for the model."""
    n = model.input_shape[0]
    if "xor" in model.name.lower():
        return [0.0, 1.0]  # XOR input (0,1) → should output ~1
    if "iris" in model.name.lower():
        return [5.1, 3.5, 1.4, 0.2]  # setosa sample
    # Default: simple pattern
    return [float(i % 10) / 10.0 for i in range(n)]


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

def compile_model(filepath: str, args) -> None:
    """Run the full Deep-MIPS compilation pipeline."""

    # 1. Parse
    print(f"\n[1/6] Parsing model: {filepath}")
    parser = ModelParser()
    graph, model_def = parser.parse(filepath)

    if args.graph:
        print("\n-- Graph BEFORE optimization --")
        print(graph.print_graph())

    # 2. Quantize (optional)
    if args.quantize or model_def.quantize:
        print("\n[2/6] Quantizing weights (Q8.8 fixed-point)...")
        q = Quantizer()
        graph = q.quantize_graph(graph)
        q.compute_quantization_error(graph)
    else:
        print("[2/6] Quantization: skipped (float mode)")

    # 3. Optimize
    print("\n[3/6] Running optimization passes...")
    opt = GraphOptimizer()
    graph = opt.optimize(graph)

    if args.graph:
        print("\n-- Graph AFTER optimization --")
        print(graph.print_graph())

    # 4. Memory plan
    print("\n[4/6] Planning memory layout...")
    mem = MemoryPlanner()
    plan = mem.plan(graph, is_quantized=graph.is_quantized, use_fpu=args.fpu)

    if args.memory:
        mem.print_plan(plan)

    # 5. Code generation
    print("\n[5/6] Generating MIPS assembly...")
    try:
        from code_generator import CodeGenerator
        cg = CodeGenerator()
        asm_body = cg.generate(graph, plan, model_def)
    except ImportError:
        # Fallback: generate a minimal stub
        asm_body = _generate_stub(graph, plan, model_def)

    # 6. Append runtime library
    rt = RuntimeLibrary()
    asm_text = asm_body + "\n" + rt.generate_all()

    # Write output
    out_path = filepath.rsplit(".", 1)[0] + ".asm"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(asm_text)

    n_lines = asm_text.count("\n") + 1
    n_bytes = len(asm_text.encode("utf-8"))
    print(f"\n[6/6] Compiled: {out_path}  ({n_lines} lines, {n_bytes} bytes)")

    if args.asm:
        print("\n-- Generated Assembly --")
        print(asm_text)

    # 7. Compare (optional)
    if args.compare:
        test_input = get_test_input(model_def)
        py_out = PythonForwardPass.forward(model_def, test_input)
        print("\n" + "=" * 55)
        print("  Python vs MIPS Comparison")
        print("=" * 55)
        print(f"  Input: {test_input}")
        print(f"  Python output: {[round(v, 6) for v in py_out]}")
        if len(py_out) > 1:
            print(f"  Predicted class: {py_out.index(max(py_out))}")
        else:
            print(f"  Predicted value: {round(py_out[0], 4)}")
        print("  (Run the .asm in MARS to see MIPS output)")
        print("=" * 55)


def _generate_stub(graph, plan, model_def):
    """Generate assembly when code_generator.py is not yet ready."""
    lines = []
    lines.append("# Deep-MIPS Inference Engine")
    lines.append(f"# Model: {model_def.name}")
    lines.append(f"# Input: {model_def.input_shape}  Output: {model_def.output_shape}")
    lines.append(f"# Quantized: {graph.is_quantized}")
    lines.append("")
    lines.append(".data")
    lines.append(plan.data_section)
    lines.append("")
    lines.append(".text")
    lines.append(".globl main")
    lines.append("")
    lines.append("main:")
    lines.append("    li    $v0, 10")
    lines.append("    syscall")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deep-MIPS Neural Network Compiler")
    ap.add_argument("model", help="Path to .json model file")
    ap.add_argument("--fpu", action="store_true", help="Use IEEE 754 floats")
    ap.add_argument("--quantize", action="store_true", help="Force quantization")
    ap.add_argument("--graph", action="store_true", help="Print computation graph")
    ap.add_argument("--memory", action="store_true", help="Print memory plan")
    ap.add_argument("--asm", action="store_true", help="Print generated assembly")
    ap.add_argument("--compare", action="store_true", help="Compare Python vs MIPS")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: '{args.model}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        compile_model(args.model, args)
    except (ModelParseError, GraphError, MemoryPlannerError,
            CodeGenError, QuantizationError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
