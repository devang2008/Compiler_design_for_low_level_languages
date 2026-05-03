"""
graph_optimizer.py — Optimisation Passes for the Computation Graph
===================================================================
Runs a sequence of graph-rewriting passes before code generation:

  Pass 1  Operator Fusion      — merge matmul+bias, matmul+bias+relu
  Pass 2  Constant Folding     — mark zero biases so code-gen can skip adds
  Pass 3  Dead Node Elimination — remove identity (linear) nodes, unreachable nodes
  Pass 4  Loop Unroll Annotation — tag layers with unroll strategy
"""

from __future__ import annotations
from typing import List, Set

from errors import GraphError
from graph import ComputationGraph, GraphNode, NodeType


class GraphOptimizer:
    """Apply optimisation passes to a ComputationGraph.

    Usage:
        opt = GraphOptimizer()
        graph = opt.optimize(graph)
    """

    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """Run all passes in order and return the modified graph."""
        graph = self.pass_operator_fusion(graph)
        graph = self.pass_constant_folding(graph)
        graph = self.pass_dead_node_elimination(graph)
        graph = self.pass_loop_unroll_annotation(graph)
        return graph

    # ── Pass 1: Operator Fusion ───────────────────────────────────

    def pass_operator_fusion(self, graph: ComputationGraph) -> ComputationGraph:
        """Fuse matmul+bias and matmul+bias+relu into single nodes.

        Pattern 1: MATMUL → BIAS_ADD  ⟹  FUSED_MATMUL_BIAS
        Pattern 2: FUSED_MATMUL_BIAS → RELU  ⟹  FUSED_MATMUL_BIAS_RELU

        Fusion eliminates intermediate memory writes.
        """
        report: List[str] = []

        # ── Pattern 1: MATMUL + BIAS_ADD → FUSED_MATMUL_BIAS ─────
        changed = True
        while changed:
            changed = False
            for nid in list(graph.nodes):
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                if node.node_type != NodeType.MATMUL:
                    continue
                # Single successor that is BIAS_ADD?
                succs = graph.successors(nid)
                if len(succs) != 1:
                    continue
                bias_node = succs[0]
                if bias_node.node_type != NodeType.BIAS_ADD:
                    continue
                # Bias node must have exactly one input
                if len(bias_node.inputs) != 1:
                    continue

                # Fuse
                fused_id = nid.replace("_matmul", "_fused_mb")
                fused = GraphNode(
                    id=fused_id,
                    node_type=NodeType.FUSED_MATMUL_BIAS,
                    input_shape=node.input_shape,
                    output_shape=bias_node.output_shape,
                    input_size=node.input_size,
                    output_size=node.output_size,
                    weight_label=node.weight_label,
                    bias_label=bias_node.bias_label,
                    fused_with=[nid, bias_node.id],
                )
                graph.add_node(fused)

                # Rewire predecessors of matmul → fused
                for pred in graph.predecessors(nid):
                    graph.replace_edge(pred.id, nid, pred.id, fused_id)

                # Rewire successors of bias → fused
                for succ in graph.successors(bias_node.id):
                    graph.replace_edge(bias_node.id, succ.id, fused_id, succ.id)

                # Remove old nodes
                graph.remove_node(nid)
                graph.remove_node(bias_node.id)

                report.append(
                    f"  Fused {nid} + {bias_node.id} -> {fused_id} "
                    f"(FUSED_MATMUL_BIAS)"
                )
                changed = True
                break

        # ── Pattern 2: FUSED_MATMUL_BIAS + RELU → FUSED_MATMUL_BIAS_RELU
        changed = True
        while changed:
            changed = False
            for nid in list(graph.nodes):
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                if node.node_type != NodeType.FUSED_MATMUL_BIAS:
                    continue
                succs = graph.successors(nid)
                if len(succs) != 1:
                    continue
                relu_node = succs[0]
                if relu_node.node_type != NodeType.RELU:
                    continue
                if len(relu_node.inputs) != 1:
                    continue

                fused_id = nid.replace("_fused_mb", "_fused_mbr")
                fused = GraphNode(
                    id=fused_id,
                    node_type=NodeType.FUSED_MATMUL_BIAS_RELU,
                    input_shape=node.input_shape,
                    output_shape=relu_node.output_shape,
                    input_size=node.input_size,
                    output_size=node.output_size,
                    weight_label=node.weight_label,
                    bias_label=node.bias_label,
                    fused_with=node.fused_with + [relu_node.id],
                )
                graph.add_node(fused)

                for pred in graph.predecessors(nid):
                    graph.replace_edge(pred.id, nid, pred.id, fused_id)
                for succ in graph.successors(relu_node.id):
                    graph.replace_edge(relu_node.id, succ.id, fused_id, succ.id)

                graph.remove_node(nid)
                graph.remove_node(relu_node.id)

                report.append(
                    f"  Fused {nid} + {relu_node.id} -> {fused_id} "
                    f"(FUSED_MATMUL_BIAS_RELU)"
                )
                changed = True
                break

        # Re-sort
        graph.topological_sort()

        print("=" * 50)
        print("  Fusion Report")
        print("=" * 50)
        if report:
            for line in report:
                print(line)
            print(f"  Total fusions applied: {len(report)}")
        else:
            print("  No fusible patterns found.")
        print("=" * 50)

        return graph

    # ── Pass 2: Constant Folding ──────────────────────────────────

    def pass_constant_folding(self, graph: ComputationGraph) -> ComputationGraph:
        """Mark zero biases so code-gen can skip the add instruction.

        Since all weights are compile-time constants, any bias == 0.0
        can be folded away (the add $t0, $t0, $zero is a no-op).
        """
        eliminated = 0
        epsilon = 1e-6

        for nid, node in graph.nodes.items():
            if node.bias_label and node.bias_label in graph.weight_store:
                bv = graph.weight_store[node.bias_label]
                zero_flags = []
                for val in bv.data:
                    is_zero = abs(val) < epsilon
                    zero_flags.append(is_zero)
                    if is_zero:
                        eliminated += 1
                node.metadata["zero_biases"] = zero_flags

        print("=" * 50)
        print("  Constant Folding Report")
        print("=" * 50)
        print(f"  Zero-bias additions eliminated: {eliminated}")
        print("=" * 50)
        return graph

    # ── Pass 3: Dead Node Elimination ─────────────────────────────

    def pass_dead_node_elimination(self, graph: ComputationGraph) -> ComputationGraph:
        """Remove identity (LINEAR) nodes and unreachable nodes.

        A LINEAR activation is identity (output = input), so we
        bypass it with a direct edge.
        """
        eliminated: List[str] = []

        # Remove LINEAR (identity) nodes by bypassing
        for nid in list(graph.nodes):
            node = graph.nodes.get(nid)
            if node is None:
                continue
            if node.node_type != NodeType.LINEAR:
                continue

            preds = graph.predecessors(nid)
            succs = graph.successors(nid)
            if len(preds) == 1 and len(succs) == 1:
                pred = preds[0]
                succ = succs[0]
                graph.replace_edge(pred.id, nid, pred.id, succ.id)
                graph.remove_node(nid)
                eliminated.append(nid)

        # Remove unreachable nodes (no path from INPUT)
        if graph.input_node_id:
            reachable: Set[str] = set()
            stack = [graph.input_node_id]
            while stack:
                curr = stack.pop()
                if curr in reachable:
                    continue
                reachable.add(curr)
                if curr in graph.nodes:
                    for succ_id in graph.nodes[curr].outputs:
                        stack.append(succ_id)

            for nid in list(graph.nodes):
                if nid not in reachable:
                    graph.remove_node(nid)
                    eliminated.append(nid)

        if eliminated:
            graph.topological_sort()

        print("=" * 50)
        print("  Dead Node Elimination Report")
        print("=" * 50)
        if eliminated:
            for nid in eliminated:
                print(f"  Eliminated: {nid}")
        else:
            print("  No dead nodes found.")
        print("=" * 50)
        return graph

    # ── Pass 4: Loop Unroll Annotation ────────────────────────────

    def pass_loop_unroll_annotation(self, graph: ComputationGraph) -> ComputationGraph:
        """Tag MATMUL and FUSED nodes with their unroll strategy.

        Strategies:
            output_size ≤  4  → "full"       zero loop overhead
            output_size ≤ 16  → "partial_4"  unroll by factor 4
            output_size >  16 → "none"       standard nested loop
        """
        report: List[str] = []
        compute_types = {
            NodeType.MATMUL,
            NodeType.FUSED_MATMUL_BIAS,
            NodeType.FUSED_MATMUL_BIAS_RELU,
        }

        for nid, node in graph.nodes.items():
            if node.node_type not in compute_types:
                continue

            out_sz = node.output_size
            if out_sz <= 4:
                strategy = "full"
                reason = f"output_size={out_sz} <= 4 -> fully unrolled"
            elif out_sz <= 16:
                strategy = "partial_4"
                reason = f"output_size={out_sz} <= 16 -> unroll by 4"
            else:
                strategy = "none"
                reason = f"output_size={out_sz} > 16 -> standard loop"

            node.metadata["unroll"] = strategy
            report.append(f"  {nid}: {reason}")

        print("=" * 50)
        print("  Loop Unroll Annotation Report")
        print("=" * 50)
        for line in report:
            print(line)
        print("=" * 50)
        return graph
