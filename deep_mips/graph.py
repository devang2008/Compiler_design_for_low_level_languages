"""
graph.py — Computation Graph for Deep-MIPS Compiler
=====================================================
The Computation Graph is the core internal representation (IR)
of the neural network.  It replaces the AST from a traditional
compiler.  Each node represents one atomic operation (matmul,
bias_add, relu, etc.) and edges represent data-flow dependencies.

Kahn's algorithm (BFS topo-sort) is used to linearise the graph
into execution order before code generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from errors import GraphError


# ──────────────────────────────────────────────────────────────────
# Node type enumeration
# ──────────────────────────────────────────────────────────────────

class NodeType(Enum):
    """Every possible atomic operation in the computation graph."""
    INPUT                   = "input"
    MATMUL                  = "matmul"
    BIAS_ADD                = "bias_add"
    RELU                    = "relu"
    SIGMOID                 = "sigmoid"
    TANH                    = "tanh"
    SOFTMAX                 = "softmax"
    LINEAR                  = "linear"
    FLATTEN                 = "flatten"
    OUTPUT                  = "output"
    FUSED_MATMUL_BIAS       = "fused_matmul_bias"
    FUSED_MATMUL_BIAS_RELU  = "fused_matmul_bias_relu"


# ──────────────────────────────────────────────────────────────────
# Graph node
# ──────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A single node in the computation graph.

    Attributes:
        id            Unique string identifier (e.g. "layer_0_matmul")
        node_type     The operation this node performs
        input_shape   Shape of data entering this node
        output_shape  Shape of data leaving this node
        inputs        IDs of predecessor nodes
        outputs       IDs of successor nodes
        weight_label  .data label for the weight matrix (or None)
        bias_label    .data label for the bias vector (or None)
        input_size    Number of input neurons / elements
        output_size   Number of output neurons / elements
        fused_with    IDs of nodes that were fused into this one
        metadata      Arbitrary extra info for optimisation passes
    """
    id: str
    node_type: NodeType
    input_shape: List[int]       = field(default_factory=list)
    output_shape: List[int]      = field(default_factory=list)
    inputs: List[str]            = field(default_factory=list)
    outputs: List[str]           = field(default_factory=list)
    weight_label: Optional[str]  = None
    bias_label: Optional[str]    = None
    input_size: int              = 0
    output_size: int             = 0
    fused_with: List[str]        = field(default_factory=list)
    metadata: Dict[str, Any]     = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Computation graph
# ──────────────────────────────────────────────────────────────────

class ComputationGraph:
    """Directed acyclic graph of neural-network operations.

    Supports:
        - Adding / removing nodes and edges
        - Topological sorting via Kahn's algorithm
        - Predecessor / successor queries
        - Pretty-printing
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.input_node_id: Optional[str] = None
        self.output_node_id: Optional[str] = None
        self.topological_order: List[str] = []
        # Attached by ModelParser — maps label → WeightMatrix / BiasVector
        self.weight_store: Dict[str, Any] = {}
        self.is_quantized: bool = False
        self.scale_factor: int = 256

    # ── mutation ──────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        """Insert a node.  Raises GraphError on duplicate id."""
        if node.id in self.nodes:
            raise GraphError(f"duplicate node id '{node.id}'", node.id)
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges."""
        if node_id not in self.nodes:
            raise GraphError(f"node '{node_id}' not found", node_id)
        self.edges = [
            (u, v) for (u, v) in self.edges
            if u != node_id and v != node_id
        ]
        # Remove references from neighbours
        for nid, n in self.nodes.items():
            if node_id in n.inputs:
                n.inputs.remove(node_id)
            if node_id in n.outputs:
                n.outputs.remove(node_id)
        del self.nodes[node_id]

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge from_id → to_id."""
        if from_id not in self.nodes:
            raise GraphError(f"source node '{from_id}' not found", from_id)
        if to_id not in self.nodes:
            raise GraphError(f"target node '{to_id}' not found", to_id)
        self.edges.append((from_id, to_id))
        self.nodes[from_id].outputs.append(to_id)
        self.nodes[to_id].inputs.append(from_id)

    def replace_edge(self, old_from: str, old_to: str,
                     new_from: str, new_to: str) -> None:
        """Replace a specific edge (old) with a new edge."""
        self.edges = [
            (new_from, new_to) if (u == old_from and v == old_to) else (u, v)
            for (u, v) in self.edges
        ]
        if old_to in self.nodes[old_from].outputs:
            self.nodes[old_from].outputs.remove(old_to)
        if old_from in self.nodes[old_to].inputs:
            self.nodes[old_to].inputs.remove(old_from)
        if new_to not in self.nodes[new_from].outputs:
            self.nodes[new_from].outputs.append(new_to)
        if new_from not in self.nodes[new_to].inputs:
            self.nodes[new_to].inputs.append(new_from)

    # ── queries ───────────────────────────────────────────────────

    def get_node(self, node_id: str) -> GraphNode:
        """Look up a node by id.  Raises GraphError if missing."""
        if node_id not in self.nodes:
            raise GraphError(f"node '{node_id}' not found", node_id)
        return self.nodes[node_id]

    def predecessors(self, node_id: str) -> List[GraphNode]:
        """Return all immediate predecessor nodes."""
        node = self.get_node(node_id)
        return [self.nodes[nid] for nid in node.inputs if nid in self.nodes]

    def successors(self, node_id: str) -> List[GraphNode]:
        """Return all immediate successor nodes."""
        node = self.get_node(node_id)
        return [self.nodes[nid] for nid in node.outputs if nid in self.nodes]

    # ── topological sort (Kahn's algorithm) ───────────────────────

    def topological_sort(self) -> None:
        """Linearise the graph using Kahn's BFS-based algorithm.

        After this call, self.topological_order contains node ids
        in a valid execution order.  Raises GraphError if a cycle
        is detected.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for (u, v) in self.edges:
            if v in in_degree:
                in_degree[v] += 1

        queue: deque = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ_id in self.nodes[nid].outputs:
                if succ_id in in_degree:
                    in_degree[succ_id] -= 1
                    if in_degree[succ_id] == 0:
                        queue.append(succ_id)

        if len(order) != len(self.nodes):
            raise GraphError("cycle detected in computation graph")

        self.topological_order = order

    # ── pretty-print ──────────────────────────────────────────────

    def print_graph(self) -> str:
        """Return a human-readable summary of the graph."""
        if not self.topological_order:
            try:
                self.topological_sort()
            except GraphError:
                pass

        lines = []
        lines.append("=" * 60)
        lines.append("  Computation Graph")
        lines.append("=" * 60)

        order = self.topological_order or list(self.nodes.keys())
        for i, nid in enumerate(order):
            n = self.nodes[nid]
            fuse_info = ""
            if n.fused_with:
                fuse_info = f"  [fused: {', '.join(n.fused_with)}]"
            unroll = n.metadata.get("unroll", "")
            unroll_info = f"  [unroll: {unroll}]" if unroll else ""
            arrow = " -> " if i < len(order) - 1 else ""
            lines.append(
                f"  [{i}] {n.node_type.value:30s}  "
                f"shape: {n.input_shape} -> {n.output_shape}"
                f"{fuse_info}{unroll_info}"
            )
            if n.weight_label:
                lines.append(f"       weights: {n.weight_label}")
            if n.bias_label:
                lines.append(f"       biases:  {n.bias_label}")

        lines.append("=" * 60)
        # Also show a compact flow line
        flow_parts = []
        for nid in order:
            n = self.nodes[nid]
            flow_parts.append(n.node_type.value.upper())
        lines.append("  Flow: " + " -> ".join(flow_parts))
        lines.append("=" * 60)
        result = "\n".join(lines)
        return result
