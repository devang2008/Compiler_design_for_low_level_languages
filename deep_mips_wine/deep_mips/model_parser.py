"""
model_parser.py — JSON → Computation Graph
============================================
Reads a trained neural-network model in JSON format, validates
every field, builds typed ModelDef dataclasses, and constructs
the ComputationGraph that the rest of the pipeline operates on.

This is the "Lexer + Parser" stage for the Deep-MIPS compiler.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List

from errors import ModelParseError
from model_schema import WeightMatrix, BiasVector, LayerDef, ModelDef
from graph import ComputationGraph, GraphNode, NodeType


# Map activation string → NodeType
_ACTIVATION_MAP = {
    "relu":    NodeType.RELU,
    "sigmoid": NodeType.SIGMOID,
    "tanh":    NodeType.TANH,
    "softmax": NodeType.SOFTMAX,
    "linear":  NodeType.LINEAR,
}

VALID_LAYER_TYPES = {"Dense", "Flatten", "Softmax"}
VALID_ACTIVATIONS = set(_ACTIVATION_MAP.keys())


class ModelParser:
    """Parse a JSON model file and produce a ComputationGraph.

    Usage:
        parser = ModelParser()
        graph, model_def = parser.parse("models/xor.json")
    """

    # ── public entry point ────────────────────────────────────────

    def parse(self, filepath: str):
        """Read *filepath*, validate, and return (ComputationGraph, ModelDef)."""
        raw = self._load_json(filepath)
        model_def = self._build_model_def(raw)
        graph = self._build_graph(model_def)
        return graph, model_def

    # ── Step 1: load and validate JSON ────────────────────────────

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON from disk and perform top-level validation."""
        if not os.path.isfile(filepath):
            raise ModelParseError(f"file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelParseError(f"invalid JSON: {e}")

        # Top-level required fields
        for key in ("name", "input_shape", "output_shape", "layers"):
            if key not in raw:
                raise ModelParseError(f"missing required field", field=key)

        if not isinstance(raw["layers"], list) or len(raw["layers"]) == 0:
            raise ModelParseError("layers must be a non-empty list", field="layers")

        # Validate each layer
        for idx, layer in enumerate(raw["layers"]):
            self._validate_layer_json(layer, idx)

        return raw

    def _validate_layer_json(self, layer: Dict, idx: int) -> None:
        """Validate a single layer's JSON structure."""
        required = ("id", "type", "input_size", "output_size",
                     "activation", "weights", "biases")
        for key in required:
            if key not in layer:
                raise ModelParseError(
                    f"layer {idx} missing required field", field=key
                )

        lid = layer["id"]
        ltype = layer["type"]
        if ltype not in VALID_LAYER_TYPES:
            raise ModelParseError(
                f"layer '{lid}' has unsupported type '{ltype}'", field="type"
            )

        act = layer["activation"]
        if act not in VALID_ACTIVATIONS:
            raise ModelParseError(
                f"layer '{lid}' has unsupported activation '{act}'",
                field="activation"
            )

        # Validate weight/bias shapes for Dense layers
        if ltype == "Dense":
            in_sz = layer["input_size"]
            out_sz = layer["output_size"]
            weights = layer["weights"]
            biases = layer["biases"]

            if not isinstance(weights, list) or len(weights) != in_sz:
                raise ModelParseError(
                    f"layer '{lid}': weights must have {in_sz} rows, "
                    f"got {len(weights) if isinstance(weights, list) else 'non-list'}",
                    field="weights"
                )
            for i, row in enumerate(weights):
                if not isinstance(row, list) or len(row) != out_sz:
                    raise ModelParseError(
                        f"layer '{lid}' weights row {i}: expected {out_sz} cols, "
                        f"got {len(row) if isinstance(row, list) else 'non-list'}",
                        field="weights"
                    )

            if not isinstance(biases, list) or len(biases) != out_sz:
                raise ModelParseError(
                    f"layer '{lid}': biases must have {out_sz} values",
                    field="biases"
                )

    # ── Step 2: build ModelDef ────────────────────────────────────

    def _build_model_def(self, raw: Dict) -> ModelDef:
        """Convert validated JSON into typed ModelDef."""
        layers: List[LayerDef] = []
        for layer_json in raw["layers"]:
            lid = layer_json["id"]
            ltype = layer_json["type"]

            if ltype == "Dense":
                wm = WeightMatrix(
                    data=layer_json["weights"],
                    input_size=layer_json["input_size"],
                    output_size=layer_json["output_size"],
                    label=f"{lid}_weights",
                )
                bv = BiasVector(
                    data=layer_json["biases"],
                    size=layer_json["output_size"],
                    label=f"{lid}_biases",
                )
            else:
                wm = None
                bv = None

            layers.append(LayerDef(
                id=lid,
                type=ltype,
                input_size=layer_json["input_size"],
                output_size=layer_json["output_size"],
                activation=layer_json["activation"],
                weights=wm,
                biases=bv,
            ))

        return ModelDef(
            name=raw["name"],
            input_shape=raw["input_shape"],
            output_shape=raw["output_shape"],
            quantize=raw.get("quantize", False),
            layers=layers,
        )

    # ── Step 3: build ComputationGraph ────────────────────────────

    def _build_graph(self, model: ModelDef) -> ComputationGraph:
        """Construct the full computation graph from a ModelDef."""
        g = ComputationGraph()

        # INPUT node
        inp_node = GraphNode(
            id="input",
            node_type=NodeType.INPUT,
            input_shape=model.input_shape,
            output_shape=model.input_shape,
            input_size=model.input_shape[0],
            output_size=model.input_shape[0],
        )
        g.add_node(inp_node)
        g.input_node_id = "input"
        prev_id = "input"

        weight_store: Dict[str, Any] = {}

        for layer in model.layers:
            if layer.type == "Dense":
                prev_id = self._add_dense_nodes(
                    g, layer, prev_id, weight_store
                )
            elif layer.type == "Flatten":
                prev_id = self._add_flatten_node(g, layer, prev_id)
            elif layer.type == "Softmax":
                prev_id = self._add_softmax_node(g, layer, prev_id)

        # OUTPUT node
        prev_node = g.get_node(prev_id)
        out_node = GraphNode(
            id="output",
            node_type=NodeType.OUTPUT,
            input_shape=prev_node.output_shape,
            output_shape=model.output_shape,
            input_size=prev_node.output_size,
            output_size=model.output_shape[0],
        )
        g.add_node(out_node)
        g.add_edge(prev_id, "output")
        g.output_node_id = "output"

        # Attach weight data
        g.weight_store = weight_store

        # Sort
        g.topological_sort()
        return g

    # ── Dense layer → 3 nodes ─────────────────────────────────────

    def _add_dense_nodes(
        self, g: ComputationGraph, layer: LayerDef,
        prev_id: str, weight_store: Dict
    ) -> str:
        """Create matmul → bias_add → activation nodes for a Dense layer."""
        lid = layer.id

        # ── matmul ────────────────────────────────────────────────
        matmul_id = f"{lid}_matmul"
        matmul_node = GraphNode(
            id=matmul_id,
            node_type=NodeType.MATMUL,
            input_shape=[layer.input_size],
            output_shape=[layer.output_size],
            input_size=layer.input_size,
            output_size=layer.output_size,
            weight_label=f"{lid}_weights",
        )
        g.add_node(matmul_node)
        g.add_edge(prev_id, matmul_id)

        # ── bias_add ──────────────────────────────────────────────
        bias_id = f"{lid}_bias"
        bias_node = GraphNode(
            id=bias_id,
            node_type=NodeType.BIAS_ADD,
            input_shape=[layer.output_size],
            output_shape=[layer.output_size],
            input_size=layer.output_size,
            output_size=layer.output_size,
            bias_label=f"{lid}_biases",
        )
        g.add_node(bias_node)
        g.add_edge(matmul_id, bias_id)

        # ── activation ────────────────────────────────────────────
        act_type = _ACTIVATION_MAP.get(layer.activation, NodeType.LINEAR)
        act_id = f"{lid}_act"
        act_node = GraphNode(
            id=act_id,
            node_type=act_type,
            input_shape=[layer.output_size],
            output_shape=[layer.output_size],
            input_size=layer.output_size,
            output_size=layer.output_size,
        )
        g.add_node(act_node)
        g.add_edge(bias_id, act_id)

        # Store weight data
        weight_store[layer.weights.label] = layer.weights
        weight_store[layer.biases.label] = layer.biases

        return act_id

    # ── Flatten layer → 1 node ────────────────────────────────────

    def _add_flatten_node(
        self, g: ComputationGraph, layer: LayerDef, prev_id: str
    ) -> str:
        fid = f"{layer.id}_flatten"
        node = GraphNode(
            id=fid,
            node_type=NodeType.FLATTEN,
            input_shape=[layer.input_size],
            output_shape=[layer.output_size],
            input_size=layer.input_size,
            output_size=layer.output_size,
        )
        g.add_node(node)
        g.add_edge(prev_id, fid)
        return fid

    # ── Softmax layer → 1 node ────────────────────────────────────

    def _add_softmax_node(
        self, g: ComputationGraph, layer: LayerDef, prev_id: str
    ) -> str:
        sid = f"{layer.id}_softmax"
        node = GraphNode(
            id=sid,
            node_type=NodeType.SOFTMAX,
            input_shape=[layer.input_size],
            output_shape=[layer.output_size],
            input_size=layer.input_size,
            output_size=layer.output_size,
        )
        g.add_node(node)
        g.add_edge(prev_id, sid)
        return sid
