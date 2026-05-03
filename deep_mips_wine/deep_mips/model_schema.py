"""
model_schema.py — Dataclass Definitions for Neural Network Models
==================================================================
These dataclasses form the typed internal representation of a model
after JSON parsing but before graph construction.

Classes:
    WeightMatrix  — 2D weight array for a Dense layer
    BiasVector    — 1D bias array for a Dense layer
    LayerDef      — Complete definition of one neural network layer
    ModelDef      — Complete definition of an entire neural network
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class WeightMatrix:
    """A 2D weight matrix for a Dense layer.

    Shape: [input_size][output_size]
    Each inner list holds one input neuron's connections to all output neurons.
    """
    data: List[List[float]]
    input_size: int
    output_size: int
    label: str                           # e.g. "layer_0_weights"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flat(self) -> List[float]:
        """Return weights in row-major order (i*output_size + j)."""
        result = []
        for row in self.data:
            result.extend(row)
        return result

    def __post_init__(self):
        if len(self.data) != self.input_size:
            raise ValueError(
                f"WeightMatrix '{self.label}': expected {self.input_size} rows, "
                f"got {len(self.data)}"
            )
        for i, row in enumerate(self.data):
            if len(row) != self.output_size:
                raise ValueError(
                    f"WeightMatrix '{self.label}' row {i}: expected {self.output_size} "
                    f"columns, got {len(row)}"
                )


@dataclass
class BiasVector:
    """A 1D bias vector for a Dense layer.

    Length: output_size
    """
    data: List[float]
    size: int
    label: str                           # e.g. "layer_0_biases"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.data) != self.size:
            raise ValueError(
                f"BiasVector '{self.label}': expected {self.size} values, "
                f"got {len(self.data)}"
            )


@dataclass
class LayerDef:
    """Complete definition of one neural network layer.

    Supported types: "Dense", "Flatten", "Softmax"
    Supported activations: "relu", "sigmoid", "tanh", "softmax", "linear"
    """
    id: str
    type: str
    input_size: int
    output_size: int
    activation: str
    weights: Optional[WeightMatrix]
    biases: Optional[BiasVector]


@dataclass
class ModelDef:
    """Complete definition of a trained neural network.

    Contains the network name, I/O shapes, quantization flag,
    and an ordered list of layer definitions with their pre-trained weights.
    """
    name: str
    input_shape: List[int]
    output_shape: List[int]
    quantize: bool
    layers: List[LayerDef]
