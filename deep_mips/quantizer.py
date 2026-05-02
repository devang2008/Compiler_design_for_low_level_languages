"""
quantizer.py — Float → Fixed-Point Conversion for Deep-MIPS
=============================================================
Converts floating-point weights and biases to Q8.8 fixed-point
integers for deployment on hardware without an FPU.

Q8.8 format:
    16-bit signed integer
    Upper 8 bits = integer part
    Lower 8 bits = fractional part
    SCALE = 256 (2^8)
    Range: approx -128.0 to +127.996

Arithmetic rules for generated MIPS code:
    Addition:   fixed_a + fixed_b           (no scaling)
    Multiply:   (fixed_a * fixed_b) >> 8    (shift right by 8)
"""

from __future__ import annotations
import math
from typing import Any, Dict, List

from errors import QuantizationError
from model_schema import WeightMatrix, BiasVector
from graph import ComputationGraph


class Quantizer:
    """Handles all float↔fixed-point conversions.

    Usage:
        q = Quantizer()
        graph = q.quantize_graph(graph)
    """

    SCALE: int = 256              # 2^8
    INT16_MIN: int = -32768
    INT16_MAX: int = 32767

    # ── scalar conversions ────────────────────────────────────────

    def float_to_fixed(self, value: float) -> int:
        """Convert a single float to Q8.8 fixed-point integer.

        Raises QuantizationError for NaN or Inf.
        """
        if math.isnan(value):
            raise QuantizationError("cannot quantize NaN")
        if math.isinf(value):
            raise QuantizationError("cannot quantize Inf")
        result = round(value * self.SCALE)
        # Clamp to 16-bit signed range
        result = max(self.INT16_MIN, min(self.INT16_MAX, result))
        return result

    def fixed_to_float(self, value: int) -> float:
        """Convert a Q8.8 fixed-point integer back to float."""
        return value / self.SCALE

    # ── matrix / vector conversions ───────────────────────────────

    def quantize_weight_matrix(self, wm: WeightMatrix) -> WeightMatrix:
        """Quantize every element in a WeightMatrix to fixed-point.

        The original float values are stored in metadata for error analysis.
        """
        original_data = [row[:] for row in wm.data]
        quantized_data = []
        for row in wm.data:
            quantized_data.append([self.float_to_fixed(v) for v in row])

        new_wm = WeightMatrix(
            data=quantized_data,
            input_size=wm.input_size,
            output_size=wm.output_size,
            label=wm.label,
            metadata={"original_float": original_data},
        )
        return new_wm

    def quantize_bias_vector(self, bv: BiasVector) -> BiasVector:
        """Quantize every element in a BiasVector to fixed-point."""
        original_data = bv.data[:]
        quantized_data = [self.float_to_fixed(v) for v in bv.data]

        new_bv = BiasVector(
            data=quantized_data,
            size=bv.size,
            label=bv.label,
            metadata={"original_float": original_data},
        )
        return new_bv

    # ── whole-graph quantization ──────────────────────────────────

    def quantize_graph(self, graph: ComputationGraph) -> ComputationGraph:
        """Walk the weight_store and quantize every matrix and vector.

        Sets graph.is_quantized = True and graph.scale_factor = SCALE.
        """
        new_store: Dict[str, Any] = {}
        for label, obj in graph.weight_store.items():
            if isinstance(obj, WeightMatrix):
                new_store[label] = self.quantize_weight_matrix(obj)
            elif isinstance(obj, BiasVector):
                new_store[label] = self.quantize_bias_vector(obj)
            else:
                new_store[label] = obj

        graph.weight_store = new_store
        graph.is_quantized = True
        graph.scale_factor = self.SCALE
        return graph

    # ── error analysis ────────────────────────────────────────────

    def compute_quantization_error(
        self, graph: ComputationGraph
    ) -> str:
        """Compute and format a per-label quantization error report.

        Returns a multi-line report string and also prints it.
        """
        lines = []
        lines.append("=" * 50)
        lines.append("  Quantization Error Report")
        lines.append("=" * 50)

        for label, obj in graph.weight_store.items():
            original = obj.metadata.get("original_float")
            if original is None:
                continue

            if isinstance(obj, WeightMatrix):
                errors = []
                for i in range(obj.input_size):
                    for j in range(obj.output_size):
                        orig_val = original[i][j]
                        quant_val = self.fixed_to_float(obj.data[i][j])
                        errors.append(abs(orig_val - quant_val))
            elif isinstance(obj, BiasVector):
                errors = []
                for k in range(obj.size):
                    orig_val = original[k]
                    quant_val = self.fixed_to_float(obj.data[k])
                    errors.append(abs(orig_val - quant_val))
            else:
                continue

            if errors:
                max_err = max(errors)
                mean_err = sum(errors) / len(errors)
                lines.append(
                    f"  {label:30s}  max_error={max_err:.6f}  "
                    f"mean_error={mean_err:.6f}"
                )

        lines.append("=" * 50)
        report = "\n".join(lines)
        print(report)
        return report
