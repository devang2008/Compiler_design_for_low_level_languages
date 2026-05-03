"""
memory_planner.py - Memory layout planner for Deep-MIPS
Supports both integer (.word) and FPU (.float) data modes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from errors import CodeGenError
from graph import ComputationGraph, NodeType
from model_schema import ModelDef


@dataclass
class MemorySlot:
    label: str
    size_bytes: int
    data_type: str          # "weights", "biases", "input", "buffer"
    layer_id: str


@dataclass
class MemoryPlan:
    data_section: str
    slots: List[MemorySlot]
    input_buffer_label: str
    buffer_a_label: str
    buffer_b_label: str
    element_size: int       # 4 for float/int, 2 for quantized half
    use_fpu: bool           # True = FPU float mode


class MemoryPlanner:
    """
    Assigns MIPS memory addresses to all weights, biases, and
    activation buffers. Supports three modes:
      - FPU mode (use_fpu=True):   weights stored as .float (IEEE 754)
      - Integer mode:              weights stored as .word (scaled int)
      - Quantized mode:            weights stored as .half (Q8.8)
    """

    # LUT for sigmoid piecewise approximation (sigmoid(x)*256 for x=-4..4)
    SIGMOID_LUT = [0, 5, 12, 25, 50, 88, 128, 168, 206, 231, 244, 249, 252, 254, 255, 256]
    # Values at x = -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4

    def plan(self,
             graph: ComputationGraph,
             is_quantized: bool,
             use_fpu: bool = False) -> MemoryPlan:
        """
        Build the full memory plan for the model.
        Returns a MemoryPlan with a ready-to-emit .data section string.
        """
        if is_quantized and use_fpu:
            raise CodeGenError("Cannot use FPU mode with quantization simultaneously.")

        element_size = 2 if is_quantized else 4
        lines: List[str] = []
        slots: List[MemorySlot] = []

        # ── Sigmoid LUT (always included — needed for sigmoid activation) ─
        lut_vals = ", ".join(str(v) for v in self.SIGMOID_LUT)
        lines.append(f"sigmoid_lut:    .word  {lut_vals}")

        # ── Collect all weight/bias data from graph.weight_store ──────────
        # Sort by size descending (largest allocation first)
        weight_items = []
        for label, data_obj in graph.weight_store.items():
            if hasattr(data_obj, 'data') and isinstance(data_obj.data[0], list):
                # WeightMatrix: flatten row-major
                flat = [v for row in data_obj.data for v in row]
            else:
                # BiasVector: already flat
                flat = list(data_obj.data)
            weight_items.append((label, flat, len(flat)))

        # Sort largest first
        weight_items.sort(key=lambda x: x[2], reverse=True)

        for label, flat_data, count in weight_items:
            size_bytes = count * element_size
            slots.append(MemorySlot(
                label=label,
                size_bytes=size_bytes,
                data_type="weights" if "weight" in label else "biases",
                layer_id=label.split("_weights")[0].split("_biases")[0]
            ))

            if use_fpu:
                # Store as IEEE 754 single-precision floats (.float directive)
                # MARS simulator supports .float natively
                lines.extend(self._emit_float_data(label, flat_data))

            elif is_quantized:
                # Store as Q8.8 fixed-point halfwords (.half directive)
                lines.extend(self._emit_half_data(label, flat_data, scale=256))

            else:
                # Store as scaled integers (.word directive)
                lines.extend(self._emit_word_data_int(label, flat_data))

        # ── Find max activation buffer size needed ────────────────────────
        max_act_size = 0
        for nid in graph.topological_order:
            node = graph.get_node(nid)
            if node.node_type not in (NodeType.INPUT, NodeType.OUTPUT):
                max_act_size = max(max_act_size, node.output_size)

        buf_bytes = max_act_size * element_size
        input_bytes = graph.get_node(graph.input_node_id).output_size * element_size

        # ── Emit activation buffers and input buffer ──────────────────────
        lines.append(f"input_buffer:   .space  {input_bytes}")
        lines.append(f"act_buffer_a:   .space  {buf_bytes}")
        lines.append(f"act_buffer_b:   .space  {buf_bytes}")

        # ── Float constants used by FPU softmax exp subroutine ────────────
        if use_fpu:
            lines.append("fpu_zero:       .float  0.0")
            lines.append("fpu_one:        .float  1.0")
            lines.append("fpu_half:       .float  0.5")
            lines.append("fpu_ln2:        .float  0.693147")
            lines.append("fpu_inv_ln2:    .float  1.442695")
            # Taylor coefficients for exp(x) on [-0.5, 0.5]:
            # e^x ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
            lines.append("exp_c0:         .float  1.0")
            lines.append("exp_c1:         .float  1.0")
            lines.append("exp_c2:         .float  0.5")
            lines.append("exp_c3:         .float  0.166667")
            lines.append("exp_c4:         .float  0.041667")
            lines.append("exp_clamp_neg:  .float  -88.0")
            lines.append("exp_clamp_pos:  .float  88.0")

        data_section = "\n".join(lines)

        return MemoryPlan(
            data_section=data_section,
            slots=slots,
            input_buffer_label="input_buffer",
            buffer_a_label="act_buffer_a",
            buffer_b_label="act_buffer_b",
            element_size=element_size,
            use_fpu=use_fpu,
        )

    # ── Data emission helpers ──────────────────────────────────────────────

    def _emit_float_data(self, label: str, values: List[float],
                         per_line: int = 8) -> List[str]:
        """
        Emit values as .float directives (IEEE 754 single precision).
        MARS supports .float natively — no bit-twiddling needed.
        """
        lines = []
        for i in range(0, len(values), per_line):
            chunk = values[i:i + per_line]
            formatted = ", ".join(f"{v:.8f}" for v in chunk)
            prefix = f"{label}:" if i == 0 else " " * (len(label) + 1)
            lines.append(f"{prefix:<24}.float  {formatted}")
        return lines

    def _emit_half_data(self, label: str, values: List[float],
                        scale: int = 256, per_line: int = 8) -> List[str]:
        """Emit values as Q8.8 fixed-point .half directives."""
        lines = []
        int_vals = [max(-32768, min(32767, round(v * scale))) for v in values]
        for i in range(0, len(int_vals), per_line):
            chunk = int_vals[i:i + per_line]
            formatted = ", ".join(str(v) for v in chunk)
            prefix = f"{label}:" if i == 0 else " " * (len(label) + 1)
            lines.append(f"{prefix:<24}.half   {formatted}")
        # Align to 4-byte boundary after .half section
        if len(values) % 2 != 0:
            lines.append(" " * 24 + ".align  2")
        return lines

    def _emit_word_data_int(self, label: str, values: List[float],
                            per_line: int = 8) -> List[str]:
        """Emit values as scaled integers in .word directives."""
        lines = []
        int_vals = [int(v) for v in values]
        for i in range(0, len(int_vals), per_line):
            chunk = int_vals[i:i + per_line]
            formatted = ", ".join(str(v) for v in chunk)
            prefix = f"{label}:" if i == 0 else " " * (len(label) + 1)
            lines.append(f"{prefix:<24}.word   {formatted}")
        return lines
