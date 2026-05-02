"""Tests for memory_planner.py"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_parser import ModelParser
from graph_optimizer import GraphOptimizer
from memory_planner import MemoryPlanner

class TestMemoryPlanner(unittest.TestCase):
    def _plan(self, name, quantized=False):
        p = ModelParser()
        g, m = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", name))
        opt = GraphOptimizer()
        g = opt.optimize(g)
        mp = MemoryPlanner()
        return mp.plan(g, is_quantized=quantized)

    def test_xor_plan(self):
        plan = self._plan("xor.json")
        self.assertEqual(plan.element_size, 4)
        self.assertIn("act_buffer_a", plan.data_section)
        self.assertIn("input_buffer", plan.data_section)

    def test_slots_exist(self):
        plan = self._plan("iris.json")
        labels = [s.label for s in plan.slots]
        self.assertIn("act_buffer_a", labels)
        self.assertIn("act_buffer_b", labels)
        self.assertIn("input_buffer", labels)

    def test_quantized_element_size(self):
        plan = self._plan("xor.json", quantized=True)
        self.assertEqual(plan.element_size, 2)

    def test_data_section_contains_weights(self):
        plan = self._plan("xor.json")
        self.assertIn("layer_0_weights", plan.data_section)
        self.assertIn("layer_1_weights", plan.data_section)

if __name__ == "__main__":
    unittest.main()
