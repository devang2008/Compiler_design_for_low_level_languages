"""Tests for model_parser.py"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_parser import ModelParser
from graph import NodeType
from errors import ModelParseError

class TestModelParser(unittest.TestCase):
    def test_xor_parse(self):
        p = ModelParser()
        g, m = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", "xor.json"))
        self.assertEqual(m.name, "xor_classifier")
        self.assertEqual(len(m.layers), 2)
        self.assertIn("input", g.nodes)
        self.assertIn("output", g.nodes)
        self.assertTrue(len(g.topological_order) > 0)

    def test_iris_parse(self):
        p = ModelParser()
        g, m = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", "iris.json"))
        self.assertEqual(m.input_shape, [4])
        self.assertEqual(m.output_shape, [3])

    def test_invalid_file(self):
        p = ModelParser()
        with self.assertRaises(ModelParseError):
            p.parse("nonexistent.json")

    def test_weight_store(self):
        p = ModelParser()
        g, _ = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", "xor.json"))
        self.assertIn("layer_0_weights", g.weight_store)
        self.assertIn("layer_0_biases", g.weight_store)

    def test_topo_order_starts_with_input(self):
        p = ModelParser()
        g, _ = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", "xor.json"))
        self.assertEqual(g.topological_order[0], "input")
        self.assertEqual(g.topological_order[-1], "output")

if __name__ == "__main__":
    unittest.main()
