"""Tests for graph_optimizer.py"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_parser import ModelParser
from graph_optimizer import GraphOptimizer
from graph import NodeType

class TestOptimizer(unittest.TestCase):
    def _get_graph(self, name):
        p = ModelParser()
        g, m = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", name))
        return g, m

    def test_fusion_xor(self):
        g, _ = self._get_graph("xor.json")
        opt = GraphOptimizer()
        g = opt.optimize(g)
        types = {n.node_type for n in g.nodes.values()}
        self.assertIn(NodeType.FUSED_MATMUL_BIAS_RELU, types)
        self.assertNotIn(NodeType.MATMUL, types)
        self.assertNotIn(NodeType.BIAS_ADD, types)

    def test_fusion_iris(self):
        g, _ = self._get_graph("iris.json")
        opt = GraphOptimizer()
        g = opt.optimize(g)
        types = {n.node_type for n in g.nodes.values()}
        self.assertIn(NodeType.FUSED_MATMUL_BIAS_RELU, types)
        self.assertIn(NodeType.SOFTMAX, types)

    def test_unroll_annotation(self):
        g, _ = self._get_graph("xor.json")
        opt = GraphOptimizer()
        g = opt.optimize(g)
        for n in g.nodes.values():
            if n.node_type in (NodeType.FUSED_MATMUL_BIAS_RELU, NodeType.FUSED_MATMUL_BIAS):
                self.assertIn("unroll", n.metadata)

    def test_dead_node_linear_removed(self):
        """LINEAR activation nodes should be bypassed."""
        g, _ = self._get_graph("xor.json")
        opt = GraphOptimizer()
        g = opt.optimize(g)
        for n in g.nodes.values():
            self.assertNotEqual(n.node_type, NodeType.LINEAR)

if __name__ == "__main__":
    unittest.main()
