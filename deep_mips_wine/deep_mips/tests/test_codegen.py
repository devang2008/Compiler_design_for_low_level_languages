"""Tests for code_generator.py (and end-to-end pipeline)"""
import sys, os, unittest, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import PythonForwardPass
from model_parser import ModelParser

class TestPythonForwardPass(unittest.TestCase):
    def _load(self, name):
        p = ModelParser()
        _, m = p.parse(os.path.join(os.path.dirname(__file__), "..", "models", name))
        return m

    def test_xor_00(self):
        m = self._load("xor.json")
        out = PythonForwardPass.forward(m, [0.0, 0.0])
        self.assertLess(out[0], 0.5)

    def test_xor_01(self):
        m = self._load("xor.json")
        out = PythonForwardPass.forward(m, [0.0, 1.0])
        self.assertGreater(out[0], 0.5)

    def test_xor_10(self):
        m = self._load("xor.json")
        out = PythonForwardPass.forward(m, [1.0, 0.0])
        self.assertGreater(out[0], 0.5)

    def test_xor_11(self):
        m = self._load("xor.json")
        out = PythonForwardPass.forward(m, [1.0, 1.0])
        self.assertLess(out[0], 0.5)

    def test_iris_output_shape(self):
        m = self._load("iris.json")
        out = PythonForwardPass.forward(m, [5.1, 3.5, 1.4, 0.2])
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(sum(out), 1.0, places=5)

    def test_iris_probabilities(self):
        m = self._load("iris.json")
        out = PythonForwardPass.forward(m, [5.1, 3.5, 1.4, 0.2])
        for v in out:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

if __name__ == "__main__":
    unittest.main()
