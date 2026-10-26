import unittest

import numpy as np

import aner
from aner.modal import forecast


class AnerTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(aner.version(), "0.1.0")

    def test_const_add_on_interpreter(self):
        graph = aner.Graph()
        state = graph.input(4)
        bias = graph.const([1.0, 2.0, 3.0, 4.0])
        graph.output(graph.add(state, bias))
        values = np.zeros((4, 3), np.float32)
        values[0, 0] = 0.5
        got = graph.reference(3, values)
        self.assertAlmostEqual(float(got[0, 0]), 1.5)
        self.assertAlmostEqual(float(got[3, 2]), 4.0)
        program = graph.compile(3)
        self.assertFalse(program.uses_neural_engine)
        ran = program.run(values)
        np.testing.assert_allclose(ran, got)

    def test_repeat_on_neural_engine(self):
        n = 64
        matrix = np.zeros((n, n), np.float32)
        np.fill_diagonal(matrix, 0.5)
        state = np.zeros((n, n), np.float32)
        state[0, 0] = 1.0
        graph = aner.Graph()
        graph.output(graph.repeat(graph.input(n), 4, matrix))
        program = graph.compile(n)
        got = program.run(state)
        self.assertTrue(program.uses_neural_engine)
        self.assertAlmostEqual(float(got[0, 0]), 0.0625, places=5)
        self.assertGreater(program.compile_ms, 0.0)
        self.assertGreater(program.eval_ms, 0.0)

    def test_two_fields_match_reference(self):
        n = 64
        left = np.linspace(-0.2, 0.2, n * n, dtype=np.float32).reshape(n, n)
        right = np.linspace(0.05, 0.3, n * n, dtype=np.float32).reshape(n, n)
        graph = aner.Graph()
        a = graph.input(n)
        b = graph.input(n)
        graph.output(graph.mul(a, b))
        ref = graph.reference(n, [left, right])
        program = graph.compile(n)
        got = program.run([left, right])
        self.assertTrue(program.uses_neural_engine)
        self.assertLess(float(np.max(np.abs(got - ref))), 2e-3)

    def test_batch_tiles_keep_values(self):
        n = 64
        batch = 130
        state = np.zeros((n, batch), np.float32)
        state[0, :] = 0.25
        graph = aner.Graph()
        graph.output(graph.scale(graph.input(n), np.ones(n, np.float32)))
        program = graph.compile(64)
        got = program.run_batch(state)
        self.assertTrue(program.uses_neural_engine)
        np.testing.assert_allclose(got[:, [0, 63, 64, 129]], state[:, [0, 63, 64, 129]])

    def test_modal_forecast(self):
        result = forecast()
        self.assertTrue(result.roll_on_neural_engine)
        self.assertTrue(result.sensor_on_neural_engine)
        self.assertLess(result.probe_error, 5e-2)
        self.assertLess(result.sensor_error, 1e-5)
        self.assertAlmostEqual(result.p05, -0.5898, delta=1e-3)
        self.assertAlmostEqual(result.p50, 0.2320, delta=1e-3)
        self.assertAlmostEqual(result.p95, 1.1060, delta=1e-3)
        self.assertAlmostEqual(result.fundamental_ratio, 1.0131, delta=1e-3)
        self.assertAlmostEqual(result.highest_ratio, 0.9944, delta=1e-3)
        self.assertAlmostEqual(result.fundamental_retained, 90, delta=0.5)
        self.assertAlmostEqual(result.highest_retained, 16, delta=0.5)

    def test_split_matches_ane(self):
        n, steps, batch = 512, 64, 4096
        matrix = np.zeros((n, n), np.float32)
        np.fill_diagonal(matrix, 0.9)
        matrix[np.arange(n), (np.arange(n) + 1) % n] = 0.03
        state = np.linspace(-0.3, 0.3, n * batch, dtype=np.float32).reshape(n, batch)

        graph = aner.Graph()
        graph.output(graph.repeat(graph.input(n), steps, matrix))
        with graph.compile(64, aner.Schedule.ANE) as on_ane, \
             graph.compile(64, aner.Schedule.SPLIT) as split:
            self.assertTrue(on_ane.uses_neural_engine)
            self.assertTrue(split.uses_neural_engine)
            self.assertTrue(split.uses_gpu)
            ane_out = on_ane.run_batch(state)
            split_out = split.run_batch(state)
        # Both engines compute in fp16, so the two agree to fp16 rounding.
        self.assertLess(float(np.max(np.abs(split_out - ane_out))), 3e-2)

    def test_horner_reference(self):
        graph = aner.Graph()
        graph.output(graph.horner(graph.input(4), [0.1, -0.25, 0.05]))
        state = np.full((4, 2), 0.5, np.float32)
        got = graph.reference(2, state)
        np.testing.assert_allclose(got, -0.0125, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
