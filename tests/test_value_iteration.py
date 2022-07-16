import unittest

from src.main import StudentMDP
from src.main import ValueFunctionTabular
from src.main import ValueIteration

class TestValueIteration(unittest.TestCase):
    def setUp(self):
        self.states = list(range(5))
        self.mdp = StudentMDP()
        self.value_function = ValueFunctionTabular(self.states)
        self.discount_rate = 1.
        self.delta_threshold = 0.2
        self.algorithm = ValueIteration(self.mdp, self.value_function, self.discount_rate, self.delta_threshold)

    def test_discount_rate_init(self):
        expected = 1.
        actual = self.algorithm.discount_rate
        self.assertEqual(expected, actual)

    def test_delta_threshold_init(self):
        expected = 0.2
        actual = self.algorithm.delta_threshold
        self.assertEqual(expected, actual)

    def test_one_iteration(self):
        expected = { 0: 0., 1: -1., 2: 0., 3: 10., 4: 0. }
        actual = self.algorithm.update_state_values()
        self.assertDictEqual(expected, actual)

    def test_two_iterations(self):
        expected = { 0: -1., 1: -1., 2: 8., 3: 10., 4: 0. }
        self.algorithm.max_iterations = 2
        value_function = self.algorithm.run()
        actual = value_function.get_values()
        self.assertDictEqual(expected, actual)

    def test_algorithm(self):
        expected = { 0: 6., 1: 6., 2: 8., 3: 10., 4: 0. }
        value_function = self.algorithm.run()
        actual = value_function.get_values()
        self.assertDictEqual(expected, actual)

