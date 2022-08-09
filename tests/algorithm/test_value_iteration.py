import unittest

from src.algorithm.value_iteration import ValueIteration
from src.function import ValueFunctionTabular
from src.mdp import StudentMDP

class TestValueIteration(unittest.TestCase):
    def setUp(self):
        mdp = StudentMDP()
        function = ValueFunctionTabular(mdp)
        self.discount_rate = 1.
        self.delta_threshold = 0.2
        self.algorithm = ValueIteration(mdp, function, self.discount_rate, self.delta_threshold)

    def test_eq(self):
        mdp = StudentMDP()
        other = ValueIteration(mdp, ValueFunctionTabular(mdp), self.discount_rate, self.delta_threshold)
        self.assertTrue(self.algorithm == other)

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

