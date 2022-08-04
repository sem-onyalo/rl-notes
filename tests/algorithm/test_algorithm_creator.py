import unittest

from src.algorithm.algorithm_creator import AlgorithmCreator
from src.algorithm.monte_carlo import MonteCarlo
from src.algorithm.policy import Policy
from src.algorithm.value_iteration import ValueIteration
from src.function import ActionValueFunctionTabular
from src.function import ValueFunctionTabular
from src.mdp import StudentMDP

class RuntimeArgs:
    pass

class TestAlgorithmCreator(unittest.TestCase):
    def test_invalid_algorithm_name(self):
        name = "not-a-valid-algorithm"
        with self.assertRaises(Exception, msg=f"The name '{name}' is not a valid algorithm name."):
            AlgorithmCreator.build(name)

    def test_value_iteration(self):
        discount_rate = 1.
        delta_threshold = 0.2
        mdp = StudentMDP()

        expected = ValueIteration(
            mdp=mdp,
            function=ValueFunctionTabular(mdp),
            discount_rate=discount_rate,
            delta_threshold=delta_threshold)

        args = RuntimeArgs()
        args.discount_rate = discount_rate
        args.delta_threshold = delta_threshold

        actual = AlgorithmCreator.build("value-iteration", args)
        self.assertEqual(expected, actual)

    def test_monte_carlo(self):
        epsilon = 0.5
        discount_rate = 1.
        mdp = StudentMDP()

        expected = MonteCarlo(
            mdp=mdp,
            function=ActionValueFunctionTabular(mdp),
            policy=Policy(mdp, epsilon),
            discount_rate=discount_rate)

        args = RuntimeArgs()
        args.epsilon = epsilon
        args.discount_rate = discount_rate
        args.episodes = 100
        args.no_glie = False

        actual = AlgorithmCreator.build("monte-carlo", args)
        self.assertEqual(expected, actual)

