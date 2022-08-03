import unittest

from src.main import ActionValueFunctionTabular
from src.main import AlgorithmCreator
from src.main import MonteCarlo
from src.main import Policy
from src.main import StudentMDP
from src.main import ValueFunctionTabular
from src.main import ValueIteration

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

        actual = AlgorithmCreator.build("value-iteration", discount_rate=discount_rate, delta_threshold=delta_threshold)
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

        actual = AlgorithmCreator.build("monte-carlo", epsilon=epsilon, discount_rate=discount_rate, max_episodes=100, do_glie=True)
        self.assertEqual(expected, actual)

