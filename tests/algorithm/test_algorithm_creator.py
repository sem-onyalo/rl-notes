import unittest

from src.algorithm.algorithm_creator import AlgorithmCreator
from src.algorithm.monte_carlo import MonteCarlo
from src.algorithm.policy import Policy
from src.algorithm.q_learning import QLearning
from src.algorithm.value_iteration import ValueIteration
from src.constants import MONTE_CARLO
from src.constants import Q_LEARNING
from src.constants import VALUE_ITERATION
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

        actual = AlgorithmCreator.build(VALUE_ITERATION, args)
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

        actual = AlgorithmCreator.build(MONTE_CARLO, args)
        self.assertEqual(expected, actual)

    def test_q_learning(self):
        epsilon = 0.5
        change_rate = 0.2
        discount_rate = 1.
        mdp = StudentMDP()

        expected = QLearning(
            mdp=mdp,
            function=ActionValueFunctionTabular(mdp),
            policy=Policy(mdp, epsilon),
            discount_rate=discount_rate,
            change_rate=change_rate)

        args = RuntimeArgs()
        args.epsilon = epsilon
        args.change_rate = change_rate
        args.discount_rate = discount_rate
        args.episodes = 100

        actual = AlgorithmCreator.build(Q_LEARNING, args)
        self.assertEqual(expected, actual)

