import unittest

from src.algorithm.q_learning import QLearning
from src.algorithm.policy import Policy
from src.function import ActionValueFunctionTabular
from src.mdp import StudentMDP

class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.epsilon = 0.5
        self.change_rate = 0.2
        self.discount_rate = 1.
        mdp = StudentMDP()
        policy = Policy(mdp, self.epsilon)
        function = ActionValueFunctionTabular(mdp)
        self.algorithm = QLearning(mdp, function, policy, self.discount_rate, self.change_rate)

    def test_eq(self):
        mdp = StudentMDP()
        expected = QLearning(
            mdp=mdp,
            function=ActionValueFunctionTabular(mdp),
            policy=Policy(mdp, self.epsilon),
            discount_rate=self.discount_rate,
            change_rate=self.change_rate)

        self.assertTrue(self.algorithm == expected)

    def test_one_iteration(self):
        policy_init = { 0: None, 1: None, 2: None, 3: None, 4: None }
        function_init = { 0: { 0: 0., 1: 0. }, 1: { 0: 0., 2: 0. }, 2: { 2: 0., 3: 0. }, 3: { 2: 0., 4: 0. }, 4: 0. }
        self.algorithm.max_episodes = 1
        function, policy = self.algorithm.run()
        self.assertNotEqual(policy_init, policy())
        self.assertNotEqual(function_init, function.get_values())

    def test_algorithm(self):
        expected = { 0: 1, 1: 2, 2: 2, 3: 2, 4: 0. }
        _, policy = self.algorithm.run()
        self.assertDictEqual(expected, policy())

