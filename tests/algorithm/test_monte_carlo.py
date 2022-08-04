import unittest

from src.algorithm.monte_carlo import MonteCarlo
from src.algorithm.policy import Policy
from src.function import ActionValueFunctionTabular
from src.mdp import StudentMDP

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.epsilon = 0.5
        self.discount_rate = 1.
        mdp = StudentMDP()
        policy = Policy(mdp, self.epsilon)
        function = ActionValueFunctionTabular(mdp)
        self.algorithm = MonteCarlo(mdp, function, policy, self.discount_rate)

    def test_eq(self):
        mdp = StudentMDP()
        expected = MonteCarlo(
            mdp=mdp,
            function=ActionValueFunctionTabular(mdp),
            policy=Policy(mdp, self.epsilon),
            discount_rate=self.discount_rate)

        self.assertTrue(self.algorithm == expected)

    def test_discount_rate_init(self):
        expected = self.discount_rate
        actual = self.algorithm.discount_rate
        self.assertEqual(expected, actual)

    def test_do_glie_init(self):
        expected = True
        actual = self.algorithm.do_glie
        self.assertEqual(expected, actual)

    def test_one_iteration(self):
        policy_init = { 0: None, 1: None, 2: None, 3: None, 4: None }
        function_init = { 0: { 0: 0., 1: 0. }, 1: { 0: 0., 2: 0. }, 2: { 2: 0., 3: 0. }, 3: { 2: 0., 4: 0. }, 4: 0. }
        self.algorithm.max_episodes = 1
        function, policy = self.algorithm.run()
        self.assertNotEqual(policy_init, policy())
        self.assertNotEqual(function_init, function.get_values())

    def test_two_iterations_no_epsilon_update(self):
        policy_init = { 0: None, 1: None, 2: None, 3: None, 4: None }
        function_init = { 0: { 0: 0., 1: 0. }, 1: { 0: 0., 2: 0. }, 2: { 2: 0., 3: 0. }, 3: { 2: 0., 4: 0. }, 4: 0. }
        self.algorithm.max_episodes = 5
        self.algorithm.do_glie = False
        self.algorithm.run()
        self.assertEqual(self.epsilon, self.algorithm.policy.epsilon)

    def test_algorithm(self):
        expected = { 0: 1, 1: 2, 2: 2, 3: 2, 4: None }
        self.algorithm.max_episodes = 100
        function, policy = self.algorithm.run()
        self.assertDictEqual(expected, policy())
        self.assertEqual(10, function(3, 2))
        self.assertAlmostEqual(8, function(2, 2), delta=1.)
        self.assertAlmostEqual(6, function(1, 2), delta=2.)
        self.assertAlmostEqual(6, function(0, 1), delta=3.)

