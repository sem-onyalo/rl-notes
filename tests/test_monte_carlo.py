import unittest

from src.main import ActionValueFunctionTabular
from src.main import MonteCarlo
from src.main import Policy
from src.main import StudentMDP

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        epsilon = 0.5
        self.discount_rate = 1.
        mdp = StudentMDP()
        policy = Policy(mdp, epsilon)
        function = ActionValueFunctionTabular(mdp)
        self.algorithm = MonteCarlo(mdp, function, self.discount_rate, policy)

    def test_discount_rate_init(self):
        expected = self.discount_rate
        actual = self.algorithm.discount_rate
        self.assertEqual(expected, actual)

    def test_one_iteration(self):
        policy_init = { 0: None, 1: None, 2: None, 3: None, 4: None }
        function_init = { 0: { 0: 0., 1: 0. }, 1: { 0: 0., 2: 0. }, 2: { 2: 0., 3: 0. }, 3: { 2: 0., 4: 0. }, 4: 0. }
        self.algorithm.max_episodes = 1
        function, policy = self.algorithm.run()
        self.assertNotEqual(policy_init, policy)
        self.assertNotEqual(function_init, function)

    def test_algorithm(self):
        expected = { 0: 1, 1: 2, 2: 2, 3: 2, 4: None }
        self.algorithm.max_episodes = 100
        function, policy = self.algorithm.run()
        self.assertDictEqual(expected, policy)
        self.assertEqual(10, function(3, 2))
        self.assertAlmostEqual(8, function(2, 2), delta=1.)
        self.assertAlmostEqual(6, function(1, 2), delta=2.)
        self.assertAlmostEqual(6, function(0, 1), delta=3.)

