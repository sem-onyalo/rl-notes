import unittest

from src.main import Policy
from src.main import StudentMDP

class TestPolicy(unittest.TestCase):
    def setUp(self):
        mdp = StudentMDP()
        self.epsilon = 0.1
        self.policy = Policy(mdp, self.epsilon)

    def test_str(self):
        expected = str({ 0: None, 1: None, 2: None, 3: None, 4: None })
        actual = str(self.policy)
        self.assertEqual(expected, actual)

    def test_eq(self):
        other = Policy(StudentMDP(), self.epsilon)
        self.assertTrue(self.policy == other)

    def test_call(self):
        expected = { 0: None, 1: None, 2: None, 3: None, 4: None }
        actual = self.policy()
        self.assertDictEqual(expected, actual)

    def test_epsilon_init(self):
        expected = self.epsilon
        actual = self.policy.epsilon
        self.assertEqual(expected, actual)

    def test_update_epsilon(self):
        expected = 0.2
        self.policy.update_epsilon(expected)
        self.assertEqual(expected, self.policy.epsilon)

    def test_get_action_greedily(self):
        expected = 4
        self.policy.epsilon = 0.0
        self.policy.policy = { 0: None, 1: None, 2: None, 3: expected, 4: None }
        actual = self.policy.get_action(3)
        self.assertEqual(expected, actual)

    def test_get_action_epsilon_greedily(self):
        expected = 2
        optimal_action = 4
        self.policy.epsilon = 0.5
        self.policy.policy = { 0: None, 1: None, 2: None, 3: optimal_action, 4: None }
        actuals = list()
        for _ in range(10):
            actuals.append(self.policy.get_action(3))
        self.assertTrue(expected in actuals)

    def test_update_policy(self):
        expected = 4
        self.policy.update_policy(3, expected)
        actual = self.policy.policy[3]
        self.assertEqual(expected, actual)

