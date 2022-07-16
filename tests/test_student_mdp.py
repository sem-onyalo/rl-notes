import unittest

from src.main import StudentMDP

class TestStudentMDP(unittest.TestCase):
    def setUp(self):
        self.mdp = StudentMDP()

    def test_states_init(self):
        expected = list(range(5))
        self.assertListEqual(expected, self.mdp.states)

    def test_actions_init(self):
        expected = list(range(5))
        self.assertListEqual(expected, self.mdp.actions)

    def test_state_actions_init(self):
        expected = { 
            0: { 0: (-1, [0]), 1: (0, [1]) }, 
            1: { 0: (-1, [0]), 2: (-2, [2]) }, 
            2: { 2: (-2, [3]), 3: (0, [4]) }, 
            3: { 2: (10, [4]), 4: (1, [1, 2, 3]) }, 
            4: None 
        }
        self.assertDictEqual(expected, self.mdp.state_actions)

    def test_action_probabilities_init(self):
        expected = { 3: { 1: 0.2, 2: 0.4, 3: 0.4 } }
        self.assertDictEqual(expected, self.mdp.action_probabilities)

    def test_state_action_result(self):
        expected = (-1, [0], False)
        actual = self.mdp.get_state_action_result(0, 0)
        self.assertTupleEqual(expected, actual)

    def test_state_action_result_terminal(self):
        expected = (None, None, True)
        actual = self.mdp.get_state_action_result(4, 0)
        self.assertTupleEqual(expected, actual)

    def test_state_action_result_invalid_action(self):
        expected = None
        actual = self.mdp.get_state_action_result(0, 2)
        self.assertEqual(expected, actual)

    def test_action_probability(self):
        expected = 0.2
        actual = self.mdp.get_action_probability(3, 1)
        self.assertEqual(expected, actual)

    def test_action_probability_default(self):
        expected = 1
        actual = self.mdp.get_action_probability(0, 1)
        self.assertEqual(expected, actual)

