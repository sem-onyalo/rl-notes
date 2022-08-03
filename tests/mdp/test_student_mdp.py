import unittest

from src.mdp.student_mdp import StudentMDP

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
            0: { 0: (-1, [0]), 1: ( 0, [1]) }, 
            1: { 0: (-1, [0]), 2: (-2, [2]) }, 
            2: { 2: (-2, [3]), 3: ( 0, [4]) }, 
            3: { 2: (10, [4]), 4: ( 1, [1, 2, 3]) }, 
            4: { 0: ( 0, [4]), 1: ( 0, [4]), 2: ( 0, [4]), 3: ( 0, [4]), 4: (0, [4]) }
        }
        self.assertDictEqual(expected, self.mdp.state_actions)

    def test_action_probabilities_init(self):
        expected = { 3: { 1: 0.2, 2: 0.4, 3: 0.4 } }
        self.assertDictEqual(expected, self.mdp.action_probabilities)

    def test_getitem(self):
        expected = { 
            0: { 0: (-1, [0]), 1: ( 0, [1]) }, 
            1: { 0: (-1, [0]), 2: (-2, [2]) }, 
            2: { 2: (-2, [3]), 3: ( 0, [4]) }, 
            3: { 2: (10, [4]), 4: ( 1, [1, 2, 3]) }, 
            4: { 0: ( 0, [4]), 1: ( 0, [4]), 2: ( 0, [4]), 3: ( 0, [4]), 4: (0, [4]) } 
        }

        self.assertDictEqual(expected[0], self.mdp[0])
        self.assertDictEqual(expected[1], self.mdp[1])
        self.assertDictEqual(expected[2], self.mdp[2])
        self.assertDictEqual(expected[3], self.mdp[3])
        self.assertEqual(expected[4], self.mdp[4])

    def test_is_terminal_state(self):
        state = 4
        actual = self.mdp.is_terminal_state(state)
        self.assertTrue(actual)

    def test_get_initial_state(self):
        expected = 0
        actual = self.mdp.get_initial_state()
        self.assertEqual(expected, actual)

    def test_get_reward_and_next_state_terminal(self):
        terminal_state = 4
        expected = (0, terminal_state, True)
        actual = self.mdp.get_reward_and_next_state(terminal_state, 0)
        self.assertTupleEqual(expected, actual)

    def test_get_reward_and_next_state_deterministic(self):
        expected = (-2, 2, False)
        actual = self.mdp.get_reward_and_next_state(1, 2)
        self.assertTupleEqual(expected, actual)

    def test_get_reward_and_next_state_stochastic(self):
        expected_actions = self.mdp.get_state_actions()[3][4][1]
        actuals = list()
        for _ in range(20):
            reward, next_state, is_terminal = self.mdp.get_reward_and_next_state(3, 4)
            actuals.append(next_state)
        actuals = sorted(set(actuals))

        self.assertEqual(1, reward)
        self.assertFalse(is_terminal)
        self.assertTrue(1 in actuals)
        self.assertTrue(2 in actuals)
        self.assertTrue(3 in actuals)
        self.assertListEqual(expected_actions, actuals)

    def test_get_reward_and_next_states(self):
        expected = (-1, [0], False)
        actual = self.mdp.get_reward_and_next_states(0, 0)
        self.assertTupleEqual(expected, actual)

    def test_get_reward_and_next_states_terminal(self):
        terminal_state = 4
        expected = (0, [terminal_state], True)
        actual = self.mdp.get_reward_and_next_states(terminal_state, 0)
        self.assertTupleEqual(expected, actual)

    def test_get_reward_and_next_states_invalid_action(self):
        expected = None
        actual = self.mdp.get_reward_and_next_states(0, 2)
        self.assertEqual(expected, actual)

    def test_get_action_probability(self):
        expected = 0.2
        actual = self.mdp.get_action_probability(3, 1)
        self.assertEqual(expected, actual)

    def test_get_action_probability_default(self):
        expected = 1
        actual = self.mdp.get_action_probability(0, 1)
        self.assertEqual(expected, actual)

