import unittest

from src.function import ActionValueFunctionTabular
from src.mdp import StudentMDP

class TestActionValueFunctionTabular(unittest.TestCase):
    def setUp(self):
        mdp = StudentMDP()
        self.action_value_function = ActionValueFunctionTabular(mdp)

    def test_str(self):
        expected = str({ 
            0: { 0: 0., 1: 0. }, 
            1: { 0: 0., 2: 0. }, 
            2: { 2: 0., 3: 0. }, 
            3: { 2: 0., 4: 0. }, 
            4: { 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. } 
        })
        actual = str(self.action_value_function)
        self.assertEqual(expected, actual)

    def test_eq(self):
        other = ActionValueFunctionTabular(StudentMDP())
        self.assertTrue(self.action_value_function == other)

    def test_call(self):
        expected = 0.
        actual = self.action_value_function(0, 0)
        self.assertEqual(expected, actual)

    def test_state_action_values_init(self):
        expected = { 
            0: { 0: 0., 1: 0. }, 
            1: { 0: 0., 2: 0. }, 
            2: { 2: 0., 3: 0. }, 
            3: { 2: 0., 4: 0. }, 
            4: { 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. } 
        }
        actual = self.action_value_function.get_values()
        self.assertDictEqual(expected, actual)

    def test_update_value(self):
        expected = 2.
        self.action_value_function.update_value(1, 2, expected)
        actual = self.action_value_function(1, 2)
        self.assertEqual(expected, actual)

    def test_get_optimal_action(self):
        self.action_value_function.update_value(0, 0, 1.)
        actual_1 = self.action_value_function.get_optimal_action(0)
        self.action_value_function.update_value(0, 1, 2.)
        actual_2 = self.action_value_function.get_optimal_action(0)
        self.assertEqual(0, actual_1)
        self.assertEqual(1, actual_2)

