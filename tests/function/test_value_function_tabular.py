import unittest

from src.function.value_function_tabular import ValueFunctionTabular
from src.mdp.student_mdp import StudentMDP

class TestValueFunctionTabular(unittest.TestCase):
    def setUp(self):
        mdp = StudentMDP()
        self.value_function = ValueFunctionTabular(mdp)

    def test_str(self):
        expected = str({ 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. })
        actual = str(self.value_function)
        self.assertEqual(expected, actual)

    def test_eq(self):
        other = ValueFunctionTabular(StudentMDP())
        self.assertTrue(self.value_function == other)

    def test_call(self):
        expected = 0.
        actual = self.value_function(3)
        self.assertEqual(expected, actual)

    def test_state_values_init(self):
        expected = { 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. }
        actual = self.value_function.get_values()
        self.assertDictEqual(expected, actual)

    def test_update_values(self):
        expected = { 0: 0., 1: 1., 2: 1., 3: 2., 4: 3. }
        self.value_function.update_values(expected)
        actual = self.value_function.get_values()
        self.assertDictEqual(expected, actual)

    def test_get_value(self):
        expected = 7.
        self.value_function.update_values({ 0: 0., 1: 1., 2: 1., 3: 2., 4: 3. })
        actual = self.value_function.get_value()
        self.assertEqual(expected, actual)

