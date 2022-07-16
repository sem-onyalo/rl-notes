import unittest

from src.main import ValueFunctionTabular

class TestValueFunctionTabular(unittest.TestCase):
    def setUp(self):
        self.states = list(range(5))
        self.value_function = ValueFunctionTabular(self.states)

    def test_state_values_init(self):
        expected = { 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. }
        actual = self.value_function.state_values
        self.assertDictEqual(expected, actual)

    def test_state_values_str(self):
        expected = str({ 0: 0., 1: 0., 2: 0., 3: 0., 4: 0. })
        actual = str(self.value_function)
        self.assertEqual(expected, actual)

    def test_state_values_call(self):
        expected = 0.
        actual = self.value_function(3)
        self.assertEqual(expected, actual)

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

