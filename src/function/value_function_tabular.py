
class ValueFunctionTabular:
    """
    This class represents a tabular value function, i.e. V(s), with initial values of zero.
    """

    def __init__(self, mdp):
        self.state_values = {
            state: 0.
            for state in mdp.get_state_actions()
        }

    def __str__(self):
        return str(self.state_values)

    def __eq__(self, other):
        if other != None:
            return self.state_values == other.get_values()

        return False

    def __call__(self, state):
        return self.state_values[state]

    def get_value(self):
        return sum([self.state_values[x] for x in self.state_values])

    def update_values(self, state_values):
        self.state_values = state_values

    def get_values(self):
        return self.state_values

