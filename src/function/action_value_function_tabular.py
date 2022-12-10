
class ActionValueFunctionTabular:
    """
    This class represents a tabular action-value function, i.e. Q(s,a), with initial values of zero.
    """

    def __init__(self, mdp):
        self.state_action_values = {
            state: {
                action: 0.
                for action in mdp[state]
            } for state in mdp.get_state_actions()
        }

    def __str__(self):
        return str(self.state_action_values)

    def __eq__(self, other):
        if other != None:
            return self.state_action_values == other.get_values()

        return False

    def __call__(self, state, action):
        return self.state_action_values[state][action]

    def get_values(self):
        return self.state_action_values

    def update_value(self, state, action, value):
        self.state_action_values[state][action] = value

    def get_optimal_action(self, state):
        optimal_action = None
        max_value = 0.
        for action in self.state_action_values[state]:
            if optimal_action == None or self.state_action_values[state][action] > max_value:
                optimal_action = action
                max_value = self.state_action_values[state][action]

        return optimal_action
