import random

class Policy:
    """
    This class represents a policy to be followed (i.e. for every state what action should be taken) with the caveat
    that in some cases the action will be random (i.e. exploratory).
    """

    def __init__(self, mdp, epsilon_probability):
        """
        Initializes a new policy.
        :param MDP mdp: The MDP in which to execute this policy.
        :param float epsilon_probability: The probability in which to take a random action (i.e. explore).
        If 0.0 then never explore (i.e. act greedily). If 1.0 then always explore.
        """
        self.mdp = mdp
        self.policy = { state: None for state in mdp.get_state_actions() }
        self.epsilon = epsilon_probability

    def __str__(self):
        return str(self.policy)

    def __eq__(self, other):
        if other != None:
            return self.policy == other()

        return False

    def __call__(self):
        return self.policy

    def update_epsilon(self, value):
        self.epsilon = value

    def get_action(self, state):
        do_explore = random.random() < self.epsilon
        if self.policy[state] == None or do_explore:
            # pick a random action from the possible actions in the specified state
            actions = list(self.mdp[state].keys())
            action_idx = random.randint(0, len(actions) - 1)
            return actions[action_idx]
        else:
            # pick the optimal action
            return self.policy[state]

    def update_policy(self, state, action):
        self.policy[state] = action

