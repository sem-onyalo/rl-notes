from mdp import MDP

class Policy:
    def __init__(self, mdp:MDP) -> None:
        self.mdp = mdp
        self.n_actions = mdp.n_actions

    def __call__(self, state:object) -> int:
        """
        Return the optimal (max) action.
        """
        pass

    def choose_action(self, state:str) -> int:
        """
        Choose an action stochastically.
        """
        pass

    def get_epsilon_greedy_action(self, state: str) -> int:
        """
        Choose an action using the epsilon-greedy algorithm.
        """
        pass

    def decay(self, value:float) -> None:
        """
        Decay the epsilon value according to the current exploration/exploitation strategy.
        """
        pass

    def transform_state(self, state:object) -> str:
        """
        Transform a state to the format the action-value function uses.
        """
        pass
