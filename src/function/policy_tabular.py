import random
from typing import List
import numpy as np

from .policy import Policy
from .function_tabular import FunctionTabular
from constants import GLIE_DECAY
from constants import EXP_DECAY
from constants import EPSILON_GREEDY_EXPLORE
from constants import UCB_EXPLORE
from mdp import MDP

class PolicyTabular(Policy):
    def __init__(self, mdp:MDP, explore_type:str, epsilon:float, decay_type:str, decay_rate:float=None, epsilon_floor:float=None) -> None:
        super().__init__(mdp)
        
        self.epsilon = epsilon
        self.decay_type = decay_type
        self.explore_type = explore_type
        self.decay_rate = decay_rate
        self.epsilon_floor = epsilon_floor

        self.function = FunctionTabular(mdp=self.mdp)

        assert self.explore_type in [EPSILON_GREEDY_EXPLORE, UCB_EXPLORE], f"{self.explore_type} is invalid or not yet implemented"

    def __call__(self, state:str) -> int:
        return np.argmax(self.function(state)) # return the optimal (max) action

    def choose_action(self, state:str) -> int:
        if self.explore_type == EPSILON_GREEDY_EXPLORE:
            return self.get_epsilon_greedy_action(state)
        else:
            raise Exception(f"{self.explore_type} not yet implemented")

    def get_epsilon_greedy_action(self, state: str) -> int:
        do_explore = random.random() < self.epsilon
        if do_explore:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.__call__(state)

    def decay(self, value:float) -> None:
        if self.decay_type == GLIE_DECAY:
            self.epsilon = 1 / value
        elif self.decay_type == EXP_DECAY:
            raise Exception("Exponential decay not yet implemented.")

    def transform_state(self, state:object) -> str:
        if isinstance(state, int):
            return str(state)
        elif isinstance(state, np.ndarray):
            return ",".join(list(map(str, state.flatten())))
        else:
            raise Exception(f"{self.name} currently does not support {type(state)} state types")
