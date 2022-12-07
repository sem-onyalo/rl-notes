import random
from typing import List
import numpy as np

from .tabular_function_v2 import TabularFunctionV2
from constants import GLIE_DECAY
from constants import EXP_DECAY
from constants import EPSILON_GREEDY_EXPLORE
from constants import UCB_EXPLORE
from mdp import MDP

class PolicyV2:
    def __init__(self, mdp:MDP, function:TabularFunctionV2, explore_type:str, epsilon:float, decay_type:str, decay_rate:float=None, epsilon_floor:float=None) -> None:
        self.n_actions = mdp.n_actions
        self.function = function
        self.epsilon = epsilon
        self.decay_type = decay_type
        self.explore_type = explore_type
        self.decay_rate = decay_rate
        self.epsilon_floor = epsilon_floor

        assert self.explore_type in [EPSILON_GREEDY_EXPLORE, UCB_EXPLORE], f"{self.explore_type} is invalid or not yet implemented"

    def __call__(self, state:str) -> int:
        return np.argmax(self.function(state)) # return the optimal (max) action

    def __len__(self):
        return len(self.function)

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

    def get_greedy(self, state:str) -> float:
        return self.function.get(state, self.__call__(state))

    def decay(self, value:float) -> None:
        if self.decay_type == GLIE_DECAY:
            self.epsilon = 1 / value
        elif self.decay_type == EXP_DECAY:
            raise Exception("Exponential decay not yet implemented.")

    def get_states_visited(self) -> List[str]:
        return self.function.value_map.keys()
