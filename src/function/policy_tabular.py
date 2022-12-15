import io
import json
import random

import numpy as np

from .policy import Policy
from .function_tabular import FunctionTabular
from constants import EPSILON_GREEDY_EXPLORE
from constants import UCB_EXPLORE
from mdp import MDP

POLICY_NAME = "tabular-function-policy"

class PolicyTabularArgs:
    explore_type:str
    epsilon:float
    epsilon_floor:float
    decay_type:str
    decay_rate:int

class PolicyTabular(Policy):
    def __init__(self, mdp:MDP, args:PolicyTabularArgs) -> None:
        super().__init__(mdp, POLICY_NAME)

        self.epsilon = args.epsilon
        self.decay_type = args.decay_type
        self.explore_type = args.explore_type
        self.decay_rate = args.decay_rate
        self.epsilon_floor = args.epsilon_floor

        self.model_file_ext = "json"
        self.function = FunctionTabular(mdp=self.mdp)

        assert self.explore_type in [EPSILON_GREEDY_EXPLORE, UCB_EXPLORE], f"{self.explore_type} is invalid or not yet implemented"

    def __call__(self, state:str) -> int:
        """
        Return the optimal (max) action.
        """
        return np.argmax(self.function(state))

    def choose_action(self, state:str) -> int:
        """
        Choose an action stochastically.
        """
        if self.explore_type == EPSILON_GREEDY_EXPLORE:
            return self.get_epsilon_greedy_action(state)
        else:
            raise Exception(f"{self.explore_type} not yet implemented")

    def get_epsilon_greedy_action(self, state: str) -> int:
        """
        Choose an action using the epsilon-greedy algorithm.
        """
        do_explore = random.random() < self.epsilon
        if do_explore:
            return random.randint(0, self.mdp.n_action - 1)
        else:
            return self.__call__(state)

    def get_value(self, state:str, action:int) -> np.ndarray:
        """
        Returns the value for a state and action.
        """
        return self.function.get(state, action)

    def get_values(self, state:object) -> np.ndarray:
        """
        Returns the values associated to each action for a given state.
        """
        transformed_state = self.transform_state(state)
        return self.function(transformed_state)

    def transform_state(self, state:object) -> str:
        """
        Transform a state to the format the action-value function uses.
        """
        if isinstance(state, np.ndarray):
            return ",".join(list(map(str, state.flatten())))
        else:
            raise Exception(f"{self.name} currently does not support {type(state)} state types")

    def update(self, *args) -> None:
        """
        Updates the action-value function.
        """
        state = args[0]
        action = args[1]
        new_value = args[2]
        self.function.update(state, action, new_value)

    def get_model(self) -> io.BytesIO:
        """
        Gets the model parameters as a bytes-like object.
        """
        buffer = io.BytesIO()
        state_dict = { state: list(self.function.value_map[state]) for state in self.function.value_map }
        buffer.write(json.dumps(state_dict, indent=4).encode("utf-8"))
        return buffer

    def load_model(self, buffer:io.BytesIO) -> None:
        """
        Loads the model parameters from a bytes-like object.
        """
        self.function.load_from_buffer(buffer)
