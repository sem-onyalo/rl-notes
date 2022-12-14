import io
import json

import numpy as np

from mdp import MDP

class FunctionTabular:
    def __init__(self, **kwargs) -> None:
        if "mdp" in kwargs:
            self.load_from_mdp(kwargs["mdp"])

    def __call__(self, state:str) -> np.ndarray:
        self.lazy_load(state)
        return self.value_map[state]

    def __len__(self):
        return len(self.value_map)

    def load_from_mdp(self, mdp:MDP) -> None:
        self.value_map = {}
        self.n_action = mdp.n_action

    def load_from_buffer(self, buffer:io.BytesIO) -> None:
        state_dict = json.loads(buffer.getvalue())
        self.value_map = {}
        for state in state_dict:
            self.value_map[state] = np.asarray(state_dict[state])
        self.n_action = len(self.value_map[next(iter(self.value_map))])

    def get(self, state:str, action:int) -> float:
        return self.__call__(state)[action]

    def update(self, state:str, action:int, value:float) -> None:
        self.lazy_load(state)
        self.value_map[state][action] = value

    def lazy_load(self, state:str) -> None:
        if not state in self.value_map:
            self.value_map[state] = np.asarray([0.] * self.n_action)
