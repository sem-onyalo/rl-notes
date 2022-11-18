import io
import json
from typing import DefaultDict

import numpy as np

from mdp import MDP

class TabularFunctionV2:
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
        self.n_actions = mdp.n_actions

    def load_from_buffer(self, buffer:io.BytesIO) -> None:
        state_dict = json.loads(buffer.getvalue())
        self.value_map = {}
        for state in state_dict:
            self.value_map[state] = np.asarray(state_dict[state])
        self.n_actions = len(self.value_map[next(iter(self.value_map))])

    def get(self, state:str, action:int) -> float:
        return self.__call__(state)[action]

    def update(self, state:str, action:int, value:float) -> None:
        self.lazy_load(state)
        self.value_map[state][action] = value

    def state_dict(self) -> DefaultDict:
        state_dict = {}
        for state in self.value_map:
            state_dict[state] = list(self.value_map[state])
        return json.dumps(state_dict, indent=4).encode("utf-8")

    def lazy_load(self, state:str) -> None:
        if not state in self.value_map:
            self.value_map[state] = np.asarray([0.] * self.n_actions)
