import time
from typing import Dict, Tuple

import numpy as np

from .mdp import MDP
from constants import HUMAN

X = 0
Y = 1

class PyGameMDP(MDP):
    operator:str
    def __init__(self) -> None:
        super().__init__()

        self.debounce_val = 100
        self.debounce = time.time_ns()

    def start(self) -> np.ndarray:
        pass

    def step(self, action:int) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        pass

    def init_display(self) -> None:
        pass

    def get_operator(self) -> str:
        return self.operator

    def set_operator(self, operator:str) -> None:
        self.operator = operator
        if self.operator == HUMAN:
            self.display = True
            self.init_display()

    def check_input(self) -> bool:
        if ((time.time_ns() - self.debounce) / 1e6) < self.debounce_val:
            return False
        else:
            self.debounce = time.time_ns()
            return True
