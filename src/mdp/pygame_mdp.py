import time
from typing import Dict, Tuple

from .mdp import MDP
from constants import HUMAN

X = 0
Y = 1
A = 2
TEXT_COLOUR = (36, 113, 163)

class PyGameMDP(MDP):
    def __init__(self) -> None:
        super().__init__()

        self.debounce_val = 100
        self.debounce = time.time_ns()

    def init_display(self) -> None:
        pass

    def set_operator(self, operator: str) -> None:
        super().set_operator(operator)
        if self.operator == HUMAN:
            self.display = True
            self.init_display()

    def check_input(self) -> bool:
        if ((time.time_ns() - self.debounce) / 1e6) < self.debounce_val:
            return False
        else:
            self.debounce = time.time_ns()
            return True

    def set_policy(self, policy) -> None:
        # TODO: remove this, the MDP should not have access to the policy object
        #       this is only used to show values in the grid target MDP
        #       need to find another way to show values in the grid target MDP
        pass
