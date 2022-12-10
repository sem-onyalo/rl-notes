
from typing import Dict, Tuple

class MDP:
    n_states:int
    n_actions:int

    def start(self) -> float:
        pass

    def step(self, action:int) -> Tuple[float, int, bool, Dict[str, object]]:
        pass

    def get_operator(self) -> str:
        pass

    def set_operator(self, operator:str) -> None:
        pass
