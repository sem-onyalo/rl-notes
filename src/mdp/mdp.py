
from typing import Dict, Tuple

class MDP:
    n_state:int
    n_action:int
    d_state:Tuple[int, int]

    def start(self) -> float:
        pass

    def step(self, action:int) -> Tuple[float, int, bool, Dict[str, object]]:
        pass

    def get_operator(self) -> str:
        pass

    def set_operator(self, operator:str) -> None:
        pass
