from typing import Dict, Tuple

class MDP:
    n_state:int
    n_action:int
    d_state:Tuple[int, int]
    operator:str

    def start(self) -> float:
        return None

    def step(self, action:int) -> Tuple[float, float, bool, Dict[str, object]]:
        return None, None, None, { "action": action }

    def get_operator(self) -> str:
        return self.operator

    def set_operator(self, operator:str) -> None:
        self.operator = operator
