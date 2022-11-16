
from typing import Dict, Tuple

class MDP:
    n_states:int
    n_actions:int

    def start(self) -> float:
        pass

    def step(self, action:int) -> Tuple[float, int, bool, Dict[str, object]]:
        pass
