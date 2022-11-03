from typing import Tuple

from .racecar_bullet_gym_mdp import RacecarBulletGymMDP
from .student_mdp import StudentMDP

class MDP:
    actions:list

    def start(self) -> float:
        pass

    def step(self, action:int) -> Tuple[float, int, bool]:
        pass
