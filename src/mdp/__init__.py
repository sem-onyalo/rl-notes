from typing import Tuple

from .drift_car_mdp import DriftCarMDP
from .racecar_bullet_gym_mdp import RacecarBulletGymMDP
from .student_mdp import StudentMDP

class MDP:
    states:list
    actions:list

    def start(self) -> float:
        pass

    def step(self, action:int) -> Tuple[float, int, bool]:
        pass
