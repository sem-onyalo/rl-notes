import logging
from typing import Tuple

import pybullet_envs.bullet.racecarGymEnv as racecar_gym_env

_logger = logging.getLogger(__name__)

class RacecarBulletGymMDP:
    def __init__(self) -> None:
        self.env = racecar_gym_env.RacecarGymEnv(isDiscrete=True, renders=False)

        self.actions = list(range(0, self.env.action_space.n))

    def start(self) -> float:
        observation = self.env.reset()
        return observation

    def step(self, action:int) -> Tuple[float, int, bool]:
        observation, reward, terminated, _ = self.env.step(action)
        return reward, observation, terminated
