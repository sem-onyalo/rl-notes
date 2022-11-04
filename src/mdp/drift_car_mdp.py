from typing import Tuple

import gym
from . import sem_gyms

class DriftCarMDP:
    def __init__(self) -> None:
        self.env = gym.make("sem_gyms/DriftCarEnv-v0")
        # self.env = gym.make("sem_gyms/DriftCarEnv-v0", renders=True)

        self.actions = list(range(0, self.env.action_space.n))

    def start(self) -> float:
        observation, _ = self.env.reset()
        return observation["agent"]

    def step(self, action:int) -> Tuple[float, int, bool]:
        observation, reward, is_terminal, _, _ = self.env.step(action)
        return reward, observation["agent"], is_terminal
