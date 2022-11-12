from typing import Tuple

import gym
from . import sem_gyms

class DriftCarMDP:
    def __init__(self, show_visual=False, max_step_count=1000) -> None:
        self.env = gym.make(
            "sem_gyms/DriftCarEnv-v0", 
            renders=show_visual, 
            max_step_count=max_step_count)

        self.states = list(range(0, self.env.observation_space.n))
        
        self.actions = list(range(0, self.env.action_space.n))

    def start(self, options=None) -> float:
        observation, _ = self.env.reset(options=options)
        return observation

    def step(self, action:int) -> Tuple[float, int, bool]:
        observation, reward, is_terminal, _, info = self.env.step(action)
        return reward, observation, is_terminal, info
