import random
from typing import Dict, Tuple

import gym
from . import sem_gyms
from .sem_gyms.drift_car_env import CAR_START_ORIENTATION
from .sem_gyms.drift_car_env import CAR_START_POSITION
from .sem_gyms.drift_car_env import DriftCarEnvV0

class DriftCarMDP:
    def __init__(
        self, 
        boundary, 
        renders=False, 
        is_discrete=True, 
        max_step_count=1000,
        randomize_start_position=False) -> None:

        self.boundary = boundary
        self.randomize_start_position = randomize_start_position
        assert self.randomize_start_position, "Non-random start position not yet implemented. Use randomize_start_position==True"

        self.default_target_position = [0, 0, 1]

        self.env:DriftCarEnvV0 = gym.make(
            "sem_gyms/DriftCarEnv-v0", 
            boundary=boundary, 
            renders=renders, 
            is_discrete=is_discrete,
            max_step_count=max_step_count)

        self.n_states = self.env.observation_space.n

        self.n_actions = self.env.action_space.n

    def start(self) -> float:
        if self.randomize_start_position:
            position, orientation = self.get_random_position()

        options = { CAR_START_POSITION: position, CAR_START_ORIENTATION: orientation }
        observation, _ = self.env.reset(options=options)
        return observation

    def step(self, action:int) -> Tuple[float, int, bool, Dict[str, object]]:
        observation, reward, is_terminal, _, info = self.env.step(action)
        return reward, observation, is_terminal, info

    def get_random_position(self, init_coord_z:float=.2) -> Tuple[int, int, int]:
        x = random.choice(list(set(range(-self.boundary, self.boundary)) - set([0])))
        y = random.choice(list(set(range(-self.boundary, self.boundary)) - set([0])))
        a = random.choice(self.env.discrete_state_angles)
        orientation = self.env.angle_to_quaternion_map[a]
        position = [
            # do not start at target position
            x + 1 if x == self.default_target_position[0] and y == self.default_target_position[1] else x, 
            y, 
            init_coord_z
        ]
        return position, orientation
