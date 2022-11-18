import logging
import math
import random
from typing import Dict, Tuple

import numpy as np

import gym
from . import sem_gyms
from .sem_gyms.drift_car_env import CAR_START_ORIENTATION
from .sem_gyms.drift_car_env import CAR_START_POSITION

_logger = logging.getLogger("drift-car-mdp")

class DriftCarMDP:
    """
    This class represents an MDP for teaching an agent to drive towards a target.
    e.g. pipenv run python src/main.py -e 250 monte-carlo-v2 drift-car-mdp --discrete --boundary 3
    """

    def __init__(
        self, 
        boundary, 
        renders=False, 
        is_discrete=True, 
        max_step_count=1000,
        randomize_start_position=False,
        seed=None) -> None:

        self.seed = seed
        self.boundary = boundary
        self.is_discrete = is_discrete
        self.max_step_count = max_step_count
        self.randomize_start_position = randomize_start_position

        self.angle_to_quaternion_map = {
            # angle: [x, y, z, w]
            0:   [0.,  0.,  0.,      1.    ], # sin(  0.0) == 0,      cos(  0.0) ==  1
            45:  [0.,  0.,  0.3827,  0.9239], # sin( 22.5) == 0.3827, cos( 22.5) ==  0.9239
            90:  [0.,  0.,  0.7071,  0.7071], # sin( 45.0) == 0.7071, cos( 45.0) ==  0.7071
            135: [0.,  0.,  0.9239,  0.3827], # sin( 67.5) == 0.9239, cos( 67.5) ==  0.3827
            180: [0.,  0.,  1.,      0.    ], # sin( 90.0) == 1,      cos( 90.0) ==  0
            225: [0.,  0.,  0.9239, -0.3827], # sin(112.5) == 0.9239, cos(112.5) == -0.3827, asin(0.9239) == 67.5, acos(-0.3827) == 112.5
            270: [0.,  0.,  0.7071, -0.7071], # sin(135.0) == 0.7071, cos(135.0) == -0.7071, asin(0.7071) == 45.0, acos(-0.7071) == 135.0
            315: [0.,  0.,  0.3827, -0.9239]  # sin(157.5) == 0.3827, cos(157.5) == -0.9239, asin(0.3827) == 22.5, acos(-0.9239) == 157.5
        } # https://wiki.alioth.net/index.php/Quaternion#Overview

        self.default_reached_target_distance = .4
        self.expected_gym_env_observation_space = 6
        self.default_start_position = [boundary - 1, -1, 180]
        self.default_target_position = [0, 0, 1]
        self.default_action_duration = 50
        self.steps = 0

        self.starting_positions  = []
        self.discrete_velocities = [   1,   1,     1]
        self.discrete_steering   = [-0.6,   0,   0.6]
        self.discrete_orientations = np.asarray([0, 45, 90, 135, 180, 225, 270, 315])
        self.n_discrete_state_space = len(range(-boundary, boundary + 1)) ** 2 * len(self.discrete_orientations)
        start_position, start_orientation = self.get_start_position_and_orientation()

        self.n_states = 5
        self.n_actions = len(self.discrete_velocities)

        self.env = gym.make(
            "sem_gyms/DriftCarEnv-v0", 
            renders=renders, 
            is_discrete=is_discrete, 
            n_action_space=self.n_actions,
            n_observation_space=self.expected_gym_env_observation_space,
            car_start_position=start_position,
            car_start_orientation=start_orientation)

    def start(self) -> float:
        position, orientation = self.get_start_position_and_orientation()
        options = { CAR_START_POSITION: position, CAR_START_ORIENTATION: orientation }
        observation, _ = self.env.reset(seed=self.seed, options=options)
        next_state = self.get_next_state(observation)
        self.steps = 0

        return next_state

    def step(self, action:int) -> Tuple[float, int, bool, Dict[str, object]]:
        gym_action = self.get_action(action), self.default_action_duration
        observation, _, _, _, _ = self.env.step(gym_action)
        is_terminal, reason = self.get_is_terminal_and_reason(observation)
        next_state = self.get_next_state(observation)
        reward = self.get_reward(observation)
        info = { "reason": reason }
        self.steps += 1

        return reward, next_state, is_terminal, info

    def get_action(self, action):
        assert self.is_discrete, "Continuous action space is not yet implemented, use is_discrete=True"

        velocity = self.discrete_velocities[action]
        steering = self.discrete_steering[action]
        return [velocity, steering]

    def get_reward(self, observation:np.ndarray) -> float:
        distance_to_target = self.get_distance_to_target(observation)
        return -round(distance_to_target, 1)

    def get_is_terminal_and_reason(self, observation:np.ndarray) -> bool:
        distance_to_target = self.get_distance_to_target(observation)

        if distance_to_target <= self.default_reached_target_distance:
            return True, "REACHED_TARGET"

        if distance_to_target > self.boundary:
            return True, "REACHED_BOUNDARY"

        if self.max_step_count > 0 and self.steps > self.max_step_count:
            return True, "REACHED_STEP_LIMIT"

        return False, ""

    def get_start_position_and_orientation(self) -> Tuple[int, int, int]:
        if self.randomize_start_position:
            return self.get_random_start_position_and_orientation()
        else:
            return self.get_default_start_position_and_orientation()

    def get_random_start_position_and_orientation(self, init_coord_z:float=.2) -> Tuple[int, int, int]:
        # TODO: right now this function generates points that are TOO random (e.g. a 
        # position where the car is pointing right at the boundary and then the MDP 
        # terminates immediately). What you need to do is generate points that are 
        # vectors pointed towards the target or at an angle towards the target (i.e. 
        # exclude positions where the vector angle >= 90 degress).
        x = random.choice(list(set(range(-self.boundary, self.boundary)) - set([0])))
        y = random.choice(list(set(range(-self.boundary, self.boundary)) - set([0])))
        a = random.choice(self.discrete_orientations)
        return self.mdp_to_env_positions(x, y, init_coord_z, a)

    def get_default_start_position_and_orientation(self, init_coord_z:float=.2) -> Tuple[int, int, int]:
        x = self.default_start_position[0]
        y = self.default_start_position[1]
        a = self.default_start_position[2]
        return self.mdp_to_env_positions(x, y, init_coord_z, a)

    def mdp_to_env_positions(self, x, y, z, a) -> Tuple[int, int, int]:
        orientation = self.angle_to_quaternion_map[a]
        position = [x, y, z]
        self.starting_positions.append((x, y, a))
        return position, orientation

    def quaternion_to_discrete_orientation(self, car_z, car_w):
        if car_z < 0 or car_w < 0:
            car_w_new = round(-car_w if car_w >= 0 else car_w)
            w = round(math.degrees(math.acos(car_w_new)), 1)
            angle = round(w * 2)
        else:
            z = round(math.degrees(math.asin(car_z)), 1)
            w = round(math.degrees(math.acos(car_w)), 1)
            angle = round(z + w)

        return self.discrete_orientations[(np.abs(self.discrete_orientations - angle)).argmin()]

    def get_distance_to_target(self, observation:np.ndarray):
        target_position_x = observation[0]
        target_position_y = observation[1]
        car_position_x = observation[2]
        car_position_y = observation[3]
        length_x = abs(car_position_x - target_position_x)
        length_y = abs(car_position_y - target_position_y)
        return math.sqrt(length_x**2 + length_y**2)

    def get_next_state(self, observation:np.ndarray):
        assert self.is_discrete, "Continuous action space is not yet implemented, use is_discrete=True"

        target_position_x = observation[0]
        target_position_y = observation[1]
        car_position_x = observation[2]
        car_position_y = observation[3]
        car_orientation_z = observation[4]
        car_orientation_w = observation[5]

        next_state = np.array([
            round(target_position_x),
            round(target_position_y),
            round(car_position_x),
            round(car_position_y),
            self.quaternion_to_discrete_orientation(car_orientation_z, car_orientation_w)
        ])

        return next_state
