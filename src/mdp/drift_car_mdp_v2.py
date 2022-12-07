import logging

import numpy as np

from .drift_car_mdp_v1 import DriftCarMDP

_logger = logging.getLogger("drift-car-mdp")

class DriftCarMDPV2(DriftCarMDP):
    """
    This class represents an MDP for teaching an agent to drive in a circle.
    """

    def __init__(self, path_radius, boundary, renders=False, is_discrete=True, max_step_count=1000, randomize_start_position=False, seed=None) -> None:
        super().__init__(boundary, renders, is_discrete, max_step_count, randomize_start_position, seed)
        self.path_radius = path_radius
        self.default_start_position = [0, path_radius, 180]
        self.path_radius_target_threshold = .3
        self.target_position_threshold = .3
        self.is_terminal_steps_buffer = 100

    def get_reward(self, observation:np.ndarray) -> float:
        distance_to_target = round(self.get_distance_to_target(observation), 1)
        is_on_path = (
            distance_to_target >= self.path_radius - self.path_radius_target_threshold and
            distance_to_target <= self.path_radius + self.path_radius_target_threshold)
        return 1 if is_on_path else -round(abs(self.path_radius - distance_to_target), 1) - 1 # * 2

    def get_is_terminal_and_reason(self, observation:np.ndarray) -> bool:
        car_position_x = observation[2]
        car_position_y = observation[3]
        car_position_a = self.quaternion_to_discrete_orientation(observation[4], observation[5])

        is_on_target = (
            round(abs(car_position_x - self.default_start_position[0]), 1) <= self.target_position_threshold and
            round(abs(car_position_y - self.default_start_position[1]), 1) <= self.target_position_threshold and
            car_position_a == self.default_start_position[2]
        )

        if self.steps > self.is_terminal_steps_buffer and is_on_target:
            return True, "REACHED_TARGET"

        distance_to_target = self.get_distance_to_target(observation)
        if distance_to_target > self.boundary:
            return True, "REACHED_BOUNDARY"

        if self.max_step_count > 0 and self.steps > self.max_step_count:
            return True, "REACHED_STEP_LIMIT"

        return False, ""

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
            round(car_position_x, 1),
            round(car_position_y, 1),
            self.quaternion_to_discrete_orientation(car_orientation_z, car_orientation_w)
        ])

        return next_state
