import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import os
import math
import time
from typing import Tuple

import gym
import numpy as np
import pybullet
import pybullet_data
from gym import spaces
from pkg_resources import parse_version
from pybullet_utils import bullet_client as bc

from .mit_racecar import MitRacecar as Racecar

RENDER_HEIGHT = 1920
RENDER_WIDTH = 1080

_logger = logging.getLogger("drift-car-env-v0")

class DriftCarEnvV0(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_bullet_clienter_second': 50}

    def __init__(
            self,
            urdf_root=pybullet_data.getDataPath(),
            action_repeat=50,
            is_discrete=True,
            renders=False,
            car_start_pos=[3, 0, .2],
            car_start_orn=[0, 0, 0, 1.],
            target_start_pos=[0, 0, 1],
            max_step_count=1000,
            boundary=5
        ):
        self.time_step = 0.01
        self.urdf_root_path = urdf_root
        self.action_repeat = action_repeat
        self._ballUniqueId = -1
        self.step_counter = 0
        self.renders = renders
        self.is_discrete = is_discrete
        self.target_start_pos = target_start_pos

        self.max_step_count = max_step_count
        self.distance_to_target_lower_bound = .4
        self.distance_to_target_upper_bound = boundary

        # self.discrete_velocity_actions = [   0,   0,     0,      1,   1,    1,]
        # self.discrete_steering_actions = [-0.6,   0,   0.6,   -0.6,   0,   0.6]
        self.discrete_velocity_actions = [   1,   1,     1]
        self.discrete_steering_actions = [-0.6,   0,   0.6]
        if self.is_discrete:
            self.action_space = spaces.Discrete(len(self.discrete_velocity_actions))
        else:
            raise Exception("Continuous action space is not yet implemented, use isDiscrete=True")

        self.observation_space = spaces.Discrete(6)

        if self.renders:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bc.BulletClient()

        self.racecar = Racecar(
            self.bullet_client, 
            self.urdf_root_path, 
            car_start_pos,
            car_start_orn,
            time_step=self.time_step)

    def __del__(self):
        self.bullet_client = 0

    def get_positions(self) -> Tuple[int, int]:
        car_pos, car_orn = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        target_pos, _  = self.bullet_client.getBasePositionAndOrientation(self._ballUniqueId)

        round_precison = 4
        car_pos_x = round(car_pos[0], round_precison)
        car_pos_y = round(car_pos[1], round_precison)
        car_orn_z = round(car_orn[2], round_precison)
        car_orn_w = round(car_orn[3], round_precison)
        target_pos_x = round(target_pos[0], round_precison)
        target_pos_y = round(target_pos[1], round_precison)

        delta_x = abs(car_pos_x - target_pos_x)
        delta_y = abs(car_pos_y - target_pos_y)

        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

        return distance_to_target, target_pos_x, target_pos_y, car_pos_x, car_pos_y, car_orn_z, car_orn_w

    def is_terminal(self):
        distance_to_target, _, _, _, _, _, _ = self.get_positions()

        if distance_to_target < self.distance_to_target_lower_bound:
            return True, "TARGET_REACHED"

        if distance_to_target > self.distance_to_target_upper_bound:
            return True, "BOUNDARY_REACHED"

        if self.max_step_count > 0 and self.step_counter > self.max_step_count:
            return True, "STEP_LIMIT_REACHED"

        return False, ""

    def get_reward(self):
        distance_to_target, _, _, _, _, _, _ = self.get_positions()

        reward = 0 if distance_to_target == 0 else -distance_to_target
        _logger.info(f"distance to target: {distance_to_target}")
        _logger.info(f"reward: {reward}")

        return reward

    def get_observation(self):
        _, target_pos_x, target_pos_y, car_pos_x, car_pos_y, car_orn_z, car_orn_w = self.get_positions()

        observation = np.array([
            target_pos_x,
            target_pos_y,
            car_pos_x,
            car_pos_y,
            car_orn_z,
            car_orn_w
        ])

        return observation

    def get_action(self, action):
        if self.is_discrete:
            velocity = self.discrete_velocity_actions[action]
            steering = self.discrete_steering_actions[action]
            return [velocity, steering]

        return action

    def apply_action(self, action):
        self.racecar.apply_action(action)
        for _ in range(self.action_repeat):
            self.bullet_client.stepSimulation()
            if self.renders:
                time.sleep(self.time_step)

            if self.is_terminal():
                break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _logger.info(f"Seed: {self.np_random}")

        self.bullet_client.resetSimulation()
        self.bullet_client.setGravity(0, 0, -10)
        self.bullet_client.setTimeStep(self.time_step)
        self.bullet_client.loadSDF(os.path.join(self.urdf_root_path, "stadium.sdf"))
        self._ballUniqueId = self.bullet_client.loadURDF(
            os.path.join(self.urdf_root_path, "sphere_small.urdf"),
            self.target_start_pos)

        self.racecar.reset(options)

        self.step_counter = 0

        for _ in range(100):
            self.bullet_client.stepSimulation()

        return self.get_observation(), {}

    def step(self, action):
        realaction = self.get_action(action)

        self.apply_action(realaction)

        self.step_counter += 1

        is_terminal, reason = self.is_terminal()

        info = {}
        reward = 0
        if is_terminal:
            info["reason"] = reason

            if reason == "TARGET_REACHED":
                reward = 1
            elif reason == "BOUNDARY_REACHED":
                reward = -1

        return self.get_observation(), reward, is_terminal, False, info

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)

        view_matrix = self.bullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0
        )

        (_, _, px, _, _) = self.bullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _step = step
