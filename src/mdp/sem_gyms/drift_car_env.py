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
from .mit_racecar import CarOptions

RENDER_WIDTH = 1080
RENDER_HEIGHT = 1920
CAR_START_POSITION = "car_start_position"
CAR_START_ORIENTATION = "car_start_orientation"

_logger = logging.getLogger("drift-car-env-v0")

class DriftCarEnvV0(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_bullet_clienter_second': 50}

    def __init__(
            self,
            urdf_root=pybullet_data.getDataPath(),
            boundary=5,
            renders=False,
            is_discrete=True,
            action_repeat=50,
            target_boundary=.4,
            max_step_count=1000,
            car_start_pos=[3, 0, .2],
            car_start_orn=[0, 0, 0, 1.],
            target_start_pos=[0, 0, 1]
        ):
        self.renders = renders
        self.is_discrete = is_discrete
        self.urdf_root_path = urdf_root
        self.action_repeat = action_repeat
        self.max_step_count = max_step_count
        self.target_start_pos = target_start_pos
        self.distance_to_target_upper_bound = boundary
        self.distance_to_target_lower_bound = target_boundary

        self.time_step = 0.01
        self.step_counter = 0
        self._ballUniqueId = -1
        self.position_round_precison = 4

        self.discrete_action_velocity = [   1,   1,     1] # [   0,   0,     0,      1,   1,    1,]
        self.discrete_action_steering = [-0.6,   0,   0.6] # [-0.6,   0,   0.6,   -0.6,   0,   0.6]
        self.discrete_state_angles = np.asarray([0, 45, 90, 135, 180, 225, 270, 315])
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

        if self.is_discrete:
            n_observation_space = 5 # target == 2, car position == 2, car orientation == 1
            self.observation_space = spaces.Discrete(n_observation_space)
            self.action_space = spaces.Discrete(len(self.discrete_action_velocity))
        else:
            n_observation_space = 6 # target == 2, car position == 2, car orientation == 2
            raise Exception("Continuous action space is not yet implemented, use is_discrete=True")

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

    def get_distance_to_target(self, car_pos_x, car_pos_y, target_pos_x, target_pos_y):
        length_x = abs(car_pos_x - target_pos_x)
        length_y = abs(car_pos_y - target_pos_y)
        return math.sqrt(length_x**2 + length_y**2)

    def get_positions(self) -> Tuple[int, int]:
        car_pos, car_orn = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        target_pos, _  = self.bullet_client.getBasePositionAndOrientation(self._ballUniqueId)

        car_pos_x = car_pos[0]
        car_pos_y = car_pos[1]
        car_orn_z = car_orn[2]
        car_orn_w = car_orn[3]
        target_pos_x = target_pos[0]
        target_pos_y = target_pos[1]

        distance_to_target = self.get_distance_to_target(car_pos_x, car_pos_y, target_pos_x, target_pos_y)

        return distance_to_target, target_pos_x, target_pos_y, car_pos_x, car_pos_y, car_orn_z, car_orn_w

    def get_discrete_angle(self, car_z, car_w):
        if car_z < 0 or car_w < 0:
            car_w_new = -car_w if car_w >= 0 else car_w
            w = round(math.degrees(math.acos(car_w_new)), 1)
            angle = round(w * 2)
        else:
            z = round(math.degrees(math.asin(car_z)), 1)
            w = round(math.degrees(math.acos(car_w)), 1)
            angle = round(z + w)

        return self.discrete_state_angles[(np.abs(self.discrete_state_angles - angle)).argmin()]

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

        if self.is_discrete:
            observation = np.array([
                round(target_pos_x),
                round(target_pos_y),
                round(car_pos_x),
                round(car_pos_y),
                self.get_discrete_angle(car_orn_z, car_orn_w)
            ])
        else:
            raise Exception("Continuous action space is not yet implemented, use is_discrete=True")

        return observation

    def get_reward(self, is_terminal:bool, reason:str) -> float:
        reward = 0
        if is_terminal:
            if reason == "TARGET_REACHED":
                reward = 1
            elif reason == "BOUNDARY_REACHED":
                reward = -1

        return reward

    def get_info(self, is_terminal:bool, reason:str) -> dict:
        info = {}
        if is_terminal:
            info["reason"] = reason
        return info

    def get_action(self, action):
        if self.is_discrete:
            velocity = self.discrete_action_velocity[action]
            steering = self.discrete_action_steering[action]
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

        racecar_options = CarOptions()
        racecar_options.start_position = options[CAR_START_POSITION]
        racecar_options.start_orientation = options[CAR_START_ORIENTATION]
        self.racecar.reset(options=racecar_options)

        self.step_counter = 0

        for _ in range(100):
            self.bullet_client.stepSimulation()

        return self.get_observation(), {}

    def step(self, action):
        realaction = self.get_action(action)

        self.apply_action(realaction)

        self.step_counter += 1

        is_terminal, reason = self.is_terminal()

        reward = self.get_reward(is_terminal, reason)

        info = self.get_info(is_terminal, reason)

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
