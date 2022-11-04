import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import os
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
            target_start_pos=[0, 0, 1]
        ):
        self.time_step = 0.01
        self.urdf_root_path = urdf_root
        self.action_repeat = action_repeat
        self._ballUniqueId = -1
        self.step_counter = 0
        self.renders = renders
        self.is_discrete = is_discrete
        self.target_start_pos = target_start_pos

        self.coords_bound = 10
        self.velocity_actions = [   0,   0,     0,      1,   1,    1,]
        self.steering_actions = [-0.6,   0,   0.6,   -0.6,   0,   0.6]
        if self.is_discrete:
            self.action_space = spaces.Discrete(len(self.velocity_actions))
        else:
            raise Exception("Continuous action space is not yet implemented, use isDiscrete=True")

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(2),
                "target": spaces.Discrete(2)
            }
        )

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

    def get_delta_positions(self) -> Tuple[int, int]:
        car_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        target_pos, _  = self.bullet_client.getBasePositionAndOrientation(self._ballUniqueId)
        x_delta = abs(car_pos[0] - target_pos[0])
        y_delta = abs(car_pos[1] - target_pos[1])

        # _logger.info(f"deltas: {x_delta},{y_delta}")

        return x_delta, y_delta

    def is_terminal(self):
        x_delta, y_delta = self.get_delta_positions()

        if x_delta > self.coords_bound:
            return True

        if y_delta > self.coords_bound:
            return True

        if self.step_counter > 300:
            return True
        
        return False

    def get_reward(self):
        x_delta, y_delta = self.get_delta_positions()

        reward = -(x_delta + y_delta) / 2

        return reward

    def get_observation(self):
        car_pos, car_orn = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        target_pos, _  = self.bullet_client.getBasePositionAndOrientation(self._ballUniqueId)

        _logger.info(f"pos: {car_pos}")
        _logger.info(f"orn: {car_orn}")

        observation = {
            "agent": np.array([car_pos[0], car_pos[1]]), #np.array([car_pos[0], car_pos[1], car_orn[0], car_orn[1], car_orn[3]]),
            "target": np.array([target_pos[0], target_pos[1]])
        }

        return observation

    def get_action(self, action):
        if self.is_discrete:
            velocity = self.velocity_actions[action]
            steering = self.steering_actions[action]
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

        self.racecar.reset()

        self.step_counter = 0

        for _ in range(100):
            self.bullet_client.stepSimulation()

        return self.get_observation(), {}

    def step(self, action):
        realaction = self.get_action(action)

        self.apply_action(realaction)

        self.step_counter += 1

        return self.get_observation(), self.get_reward(), self.is_terminal(), False, {}

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
