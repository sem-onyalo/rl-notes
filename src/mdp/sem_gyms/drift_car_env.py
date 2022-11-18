import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import os
import time

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
            renders=False,
            is_discrete=True,
            n_action_space=3,
            n_observation_space=6,
            car_start_position=[0, 0, .2],
            car_start_orientation=[0, 0, 0, 1.],
            target_start_position=[0, 0, 1]
        ):
        self.renders = renders
        self.is_discrete = is_discrete
        self.urdf_root_path = urdf_root
        self.target_start_position = target_start_position

        self.time_step = 0.01
        self.step_counter = 0
        self.reset_steps = 100
        self.target_object = -1

        assert self.is_discrete, "Continuous state/action space is not yet implemented"
        self.observation_space = spaces.Discrete(n_observation_space)
        self.action_space = spaces.Discrete(n_action_space)

        if self.renders:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bc.BulletClient()

        self.racecar = Racecar(
            self.bullet_client, 
            self.urdf_root_path, 
            car_start_position,
            car_start_orientation,
            time_step=self.time_step)

    def __del__(self):
        self.bullet_client = 0

    def get_observation(self) -> np.ndarray:
        car_position, car_orientation = self.bullet_client.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        target_position, _  = self.bullet_client.getBasePositionAndOrientation(self.target_object)

        observation = np.array([
            target_position[0],  # target_position_x
            target_position[1],  # target_position_y
            car_position[0],     # car_position_x
            car_position[1],     # car_position_y
            car_orientation[2],  # car_orientation_z
            car_orientation[3]   # car_orientation_w
        ])

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bullet_client.resetSimulation()
        self.bullet_client.setGravity(0, 0, -10)
        self.bullet_client.setTimeStep(self.time_step)
        self.bullet_client.loadSDF(os.path.join(self.urdf_root_path, "stadium.sdf"))
        self.target_object = self.bullet_client.loadURDF(
            os.path.join(self.urdf_root_path, "sphere_small.urdf"),
            self.target_start_position)

        racecar_options = CarOptions()
        racecar_options.start_position = options[CAR_START_POSITION]
        racecar_options.start_orientation = options[CAR_START_ORIENTATION]
        self.racecar.reset(options=racecar_options)
        self.step_counter = 0

        for _ in range(self.reset_steps):
            self.bullet_client.stepSimulation()

        observation = self.get_observation()
        info = {}

        return observation, info

    def step(self, action):
        gym_action, duration = action
        self.apply_action(gym_action, duration)

        observation = self.get_observation()
        reward = 0 # MDP handles calculating reward
        is_terminal = True # MDP handles determining is_terminal
        info = {} # MDP handles sending info to the agent

        return observation, reward, is_terminal, False, info

    def apply_action(self, action, duration):
        self.racecar.apply_action(action)
        for _ in range(duration):
            self.bullet_client.stepSimulation()
            if self.renders:
                time.sleep(self.time_step)

            # if self.is_terminal(self.get_positions()[0])[0]:
            #     break

        self.step_counter += 1

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
