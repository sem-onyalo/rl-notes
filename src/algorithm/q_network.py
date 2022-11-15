import io
import logging
import math
import time
import traceback
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .algorithm_net import AlgorithmNet
from mdp import MDP
from model import EpsilonDecay
from model import ExperienceMemory
from model import Transition
from registry import plot_training_metrics
# from registry import save_model

ALGORITHM_NAME = "q-network"
TRAINED_MODEL_FILENAME = f"{ALGORITHM_NAME}.pth"

_logger = logging.getLogger(ALGORITHM_NAME)

class QNetwork(AlgorithmNet):
    """
    This class represents a Deep Q-Network (DQN) algorithm with experience replay and fixed Q-targets.
    """

    def __init__(self, mdp:MDP, epsilon_decay:EpsilonDecay, layers:str, discount_rate=1., change_rate=.2, max_episodes=1000, batch_size=4, memory_capacity=10000, target_update_freq=500, no_plot=False) -> None:
        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.epsilon_decay = epsilon_decay
        self.discount_rate = discount_rate
        self.change_rate = change_rate
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.no_plot = no_plot
        self.memory = ExperienceMemory(memory_capacity)
        self.target_function = self.build_linear_function(layers)
        self.behaviour_function = self.build_linear_function(layers)
        self.optimizer = optim.RMSprop(self.behaviour_function.parameters(), lr=self.change_rate)
        self.loss_function = nn.SmoothL1Loss()

        # align the weights of the two functions on init
        self.target_function.load_state_dict(self.behaviour_function.state_dict())
        self.target_function.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.angles = np.asarray([0, 45, 90, 135, 180, 225, 270, 315])
        self.orn_z_values = np.asarray([0, 0.3827, 0.7071, 0.9239, 1])
        self.orn_w_values = np.asarray([-0.9239, -0.7071, -0.3827, 0, 0.3827, 0.9239, 1])
        self.orientations = {
            0:   [0, 0,  0,       1     ], # sin(  0.0) == 0,      cos(  0.0) ==  1
            45:  [0, 0,  0.3827,  0.9239], # sin( 22.5) == 0.3827, cos( 22.5) ==  0.9239
            90:  [0, 0,  0.7071,  0.7071], # sin( 45.0) == 0.7071, cos( 45.0) ==  0.7071
            135: [0, 0,  0.9239,  0.3827], # sin( 67.5) == 0.9239, cos( 67.5) ==  0.3827
            180: [0, 0,  1,       0     ], # sin( 90.0) == 1,      cos( 90.0) ==  0
            225: [0, 0,  0.9239, -0.3827], # sin(112.5) == 0.9239, cos(112.5) == -0.3827, asin(0.9239) == 67.5, acos(-0.3827) == 112.5
            270: [0, 0,  0.7071, -0.7071], # sin(135.0) == 0.7071, cos(135.0) == -0.7071, asin(0.7071) == 45.0, acos(-0.7071) == 135.0
            315: [0, 0,  0.3827, -0.9239]  # sin(157.5) == 0.3827, cos(157.5) == -0.9239, asin(0.3827) == 22.5, acos(-0.9239) == 157.5
        }
        self.q_angles = {
            (0,       1     ):   0,
            (0.3827,  0.9239):  45,
            (0.7071,  0.7071):  90,
            (0.9239,  0.3827): 135,
            (1,       0     ): 180,
            (0.9239, -0.3827): 225,
            (0.7071, -0.7071): 270,
            (0.3827, -0.9239): 315
        }

        # for a in self.angles:
        #     q = self.orientations[a]
        #     orientation = None
        #     if q[3] >= 0:
        #         z = round(math.degrees(math.asin(q[2])), 1)
        #         w = round(math.degrees(math.acos(q[3])), 1)
        #         orientation = z + w
        #     else:
        #         w = round(math.degrees(math.acos(q[3])), 1)
        #         orientation = w * 2
        #     _logger.info(f"({q[2]}, {q[3]}): {orientation}")
        # quit()

        # subentities =
        # (
        #     "torus_pod  0 0 0 1 0 0 0",                //   0 degree
        #     "torus_pod  0 0 0 0.9239 0.0 0.0 0.3827",  //  45 degrees 
        #     "torus_pod  0 0 0 0.7071 0.0 0.0 0.7071",  //  90 degrees
        #     "torus_pod  0 0 0 0.3827 0.0 0.0 0.9239",  // 135 degrees
        #     "torus_pod  0 0 0 0 0 0 1",                // 180 degrees
        #     "torus_pod  0 0 0 -0.3827 0.0 0.0 0.9239", // 225 degrees
        #     "torus_pod  0 0 0 -0.7071 0.0 0.0 0.7071", // 270 degrees
        #     "torus_pod  0 0 0 -0.9239 0.0 0.0 0.3827", // 315 degrees
        # ) # https://wiki.alioth.net/index.php/Quaternion#Overview

    def get_discrete_state(self, state):
        target_x = int(state[0])
        target_y = int(state[1])
        car_x = round(state[2])
        car_y = round(state[3])
        car_z = round(state[4], 4)
        car_w = round(state[5], 4)
        # car_z = self.orn_z_values[(np.abs(self.orn_z_values - round(state[4], 4))).argmin()]
        # car_w = self.orn_w_values[(np.abs(self.orn_w_values - round(state[5], 4))).argmin()]
        # if (car_z, car_w) in self.q_angles:
        #     orientation = self.q_angles[(car_z, car_w)]
        angle = None
        if car_z < 0 or car_w < 0:
            car_w_new = -car_w if car_w >= 0 else car_w
            w = round(math.degrees(math.acos(car_w_new)), 1)
            angle = round(w * 2)
        else:
            z = round(math.degrees(math.asin(car_z)), 1)
            w = round(math.degrees(math.acos(car_w)), 1)
            angle = round(z + w)
        discrete_angle = self.angles[(np.abs(self.angles - angle)).argmin()]
        # _logger.info(f"state: target({target_x}, {target_y}), car({car_x}, {car_y}, {car_z}, {car_w})")
        # _logger.info(f"state: target({target_x}, {target_y}), car({car_x}, {car_y}, {car_z_2}, {car_w_2})")
        return target_x, target_y, car_x, car_y, discrete_angle

    def run(self, max_episodes=0):
        # ----------------------------------------------------------------------------------------------------
        i = 0
        action = 0
        steps = 50000
        boundary = 5
        while True:
            # x, y, a = self.get_random_position(boundary)
            # options = { "start_pos": [x, y, .2], "start_orn": self.orientations[a] }
            # --------------------------------------------------
            a = self.angles[i]
            options = { "start_pos": [1, 1, .2], "start_orn": self.orientations[a] }
            # ==================================================
            self.mdp.start(options)
            for _ in range(0, steps):
                _, obs, _, _ = self.mdp.step(action)
                # z = round(math.degrees(math.asin(obs[4])), 1)
                # w = round(math.degrees(math.acos(obs[5])), 1)
                # _logger.info(f"{a}: ({obs[4]}, {obs[5]}) ({z}, {w}, {z + w})")
                dis = self.get_discrete_state(obs)
                _logger.info(f"{dis[2]}, {dis[3]}: {dis[4]}")
            # ==================================================
            # i = 0 if (i+1) >= len(self.angles) else (i+1)
            # i += 1
            # if i >= len(self.angles):
            #     quit()
            # quit()
        # ----------------------------------------------------------------------------------------------------

        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.steps = 0
        self.terminal_reasons = {}
        total_rewards = []
        max_rewards = []
        epsilon = []

        t0 = time.time()
        max_reward = None
        for i in range(0, max_episodes):
            _logger.info("-" * 50)
            _logger.info(f"Episode {i + 1}")
            _logger.info("-" * 50)

            try:
                total_reward = self.run_episode()
            except Exception as e:
                _logger.error(f"Run episode failed on episode {i + 1} at step count {self.steps}: {e}")
                _logger.debug(''.join(traceback.format_exception(None, e, e.__traceback__)))
                max_episodes = i
                break

            max_reward = total_reward if max_reward == None or total_reward > max_reward else max_reward
            total_rewards.append(total_reward)
            max_rewards.append(max_reward)
            epsilon.append(self.epsilon_decay.get_epsilon())
            self.log_episode_metrics([], total_reward, max_reward)

        _logger.info(f"Training elapsed time: {time.time() - t0}")
        
        _logger.info("-" * 50)
        _logger.info(f"Terminal reasons")
        _logger.info("-" * 50)

        for key in self.terminal_reasons:
            _logger.info(f"{key}: {self.terminal_reasons[key]}")

        if not self.no_plot and max_episodes > 0:
            plot_training_metrics(ALGORITHM_NAME, max_episodes, total_rewards, max_rewards, epsilon)
            self.save_model()

    def run_episode(self):
        # if the number of steps in the episode is less than the batch size 
        # then optimize the model at the end of the epiosde
        ran_function_update = False

        t0 = time.time()
        total_reward = 0
        is_terminal = False
        boundary = 5
        x, y, a = self.get_random_position(boundary)
        options = { "start_pos": [x, y, .2], "start_orn": self.orientations[a] }
        state = self.mdp.start(options=options)

        while not is_terminal:
            state = self.get_discrete_state(state)
            action = self.get_action(self.normalize_state(state, boundary))
            reward, next_state, is_terminal, info = self.mdp.step(action)
            self.memory.push(Transition(state, action, next_state, reward))
            total_reward += reward
            state = next_state

            if len(self.memory) >= self.batch_size:
                self.update_function(boundary)
                ran_function_update = True

            if self.steps > 0 and self.steps % self.target_update_freq == 0:
                self.target_function.load_state_dict(self.behaviour_function.state_dict())
            self.steps += 1

            if "reason" in info:
                if info["reason"] in self.terminal_reasons:
                    self.terminal_reasons[info["reason"]] += 1
                else:
                    self.terminal_reasons[info["reason"]] = 1

        if not ran_function_update:
            self.update_function(boundary)

        self.epsilon_decay.update_epsilon(self.steps)

        _logger.info(f"Episode elapsed time: {time.time() - t0}")

        return total_reward

    def normalize_state(self, state, boundary):
        expanded = np.expand_dims(state, axis=0)
        clipped = np.clip(expanded, -boundary, boundary)
        normed = 2 * ((clipped - -boundary) / (boundary - -boundary)) - 1 # normalize to -1,1
        return torch.tensor(normed, dtype=torch.float32, device=self.device)

    def normalize_state_batch(self, state, boundary):
        clipped = np.clip(state, -boundary, boundary)
        normed = 2 * ((clipped - -boundary) / (boundary - -boundary)) - 1 # normalize to -1,1
        return torch.tensor(normed, dtype=torch.float32, device=self.device)

    def get_action(self, state):
        _logger.debug(f"getting action from state: {state}")
        do_explore = random.random() < self.epsilon_decay.get_epsilon()
        if do_explore:
            _logger.debug(f"getting action by exploration")
            return random.randrange(len(self.mdp.actions))
        else:
            _logger.debug(f"getting action by exploitation")
            with torch.no_grad():
                self.behaviour_function.eval()
                predictions = self.behaviour_function(state)
                _logger.debug(f"preds for state {state}: {predictions}")
                self.behaviour_function.train()
                return predictions.max(1)[1].item()

    def update_function(self, boundary):
        transitions = self.memory.sample(self.batch_size)

        states, actions, next_states, rewards = self.transitions_to_batches(transitions)

        state_batch = self.normalize_state_batch([list(i) for i in states], boundary)
        action_batch = torch.tensor([[i] for i in actions], dtype=torch.long, device=self.device)
        next_state_batch = self.normalize_state_batch([list(i) for i in next_states], boundary)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)

        current_values = self.behaviour_function(state_batch).gather(1, action_batch)
        _logger.debug(f"current values: {current_values}")

        next_state_max_values = self.target_function(next_state_batch).max(1)[0].detach()
        _logger.debug(f"next state max values: {next_state_max_values}")

        expected_values = (reward_batch + (self.discount_rate * next_state_max_values)).unsqueeze(1)
        _logger.debug(f"expected values: {expected_values}")

        loss = self.loss_function(current_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.behaviour_function.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        new_values = self.behaviour_function(state_batch).gather(1, action_batch)
        _logger.debug(f"new values: {new_values}")

    def save_model(self) -> None:
        buffer = io.BytesIO()
        model_state_dict = self.behaviour_function.state_dict()
        torch.save(model_state_dict, buffer)
        save_model(TRAINED_MODEL_FILENAME, buffer)

    def get_random_position(self, boundary:int) -> Tuple[int, int, int]:
        x = random.choice(list(set(range(-boundary, boundary)) - set([0])))
        y = random.choice(list(set(range(-boundary, boundary)) - set([0])))
        a = random.choice(self.angles)
        return x, y, a
