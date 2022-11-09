import io
import logging
import time
import traceback
import random

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
from registry import save_model

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

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.terminal_reasons = {}
        total_rewards = []
        max_rewards = []
        max_reward = None
        self.steps = 0

        t0 = time.time()
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
            self.log_episode_metrics({}, total_reward, max_reward)

        _logger.info(f"Training elapsed time: {time.time() - t0}")
        
        _logger.info("-" * 50)
        _logger.info(f"Terminal reasons")
        _logger.info("-" * 50)

        for key in self.terminal_reasons:
            _logger.info(f"{key}: {self.terminal_reasons[key]}")

        if not self.no_plot and max_episodes > 0:
            plot_training_metrics(ALGORITHM_NAME, max_episodes, total_rewards, max_rewards)
            self.save_model()

    def run_episode(self):
        # if the number of steps in the episode is less than the batch size 
        # then optimize the model at the end of the epiosde
        ran_function_update = False

        t0 = time.time()
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()

        while not is_terminal:
            action = self.get_action(self.normalize_state(state))
            reward, next_state, is_terminal, _ = self.mdp.step(action)
            self.memory.push(Transition(state, action, next_state, reward))
            total_reward += reward

            if len(self.memory) >= self.batch_size:
                self.update_function()
                ran_function_update = True

            state = next_state

            self.epsilon_decay.update_epsilon(self.steps)
            if self.steps > 0 and self.steps % self.target_update_freq == 0:
                self.target_function.load_state_dict(self.behaviour_function.state_dict())
            self.steps += 1

        if not ran_function_update:
            self.update_function()

        _logger.info(f"Episode elapsed time: {time.time() - t0}")

        return total_reward

    def normalize_state(self, state):
        bound = 10.
        expanded = np.expand_dims(state, axis=0)
        clipped = np.clip(expanded, -bound, bound)
        normed = 2 * ((clipped - -bound) / (bound - -bound)) - 1 # normalize to -1,1
        return torch.tensor(normed, dtype=torch.float32, device=self.device)

    def normalize_state_batch(self, state):
        bound = 10.
        clipped = np.clip(state, -bound, bound)
        normed = 2 * ((clipped - -bound) / (bound - -bound)) - 1 # normalize to -1,1
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

    def update_function(self):
        transitions = self.memory.sample(self.batch_size)

        states, actions, next_states, rewards = self.transitions_to_batches(transitions)

        state_batch = self.normalize_state_batch([list(i) for i in states])
        action_batch = torch.tensor([[i] for i in actions], dtype=torch.long, device=self.device)
        next_state_batch = self.normalize_state_batch([list(i) for i in next_states])
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
