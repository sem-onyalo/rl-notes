import io
import json
import logging
import random
from datetime import datetime
from typing import DefaultDict, List
from uuid import uuid4

import numpy as np

from .algorithm import Algorithm
from mdp import MDP
from model import ExperienceMemory
from model import Transition
from registry import Registry

ALGORITHM_NAME = "monte-carlo-v2"

_logger = logging.getLogger(ALGORITHM_NAME)

class TabularFunctionV2:
    def __init__(self, **kwargs) -> None:
        if "mdp" in kwargs:
            self.load_from_mdp(kwargs["mdp"])

    def load_from_mdp(self, mdp:MDP) -> None:
        self.value_map = {
            state: np.asarray([0.] * mdp.n_actions)
            for state in range(0, mdp.n_states)
        }

    def load_from_buffer(self, buffer:io.BytesIO) -> None:
        state_dict = json.loads(buffer.getvalue())
        self.value_map = {}
        for state in state_dict:
            self.value_map[int(state)] = np.asarray(state_dict[state])

    def __call__(self, state:int) -> np.ndarray:
        return self.value_map[state]

    def get(self, state:int, action:int) -> float:
        return self.__call__(state)[action]

    def update(self, state:int, action:int, value:float) -> None:
        self.value_map[state][action] = value

    def state_dict(self) -> DefaultDict:
        state_dict = {}
        for state in self.value_map:
            state_dict[state] = list(self.value_map[state])
        return json.dumps(state_dict, indent=4).encode("utf-8")

class PolicyV2:
    def __init__(self, mdp:MDP, function:TabularFunctionV2, epsilon:float, decay_type:str, decay_rate:float=None, epsilon_floor:float=None) -> None:
        self.n_actions = mdp.n_actions
        self.function = function
        self.epsilon = epsilon
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.epsilon_floor = epsilon_floor

    def __call__(self, state:int) -> int:
        return np.argmax(self.function(state)) # return the optimal (max) action

    def get_stochastic(self, state:int) -> int:
        do_explore = random.random() < self.epsilon
        if do_explore:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.__call__(state)

    def decay(self, value:float) -> None:
        if self.decay_type == "glie": # TODO get from constants
            self.epsilon = 1 / value
        elif self.decay_type == "exp":
            raise Exception("Exponential decay not yet implemented.")

class RunHistory:
    def __init__(self, episodes:int) -> None:
        self.run_id = self.new_run_id()
        self.episodes = episodes
        self.total_rewards = []
        self.max_rewards = []
        self.epsilon = []
        self.visits = {}
        self.max_reward = None
        self.steps = 0

    def add(self, total_reward:float, epsilon:float) -> None:
        self.max_reward = total_reward if self.max_reward == None or total_reward > self.max_reward else self.max_reward
        self.total_rewards.append(total_reward)
        self.max_rewards.append(self.max_reward)
        self.epsilon.append(epsilon)

    def new_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4()}"

class MonteCarloV2(Algorithm):
    trained_model_filename = "monte-carlo.json"

    def __init__(
        self, 
        mdp:MDP, 
        function:TabularFunctionV2, 
        policy:PolicyV2, 
        registry:Registry=None, 
        discount_rate=1., 
        max_episodes=1000, 
        memory_capacity=10000, 
        max_steps_per_episode=5000) -> None:

        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.function = function
        self.policy = policy
        self.registry = registry
        self.discount_rate = discount_rate
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.memory = ExperienceMemory(memory_capacity)

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        for episode in range(1, max_episodes + 1):
            _logger.info("-" * 50)
            _logger.info(f"Episode {episode}")
            _logger.info("-" * 50)

            rewards, total_reward = self.run_episode(episode, self.run_history.visits)

            self.update_function(self.run_history.visits, rewards)

            self.policy.decay(episode)

            self.run_history.add(total_reward, self.policy.epsilon)

            self.log_episode_metrics(total_reward, self.run_history.max_reward)

        if self.registry != None and max_episodes > 0:
            self.registry.save_run_history(ALGORITHM_NAME, self.run_history)
            self.save_model()

    def run_episode(self, episode:int, visits:dict):
        rewards = {}
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        episode_start = self.run_history.steps
        _logger.info(f"{episode}> init state: {state}")

        while not is_terminal:
            action = self.policy.get_stochastic(state)

            reward, next_state, is_terminal = self.mdp.step(action)

            self.update_history(state, action, next_state, reward, visits, rewards)

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - episode_start

            _logger.info(f"{episode}> state: {state}, action: {action}, reward: {reward}, steps: {steps}")

            if steps >= self.max_steps_per_episode:
                _logger.info(f"max steps for episode reached")
                break

        return rewards, total_reward

    def update_history(self, state:int, action:int, next_state:int, reward:float, visits:DefaultDict[str,int], rewards:DefaultDict[str,List[int]]) -> None:
        if not (state, action) in visits:
            visits[(state, action)] = 0

        if not (state, action) in rewards:
            rewards[(state, action)] = []

        visits[(state, action)] += 1

        for key in rewards:
            rewards[key].append(reward)

        self.memory.push(Transition(state, action, next_state, reward))

    def update_function(self, visits:dict, rewards:dict) -> None:
        for (state, action) in rewards:
            visit_count = visits[(state, action)]
            transition_rewards = rewards[(state, action)]
            current_value = self.function.get(state, action)
            total_reward = self.get_total_discounted_reward(transition_rewards)
            new_value = current_value + (1 / visit_count * (total_reward - current_value))
            self.function.update(state, action, new_value)

    def get_total_discounted_reward(self, rewards):
        # G_t = SUM(t=0,t=T-1) gamma**t * R
        # G_t = R_t+1 + gamma*R_t+2 + ... + gamma**T-1*R_T
        return sum([self.discount_rate**step * reward for step, reward in enumerate(rewards)])

    def log_episode_metrics(self, total_reward:float, max_reward:float) -> None:
        _logger.info(f"total reward: {total_reward}")
        _logger.info(f"max reward: {max_reward}")

    def save_model(self) -> None:
        buffer = io.BytesIO()
        model_state_dict = self.function.state_dict()
        buffer.write(model_state_dict)
        self.registry.save_model(self.trained_model_filename, buffer)

class MonteCarloV2Inf(MonteCarloV2):
    def __init__(self, mdp:MDP, policy:PolicyV2, registry:Registry, max_steps=5000) -> None:
        self.mdp = mdp
        self.policy = policy
        self.registry = registry
        self.max_steps = max_steps
        self.memory = ExperienceMemory(0)
        self.policy.function.load_from_buffer(registry.load_model(self.trained_model_filename))

    def run(self):
        steps = 0
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        _logger.info(f"init state: {state}")

        while not is_terminal:
            action = self.policy(state)

            reward, next_state, is_terminal = self.mdp.step(action)

            self.memory.push(Transition(state, action, next_state, reward))

            steps += 1

            state = next_state

            total_reward += reward

            _logger.info(f"state: {state}, action: {action}, reward: {reward}")

            if steps >= self.max_steps:
                _logger.info(f"Aborting, max steps reached!")
                return

        # TODO: write memory to csv file

        _logger.info(f"Total reward: {reward}")
