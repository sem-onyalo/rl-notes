import time
from datetime import datetime

from .algorithm import Algorithm
from constants import *
from function import Policy
from mdp import MDP
from model import ExperienceMemory
from model import RunHistory
from model import Transition
from registry import Registry

ALGORITHM_NAME = "q-network-v2"

class QNetworkArgs:
    run_id:str
    episodes:int
    discount_rate:float
    change_rate:float
    target_update_frequency:int
    batch_size:int

class QNetworkV2(Algorithm):
    """
    This class represents a Deep Q-Network (DQN) algorithm with experience replay and fixed Q-targets.
    """

    def __init__(self, mdp:MDP, policy:Policy, registry:Registry, args:QNetworkArgs) -> None:
        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.policy = policy
        self.registry = registry
        self.max_episodes = args.episodes
        self.discount_rate = args.discount_rate
        self.change_rate = args.change_rate
        self.target_update_frequency = args.target_update_frequency
        self.batch_size = args.batch_size
        self.memory = ExperienceMemory(10000)

        if args.run_id == None:
            self.mdp.set_operator(MACHINE_TRAINING)
        else:
            buffer = registry.load_model(f"{self.name}-{args.run_id}.{self.model_file_ext}")
            self.policy.function.load_from_buffer(buffer)
            self.mdp.set_operator(MACHINE)

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        self.t0 = datetime.utcnow()

        if self.mdp.get_operator() == MACHINE_TRAINING:
            self.run_training(max_episodes)
        # else:
        #     self.run_episode(0)
        #     while True:
        #         time.sleep(5)

        # self.save(max_episodes)

        # self.log_run_metrics()

    def run_training(self, episodes:int):
        for episode in range(1, episodes + 1):
            t0 = self.init_new_episode(episode)

            _, total_reward = self.run_episode(episode)

            self.policy.decay(episode)

            self.run_history.add(episode, total_reward, self.policy.epsilon)

            # self.log_episode_metrics(t0=t0, episode=episode, reward=total_reward)

        self.logger.info("-" * 50)
        self.logger.info(f"run id: {self.run_history.run_id}")

    def run_episode(self, episode:int):
        rewards = {}
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        episode_start = self.run_history.steps
        self.logger.info(f"{episode}> init state:\n{state}")

        while not is_terminal:
            transformed_state = self.policy.transform_state(state)

            # action = self.choose_action(transformed_state)

            # reward, next_state, is_terminal, info = self.mdp.step(action)

            # transformed_next_state = self.transform_state(next_state)

            # self.update_function(transformed_state, action, transformed_next_state, reward)

            # self.update_history(transformed_state, action, transformed_next_state, reward, rewards, info)

            # state = next_state

            # total_reward += reward

            # self.run_history.steps += 1

            # steps = self.run_history.steps - episode_start

            self.logger.info(f"{episode}> action: {action}, reward: {reward}, steps: {steps}")

            self.logger.debug(f"{episode}> state:\n{state}")

        return rewards, total_reward
