import time
from datetime import datetime

from .algorithm import Algorithm
from constants import *
from function import Policy
from mdp import MDP
from model import ExperienceMemory
from model import RunHistory
from registry import Registry

ALGORITHM_NAME = "q-network"

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
        self.batch_size = 1
        self.memory = ExperienceMemory(10000)

        self.load_model(args.run_id)
        self.mdp.set_policy(policy)

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        self.t0 = datetime.utcnow()

        if self.mdp.get_operator() == MACHINE_TRAINING:
            self.run_training(max_episodes)
        else:
            while True:
                self.run_policy()
                time.sleep(5)

    def run_training(self, episodes:int):
        for episode in range(1, episodes + 1):
            t0 = self.init_new_episode(episode)

            _, total_reward = self.run_episode(episode)

            self.policy.decay(episode)

            self.run_history.add(episode, total_reward, self.policy.epsilon)

            self.log_episode_metrics(t0=t0, episode=episode, reward=total_reward)

        self.logger.info("-" * 50)
        self.logger.info(f"run id: {self.run_history.run_id}")

        self.save_model()

        self.log_run_metrics()

    def run_episode(self, episode:int):
        rewards = {}
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        start_step = self.run_history.steps
        self.logger.debug(f"{episode}> init state:\n{state}")

        while not is_terminal:
            transformed_state = self.policy.transform_state(state)

            action = self.choose_action(transformed_state)

            reward, next_state, is_terminal, info = self.mdp.step(action)

            # transformed_next_state = self.policy.transform_state(next_state)

            self.update_history(tuple(state.flatten()), action, tuple(next_state.flatten()), reward, rewards, info)

            self.update_function()

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - start_step

            self.logger.info(f"{episode}> action: {action}, reward: {reward}, steps: {steps}, {self.run_history.steps}")

            self.logger.debug(f"{episode}> state:\n{state}")

            if self.run_history.steps - start_step >= 10000:
                break

        return rewards, total_reward

    def choose_action(self, transformed_state):
        if self.mdp.get_operator() == MACHINE_TRAINING:
            return self.policy.choose_action(transformed_state)
        else:
            return self.policy(transformed_state)

    def update_function(self):
        update_target_function = False
        if self.run_history.steps > 0 and self.run_history.steps % self.target_update_frequency == 0:
            update_target_function = True
            self.logger.debug("updating target function")

        transition = self.memory.sample(self.batch_size)[0]
        self.policy.update(transition, self.discount_rate, update_target_function)

    def log_episode_metrics(self, *args, **kwargs):
        t0 = kwargs["t0"]
        episode:int = kwargs["episode"]
        reward:float = kwargs["reward"]

        max_reward_info = self.run_history.get_latest_max_reward_info()

        self.logger.info("-" * 50)
        self.logger.info(f"{episode}> total reward: {reward}")
        self.logger.info(f"{episode}> max reward ({max_reward_info[0]}): {max_reward_info[1]:.2f}")
        self.logger.info(f"{episode}> episode elapsed time: {datetime.utcnow() - t0}")
        self.logger.info(f"{episode}> elapsed time: {datetime.utcnow() - self.t0}")

        self.registry.write_plot(
            x_list=list(range(1, episode + 1)),
            y_lists=[self.run_history.total_rewards, self.run_history.mean_total_rewards],
            plot_labels=["Total reward", "Total mean reward"],
            x_label="Episode",
            y_label="Reward",
            title=f"Training: Episode Rewards ({self.name})",
            filename=f"{self.name}-{self.run_history.run_id}-total-reward.png"
        )

    def log_run_metrics(self, *args, **kwargs):
        self.logger.info(f"is terminal history: {self.run_history.is_terminal_history}")
        self.logger.info(f"max reward history: {self.run_history.max_rewards_history}")
        self.logger.info(f"elapsed time: {datetime.utcnow() - self.t0}")
