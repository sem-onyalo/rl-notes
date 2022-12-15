import time
from datetime import datetime

from .algorithm import Algorithm
from constants import *
from function import Policy
from mdp import MDP
from model import ExperienceMemory
from model import RunHistory
from registry import Registry

ALGORITHM_NAME = "monte-carlo"

class MonteCarloArgs:
    run_id:str
    episodes:int
    discount_rate:float

class MonteCarloV2(Algorithm):
    """
    This class represents the GLIE (Greedy in the Limit with Infinite Exploration) Monte-Carlo control algorithm.
    """

    def __init__(self, mdp:MDP, policy:Policy, registry:Registry, args:MonteCarloArgs) -> None:
        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.policy = policy
        self.registry = registry
        self.max_episodes = args.episodes
        self.discount_rate = args.discount_rate
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

            rewards, total_reward = self.run_episode(episode)

            self.update_function(rewards)

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

            self.update_history(transformed_state, action, next_state, reward, rewards, info)

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - start_step

            self.logger.info(f"{episode}> action: {action}, reward: {reward}, steps: {steps}")

            self.logger.debug(f"{episode}> state:\n{state}")

            if self.run_history.steps - start_step >= 10000:
                break

        return rewards, total_reward

    def choose_action(self, transformed_state:str) -> int:
        if self.mdp.get_operator() == MACHINE_TRAINING:
            return self.policy.choose_action(transformed_state)
        else:
            return self.policy(transformed_state)

    def update_function(self, rewards:dict) -> None:
        for (state, action) in rewards:
            value = self.policy.get_value(state, action)
            state_action_rewards = rewards[(state, action)]
            state_action_visits = self.run_history.visits[(state, action)]
            total_reward = self.get_total_discounted_reward(state_action_rewards)
            new_value = value + (1 / state_action_visits * (total_reward - value))
            self.policy.update(state, action, new_value)

    def get_total_discounted_reward(self, rewards):
        # G_t = SUM(t=0,t=T-1) gamma**t * R
        # G_t = R_t+1 + gamma*R_t+2 + ... + gamma**T-1*R_T
        return sum([self.discount_rate**step * reward for step, reward in enumerate(rewards)])

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
