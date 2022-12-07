from datetime import datetime
from itertools import groupby

import numpy as np

from .algorithm import Algorithm
from constants import MACHINE
from function import PolicyV2
from mdp import MDP
from mdp import DriftCarMDP
from model import ExperienceMemory
from model import RunHistory
from model import Transition
from registry import Registry

ALGORITHM_NAME = "monte-carlo-v2"

class MonteCarloV2(Algorithm):
    """
    This class represents the GLIE (Greedy in the Limit with Infinite Exploration) Monte-Carlo control algorithm.
    """

    def __init__(
        self, 
        mdp:MDP, 
        policy:PolicyV2, 
        registry:Registry=None, 
        discount_rate=1., 
        max_episodes=1000, 
        memory_capacity=10000, 
        max_steps_per_episode=5000) -> None:
        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.policy = policy
        self.registry = registry
        self.discount_rate = discount_rate
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.memory = ExperienceMemory(memory_capacity)
        self.init_experimental_stuff()

        self.mdp.set_operator(MACHINE)

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        self.t0 = datetime.utcnow()

        for episode in range(1, max_episodes + 1):
            t0 = self.init_new_episode(episode)

            rewards, total_reward = self.run_episode(episode)

            self.update_function(rewards)

            self.policy.decay(episode)

            self.run_history.add(episode, total_reward, self.policy.epsilon)

            self.do_end_of_episode_stuff(t0=t0, episode=episode, reward=total_reward)

        self.save(max_episodes)

        self.do_end_of_run_stuff()

    def run_episode(self, episode:int):
        rewards = {}
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        episode_start = self.run_history.steps
        self.logger.info(f"{episode}> init state:\n{state}")

        while not is_terminal:
            transformed_state = self.transform_state(state)

            action = self.policy.choose_action(transformed_state)

            reward, next_state, is_terminal, info = self.mdp.step(action)

            self.update_history(transformed_state, action, next_state, reward, rewards, info)

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - episode_start

            self.logger.info(f"{episode}> action: {action}, reward: {reward}, steps: {steps}")
            self.logger.debug(f"{episode}> state: {state}")

            if steps >= self.max_steps_per_episode:
                self.logger.info(f"max steps for episode reached")
                break

        return rewards, total_reward

    def update_function(self, rewards:dict) -> None:
        for (state, action) in rewards:
            value = self.policy.function.get(state, action)
            state_action_rewards = rewards[(state, action)]
            state_action_visits = self.run_history.visits[(state, action)]
            total_reward = self.get_total_discounted_reward(state_action_rewards)
            new_value = value + (1 / state_action_visits * (total_reward - value))
            self.policy.function.update(state, action, new_value)

    def get_total_discounted_reward(self, rewards):
        # G_t = SUM(t=0,t=T-1) gamma**t * R
        # G_t = R_t+1 + gamma*R_t+2 + ... + gamma**T-1*R_T
        return sum([self.discount_rate**step * reward for step, reward in enumerate(rewards)])

    def save(self, max_episodes:int):
        if self.registry != None and max_episodes > 0:
            self.registry.save_run_history(self.name, self.run_history)
            self.save_model(self.run_history.run_id)

    def init_experimental_stuff(self):
        self.states_explored_pct = []
        self.starting_states_explored_pct = []

    def do_end_of_episode_stuff(self, *args, **kwargs):
        t0 = kwargs["t0"]
        mdp:DriftCarMDP = self.mdp
        episode:int = kwargs["episode"]
        reward:float = kwargs["reward"]

        self.logger.info(f"{episode}> total reward: {reward}")
        max_reward_info = self.run_history.get_latest_max_reward_info()
        self.logger.info(f"{episode}> max reward ({max_reward_info[0]}): {max_reward_info[1]:.2f}")
        self.logger.info(f"{episode}> episode elapsed time: {datetime.utcnow() - t0}")
        self.logger.info(f"{episode}> elapsed time: {datetime.utcnow() - self.t0}")

        # plots

        # explored_pct = round(len(self.policy) / mdp.n_discrete_state_space, 2)
        # self.states_explored_pct.append(explored_pct)

        # starting_states_explored_count = { k: len(list(g)) for k, g in groupby(sorted(mdp.starting_positions))}
        # starting_explored_pct = round(len(starting_states_explored_count) / mdp.n_discrete_state_space, 2)
        # self.starting_states_explored_pct.append(starting_explored_pct)

        # self.registry.write_plot(
        #     x_list=list(range(1, episode + 1)),
        #     y_lists=[self.states_explored_pct, self.starting_states_explored_pct],
        #     plot_labels=["States explored", "Starting states explored"],
        #     x_label="Episode",
        #     y_label="Exploration %",
        #     title=f"Training: State Space Explored % ({self.name})",
        #     filename=f"{self.name}-{self.run_history.run_id}-state-space-explored.png"
        # )

        self.registry.write_plot(
            x_list=list(range(1, episode + 1)),
            y_lists=[self.run_history.total_rewards, self.run_history.mean_total_rewards],
            plot_labels=["Total reward", "Total mean reward"],
            x_label="Episode",
            y_label="Reward",
            title=f"Training: Episode Rewards ({self.name})",
            filename=f"{self.name}-{self.run_history.run_id}-total-reward.png"
        )

    def do_end_of_run_stuff(self, *args, **kwargs):
        self.logger.info(f"is terminal history: {self.run_history.is_terminal_history}")
        self.logger.info(f"max reward history: {self.run_history.max_rewards_history}")
        self.logger.info(f"elapsed time: {datetime.utcnow() - self.t0}")

class MonteCarloV2Inf(Algorithm):
    def __init__(self, mdp:MDP, policy:PolicyV2, registry:Registry, run_id:str, max_steps=5000) -> None:
        super().__init__(ALGORITHM_NAME)

        assert run_id != None, "Please supply the run id"

        self.mdp = mdp
        self.policy = policy
        self.run_id = run_id
        self.registry = registry
        self.max_steps = max_steps
        self.memory = ExperienceMemory(0)
        self.policy.function.load_from_buffer(registry.load_model(f"{self.name}-{self.run_id}.{self.model_file_ext}"))

    def run(self):
        steps = 0
        while steps < self.max_steps:
            total_reward = 0
            is_terminal = False
            state = self.mdp.start()
            self.logger.info(f"init state: {state}")
            while not is_terminal:
                transformed_state = self.transform_state(state)

                # TODO: since function value map is lazy loaded we may get actions that
                #       haven't been see during training, which means we would be getting
                #       an arbitrary action back. It would probably be best to handle this
                #       somehow.
                action = self.policy(transformed_state)

                reward, next_state, is_terminal, info = self.mdp.step(action)

                self.memory.push(Transition(state, action, next_state, reward))

                total_reward += reward

                state = next_state

                steps += 1

                self.logger.info(f"state: {state}, action: {action}, reward: {reward}")

            self.logger.info(f"total reward: {total_reward}")
            self.logger.info(f"terminal reason: {info['reason']}")

        self.logger.info(f"Max steps reached, exiting run!")
        # TODO: write memory to csv file
