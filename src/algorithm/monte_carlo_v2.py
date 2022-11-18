from collections import OrderedDict
from itertools import groupby

import numpy as np

from .algorithm import Algorithm
from function import PolicyV2
from function import TabularFunctionV2
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
        self.init_experimental_stuff()

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        for episode in range(1, max_episodes + 1):
            self.init_new_episode(episode)

            rewards, total_reward = self.run_episode(episode)

            self.update_function(self.run_history.visits, rewards)

            self.policy.decay(episode)

            self.run_history.add(total_reward, self.policy.epsilon)

            self.log_episode_metrics(total_reward, self.run_history.max_reward)

            self.do_experimental_end_of_episode_stuff(episode=episode, reward=total_reward)

        self.save(max_episodes)

        self.do_experimental_end_of_run_stuff()

    def run_episode(self, episode:int):
        rewards = {}
        total_reward = 0
        is_terminal = False
        state = self.mdp.start()
        episode_start = self.run_history.steps
        self.logger.info(f"{episode}> init state: {state}")

        while not is_terminal:
            transformed_state = self.transform_state(state)

            action = self.policy.get_stochastic(transformed_state)

            reward, next_state, is_terminal, info = self.mdp.step(action)

            self.update_history(transformed_state, action, next_state, reward, rewards, info)

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - episode_start

            self.logger.info(f"{episode}> state: {state}, action: {action}, reward: {reward}, steps: {steps}")

            if steps >= self.max_steps_per_episode:
                self.logger.info(f"max steps for episode reached")
                break

        self.logger.info(f"{episode}> terminal reason: {info['reason']}")

        return rewards, total_reward

    def update_function(self, visits:dict, rewards:dict) -> None:
        for (state, action) in rewards:
            value = self.function.get(state, action)
            state_action_visits = visits[(state, action)]
            state_action_rewards = rewards[(state, action)]
            total_reward = self.get_total_discounted_reward(state_action_rewards)
            new_value = value + (1 / state_action_visits * (total_reward - value))
            self.function.update(state, action, new_value)

    def get_total_discounted_reward(self, rewards):
        # G_t = SUM(t=0,t=T-1) gamma**t * R
        # G_t = R_t+1 + gamma*R_t+2 + ... + gamma**T-1*R_T
        return sum([self.discount_rate**step * reward for step, reward in enumerate(rewards)])

    def log_episode_metrics(self, total_reward:float, max_reward:float) -> None:
        self.logger.info(f"total reward: {total_reward}")
        self.logger.info(f"max reward: {max_reward}")

    def save(self, max_episodes:int):
        if self.registry != None and max_episodes > 0:
            self.registry.save_run_history(self.name, self.run_history)
            self.save_model(self.run_history.run_id)

    def init_experimental_stuff(self):
        self.states_explored_pct = []
        self.starting_states_explored_pct = []
        # self.states_explored_count = OrderedDict()
        # self.starting_states_explored_count = OrderedDict()
        self.total_rewards_per_episode = []
        self.mean_total_rewards_per_episode = []
        mdp:DriftCarMDP = self.mdp

        # for x in range(-mdp.boundary, mdp.boundary + 1):
        #     for y in range(-mdp.boundary, mdp.boundary + 1):
        #         for a in mdp.discrete_orientations:
        #             self.states_explored_count[(x, y, a)] = 0
        #             self.starting_states_explored_count[(x, y, a)] = 0

    def do_experimental_end_of_episode_stuff(self, *args, **kwargs):
        episode:int = kwargs["episode"]
        reward:float = kwargs["reward"]
        mdp:DriftCarMDP = self.mdp

        explored_pct = round(len(self.policy) / mdp.n_discrete_state_space, 2)
        self.states_explored_pct.append(explored_pct)

        starting_states_explored_count = { k: len(list(g)) for k, g in groupby(sorted(mdp.starting_positions))}
        starting_explored_pct = round(len(starting_states_explored_count) / mdp.n_discrete_state_space, 2)
        self.starting_states_explored_pct.append(starting_explored_pct)

        # states_explored_count = { tuple(list(map(int, k.split(",")))[2:]): len(list(g)) for k, g in groupby(sorted(self.policy.get_states_visited()))}
        # for k in states_explored_count:
        #     if k not in self.states_explored_count:
        #         self.logger.info(f"found state: {k}")
        #         quit()
        #     self.states_explored_count[k] = states_explored_count[k]
        # for k in starting_states_explored_count:
        #     if k not in self.starting_states_explored_count:
        #         self.logger.info(f"found state: {k}")
        #         quit()
        #     self.starting_states_explored_count[k] = starting_states_explored_count[k]

        self.total_rewards_per_episode.append(reward)
        self.mean_total_rewards_per_episode.append(np.asarray(self.total_rewards_per_episode).mean())

        self.registry.write_plot(
            x_list=list(range(1, episode + 1)),
            y_lists=[self.states_explored_pct, self.starting_states_explored_pct],
            plot_labels=["States explored", "Starting states explored"],
            x_label="Episode",
            y_label="Exploration %",
            title=f"Training: State Space Explored % ({self.name})",
            filename=f"{self.name}-{self.run_history.run_id}-state-space-explored.png"
        )

        # self.registry.write_plot(
        #     x_list=list(range(0, mdp.n_discrete_state_space)),
        #     y_lists=[self.states_explored_count.values(), self.starting_states_explored_count.values()],
        #     plot_labels=["States explored count", "Starting states explored count"],
        #     x_label="States",
        #     y_label="Count",
        #     title=f"Training: State Exploration Count ({self.name})",
        #     filename=f"{self.name}-{self.run_history.run_id}-state-exploration-count.png"
        # )

        self.registry.write_plot(
            x_list=list(range(1, episode + 1)),
            y_lists=[self.total_rewards_per_episode, self.mean_total_rewards_per_episode],
            plot_labels=["Total reward", "Total mean reward"],
            x_label="Episode",
            y_label="Reward",
            title=f"Training: Episode Rewards ({self.name})",
            filename=f"{self.name}-{self.run_history.run_id}-total-reward.png"
        )

    def do_experimental_end_of_run_stuff(self, *args, **kwargs):
        self.logger.info(f"terminal info: {self.run_history.terminal_info}")

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

            self.logger.info(f"terminal reason: {info['reason']}")

        self.logger.info(f"Max steps reached, exiting run!")
        # self.logger.info(f"Total reward: {reward}")

        # TODO: write memory to csv file
