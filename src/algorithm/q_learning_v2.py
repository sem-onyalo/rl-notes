from .algorithm import Algorithm
from function import PolicyV2
from function import TabularFunctionV2
from mdp import MDP
from model import ExperienceMemory
from model import RunHistory
from model import Transition
from registry import Registry

ALGORITHM_NAME = "q-learning-v2"

class QLearningV2(Algorithm):
    """
    This class represents the Q-Learning off-policy control algorithm.
    """

    def __init__(
        self, 
        mdp:MDP, 
        function:TabularFunctionV2, 
        policy:PolicyV2, 
        registry:Registry=None, 
        discount_rate=1., 
        change_rate=.2,
        max_episodes=1000, 
        memory_capacity=10000, 
        max_steps_per_episode=5000) -> None:

        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.function = function
        self.policy = policy
        self.registry = registry
        self.discount_rate = discount_rate
        self.change_rate = change_rate
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.memory = ExperienceMemory(memory_capacity)

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        self.run_history = RunHistory(max_episodes)

        for episode in range(1, max_episodes + 1):
            self.init_new_episode(episode)

            total_reward = self.run_episode(episode)

            self.policy.decay(episode)

            self.run_history.add(total_reward, self.policy.epsilon)

            self.log_episode_metrics(total_reward, self.run_history.max_reward)

        if self.registry != None and max_episodes > 0:
            self.registry.save_run_history(ALGORITHM_NAME, self.run_history)
            self.save_model(self.run_history.run_id)

    def run_episode(self, episode:int):
        rewards = {}
        total_reward = 0
        episode_start = self.run_history.steps

        is_terminal = False
        state = self.mdp.start()
        self.logger.info(f"{episode}> init state: {state}")
        while not is_terminal:
            transformed_state = self.transform_state(state)

            action = self.policy.choose_action(transformed_state)

            reward, next_state, is_terminal, _ = self.mdp.step(action)

            self.update_function(transformed_state, action, next_state, reward)

            self.update_history(transformed_state, action, next_state, reward, rewards)

            state = next_state

            total_reward += reward

            self.run_history.steps += 1

            steps = self.run_history.steps - episode_start

            self.logger.info(f"{episode}> state: {state}, action: {action}, reward: {reward}, steps: {steps}")

            if steps >= self.max_steps_per_episode:
                self.logger.info(f"max steps for episode reached")
                break

        return rewards, total_reward

    def update_function(self, state:str, action:int, next_state:str, reward:float) -> None:
        value = self.function.get(state, action)
        max_value = self.policy.get_greedy(next_state)
        # Q(S,A) = Q(S,A) + a * (R + y * max_a[Q(S',a)] - Q(S,A))
        new_value = value + self.change_rate * (reward + (self.discount_rate * max_value) - value)
        self.function.update(state, action, new_value)
