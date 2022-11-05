import io
import logging

import numpy as np
import torch
import torch.optim as optim

from .algorithm_net import AlgorithmNet
from mdp import MDP
from model import ExperienceMemory
from model import StateActionPair
from model import Transition
from registry import plot_training_metrics
from registry import save_model

ALGORITHM_NAME = "monte-carlo-policy-gradient"
TRAINED_MODEL_FILENAME = f"{ALGORITHM_NAME}.pth"

_logger = logging.getLogger(ALGORITHM_NAME)

class MonteCarloPolicyGradient(AlgorithmNet):
    """
    This class represents a Monte-Carlo policy gradient (REINFORCE) prediction algorithm.
    """

    def __init__(self, mdp:MDP, layers:str, discount_rate=1., change_rate=.2, batch_size=8, max_episodes=1000, no_plot=False) -> None:
        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.policy_function = self.build_linear_softmax_function(layers)
        self.discount_rate = discount_rate
        self.change_rate = change_rate
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.no_plot = no_plot
        self.memory = ExperienceMemory(None)
        self.optimizer = optim.Adam(self.policy_function.parameters(), lr=self.change_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes

        total_rewards = []
        max_rewards = []
        max_reward = None
        self.steps = 0
        
        for i in range(0, max_episodes):
            _logger.info("-" * 50)
            _logger.info(f"Episode {i + 1}")
            _logger.info("-" * 50)

            try:
                total_reward, state_action_path = self.run_episode()
            except Exception as e:
                _logger.error(f"Run episode failed on episode {i + 1} at step count {self.steps}: {e}")
                max_episodes = i
                break

            self.update_policy(i + 1)

            max_reward = total_reward if max_reward == None or total_reward > max_reward else max_reward
            total_rewards.append(total_reward)
            max_rewards.append(max_reward)
            self.log_episode_metrics(state_action_path, total_reward, max_reward)

        if not self.no_plot and max_episodes > 0:
            plot_training_metrics(ALGORITHM_NAME, max_episodes, total_rewards, max_rewards)
            self.save_model()

    def run_episode(self):
        total_reward = 0
        is_terminal = False
        state_action_path = []
        state = self.mdp.start()

        while not is_terminal:
            action = self.get_action(state)
            reward, next_state, is_terminal = self.mdp.step(action)
            self.memory.push(Transition(state, action, next_state, reward))
            total_reward += reward

            # may not need this anymore since we have memory now
            _logger.info(f"state: {state}, action: {action}, reward: {reward}")
            state_action_path.append(StateActionPair(state, action))
            state = next_state

        return total_reward, state_action_path

    def normalize_state(self, state):
        bound = 10.
        clipped = np.clip(state, -bound, bound)
        # normed = (clipped - (bound/2)) / (bound/2) # normalize to -1,1
        normed = 2 * ((clipped - -bound) / (bound - -bound)) - 1 # normalize to -1,1
        # _logger.info(f"state: {state}")
        # _logger.info(f"clipped: {clipped}")
        # _logger.info(f"normed: {normed}")
        return normed

    def get_action(self, state:list):
        state_expanded = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state_expanded, dtype=torch.float32, device=self.device)
        state_tensor = self.normalize_state(state_tensor)
        action_probs = self.policy_function(state_tensor)
        action_probs = action_probs.detach().numpy().squeeze()
        _logger.debug(f"state tensor: {state_tensor}")
        _logger.debug(f"actions probs: {action_probs}")
        action = np.random.choice(self.mdp.actions, p=action_probs)
        return action

    def update_policy(self, episode:int):
        if episode % self.batch_size != 0:
            return

        _logger.info("updating policy")

        transitions = self.memory.all()

        states, actions, _, rewards = self.transitions_to_batches(transitions)
        total_rewards = self.get_total_discounted_rewards(rewards, self.discount_rate)

        _logger.info(f"updating policy with {len(states)} transitions")

        state_batch = torch.tensor([self.normalize_state(i) for i in states], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([[i] for i in actions], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([[i] for i in total_rewards], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        log_probs = torch.log(self.policy_function(state_batch))
        selected_logprobs = reward_batch * torch.gather(log_probs, 1, action_batch).squeeze()
        loss = -selected_logprobs.mean()
        loss.backward()
        self.optimizer.step()

        self.memory.clear()

    def save_model(self) -> None:
        buffer = io.BytesIO()
        model_state_dict = self.policy_function.state_dict()
        torch.save(model_state_dict, buffer)
        save_model(TRAINED_MODEL_FILENAME, buffer)
