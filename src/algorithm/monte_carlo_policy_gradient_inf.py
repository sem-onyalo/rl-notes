import logging

import numpy as np
import torch
import torch.nn as nn

from .monte_carlo_policy_gradient import ALGORITHM_NAME
from .monte_carlo_policy_gradient import MonteCarloPolicyGradient
from .monte_carlo_policy_gradient import TRAINED_MODEL_FILENAME
from mdp import MDP
from registry import load_model

_logger = logging.getLogger(f"{ALGORITHM_NAME}-inf")

class MonteCarloPolicyGradientInf(MonteCarloPolicyGradient):
    def __init__(self, mdp:MDP, layers:str) -> None:
        self.mdp = mdp
        self.function = self.build_function(layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, max_episodes=0):
        # override the max episodes for now
        if max_episodes != 1:
            _logger.info(f"Overriding max episodes {max_episodes} to 1")
            max_episodes = 1

        # while True:
        for _ in range(0, max_episodes):
            is_terminal = False
            state = self.mdp.start()
            while not is_terminal:
                action = self.get_action(state)
                _, next_state, is_terminal = self.mdp.step(action)
                state = next_state

    def get_action(self, state):
        with torch.no_grad():
            state_expanded = np.expand_dims(state, axis=0)
            state_tensor = torch.tensor(state_expanded, dtype=torch.float32, device=self.device)
            state_tensor = self.normalize_state(state_tensor)
            action_probs = self.function(state_tensor)
            return action_probs.max(1)[1].item()

    def build_function(self, layers:str) -> nn.Module:
        model = self.build_linear_softmax_function(layers)
        buffer = load_model(TRAINED_MODEL_FILENAME)
        model.load_state_dict(torch.load(buffer))
        model.eval()
        return model
