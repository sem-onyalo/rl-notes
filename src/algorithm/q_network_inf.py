import logging

import torch
import torch.nn as nn

from .q_network import ALGORITHM_NAME
from .q_network import QNetwork
from .q_network import TRAINED_MODEL_FILENAME
from mdp import MDP
# from registry import load_model

_logger = logging.getLogger(f"{ALGORITHM_NAME}-inf")

class QNetworkInf(QNetwork):
    def __init__(self, mdp:MDP, layers:str) -> None:
        self.mdp = mdp
        self.function = self.build_function(layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, max_episodes=0):
        # # override the max episodes for now
        # if max_episodes != 1:
        #     _logger.info(f"Overriding max episodes {max_episodes} to 1")
        #     max_episodes = 1

        while True: # for _ in range(0, max_episodes):
            is_terminal = False
            state = self.mdp.start()
            while not is_terminal:
                action = self.get_action(state)
                _, next_state, is_terminal, _ = self.mdp.step(action)
                state = next_state

    def get_action(self, state):
        state_normed = self.normalize_state(state)
        with torch.no_grad():
            predictions = self.function(state_normed)
            return predictions.max(1)[1].item()

    def build_function(self, layers:str) -> nn.Module:
        model = self.build_linear_function(layers)
        buffer = load_model(TRAINED_MODEL_FILENAME)
        model.load_state_dict(torch.load(buffer))
        model.eval()
        return model
