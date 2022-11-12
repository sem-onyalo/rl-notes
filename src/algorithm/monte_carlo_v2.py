import logging
from typing import List

from .algorithm import Algorithm
from function import ActionValueFunctionTabular
from policy import Policy
from mdp import MDP
from model import EpsilonDecay
# from registry import Registry

ALGORITHM_NAME = "monte-carlo-v2"

_logger = logging.getLogger(ALGORITHM_NAME)

class MonteCarloV2(Algorithm):
    def __init__(
        self, 
        mdp:MDP, 
        function:ActionValueFunctionTabular, 
        policy:Policy, 
        epsilon_decay:EpsilonDecay, 
        # registry:Registry=None, 
        discount_rate=1., 
        max_episodes=1000) -> None:

        super().__init__(ALGORITHM_NAME)

        self.mdp = mdp
        self.function = function
        self.policy = policy
        self.epsilon_decay = epsilon_decay
        # self.registry = registry
        self.discount_rate = discount_rate
        self.max_episodes = max_episodes

    def run(self, max_episodes=0):
        max_episodes = self.max_episodes if max_episodes == 0 else max_episodes
