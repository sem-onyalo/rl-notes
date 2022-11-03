import logging
from typing import List

from model import StateActionPair

class Algorithm:
    def __init__(self, name) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def run(self, max_episodes=0):
        pass

    def log_episode_metrics(self, path:List[StateActionPair], total_reward:float, max_reward:float) -> None:
        # self.logger.info("State path: " + " -> ".join([f"[{sa.state},{self.mdp.items[sa.action] if sa.action != None else None}]" for sa in path]))
        self.logger.info(f"Total reward (G_t): {total_reward}, Max reward: {max_reward}")
