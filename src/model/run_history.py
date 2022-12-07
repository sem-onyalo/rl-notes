from datetime import datetime
from typing import Tuple
from uuid import uuid4

import numpy as np

class RunHistory:
    def __init__(self, episodes:int) -> None:
        self.episodes = episodes
        self.run_id = self.new_run_id()
        self.is_terminal_history = {}
        self.epsilon = []
        self.visits = {}
        self.steps = 0

        self.max_rewards_history = []
        self.mean_total_rewards = []
        self.total_rewards = []
        self.max_rewards = []
        self.max_reward = None

    def add(self, episode:int, total_reward:float, epsilon:float) -> None:
        self.epsilon.append(epsilon)
        self.total_rewards.append(total_reward)
        self.update_max_reward(episode, total_reward)
        self.mean_total_rewards.append(np.asarray(self.total_rewards).mean())

    def update_max_reward(self, episode:int, total_reward:float) -> None:
        if self.max_reward == None or total_reward > self.max_reward:
            self.max_reward = total_reward
            self.max_rewards_history.insert(0, (episode, total_reward))
        self.max_rewards.append(self.max_reward)

    def get_latest_max_reward_info(self) -> Tuple[int, float]:
        return self.max_rewards_history[0]

    def new_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4()}"
