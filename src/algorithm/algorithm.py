import io
import logging
from datetime import datetime
from typing import DefaultDict
from typing import List

import numpy as np

from function import PolicyV2
from mdp import MDP
from model import ExperienceMemory
from model import Transition
from model import RunHistory
from registry import Registry

class Algorithm:
    mdp:MDP
    policy:PolicyV2
    memory:ExperienceMemory
    registry:Registry
    run_history:RunHistory
    model_file_ext:str = "json"

    def __init__(self, name) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def run(self, max_episodes=0):
        pass

    def init_new_episode(self, episode:int) -> datetime:
        self.logger.info("-" * 50)
        self.logger.info(f"Episode {episode}")
        self.logger.info("-" * 50)
        return datetime.utcnow()

    def log_episode_metrics(self, path:List[object], total_reward:float, max_reward:float) -> None:
        self.logger.info(f"Total reward (G_t): {total_reward}, Max reward: {max_reward}")

    def transform_state(self, state:object) -> str:
        if isinstance(state, int):
            return str(state)
        elif isinstance(state, np.ndarray):
            return ",".join(list(map(str, state.flatten())))
        else:
            raise Exception(f"{self.name} currently does not support {type(state)} state types")

    def update_history(self, state:int, action:int, next_state:int, reward:float, rewards:DefaultDict[str,List[int]], info:DefaultDict[str,str]) -> None:
        if not (state, action) in rewards:
            rewards[(state, action)] = []
        for key in rewards:
            rewards[key].append(reward)

        if not (state, action) in self.run_history.visits:
            self.run_history.visits[(state, action)] = 0
        self.run_history.visits[(state, action)] += 1

        if "reason" in info and info["reason"] != "":
            if info["reason"] not in self.run_history.is_terminal_history:
                self.run_history.is_terminal_history[info["reason"]] = 0
            self.run_history.is_terminal_history[info["reason"]] += 1

        self.memory.push(Transition(state, action, next_state, reward))

    def save_model(self, run_id:str) -> None:
        buffer = io.BytesIO()
        model_state_dict = self.policy.function.state_dict()
        buffer.write(model_state_dict)
        self.registry.save_model(f"{self.name}-{run_id}.{self.model_file_ext}", buffer)
