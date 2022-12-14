import io
import logging
from datetime import datetime
from typing import DefaultDict
from typing import List

from constants import *
from function import Policy
from mdp import MDP
from model import ExperienceMemory
from model import Transition
from model import RunHistory
from registry import Registry

class Algorithm:
    mdp:MDP
    policy:Policy
    memory:ExperienceMemory
    registry:Registry
    run_history:RunHistory

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

    def load_model(self, run_id:str) -> None:
        if run_id == None:
            self.mdp.set_operator(MACHINE_TRAINING)
        else:
            buffer = self.registry.load_model(f"{self.name}-{run_id}.{self.policy.model_file_ext}")
            self.policy.load_model(buffer)
            self.mdp.set_operator(MACHINE)

    def save_model(self):
        if self.mdp.get_operator() == MACHINE_TRAINING and self.registry != None:
            self.registry.save_run_history(self.name, self.run_history)
            buffer = self.policy.get_model()
            self.registry.save_model(f"{self.name}-{self.run_history.run_id}.{self.policy.model_file_ext}", buffer)

    def run_policy(self) -> None:
        is_terminal = False
        state = self.mdp.start()
        start_step = self.run_history.steps

        while not is_terminal:
            transformed_state = self.policy.transform_state(state)

            action = self.policy(transformed_state)

            _, next_state, is_terminal, _ = self.mdp.step(action)

            state = next_state

            self.run_history.steps += 1

            if self.run_history.steps - start_step >= 10000:
                break
