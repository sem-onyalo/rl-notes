from datetime import datetime
from uuid import uuid4

class RunHistory:
    def __init__(self, episodes:int) -> None:
        self.run_id = self.new_run_id()
        self.episodes = episodes
        self.total_rewards = []
        self.max_rewards = []
        self.epsilon = []
        self.visits = {}
        self.max_reward = None
        self.steps = 0

    def add(self, total_reward:float, epsilon:float) -> None:
        self.max_reward = total_reward if self.max_reward == None or total_reward > self.max_reward else self.max_reward
        self.total_rewards.append(total_reward)
        self.max_rewards.append(self.max_reward)
        self.epsilon.append(epsilon)

    def new_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4()}"
