import copy
import random
from collections import deque

from .transition import Transition

class ExperienceMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition:Transition):
        assert isinstance(transition, Transition), f"Error: object being pushed must be of type {Transition}"
        self.memory.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size) if len(self.memory) > batch_size else list(self.memory)
        return [t.to_tuple() for t in samples]

    def all(self):
        return [t.to_tuple() for t in self.memory]

    def clear(self):
        self.memory.clear()
