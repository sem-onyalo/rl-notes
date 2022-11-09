import math

class EpsilonDecay:
    def __init__(self, epsilon:float) -> None:
        self.epsilon = epsilon

    def update_epsilon(self, value:object) -> None:
        pass

    def get_epsilon(self) -> float:
        return self.epsilon

class EpsilonDecayGlie(EpsilonDecay):
    def update_epsilon(self, value:int) -> None:
        self.epsilon = 1 / value

class EpsilonDecayExp(EpsilonDecay):
    def __init__(self, start:float, end:float, decay_rate:int) -> None:
        super().__init__(start)
        self.end = end
        self.decay_rate = decay_rate

    def update_epsilon(self, value:int) -> None:
        self.epsilon = self.end + (self.epsilon - self.end) * math.exp(-1. * value / self.decay_rate)
