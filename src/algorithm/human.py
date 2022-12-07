from .algorithm import Algorithm
from constants import *
from mdp import MDP

class Human(Algorithm):
    """
    This class facilitates a human operating in an MDP.
    """

    def __init__(self, mdp:MDP) -> None:
        super().__init__(HUMAN)
        self.mdp = mdp
        self.mdp.set_operator(HUMAN)

    def run(self) -> None:
        self.mdp.start()
        while True:
            self.mdp.step(0)
