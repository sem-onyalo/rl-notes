from .algorithm import Algorithm
from .algorithm import AlgorithmArgs
from constants import *
from mdp import MDP
from registry import Registry

class Human(Algorithm):
    """
    This class facilitates a human operating in an MDP.
    """

    def __init__(self, mdp:MDP, registry:Registry, args:AlgorithmArgs) -> None:
        super().__init__(HUMAN, args)
        self.mdp = mdp
        self.registry = registry
        self.mdp.set_operator(HUMAN)
