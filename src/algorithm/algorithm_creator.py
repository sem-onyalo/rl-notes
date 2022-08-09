import constants as ct
from .monte_carlo import MonteCarlo
from .policy import Policy
from .q_learning import QLearning
from .value_iteration import ValueIteration
from function import ActionValueFunctionTabular
from function import ValueFunctionTabular
from mdp import StudentMDP

class AlgorithmCreator:
    """
    This class represents a factory for building a specified algorithm.
    """

    @staticmethod
    def build(algorithm_name, args):
        mdp = StudentMDP()
        if algorithm_name == ct.VALUE_ITERATION:
            function = ValueFunctionTabular(mdp)
            algorithm = ValueIteration(mdp, function, args.discount_rate, args.delta_threshold)
            return algorithm
        elif algorithm_name == ct.MONTE_CARLO:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = MonteCarlo(mdp, function, policy, args.discount_rate, args.episodes, (not args.no_glie))
            return algorithm
        elif algorithm_name == ct.Q_LEARNING:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = QLearning(mdp, function, policy, args.discount_rate, args.change_rate, args.episodes)
            return algorithm
        else:
            raise Exception(f"The name '{algorithm_name}' is not a valid algorithm name.")

