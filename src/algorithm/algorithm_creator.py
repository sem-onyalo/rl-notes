import constants as ct
from .monte_carlo import MonteCarlo
from .monte_carlo_policy_gradient import MonteCarloPolicyGradient
from .monte_carlo_policy_gradient_inf import MonteCarloPolicyGradientInf
from .policy import Policy
from .q_learning import QLearning
from .value_iteration import ValueIteration
from function import ActionValueFunctionTabular
from function import ValueFunctionTabular
from mdp import DriftCarMDP
from mdp import RacecarBulletGymMDP
from mdp import StudentMDP

class AlgorithmCreator:
    """
    This class represents a factory for building a specified algorithm.
    """

    @staticmethod
    def build(algorithm_name, mdp_name, args):
        if mdp_name == ct.STUDENT_MDP:
            mdp = StudentMDP()
        elif mdp_name == ct.RACECAR_MDP:
            mdp = RacecarBulletGymMDP()
        elif mdp_name == ct.DRIFT_CAR_MDP:
            mdp = DriftCarMDP(show_visual=args.inference)
        else:
            raise Exception(f"The MDP '{mdp_name}' is invalid or not yet implemented")

        if algorithm_name == ct.VALUE_ITERATION:
            function = ValueFunctionTabular(mdp)
            algorithm = ValueIteration(mdp, function, args.discount_rate, args.delta_threshold)
        elif algorithm_name == ct.MONTE_CARLO:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = MonteCarlo(mdp, function, policy, args.discount_rate, args.episodes, (not args.no_glie))
        elif algorithm_name == ct.Q_LEARNING:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = QLearning(mdp, function, policy, args.discount_rate, args.change_rate, args.episodes)
        elif algorithm_name == ct.MONTE_CARLO_POLICY_GRADIENT:
            if args.inference:
                algorithm = MonteCarloPolicyGradientInf(mdp, args.layers)
            else:
                algorithm = MonteCarloPolicyGradient(mdp, args.layers, args.discount_rate, args.change_rate, args.batch_size, args.episodes)
        else:
            raise Exception(f"The name '{algorithm_name}' is not a valid algorithm name.")

        return algorithm
