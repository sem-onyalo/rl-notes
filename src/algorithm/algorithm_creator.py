from .monte_carlo import MonteCarlo
from .monte_carlo_v2 import MonteCarloV2
from .monte_carlo_v2 import MonteCarloV2Inf
from .monte_carlo_v2 import PolicyV2
from .monte_carlo_v2 import TabularFunctionV2
from .monte_carlo_policy_gradient import MonteCarloPolicyGradient
from .monte_carlo_policy_gradient_inf import MonteCarloPolicyGradientInf
from .policy import Policy
from .q_learning import QLearning
from .q_network import QNetwork
from .q_network_inf import QNetworkInf
from .value_iteration import ValueIteration
from constants import *
from function import ActionValueFunctionTabular
from function import ValueFunctionTabular
from mdp import DriftCarMDP
from mdp import RacecarBulletGymMDP
from mdp import StudentMDP
from mdp import StudentMDPV2
from model import EpsilonDecayGlie
from registry import LocalRegistry

class AlgorithmCreator:
    """
    This class represents a factory for building a specified algorithm.
    """

    @staticmethod
    def build(algorithm_name, mdp_name, args):
        if mdp_name == STUDENT_MDP:
            mdp = StudentMDP()
        elif mdp_name == STUDENT_MDP_V2:
            mdp = StudentMDPV2()
        elif mdp_name == RACECAR_MDP:
            mdp = RacecarBulletGymMDP()
        elif mdp_name == DRIFT_CAR_MDP:
            mdp = DriftCarMDP(show_visual=args.render, max_step_count=args.max_steps)
        else:
            raise Exception(f"The MDP '{mdp_name}' is invalid or not yet implemented")

        if args.registry_type == LOCAL_REGISTRY:
            registry = LocalRegistry(args.eval_root)
        else:
            raise Exception(f"The registry {args.registry_type} is invalid or not yet implemented")

        if algorithm_name == VALUE_ITERATION:
            function = ValueFunctionTabular(mdp)
            algorithm = ValueIteration(mdp, function, args.discount_rate, args.delta_threshold)
        elif algorithm_name == MONTE_CARLO:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = MonteCarlo(mdp, function, policy, args.discount_rate, args.episodes, (not args.no_glie))
        elif algorithm_name == MONTE_CARLO_V2:
            if args.inference:
                function = TabularFunctionV2()
                policy = PolicyV2(mdp, function, args.epsilon, args.decay_type)
                algorithm = MonteCarloV2Inf(mdp, policy, registry, args.max_steps)
            else:
                function = TabularFunctionV2(mdp=mdp)
                policy = PolicyV2(mdp, function, args.epsilon, args.decay_type)
                algorithm = MonteCarloV2(mdp, function, policy, registry, args.discount_rate, args.episodes, max_steps_per_episode=args.max_steps)
        elif algorithm_name == Q_LEARNING:
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = QLearning(mdp, function, policy, args.discount_rate, args.change_rate, args.episodes)
        elif algorithm_name == Q_NETWORK:
            if args.inference:
                algorithm = QNetworkInf(mdp, args.layers)
            else:
                epsilon_decay = EpsilonDecayGlie(args.epsilon)
                # epsilon_decay = EpsilonDecayExp(args.epsilon, args.epsilon_end, args.epsilon_decay_rate)
                algorithm = QNetwork(mdp, epsilon_decay, args.layers, discount_rate=args.discount_rate, change_rate=args.change_rate, max_episodes=args.episodes, batch_size=args.batch_size)
        elif algorithm_name == MONTE_CARLO_POLICY_GRADIENT:
            if args.inference:
                algorithm = MonteCarloPolicyGradientInf(mdp, args.layers)
            else:
                algorithm = MonteCarloPolicyGradient(mdp, args.layers, args.discount_rate, args.change_rate, args.batch_size, args.episodes)
        else:
            raise Exception(f"The name '{algorithm_name}' is not a valid algorithm name.")

        return algorithm
