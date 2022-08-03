from algorithm.monte_carlo import MonteCarlo
from algorithm.policy import Policy
from algorithm.value_iteration import ValueIteration
from function.action_value_function_tabular import ActionValueFunctionTabular
from function.value_function_tabular import ValueFunctionTabular
from mdp.student_mdp import StudentMDP

class AlgorithmCreator:
    """
    This class represents a factory for building a specified algorithm.
    """

    @staticmethod
    def build(algorithm_name, args):
        mdp = StudentMDP()
        if algorithm_name == "value-iteration":
            function = ValueFunctionTabular(mdp)
            algorithm = ValueIteration(mdp, function, args.discount_rate, args.delta_threshold)
            return algorithm
        elif algorithm_name == "monte-carlo":
            function = ActionValueFunctionTabular(mdp)
            policy = Policy(mdp, args.epsilon)
            algorithm = MonteCarlo(mdp, function, policy, args.discount_rate, args.episodes, (not args.no_glie))
            return algorithm
        else:
            raise Exception(f"The name '{algorithm_name}' is not a valid algorithm name.")

