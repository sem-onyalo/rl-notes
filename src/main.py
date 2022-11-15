import argparse
import logging

from algorithm import AlgorithmCreator
from constants import *

_logger = logging.getLogger(__name__)

def init_logger(level:str):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.getLevelName(level.upper())
    )

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mdp", type=str, default=STUDENT_MDP, help="The MDP to train the algorithm on.")
    parser.add_argument("-l", "--log-level", type=str, default="INFO", help="The logging level to use.")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
    parser.add_argument("-s", "--max-steps", type=int, default=5000, help="The maximum number of steps before the episode is considered terminal.")
    parser.add_argument("-d", "--decay-type", type=str, default=None, help="The epsilon decay function to use.")
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("--registry-type", type=str, default=LOCAL_REGISTRY, help="The type of registry to use for reading and writing artifacts.")
    parser.add_argument("--eval-root", type=str, default="/eval", help="The root directory where training artifacts are written to.")
    parser.add_argument("--data-root", type=str, default="/data", help="The root directory where model artifacts are read from.")

    subparser = parser.add_subparsers(dest="algorithm")

    value_iteration_parser = subparser.add_parser(VALUE_ITERATION, help="Value iteration DP algorithm.")
    value_iteration_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    value_iteration_parser.add_argument("--delta-threshold", type=float, default=0.2, 
        help="The delta of the total value of the value function between adjacent iterations at which to stop the algorithm.")

    monte_carlo_parser = subparser.add_parser(MONTE_CARLO, help="GLIE Monte-Carlo control algorithm.")
    monte_carlo_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    monte_carlo_parser.add_argument("--no_glie", action="store_true", help="Don't use GLIE policy improvement during run.")
    monte_carlo_parser.add_argument("--epsilon", type=float, default=0.9, 
        help="The epsilon value to use when implementing the epsilon-greedy policy. Ignored when using GLIE.")

    monte_carlo_v2_parser = subparser.add_parser(MONTE_CARLO_V2, help="Monte-Carlo control algorithm.")
    monte_carlo_v2_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    monte_carlo_v2_parser.add_argument("--no_glie", action="store_true", help="Don't use GLIE policy improvement during run.")
    monte_carlo_v2_parser.add_argument("--epsilon", type=float, default=0.9, help="The starting epsilon value to use for epsilon-greedy exploration.")

    monte_carlo_policy_gradient_parser = subparser.add_parser(MONTE_CARLO_POLICY_GRADIENT, help="Run the Monte-Carlo policy gradient (REINFORCE) algorithm.")
    monte_carlo_policy_gradient_parser.add_argument("--layers", type=str, default="1,16,5", help="A comma-separated list representing the neural network architecture.")
    monte_carlo_policy_gradient_parser.add_argument("--change-rate", type=float, default=.2, help="The change-rate parameter.")
    monte_carlo_policy_gradient_parser.add_argument("--discount-rate", type=float, default=.9, help="The discount-rate parameter.")
    monte_carlo_policy_gradient_parser.add_argument("--batch-size", type=int, default=4, help="The number of episodes to batch together when updating the policy approximator function.")

    q_learning_parser = subparser.add_parser(Q_LEARNING, help="Q-Learning off-policy control algorithm.")
    q_learning_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    q_learning_parser.add_argument("--change-rate", type=float, default=0.2, help="The change-rate parameter.")
    q_learning_parser.add_argument("--epsilon", type=float, default=0.6, help="The epsilon value for the epsilon-greedy policy.")

    q_network_parser = subparser.add_parser(Q_NETWORK, help="Run the Deep Q-Network (DQN) control algorithm.")
    q_network_parser.add_argument("--layers", type=str, default="1,3,9,5", help="A comma-separated list representing the neural network architecture.")
    q_network_parser.add_argument("--epsilon", type=float, default=.9, help="The parameter representing the upper bound and starting value of the exploration probability value.")
    q_network_parser.add_argument("--epsilon-end", type=float, default=.05, help="The parameter representing the lower bound of the exploration probability value.")
    q_network_parser.add_argument("--epsilon-decay-rate", type=int, default=100, help="The parameter representing the decay rate of the exploration probability value.")
    q_network_parser.add_argument("--change-rate", type=float, default=.2, help="The change-rate parameter.")
    q_network_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    q_network_parser.add_argument("--batch-size", type=int, default=16, help="The number of transitions to use when training the action-value approximator functions.")
    q_network_parser.add_argument("--target-update-freq", type=int, default=200, help="The frequency (in steps) to synchronize the target function weights with the behaviour weights.")

    return parser.parse_args()

def main(args):
    algorithm = AlgorithmCreator.build(args.algorithm, args.mdp, args)

    algorithm.run()

    _logger.info("Done!")

if __name__ == "__main__":
    args = get_runtime_args()
    init_logger(args.log_level)
    main(args)
