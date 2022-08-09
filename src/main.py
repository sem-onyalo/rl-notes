import argparse
import logging

import constants as ct
from algorithm import AlgorithmCreator

def init_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

def get_runtime_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="algorithm")

    value_iteration_parser = subparser.add_parser(ct.VALUE_ITERATION, help="Value iteration DP algorithm.")
    value_iteration_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    value_iteration_parser.add_argument("--delta-threshold", type=float, default=0.2, 
        help="The delta of the total value of the value function between adjacent iterations at which to stop the algorithm.")

    monte_carlo_parser = subparser.add_parser(ct.MONTE_CARLO, help="GLIE Monte-Carlo control algorithm.")
    monte_carlo_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    monte_carlo_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
    monte_carlo_parser.add_argument("--no_glie", action="store_true", help="Don't use GLIE policy improvement during run.")
    monte_carlo_parser.add_argument("--epsilon", type=float, default=0.9, 
        help="The epsilon value to use when implementing the epsilon-greedy policy. Ignored when using GLIE.")

    monte_carlo_parser = subparser.add_parser(ct.Q_LEARNING, help="Q-Learning off-policy control algorithm.")
    monte_carlo_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    monte_carlo_parser.add_argument("--change-rate", type=float, default=0.2, help="The change-rate parameter.")
    monte_carlo_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
    monte_carlo_parser.add_argument("--epsilon", type=float, default=0.6, help="The epsilon value for the epsilon-greedy policy.")
    return parser.parse_args()

def main(args):
    algorithm = AlgorithmCreator.build(args.algorithm, args)
    algorithm.run()

if __name__ == "__main__":
    init_logger()
    args = get_runtime_args()
    main(args)

