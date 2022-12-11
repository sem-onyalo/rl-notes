#
# TODO: integrate this code into src/main.py
#

import argparse
import logging

from util import init_logger

from algorithm import Human
from algorithm.monte_carlo_v2 import MonteCarloV2 as MonteCarlo
from algorithm.q_learning_v2 import QLearningV2 as QLearning
from constants import *
from function import PolicyTabular as Policy
from mdp import GridTargetMDP
from registry import LocalRegistry

_logger = logging.getLogger(__name__)

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-level", type=str, default="INFO", help="The logging level to use.")

    # registry args
    parser.add_argument("--eval-root", type=str, default="/eval", help="The root directory where training artifacts are written to.")

    # policy args
    parser.add_argument("--explore-type", type=str, default=EPSILON_GREEDY_EXPLORE, help="Denotes how the agent will explore the MDP space (i.e. explore/exploit balance)")
    parser.add_argument("--epsilon", type=float, default=0.9, help="The starting epsilon value to use for epsilon-greedy exploration.")
    parser.add_argument("--decay-type", type=str, default=GLIE_DECAY, help="The epsilon decay function to use.")
    parser.add_argument("--run-id", type=str, default=None, help="The run ID to supply to load a trained model from.")
    
    mdp_parser = parser.add_subparsers(dest="mdp")

    grid_target_parser = mdp_parser.add_parser(GRID_TARGET_MDP, help="The grid target MDP.")
    grid_target_parser.add_argument("-d", "--dim", type=int, default=10, help="The grid dimension.")
    grid_target_parser.add_argument("-f", "--fps", type=int, default=60, help="The frames per second.")
    grid_target_parser.add_argument("--width", type=int, default=1920, help="The display width.")
    grid_target_parser.add_argument("--height", type=int, default=1080, help="The display height.")
    grid_target_parser.add_argument("--agent-start-position", type=str, default="9,2", help="The display height.")
    grid_target_parser.add_argument("--target-start-position", type=str, default="2,9", help="The display height.")
    grid_target_parser.add_argument("--display", action="store_true", help="Display the grid on screen.")
    grid_target_parser.add_argument("--trail", action="store_true", help="Display a trail of the agent's path through the MDP.")

    agent_parser = grid_target_parser.add_subparsers(dest="agent")
    agent_parser.add_parser(HUMAN)

    monte_carlo_parser = agent_parser.add_parser(MONTE_CARLO)
    monte_carlo_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
    monte_carlo_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")

    q_learning_parser = agent_parser.add_parser(Q_LEARNING)
    q_learning_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
    q_learning_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
    q_learning_parser.add_argument("--change-rate", type=float, default=1., help="The change-rate parameter.")

    return parser.parse_args()

def main(args):
    _logger.info("Building registry")
    registry = LocalRegistry(args.eval_root)

    _logger.info("Building MDP")
    mdp = GridTargetMDP(args.dim, args.fps, args.width, args.height, tuple(map(int, args.agent_start_position.split(","))), tuple(map(int, args.target_start_position.split(","))), args.display, args.trail)

    if args.agent == HUMAN:
        _logger.info("Building agent")
        agent = Human(mdp)
    elif args.agent in [MONTE_CARLO, Q_LEARNING]:
        _logger.info("Builing policy")
        policy = Policy(mdp, args.explore_type, args.epsilon, args.decay_type)

        if args.agent == MONTE_CARLO:
            _logger.info("Building agent")
            agent = MonteCarlo(mdp, policy, registry, args)
        elif args.agent == Q_LEARNING:
            _logger.info("Building agent")
            agent = QLearning(mdp, policy, registry, args)
        else:
            raise Exception(f"Agent {args.agent} invalid or not yet implemented.")
    else:
        raise Exception(f"Agent {args.agent} invalid or not yet implemented.")

    _logger.info("Running agent")
    agent.run()

    _logger.info("Done!")

if __name__ == "__main__":
    args = get_runtime_args()
    init_logger(args.log_level)
    main(args)
