#
# TODO: integrate this code into src/main.py
#

import argparse
import logging

from util import init_logger

from algorithm import Human
from algorithm.monte_carlo_v2 import MonteCarloV2 as MonteCarlo
from algorithm.q_learning_v2 import QLearningV2 as QLearning
from algorithm.q_network_v2 import QNetworkV2 as QNetwork
from constants import *
from function import PolicyApproximator
from function import PolicyTabular
from mdp import DiscreteCarMDP
from mdp import GridTargetMDP
from registry import LocalRegistry

_logger = logging.getLogger(__name__)

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-level", type=str, default="INFO", help="The logging level to use.")
    parser.add_argument("--plan", action="store_true", help="Build the objects but do not run the agent.")

    # registry args
    parser.add_argument("--eval-root", type=str, default="/eval", help="The root directory where training artifacts are written to.")

    # policy args
    parser.add_argument("--explore-type", type=str, default=EPSILON_GREEDY_EXPLORE, help="Denotes how the agent will explore the MDP space (i.e. explore/exploit balance)")
    parser.add_argument("--epsilon", type=float, default=0.9, help="The starting epsilon value to use for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-floor", type=float, default=0.05, help="The ending epsilon value to use for epsilon-greedy exploration.")
    parser.add_argument("--decay-type", type=str, default=GLIE_DECAY, help="The way in which to decay the epsilon value.")
    parser.add_argument("--decay-rate", type=str, default=None, help="The rate at which to decay the epsilon value.")
    parser.add_argument("--run-id", type=str, default=None, help="The run ID to supply to load a trained model from.")
    
    mdp_parser = parser.add_subparsers(dest="mdp")

    grid_target_parser = mdp_parser.add_parser(GRID_TARGET_MDP, help="The grid target MDP.")
    grid_target_parser.add_argument("-d", "--dim", type=int, default=5, help="The grid dimension.")
    grid_target_parser.add_argument("-f", "--fps", type=int, default=60, help="The frames per second.")
    grid_target_parser.add_argument("--width", type=int, default=1920, help="The display width.")
    grid_target_parser.add_argument("--height", type=int, default=1080, help="The display height.")
    grid_target_parser.add_argument("--agent-start-position", type=str, default="4,2", help="The display height.")
    grid_target_parser.add_argument("--target-start-position", type=str, default="2,4", help="The display height.")
    grid_target_parser.add_argument("--display", action="store_true", help="Display the grid on screen.")
    grid_target_parser.add_argument("--trail", action="store_true", help="Display a trail of the agent's path through the MDP.")

    discrete_car_parser = mdp_parser.add_parser(DISCRETE_CAR_MDP, help="The discrete car MDP.")
    discrete_car_parser.add_argument("-r", "--rad", type=int, default=250, help="The track radius.")
    discrete_car_parser.add_argument("-f", "--fps", type=int, default=60, help="The frames per second.")
    discrete_car_parser.add_argument("--width", type=int, default=1920, help="The display width.")
    discrete_car_parser.add_argument("--height", type=int, default=1080, help="The display height.")
    discrete_car_parser.add_argument("--display", action="store_true", help="Display the grid on screen.")
    discrete_car_parser.add_argument("--trail", action="store_true", help="Display a trail of the agent's path through the MDP.")

    for mdp_parser in [grid_target_parser, discrete_car_parser]:
        agent_parser = mdp_parser.add_subparsers(dest="agent")
        agent_parser.add_parser(HUMAN)

        monte_carlo_parser = agent_parser.add_parser(MONTE_CARLO)
        monte_carlo_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
        monte_carlo_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")

        q_learning_parser = agent_parser.add_parser(Q_LEARNING)
        q_learning_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
        q_learning_parser.add_argument("--discount-rate", type=float, default=1., help="The discount-rate parameter.")
        q_learning_parser.add_argument("--change-rate", type=float, default=1., help="The change-rate parameter.")

        q_network_parser = agent_parser.add_parser(Q_NETWORK)
        q_network_parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to run the algorithm for.")
        q_network_parser.add_argument("--discount-rate", type=float, default=.9, help="The discount-rate parameter.")
        q_network_parser.add_argument("--change-rate", type=float, default=.01, help="The change-rate parameter.")
        q_network_parser.add_argument("--target-update-frequency", type=int, default=500, help="The frequency (in steps) to synchronize the target function weights with the behaviour weights.")

    return parser.parse_args()

def main(args):
    _logger.info("Building registry")
    registry = LocalRegistry(args.eval_root)

    _logger.info("Building MDP")
    mdp_name:str = args.mdp
    if mdp_name == GRID_TARGET_MDP:
        mdp = GridTargetMDP(args.dim, args.fps, args.width, args.height, tuple(map(int, args.agent_start_position.split(","))), tuple(map(int, args.target_start_position.split(","))), args.display, args.trail)
    elif mdp_name == DISCRETE_CAR_MDP:
        mdp = DiscreteCarMDP(args.rad, args.fps, args.width, args.height, args.display, args.trail)
    else:
        raise Exception(f"MDP {mdp_name} invalid or not yet implemented.")

    agent_name:str = args.agent
    if agent_name == HUMAN:
        _logger.info("Building agent")
        agent = Human(mdp)
    elif agent_name in [MONTE_CARLO, Q_LEARNING, Q_NETWORK]:
        if agent_name in [MONTE_CARLO, Q_LEARNING]:
            _logger.info("Builing tabular policy")
            policy = PolicyTabular(mdp, args)
        elif agent_name in [Q_NETWORK]:
            _logger.info("Builing approximator policy")
            policy = PolicyApproximator(mdp, args)

        _logger.info(f"Building {agent_name} agent")
        if agent_name == MONTE_CARLO:
            agent = MonteCarlo(mdp, policy, registry, args)
        elif agent_name == Q_LEARNING:
            agent = QLearning(mdp, policy, registry, args)
        elif agent_name == Q_NETWORK:
            agent = QNetwork(mdp, policy, registry, args)
    else:
        raise Exception(f"Agent {agent_name} invalid or not yet implemented.")

    if not args.plan:
        _logger.info("Running agent")
        agent.run()

    _logger.info("Done!")

if __name__ == "__main__":
    args = get_runtime_args()
    init_logger(args.log_level)
    main(args)
