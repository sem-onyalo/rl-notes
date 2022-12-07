import logging

from algorithm import AlgorithmCreator
from util import get_runtime_args
from util import init_logger

_logger = logging.getLogger(__name__)

def main(args):
    _logger.info("Building algorithm")
    algorithm = AlgorithmCreator.build(args.algorithm, args.mdp, args)

    _logger.info("Running algorithm")
    algorithm.run()

    _logger.info("Done!")

if __name__ == "__main__":
    args = get_runtime_args()
    init_logger(args.log_level)
    main(args)
