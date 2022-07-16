import argparse
import logging

class StudentMDP:
    """
    This class represents the student MDP example.
    """

    def __init__(self):
        self.states = list(range(5)) # TikTok, Class1, Class2, Class3, End
        self.actions = list(range(5)) # TikTok, Quit, Study, Sleep, Pub

        self.state_actions = {
            # schema: { state: { action: (reward, [next_state]) } }
            0: {
                0: (-1, [0]),
                1: ( 0, [1])
            },
            1: {
                0: (-1, [0]),
                2: (-2, [2])
            },
            2: {
                2: (-2, [3]),
                3: ( 0, [4])
            },
            3: {
                2: (10, [4]),
                4: ( 1, [1, 2, 3])
            },
            4: None # terminal state
        }

        self.action_probabilities = {
            # schema: {state: {next_state: probability}}
            3: {
                1: 0.2,
                2: 0.4,
                3: 0.4
            }
        }

    def get_state_action_result(self, state, action):
        """
        Gets the result of the specified action when in the specified state.
        :returns: reward, next_states, is_terminal_state OR None if state-action pair DNE.
        """
        if state in self.state_actions:
            if self.state_actions[state] == None:
                return None, None, True
            elif action in self.state_actions[state]:
                reward, next_states = self.state_actions[state][action]
                return reward, next_states, False

        return None

    def get_action_probability(self, state, next_state):
        """
        This _might_ get pull out of the MDP and turn into a policy function.
        """
        if state in self.action_probabilities and next_state in self.action_probabilities[state]:
            return self.action_probabilities[state][next_state]
        else:
            return 1 # if probability isn't explicitly set then always take that action (i.e. act greedily)

class ValueFunctionTabular:
    """
    This class represents a tabular value function with initial values of zero.
    """

    def __init__(self, states):
        self.state_values = { x: 0. for x in states }

    def __str__(self):
        return str(self.state_values)

    def __call__(self, state):
        return self.state_values[state]

    def get_value(self):
        return sum([self.state_values[x] for x in self.state_values])

    def update_values(self, state_values):
        self.state_values = state_values

    def get_values(self):
        return self.state_values

class ValueIteration:
    """
    This class represents the dynamic programming value iteration algorithm.
    """

    def __init__(self, mdp, value_function, discount_rate, delta_threshold, max_iterations=100):
        self.mdp = mdp
        self.value_function = value_function
        self.discount_rate = discount_rate
        self.delta_threshold = delta_threshold
        self.max_iterations = max_iterations

    def run(self):
        current_delta = 0
        logging.info(f"0, {self.value_function.get_value()}, {current_delta}")

        for i in range(self.max_iterations):
            value = self.value_function.get_value()
            new_state_values = self.update_state_values()
            self.value_function.update_values(new_state_values)
            current_delta = abs(value - self.value_function.get_value())

            logging.info(f"{i + 1}, {value}, {current_delta}")
            logging.debug(f"value function: {self.value_function}, delta: {current_delta}")

            if current_delta < self.delta_threshold:
                logging.info("threshold reached, exiting algorithm")
                break

        return self.value_function

    def update_state_values(self):
        """
        Gets new values for each state (i.e. represents a single iteration for updating the values for all states).
        """

        # store a dictionary of values for each action from a particular state
        action_values = { x: list() for x in self.mdp.states }

        for state in self.mdp.states:
            # store the reward and next possible states for each state-action pair
            state_action_results = list()

            for action in self.mdp.actions:
                result = self.mdp.get_state_action_result(state, action)
                if result != None:
                    reward, next_states, is_terminal = result
                    if not is_terminal:
                        state_action_results.append((state, action, reward, next_states))
                    else:
                        state_action_results.append((state, None, reward, next_states))
                        break # no need to check other actions if terminal state
            logging.debug(f"state, action, reward, next states: {state_action_results}")

            for state, action, reward, next_states in state_action_results:
                # calculate the expected reward for each state-action pair
                # i.e. v(s) = R + y * sum(P * v(s'))
                value = reward + self.discount_rate * sum(
                    [
                        self.mdp.get_action_probability(state, next_state) * self.value_function(next_state)
                        for next_state in next_states
                    ]) if reward != None else 0. # reward == None means a terminal state, so just store 0
                logging.debug(f"state, action, reward, next states, value: {state}, {action}, {reward}, {next_states}, {value}")
                action_values[state].append(value)
            logging.debug(f"state action values: {state}: {action_values[state]}")

        # get the max value for each state
        # i.e. v(s) = max(R + y * sum(P * v(s'))), for all actions
        state_values = { x: max(action_values[x]) for x in action_values }
        logging.debug(f"updated state values: {state_values}")

        return state_values

def init_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount-rate", type=float, default=1.0, help="The discount-rate parameter.")
    parser.add_argument("--delta-threshold", type=float, default=0.2, help="The delta threshold to stop the algorithm.")
    return parser.parse_args()

def main(args):
    mdp = StudentMDP()
    value_function = ValueFunctionTabular(mdp.states)
    algorithm = ValueIteration(mdp, value_function, args.discount_rate, args.delta_threshold)
    algorithm.run()

if __name__ == "__main__":
    init_logger()
    args = get_runtime_args()
    main(args)

