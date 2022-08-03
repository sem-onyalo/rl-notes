import logging
from statistics import mean

class ValueIteration:
    """
    This class represents the dynamic programming value iteration algorithm.
    """

    def __init__(self, mdp, function, discount_rate, delta_threshold, max_iterations=100):
        self.mdp = mdp
        self.function = function
        self.discount_rate = discount_rate
        self.delta_threshold = delta_threshold
        self.max_iterations = max_iterations

    def __eq__(self, other):
        if other != None:
            return (
                self.mdp == other.mdp
                and self.function == other.function
                and self.discount_rate == other.discount_rate
                and self.delta_threshold == other.delta_threshold
                and self.max_iterations == other.max_iterations
            )

        return False

    def run(self):
        current_delta = 0
        logging.info(f"0, {self.function.get_value()}, {current_delta}")

        for i in range(self.max_iterations):
            value = self.function.get_value()
            new_state_values = self.update_state_values()
            self.function.update_values(new_state_values)
            current_delta = abs(value - self.function.get_value())

            logging.info(f"{i + 1}, {value}, {current_delta}")
            logging.debug(f"value function: {self.function}, delta: {current_delta}")

            if current_delta < self.delta_threshold:
                logging.info("threshold reached, exiting algorithm")
                break

        # TODO: build policy from value function and return it as 0th tuple element
        return self.function

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
                result = self.mdp.get_reward_and_next_states(state, action)
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
                        self.mdp.get_action_probability(state, next_state) * self.function(next_state)
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

