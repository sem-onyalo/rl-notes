import argparse
import logging
import random
from statistics import mean

class StudentMDP:
    """
    This class represents the student MDP example.
    """

    def __init__(self):
        self.states = list(range(5)) # TikTok, Class1, Class2, Class3, End
        self.actions = list(range(5)) # TikTok, Quit, Study, Sleep, Pub
        self.terminal_state = 4

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
            4: {
                0: (0, [4]),
                1: (0, [4]),
                2: (0, [4]),
                3: (0, [4]),
                4: (0, [4]),
            } # terminal state, i.e. all actions give zero rewards result to being in the same state
        }

        self.action_probabilities = {
            # schema: {state: {next_state: probability}}
            3: {
                1: 0.2,
                2: 0.4,
                3: 0.4
            }
        }

    def __getitem__(self, i):
        return self.state_actions[i]

    def is_terminal_state(self, state):
        return state == self.terminal_state

    def get_initial_state(self):
        return 0

    def get_state_actions(self):
        return self.state_actions

    def get_reward_and_next_state(self, state, action):
        if self.is_terminal_state(state):
            is_terminal_state = True
            next_state = state
            reward = 0
        else:
            is_terminal_state = False
            reward, next_states = self.state_actions[state][action]
            if len(next_states) > 1:
                probability_total = 0
                probability_value = random.random()
                for possible_next_state in self.action_probabilities[state]:
                    probability_total += self.action_probabilities[state][possible_next_state]
                    if probability_value < probability_total:
                        next_state = possible_next_state
                        break
            else:
                next_state = next_states[0]

        return reward, next_state, is_terminal_state

    def get_reward_and_next_states(self, state, action):
        # TODO: change function description to something like "get_reward_and_next_states"
        """
        Gets the result of the specified action when in the specified state.
        :returns: reward, next_states, is_terminal_state OR None if state-action pair DNE.
        """
        if state in self.state_actions and action in self.state_actions[state]:
            reward, next_states = self.state_actions[state][action]
            return reward, next_states, self.is_terminal_state(state)

        return None

    def get_action_probability(self, state, next_state):
        """
        This _might_ get pull out of the MDP and turn into a policy function.
        """
        if state in self.action_probabilities and next_state in self.action_probabilities[state]:
            return self.action_probabilities[state][next_state]
        else:
            return 1 # if probability isn't explicitly set then always take that action (i.e. act greedily)

class ActionValueFunctionTabular:
    """
    This class represents a tabular action-value function, i.e. Q(s,a), with initial values of zero.
    """

    def __init__(self, mdp):
        self.state_action_values = {
            state: {
                action: 0.
                for action in mdp[state]
            } for state in mdp.get_state_actions()
        }

    def __str__(self):
        return str(self.state_action_values)

    def __call__(self, state, action):
        return self.state_action_values[state][action]

    def get_values(self):
        return self.state_action_values

    def update_value(self, state, action, value):
        self.state_action_values[state][action] = value

    def get_optimal_action(self, state):
        optimal_action = None
        max_value = 0.
        for action in self.state_action_values[state]:
            if optimal_action == None or self.state_action_values[state][action] > max_value:
                optimal_action = action
                max_value = self.state_action_values[state][action]

        return optimal_action

class ValueFunctionTabular:
    """
    This class represents a tabular value function, i.e. V(s), with initial values of zero.
    """

    def __init__(self, mdp):
        self.state_values = {
            state: 0.
            for state in mdp.get_state_actions()
        }

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

class Policy:
    """
    This class represents a policy to be followed (i.e. for every state what action should be taken) with the caveat
    that in some cases the action will be random (i.e. exploratory)
    """

    def __init__(self, mdp, epsilon_probability):
        """
        Initializes a new policy.
        :param MDP mdp: The MDP in which to execute this policy.
        :param float epsilon_probability: The probability in which to take a random action (i.e. explore).
        If 0.0 then never explore (i.e. act greedily). If 1.0 then always explore.
        """
        self.mdp = mdp
        self.policy = { state: None for state in mdp.get_state_actions() }
        self.epsilon = epsilon_probability

    def __str__(self):
        return str(self.policy)

    def __call__(self):
        return self.policy

    def update_epsilon(self, value):
        self.epsilon = value

    def get_action(self, state):
        do_explore = random.random() < self.epsilon
        if self.policy[state] == None or do_explore:
            # pick a random action from the possible actions in the specified state
            actions = list(self.mdp[state].keys())
            action_idx = random.randint(0, len(actions) - 1)
            return actions[action_idx]
        else:
            # pick the optimal action
            return self.policy[state]

    def update_policy(self, state, action):
        self.policy[state] = action

class MonteCarlo:
    """
    This class represents the first-visit Monte Carlo exploring starts control (policy optimization) algorithm.
    """

    def __init__(self, mdp, action_value_function, discount_rate, policy, max_episodes=100, update_epsilon=True):
        self.mdp = mdp
        self.action_value_function = action_value_function
        self.discount_rate = discount_rate
        self.policy = policy
        self.max_episodes = max_episodes
        self.update_epsilon = update_epsilon

    def run(self):
        total_visits = {
            state: {
                action: 0
                for action in self.mdp[state]
            } for state in self.mdp.get_state_actions()
        }

        for i in range(1, self.max_episodes + 1):
            logging.info(f"Episode {i}")
            logging.info("-" * 50)

            episode_visit = {
                state: {
                    action: False
                    for action in self.mdp[state]
                } for state in self.mdp.get_state_actions()
            }

            returns = {
                state: {
                    action: 0.
                    for action in self.mdp[state]
                } for state in self.mdp.get_state_actions()
            }

            path = list()
            is_terminal = False
            state = self.mdp.get_initial_state()
            while not is_terminal:
                action = self.policy.get_action(state)
                reward, next_state, is_terminal = self.mdp.get_reward_and_next_state(state, action)
                total_visits[state][action] += 1

                if not episode_visit[state][action]:
                    episode_visit[state][action] = True

                # update the total rewards for all the state-action pairs that have been taken in this episode
                for state_visited in returns:
                    if returns[state_visited] != None:
                        for action_taken in returns[state_visited]:
                            if episode_visit[state_visited][action_taken]:
                                # calculate the sum of all discounted rewards (i.e. G = y * G + R[t+1])
                                total_reward = self.discount_rate * returns[state_visited][action_taken] + reward

                                # calculate the incremental mean of the action-value function
                                # (i.e. Q(S,A) = Q(S,A) + 1/N(S,A) * (Gt - Q(S,A)))
                                current_value = self.action_value_function(state_visited, action_taken)
                                visit_count = total_visits[state_visited][action_taken]
                                new_value = current_value + (1/visit_count * (total_reward - current_value))
                                returns[state_visited][action_taken] = round(new_value, 1)
                                self.action_value_function.update_value(state_visited, action_taken, new_value)

                path.append(state)
                state = next_state

            # update policy
            if self.update_epsilon:
                self.policy.update_epsilon(1/i)

            for state in self.mdp.get_state_actions():
                if not self.mdp.is_terminal_state(state):
                    optimal_action = self.action_value_function.get_optimal_action(state)
                    self.policy.update_policy(state, optimal_action)

            logging.debug(f"{path}")
            logging.debug(f"{total_visits}")
            logging.debug(f"{returns}")
            logging.info(f"{self.action_value_function}")
            logging.info("")

        return self.action_value_function, self.policy()

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

        # TODO: build policy from value function and return it as 0th tuple element
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
    value_function = ValueFunctionTabular(mdp)
    algorithm = ValueIteration(mdp, value_function, args.discount_rate, args.delta_threshold)
    algorithm.run()

if __name__ == "__main__":
    init_logger()
    args = get_runtime_args()
    main(args)

